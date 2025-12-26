import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2  # 用于后续crop（如果用两阶段）

# ==================== 1. 配置路径 ====================
DATA_DIR = '.'  # 修改为你的数据根目录
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_GT_CSV = os.path.join(DATA_DIR, 'fovea_localization_train_GT.csv')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')


# ==================== 2. 数据集定义 ====================
class FoveaDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, transform=None, is_test=False):
        self.img_dir = img_dir
        self.is_test = is_test
        self.transform = transform

        if not is_test:
            self.df = pd.read_csv(csv_file)
            self.image_names = self.df['data'].tolist()
            self.coords = self.df[['Fovea_X', 'Fovea_Y']].values.astype(np.float32)
        else:
            self.image_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.image_names.sort()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_name
        else:
            coord = self.coords[idx]
            # 如果坐标是(0,0)，表示不可见，保持原样
            return image, torch.tensor(coord, dtype=torch.float32)


# ==================== 3. 数据预处理 ====================
IMG_SIZE = 512  # 统一resize到512x512，保留长宽比可进一步提升

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==================== 4. 模型定义（ResNet50回归） ====================
class FoveaRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super(FoveaRegressor, self).__init__()
        backbone = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # 去除avgpool和fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 输出 (x, y)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


# ==================== 5. 训练函数 ====================
def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_model_path = 'best_fovea_model.pth'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, coords in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]'):
            images = images.to(device)
            coords = coords.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, coords in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                images = images.to(device)
                coords = coords.to(device)
                outputs = model(images)
                loss = criterion(outputs, coords)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}: Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'  >>> Best model saved with Val MSE: {best_val_loss:.4f}')

    return best_model_path


# ==================== 6. 主流程 ====================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载训练数据
    full_dataset = FoveaDataset(TRAIN_IMG_DIR, TRAIN_GT_CSV, transform=train_transform, is_test=False)

    # 划分训练/验证（80张太少，建议8:2）
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)

    # 验证集使用相同transform（不增强）
    val_dataset = FoveaDataset(TRAIN_IMG_DIR, TRAIN_GT_CSV, transform=test_transform, is_test=False)
    val_subset_noaug = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset_noaug, batch_size=8, shuffle=False, num_workers=4)

    # 训练模型
    model = FoveaRegressor(pretrained=True)
    best_model_path = train_model(model, train_loader, val_loader, num_epochs=150, device=device)

    # ==================== 7. 测试集预测 ====================
    print('Loading best model for inference...')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_dataset = FoveaDataset(TEST_IMG_DIR, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for image, img_name in tqdm(test_loader, desc='Predicting test'):
            image = image.to(device)
            pred = model(image).cpu().numpy()[0]

            # 坐标反归一化？本方案直接预测绝对坐标（因为resize后训练）
            # 注意：我们resize到512x512后训练，所以预测的是512x512图像上的坐标
            # 需要映射回原图大小！！（重要！）
            orig_path = os.path.join(TEST_IMG_DIR, img_name[0])
            orig_w, orig_h = Image.open(orig_path).size

            scale_x = orig_w / IMG_SIZE
            scale_y = orig_h / IMG_SIZE

            pred_x = pred[0] * scale_x
            pred_y = pred[1] * scale_y

            # 如果预测坐标接近(0,0)或超出图像，可加后处理判断是否为不可见
            if pred_x < 50 or pred_y < 50 or pred_x > orig_w - 50 or pred_y > orig_h - 50:
                pred_x, pred_y = 0.0, 0.0  # 可根据验证集调阈值

            predictions.append({
                'image': img_name[0],
                'Fovea_X': pred_x,
                'Fovea_Y': pred_y
            })

    # ==================== 8. 生成提交文件 ====================
    submit_rows = []
    for pred in predictions:
        img_name = pred['image'].rsplit('.', 1)[0]  # 去掉后缀
        submit_rows.append({'ImageID': f'{img_name}_Fovea_X', 'value': pred['Fovea_X']})
        submit_rows.append({'ImageID': f'{img_name}_Fovea_Y', 'value': pred['Fovea_Y']})

    submit_df = pd.DataFrame(submit_rows)
    submit_df.to_csv('submission.csv', index=False)
    print('submission.csv 已生成！')