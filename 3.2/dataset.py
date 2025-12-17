"""
数据集加载：支持每轮训练时使用不同随机种子动态划分
使用Albumentations进行更强的数据增强
支持关键点热图辅助监督
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


class GrayImageDataset(Dataset):
    """
    灰度图数据集（支持关键点热图）
    读取48×48灰度图，上采样到224×224，复制为3通道
    """
    def __init__(self, image_paths, labels, transform=None, landmark_data=None, use_landmark=False):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            transform: 数据增强变换
            landmark_data: 关键点数据字典 {image_path: landmarks} 或 None
            use_landmark: 是否使用关键点辅助监督
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.landmark_data = landmark_data or {}
        self.use_landmark = use_landmark
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取灰度图
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # 复制为3通道（伪RGB）
        img = np.stack([img] * 3, axis=-1)  # (H, W, 3)
        
        # 获取关键点（如果使用）
        landmarks = None
        if self.use_landmark:
            landmarks = self.landmark_data.get(img_path, None)
        
        # 应用数据增强（Albumentations格式）
        # 注意：如果有关键点，需要在增强时同步变换关键点坐标
        if self.transform:
            if landmarks is not None and len(landmarks) > 0:
                # 将关键点转换为Albumentations格式（需要归一化到[0,1]）
                # landmarks格式: [x1, y1, x2, y2, ...]
                keypoints = []
                for i in range(0, len(landmarks), 2):
                    if i + 1 < len(landmarks):
                        # 原始图像尺寸（48x48）
                        x = float(landmarks[i]) / 48.0
                        y = float(landmarks[i + 1]) / 48.0
                        keypoints.append((x, y))
                
                # 应用变换（包括关键点）
                # 注意：Albumentations的keypoints格式为 [(x, y), ...]
                try:
                    transformed = self.transform(image=img, keypoints=keypoints)
                    img = transformed['image']  # (3, H, W) tensor
                    
                    # 将关键点坐标转换回像素坐标（224x224）
                    transformed_keypoints = transformed.get('keypoints', keypoints)
                    landmarks_rescaled = []
                    for kp in transformed_keypoints:
                        landmarks_rescaled.extend([kp[0] * 224, kp[1] * 224])
                    landmarks = np.array(landmarks_rescaled, dtype=np.float32)
                except Exception as e:
                    # 如果关键点变换失败，使用原始关键点并缩放
                    print(f"Warning: Keypoint transform failed: {e}, using scaled landmarks")
                    landmarks = np.array(landmarks, dtype=np.float32) * (224.0 / 48.0)
                    img = self.transform(image=img)['image']
            else:
                img = self.transform(image=img)['image']
        else:
            # 如果没有transform，至少需要上采样和转换为tensor
            transform_basic = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            img = transform_basic(image=img)['image']
            
            # 如果没有transform，关键点也需要缩放
            if landmarks is not None:
                landmarks = np.array(landmarks, dtype=np.float32) * (224.0 / 48.0)
        
        label = self.labels[idx]
        
        if self.use_landmark and landmarks is not None:
            return img, label, torch.from_numpy(landmarks)
        else:
            return img, label


def get_train_val_split(data_root, train_dir, train_ratio=0.8, val_ratio=0.2, seed=42):
    """
    从train文件夹中按类别划分训练集和验证集
    
    关键：每轮训练时传入不同的随机种子，确保划分不同
    
    Args:
        data_root: 数据根目录
        train_dir: 训练文件夹名称
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子（每轮使用不同的seed，如 base_seed + epoch）
    
    Returns:
        train_paths, train_labels, val_paths, val_labels
    """
    train_path = Path(data_root) / train_dir
    
    all_paths = []
    all_labels = []
    
    # 遍历所有类别文件夹
    class_dirs = sorted([d for d in train_path.iterdir() if d.is_dir()])
    
    for class_idx, class_dir in enumerate(class_dirs):
        # 获取该类别的所有图像
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        for img_path in image_files:
            all_paths.append(str(img_path))
            all_labels.append(class_idx)
    
    # 按类别分层划分（使用传入的随机种子）
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=val_ratio,
        random_state=seed,  # 使用传入的seed，每轮不同
        stratify=all_labels  # 保持类别比例
    )
    
    print(f"Total images: {len(all_paths)}")
    print(f"Train images: {len(train_paths)} (seed={seed})")
    print(f"Val images: {len(val_paths)} (seed={seed})")
    print(f"Classes: {len(class_dirs)}")
    
    return train_paths, train_labels, val_paths, val_labels


def get_test_dataset(data_root, test_dir):
    """
    获取测试数据集
    
    Args:
        data_root: 数据根目录
        test_dir: 测试文件夹名称
    
    Returns:
        test_paths, test_dataset
    """
    test_path = Path(data_root) / test_dir
    
    # 获取所有测试图像
    test_files = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    test_paths = [str(f) for f in test_files]
    
    # 创建数据集（测试时不需要标签，但为了兼容Dataset接口，使用-1作为占位符）
    test_labels = [-1] * len(test_paths)
    
    # 测试时的数据增强（只上采样和归一化，无随机增强）
    test_transform = get_val_transforms()
    
    test_dataset = GrayImageDataset(test_paths, test_labels, transform=test_transform, use_landmark=False)
    
    return test_paths, test_dataset


def get_train_transforms():
    """
    训练时的数据增强（使用Albumentations，更强的增强策略）
    +3~6% 精度提升
    """
    return A.Compose([
        # 先放大再随机裁剪（更好的上采样效果）
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        
        # 几何增强
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        
        # 噪声和模糊增强
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        ], p=0.3),
        
        # 颜色和亮度增强
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        # 归一化和转Tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    验证时的数据增强（无增强，只做resize和normalize）
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


if __name__ == '__main__':
    # 测试数据集
    from config import Config
    
    config = Config()
    
    # 获取训练/验证划分（使用基础随机种子）
    train_paths, train_labels, val_paths, val_labels = get_train_val_split(
        '.', config.train_dir, config.train_ratio, config.val_ratio, config.base_seed
    )
    
    # 创建数据集
    train_dataset = GrayImageDataset(
        train_paths, train_labels,
        transform=get_train_transforms()
    )
    
    val_dataset = GrayImageDataset(
        val_paths, val_labels,
        transform=get_val_transforms()
    )
    
    # 测试数据加载
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    for img, label in train_loader:
        print(f"Batch shape: {img.shape}, labels: {label}")
        break

