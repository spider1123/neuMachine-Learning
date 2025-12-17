"""
训练脚本：每轮训练时使用不同的随机种子重新划分数据
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import Config
from model import create_resnet50
from dataset import get_train_val_split, GrayImageDataset, get_train_transforms, get_val_transforms


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss / len(train_loader):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc='Validating')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss / len(val_loader):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    config = Config()
    
    # 设置随机种子（用于模型初始化等）
    torch.manual_seed(config.base_seed)
    np.random.seed(config.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.base_seed)
    
    print(f'Using device: {config.device}')
    print(f'Base seed: {config.base_seed}')
    
    # 创建输出目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 创建模型
    model = create_resnet50(num_classes=config.num_classes, pretrained=config.pretrained)
    model = model.to(config.device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # 训练循环
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch+1}/{config.epochs}')
        print('-' * 50)
        
        # 关键：每轮训练开始时，使用不同的随机种子重新划分数据
        # 使用 base_seed + epoch 确保每轮的划分不同
        current_seed = config.base_seed + epoch
        print(f'Using seed: {current_seed} for data split')
        
        # 重新划分训练集和验证集
        train_paths, train_labels, val_paths, val_labels = get_train_val_split(
            '.', config.train_dir, 
            config.train_ratio, config.val_ratio, 
            seed=current_seed  # 每轮使用不同的随机种子
        )
        
        # 创建数据集和数据加载器
        train_dataset = GrayImageDataset(
            train_paths, train_labels,
            transform=get_train_transforms()
        )
        
        val_dataset = GrayImageDataset(
            val_paths, val_labels,
            transform=get_val_transforms()
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)
        
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, config.best_model_path)
            print(f'✓ Saved best model (Val Acc: {val_acc:.2f}%)')
    
    print(f'\nTraining completed!')
    print(f'Best Val Accuracy: {best_val_acc:.2f}%')


if __name__ == '__main__':
    main()


