"""
数据集加载：支持每轮训练时使用不同随机种子动态划分
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image


class GrayImageDataset(Dataset):
    """
    灰度图数据集
    读取48×48灰度图，上采样到224×224，复制为3通道
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            transform: 数据增强变换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取灰度图
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # 转换为PIL Image（便于使用torchvision的transforms）
        # 注意：PIL的灰度图是'L'模式，需要转换为'RGB'模式以便Normalize正常工作
        img = Image.fromarray(img).convert('RGB')  # 转换为RGB（3通道，但值相同）
        
        # 应用数据增强（包括上采样到224×224和Normalize）
        if self.transform:
            img = self.transform(img)
        else:
            # 如果没有transform，至少需要上采样和转换为tensor
            transform_basic = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            img = transform_basic(img)
        
        # 此时img应该是(3, 224, 224)的tensor（因为已经转换为RGB）
        # 如果仍然是单通道（不应该发生），复制为3通道
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)  # (1, 224, 224) -> (3, 224, 224)
        
        label = self.labels[idx]
        
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
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = GrayImageDataset(test_paths, test_labels, transform=test_transform)
    
    return test_paths, test_dataset


def get_train_transforms():
    """
    训练时的数据增强
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """
    验证时的数据增强（无随机增强）
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

