# Augmentation.py

import torchvision.transforms as T
from config import cfg


def get_augmentation_transforms(num_augmentations=4):
    """
    生成多个不同的数据增强变换组合
    
    参数:
        num_augmentations: 需要生成的增强变换数量（默认4）
    
    返回:
        transforms_list: 包含原始变换和多个增强变换的列表
        第一个是原始图片的变换（只做基础预处理），后续是各种增强变换
    """
    # 基础预处理（用于原始图片）：只做 resize、crop 和归一化，不做随机增强
    base_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(cfg.IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])
    
    # 定义多种增强策略
    augmentation_strategies = [
        # 策略1: 随机旋转 + 颜色抖动
        T.Compose([
            T.RandomRotation(degrees=15),  # 随机旋转 ±15 度
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略2: 强颜色变换 + 翻转
        T.Compose([
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略3: 随机裁剪 + 旋转
        T.Compose([
            T.RandomRotation(degrees=20),
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略4: 多尺度裁剪 + 颜色变换
        T.Compose([
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略5: 旋转 + 强裁剪
        T.Compose([
            T.RandomRotation(degrees=25),
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略6: 轻微增强（保守策略）
        T.Compose([
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
    ]
    
    # 如果需要的增强数量超过预定义的策略数量，则循环使用
    transforms_list = [base_transform]  # 第一个是原始图片的变换
    
    for i in range(num_augmentations):
        strategy_idx = i % len(augmentation_strategies)
        transforms_list.append(augmentation_strategies[strategy_idx])
    
    return transforms_list


def get_augmentation_transform_by_index(index, num_augmentations=4):
    """
    根据索引获取对应的增强变换
    
    参数:
        index: 变换索引（0 表示原始图片，1-N 表示不同的增强）
        num_augmentations: 增强数量
    
    返回:
        transform: 对应的变换
    """
    transforms_list = get_augmentation_transforms(num_augmentations)
    if index < len(transforms_list):
        return transforms_list[index]
    else:
        # 如果索引超出范围，返回最后一个变换
        return transforms_list[-1]

