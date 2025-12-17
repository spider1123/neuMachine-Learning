# Augmentation.py

import torchvision.transforms as T
from config import cfg


def get_augmentation_transforms(num_augmentations=4):
    """
    生成多个不同的数据增强变换组合（温和版本，适合植物图像）
    
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
    
    # 定义多种温和的增强策略（针对植物图像优化）
    augmentation_strategies = [
        # 策略1: 轻微水平翻转 + 轻微颜色调整（最保守）
        T.Compose([
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.9, 1.0)),  # 保守裁剪
            T.RandomHorizontalFlip(p=0.5),  # 只做水平翻转
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),  # 轻微颜色调整
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略2: 轻微裁剪 + 轻微亮度对比度调整
        T.Compose([
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.85, 1.0)),  # 轻微裁剪
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),  # 只调整亮度和对比度
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略3: 小角度旋转 + 轻微裁剪
        T.Compose([
            T.RandomRotation(degrees=8),  # 小角度旋转（±8度）
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.88, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略4: 轻微缩放 + 轻微颜色调整
        T.Compose([
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.87, 1.0), ratio=(0.95, 1.05)),  # 轻微缩放
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略5: 非常轻微的旋转 + 保守裁剪
        T.Compose([
            T.RandomRotation(degrees=5),  # 非常小的旋转（±5度）
            T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            T.ToTensor(),
            T.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]),
        
        # 策略6: 只做水平翻转和轻微颜色调整（最保守）
        T.Compose([
            T.Resize(256),
            T.RandomCrop(cfg.IMG_SIZE),  # 随机裁剪但保持尺度
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 非常轻微
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
