# dataloader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import Image
from data_preprocess import get_train_transform, get_val_test_transform
from Augmentation import get_augmentation_transforms
from color_segmentation import (
    preprocess_train_data,
    preprocess_test_data,
    check_preprocessing_status
)
from config import cfg


class TestDataset(Dataset):
    """自定义测试集 Dataset，用于加载无标签的测试图片"""
    
    def __init__(self, test_dir, transform=None):
        self.original_test_dir = test_dir
        
        # 检查是否需要颜色分割预处理
        if cfg.ENABLE_COLOR_SEGMENTATION:
            preprocessed_dir = os.path.join(cfg.COLOR_SEGMENTATION_DIR, "test")
            status = check_preprocessing_status()
            
            # 如果需要预处理或强制重新处理
            if not status['test']['has_files'] or cfg.FORCE_REPROCESS:
                print("\n开始颜色分割预处理（测试集）...")
                preprocess_test_data(force_reprocess=cfg.FORCE_REPROCESS)
                # 重新检查状态
                status = check_preprocessing_status()
            
            # 使用预处理后的目录
            if os.path.exists(preprocessed_dir) and status['test']['has_files']:
                test_dir = preprocessed_dir
                print(f"使用颜色分割预处理后的测试数据: {test_dir}")
            else:
                print(f"警告: 预处理目录不存在或无文件，使用原始测试数据")
        
        self.test_dir = test_dir
        self.transform = transform
        
        # 收集所有图片文件
        self.image_files = []
        for f in sorted(os.listdir(test_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(f)
        
        print(f"测试集共 {len(self.image_files)} 张图片")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.test_dir, filename)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, filename


def get_full_train_dataset(class_to_idx):
    """
    创建完整的训练数据集（从 train/ 目录读取所有图片）
    如果启用了颜色分割预处理，则从预处理后的目录加载
    返回 ImageFolder 数据集对象，不应用任何 transform（原始 PIL Image）
    """
    # 检查是否需要颜色分割预处理
    data_dir = cfg.RAW_DATA_DIR
    
    if cfg.ENABLE_COLOR_SEGMENTATION:
        # 检查预处理状态
        status = check_preprocessing_status()
        preprocessed_dir = os.path.join(cfg.COLOR_SEGMENTATION_DIR, "train")
        
        # 如果需要预处理或强制重新处理
        if not status['train']['has_files'] or cfg.FORCE_REPROCESS:
            print("\n开始颜色分割预处理（训练集）...")
            preprocess_train_data(force_reprocess=cfg.FORCE_REPROCESS)
            # 重新检查状态
            status = check_preprocessing_status()
        
        # 使用预处理后的目录
        if os.path.exists(preprocessed_dir) and status['train']['has_files']:
            data_dir = preprocessed_dir
            print(f"使用颜色分割预处理后的训练数据: {data_dir}")
        else:
            print(f"警告: 预处理目录不存在或无文件，使用原始数据: {cfg.RAW_DATA_DIR}")
    
    # ImageFolder 在不使用 transform 时返回 (PIL.Image, label) 元组
    full_dataset = ImageFolder(
        root=data_dir,
        transform=None  # 不应用任何变换，返回原始 PIL Image
    )
    
    print(f"完整训练数据集共 {len(full_dataset)} 张图片")
    return full_dataset


def create_epoch_loaders(full_dataset, epoch, class_to_idx):
    """
    为当前 epoch 创建训练集和验证集的 DataLoader
    使用分层随机划分，确保每个类别在 train/val 中的比例基本一致
    
    参数:
        full_dataset: 完整的训练数据集（ImageFolder 对象）
        epoch: 当前 epoch 编号（用于生成不同的随机种子）
        class_to_idx: 类别到索引的映射字典
    
    返回:
        train_loader, val_loader
    """
    # 使用 epoch 作为随机种子的一部分，确保每轮划分都不同
    random_seed = cfg.SEED + epoch
    
    # 获取所有样本的索引和标签
    all_indices = list(range(len(full_dataset)))
    
    # 获取标签：优先使用 targets 属性，如果没有则从样本中获取
    if hasattr(full_dataset, 'targets'):
        all_labels = full_dataset.targets
    else:
        # 兼容性处理：如果 ImageFolder 没有 targets 属性，从样本中获取
        all_labels = []
        for idx in all_indices:
            _, label = full_dataset[idx]
            all_labels.append(label)
    
    # 构建类别索引到样本索引的映射
    class_to_indices = {}
    for idx, label in enumerate(all_labels):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    # 对每个类别分别进行分层划分
    train_indices = []
    val_indices = []
    
    for class_idx, indices in class_to_indices.items():
        # 对每个类别按比例划分
        class_train, class_val = train_test_split(
            indices,
            test_size=cfg.VAL_RATIO,
            random_state=random_seed,
            shuffle=True
        )
        train_indices.extend(class_train)
        val_indices.extend(class_val)
    
    # 对索引列表进行排序（可选，但有助于可重复性）
    train_indices.sort()
    val_indices.sort()
    
    # 创建验证集的 transform（验证集不做增强）
    val_transform = get_val_test_transform()
    
    # 为训练集准备数据增强
    # 每张原始图片会生成 (1 + NUM_AUGMENTATIONS) 个样本：1个原始 + NUM_AUGMENTATIONS 个增强
    augmentation_transforms = get_augmentation_transforms(cfg.NUM_AUGMENTATIONS)
    
    # 创建一个支持数据增强的 Dataset 类
    class AugmentedTrainDataset(Dataset):
        def __init__(self, dataset, indices, augmentation_transforms):
            """
            参数:
                dataset: 原始数据集
                indices: 训练集的样本索引列表
                augmentation_transforms: 增强变换列表，第一个是原始图片的变换，后续是增强变换
            """
            self.dataset = dataset
            self.original_indices = indices  # 原始样本索引
            self.augmentation_transforms = augmentation_transforms
            # 每个原始样本会生成 (1 + NUM_AUGMENTATIONS) 个训练样本
            self.samples_per_original = len(augmentation_transforms)
        
        def __len__(self):
            return len(self.original_indices) * self.samples_per_original
        
        def __getitem__(self, idx):
            # 计算这是第几个原始样本，以及是该原始样本的第几个版本（0=原始，1-N=增强）
            original_idx_in_list = idx // self.samples_per_original
            version_idx = idx % self.samples_per_original
            
            # 获取原始数据集中的样本索引
            original_idx = self.original_indices[original_idx_in_list]
            
            # 从原始数据集获取样本（返回 PIL Image 和标签）
            image, label = self.dataset[original_idx]
            
            # 应用对应的 transform（原始图片或某个增强版本）
            transform = self.augmentation_transforms[version_idx]
            image = transform(image)
            
            return image, label
    
    # 创建验证集的 Dataset（不做增强）
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            # 获取原始数据集中的样本索引
            original_idx = self.indices[idx]
            # 从原始数据集获取样本（返回 PIL Image 和标签）
            image, label = self.dataset[original_idx]
            
            # 确保 transform 总是被应用（transform 会将 PIL Image 转换为 tensor）
            if self.transform is not None:
                image = self.transform(image)
            else:
                # 如果没有 transform，至少需要转换为 tensor
                from torchvision.transforms import ToTensor
                image = ToTensor()(image)
            
            return image, label
    
    # 创建训练集（带增强）和验证集
    train_subset = AugmentedTrainDataset(full_dataset, train_indices, augmentation_transforms)
    val_subset = TransformSubset(full_dataset, val_indices, val_transform)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_test_loader():
    """
    创建测试集的 DataLoader
    """
    test_dataset = TestDataset(
        test_dir=cfg.TEST_DATA_DIR,
        transform=get_val_test_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    return test_loader


def get_class_mappings(class_to_idx):
    """
    获取类别映射信息
    """
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return num_classes, idx_to_class
