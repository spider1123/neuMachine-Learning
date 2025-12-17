# data_preprocess.py

import torchvision.transforms as T
from config import cfg


def get_train_transform():
    """
    训练集图像增强与预处理流水线。

    主要目的：
    1. 通过随机裁剪、翻转和颜色扰动，提升模型对尺度、方向和光照变化的鲁棒性（Data Augmentation）。
    2. 将图像转为张量并按 ImageNet 统计量做归一化，便于使用预训练模型。
    """
    return T.Compose([
        # 随机裁剪并缩放到 cfg.IMG_SIZE×cfg.IMG_SIZE
        # scale=(0.8, 1.0) 表示随机裁掉 0%~20% 的区域，相当于做随机缩放与平移
        T.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.8, 1.0)),

        # 以 50% 概率做水平翻转，增强模型对左右翻转的鲁棒性
        T.RandomHorizontalFlip(p=0.5),

        # 以 50% 概率做垂直翻转，有些植物叶片上下翻转不会改变类别，适当增强泛化能力
        T.RandomVerticalFlip(p=0.5),

        # 颜色抖动：随机改变亮度、对比度和饱和度
        # 可以模拟不同光照条件和拍摄设备，防止模型过拟合特定光照
        T.ColorJitter(
            brightness=0.4,   # 亮度变化范围
            contrast=0.4,     # 对比度变化范围
            saturation=0.4    # 饱和度变化范围
        ),

        # 将 PIL Image / numpy.ndarray 转换为 [C, H, W] 格式的张量，并归一化到 [0, 1]
        T.ToTensor(),

        # 使用 ImageNet 的均值和方差做标准化
        # 这样可以更好地利用在 ImageNet 上预训练的模型（如 ResNet、EfficientNet 等）
        T.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


def get_val_test_transform():
    """
    验证集和测试集的预处理流水线（不做随机增强，只做确定性变换）。

    主要目的：
    1. 保持输入稳定、可复现，方便评估模型真实性能。
    2. 尺度调整到与训练阶段一致的大小，并做同样的归一化。
    """
    return T.Compose([
        # 先将短边缩放到 256 像素，长边按比例缩放
        # 这样可以避免图像被拉伸变形，同时减小后续裁剪的尺度差异
        T.Resize(256),

        # 从中心裁剪出 cfg.IMG_SIZE×cfg.IMG_SIZE 的区域
        # 与训练时的随机裁剪在尺度上保持一致，但不再随机，保证评估稳定
        T.CenterCrop(cfg.IMG_SIZE),

        # 转换为张量，并将像素值从 [0, 255] 归一化到 [0, 1]
        T.ToTensor(),

        # 使用与训练集相同的均值和方差标准化，保证特征分布一致
        T.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])

