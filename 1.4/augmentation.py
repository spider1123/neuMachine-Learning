# augmentation.py
import cv2
import numpy as np
from typing import Tuple, Optional
import random


def random_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """随机水平或垂直翻转图像"""
    if random.random() < p:
        flip_code = random.choice([-1, 0, 1])  # -1: 水平+垂直, 0: 垂直, 1: 水平
        return cv2.flip(image, flip_code)
    return image


def random_rotate(image: np.ndarray, angle_range: Tuple[int, int] = (-30, 30), p: float = 0.5) -> np.ndarray:
    """随机旋转图像"""
    if random.random() < p:
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
    return image


def random_crop_and_resize(image: np.ndarray, target_size: Tuple[int, int], 
                          scale_range: Tuple[float, float] = (0.7, 1.0), p: float = 0.8) -> np.ndarray:
    """随机裁剪并resize到目标尺寸"""
    if random.random() < p:
        h, w = image.shape[:2]
        scale = random.uniform(scale_range[0], scale_range[1])
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 确保裁剪尺寸不超过原图
        new_h = min(new_h, h)
        new_w = min(new_w, w)
        
        # 随机选择裁剪起点
        y = random.randint(0, max(0, h - new_h))
        x = random.randint(0, max(0, w - new_w))
        
        # 裁剪
        cropped = image[y:y+new_h, x:x+new_w]
        # Resize到目标尺寸
        return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        # 直接resize
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def random_affine(image: np.ndarray, p: float = 0.7) -> np.ndarray:
    """随机仿射变换（平移、缩放、旋转）"""
    if random.random() < p:
        h, w = image.shape[:2]
        
        # 随机参数
        shift_limit = 0.1
        scale_limit = 0.1
        rotate_limit = 30
        
        # 平移
        tx = random.uniform(-shift_limit, shift_limit) * w
        ty = random.uniform(-shift_limit, shift_limit) * h
        
        # 缩放
        scale = random.uniform(1 - scale_limit, 1 + scale_limit)
        
        # 旋转
        angle = random.uniform(-rotate_limit, rotate_limit)
        
        # 构建变换矩阵
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
    return image


def adjust_brightness_contrast(image: np.ndarray, 
                               brightness_limit: float = 0.3, 
                               contrast_limit: float = 0.3, 
                               p: float = 0.7) -> np.ndarray:
    """调整亮度和对比度"""
    if random.random() < p:
        # 转换为浮点数进行计算
        img_float = image.astype(np.float32)
        
        # 随机亮度和对比度调整
        brightness = random.uniform(-brightness_limit, brightness_limit) * 255
        contrast = random.uniform(1 - contrast_limit, 1 + contrast_limit)
        
        # 应用调整
        img_float = img_float * contrast + brightness
        
        # 裁剪到有效范围并转换回uint8
        img_float = np.clip(img_float, 0, 255)
        return img_float.astype(np.uint8)
    return image


def adjust_hsv(image: np.ndarray, 
              hue_shift_limit: int = 20, 
              sat_shift_limit: int = 30, 
              val_shift_limit: int = 20, 
              p: float = 0.7) -> np.ndarray:
    """调整HSV颜色空间"""
    if random.random() < p:
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 随机调整
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue_shift_limit, hue_shift_limit)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.uniform(-sat_shift_limit, sat_shift_limit), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.uniform(-val_shift_limit, val_shift_limit), 0, 255)
        
        # 转换回BGR
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image


def channel_shuffle(image: np.ndarray, p: float = 0.2) -> np.ndarray:
    """随机通道混洗"""
    if random.random() < p and len(image.shape) == 3:
        channels = [0, 1, 2]
        random.shuffle(channels)
        return image[:, :, channels]
    return image


def add_gaussian_noise(image: np.ndarray, var_limit: Tuple[int, int] = (10, 50), p: float = 0.3) -> np.ndarray:
    """添加高斯噪声"""
    if random.random() < p:
        var = random.uniform(var_limit[0], var_limit[1])
        noise = np.random.normal(0, var ** 0.5, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return image


def gaussian_blur(image: np.ndarray, blur_limit: int = 3, p: float = 0.3) -> np.ndarray:
    """高斯模糊"""
    if random.random() < p:
        # 确保核大小为奇数
        ksize = random.choice([3, 5]) if blur_limit >= 3 else 3
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    return image


def motion_blur(image: np.ndarray, blur_limit: int = 3, p: float = 0.3) -> np.ndarray:
    """运动模糊"""
    if random.random() < p:
        ksize = random.choice([3, 5]) if blur_limit >= 3 else 3
        # 创建运动模糊核
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize-1)/2), :] = np.ones(ksize)
        kernel = kernel / ksize
        return cv2.filter2D(image, -1, kernel)
    return image


def augment_image(image: np.ndarray, target_size: Tuple[int, int] = (128, 128), 
                 is_training: bool = True) -> np.ndarray:
    """
    对图像应用数据增强
    
    参数:
        image: 输入图像 (BGR格式，numpy array)
        target_size: 目标尺寸 (height, width)
        is_training: 是否为训练模式（True: 应用随机增强, False: 只resize）
    
    返回:
        增强后的图像
    """
    if not is_training:
        # 验证/测试模式：只做resize
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 训练模式：应用随机增强
    # 1. 随机裁剪并resize（在早期应用，避免后续变换导致尺寸问题）
    img = random_crop_and_resize(image, target_size, scale_range=(0.7, 1.0), p=0.8)
    
    # 2. 几何变换
    img = random_flip(img, p=0.5)
    img = random_rotate(img, angle_range=(-30, 30), p=0.5)
    img = random_affine(img, p=0.7)
    
    # 3. 颜色/光照增强
    img = adjust_brightness_contrast(img, brightness_limit=0.3, contrast_limit=0.3, p=0.7)
    img = adjust_hsv(img, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7)
    img = channel_shuffle(img, p=0.2)
    
    # 4. 噪声/模糊（随机选择一种）
    if random.random() < 0.3:
        if random.random() < 0.5:
            img = gaussian_blur(img, blur_limit=3, p=1.0)
        else:
            img = motion_blur(img, blur_limit=3, p=1.0)
    
    # 5. 添加噪声
    img = add_gaussian_noise(img, var_limit=(10, 50), p=0.3)
    
    # 6. 确保最终尺寸正确（防止前面的操作改变尺寸）
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    return img


def augment_image_light(image: np.ndarray, target_size: Tuple[int, int] = (128, 128), 
                        is_training: bool = True) -> np.ndarray:
    """
    轻量版数据增强（适合传统特征提取方法）
    
    只保留对模型有帮助、且会在真实数据中出现的变化：
    - 轻微裁剪 + 缩放
    - 水平翻转
    - 小范围亮度/对比度调整
    - 适度 HSV 调整
    
    不做：大角度旋转、强仿射、通道打乱、模糊、噪声、强 HSV 抖动
    
    参数:
        image: 输入图像 (BGR格式，numpy array)
        target_size: 目标尺寸 (height, width)
        is_training: 是否为训练模式（True: 应用随机增强, False: 只resize）
    
    返回:
        增强后的图像
    """
    if not is_training:
        # 验证/测试模式：只做resize
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 训练模式：应用轻量增强
    # 1. 轻微随机裁剪 + resize（缩放范围缩小，避免过度裁剪）
    img = random_crop_and_resize(image, target_size, scale_range=(0.9, 1.0), p=0.5)
    
    # 2. 水平翻转（不做垂直+双向，因为植物叶片通常有方向性）
    # 只做水平翻转，概率降低
    if random.random() < 0.5:
        img = cv2.flip(img, 1)  # 1 = 水平翻转
    
    # 3. 非常小的旋转（不超过 10 度），概率降低
    img = random_rotate(img, angle_range=(-10, 10), p=0.3)
    
    # 4. 亮度/对比度调整范围缩小（避免过度改变）
    img = adjust_brightness_contrast(img,
                                     brightness_limit=0.1,  # 原来是 0.3
                                     contrast_limit=0.1,    # 原来是 0.3
                                     p=0.5)                 # 概率降低
    
    # 5. 适度 HSV 调整（范围缩小，避免过度改变颜色）
    img = adjust_hsv(img,
                     hue_shift_limit=10,   # 原来是 20
                     sat_shift_limit=15,   # 原来是 30
                     val_shift_limit=10,   # 原来是 20
                     p=0.3)                # 概率降低
    
    # 确保最终尺寸正确
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    return img


def get_train_transforms(target_size: Tuple[int, int] = (128, 128)):
    """
    获取训练集增强函数
    
    参数:
        target_size: 目标尺寸 (height, width)
    
    返回:
        增强函数，接受 (image, is_training=True) 参数
    """
    def transform(image: np.ndarray) -> np.ndarray:
        return augment_image(image, target_size=target_size, is_training=True)
    return transform


def get_val_transforms(target_size: Tuple[int, int] = (128, 128)):
    """
    获取验证集/测试集变换函数
    
    参数:
        target_size: 目标尺寸 (height, width)
    
    返回:
        变换函数，接受 (image, is_training=False) 参数
    """
    def transform(image: np.ndarray) -> np.ndarray:
        return augment_image(image, target_size=target_size, is_training=False)
    return transform


