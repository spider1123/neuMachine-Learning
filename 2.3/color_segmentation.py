# color_segmentation.py

import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from config import cfg


def extract_green_plant(image, hsv_lower=None, hsv_upper=None, use_morphology=None):
    """
    从图像中提取绿色植株，将非绿色区域置为白色
    
    参数:
        image: PIL Image 对象（RGB格式）
        hsv_lower: HSV下界 [H, S, V]，如果为None则使用cfg中的配置
        hsv_upper: HSV上界 [H, S, V]，如果为None则使用cfg中的配置
        use_morphology: 是否使用形态学操作，如果为None则使用cfg中的配置
    
    返回:
        PIL Image 对象（RGB格式），非绿色区域为白色
    """
    # 使用配置中的默认值
    if hsv_lower is None:
        hsv_lower = np.array(cfg.GREEN_HSV_LOWER, dtype=np.uint8)
    else:
        hsv_lower = np.array(hsv_lower, dtype=np.uint8)
    
    if hsv_upper is None:
        hsv_upper = np.array(cfg.GREEN_HSV_UPPER, dtype=np.uint8)
    else:
        hsv_upper = np.array(hsv_upper, dtype=np.uint8)
    
    if use_morphology is None:
        use_morphology = cfg.USE_MORPHOLOGY
    
    # 将PIL Image转换为numpy数组（RGB格式）
    img_array = np.array(image.convert('RGB'))
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # 创建绿色掩码
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    
    # 可选：使用形态学操作优化掩码
    if use_morphology:
        # 开运算：去除小噪点
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # 闭运算：填充小孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 将掩码转换为3通道，便于与RGB图像混合
    mask_3channel = np.stack([mask, mask, mask], axis=2) / 255.0
    
    # 创建白色背景
    white_background = np.ones_like(img_array) * 255
    
    # 混合：绿色区域保持原图，非绿色区域为白色
    result = (img_array * mask_3channel + white_background * (1 - mask_3channel)).astype(np.uint8)
    
    # 转换回PIL Image
    result_image = Image.fromarray(result, 'RGB')
    
    return result_image


def preprocess_directory(input_dir, output_dir, force_reprocess=False):
    """
    对目录中的所有图片进行颜色分割预处理
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        force_reprocess: 是否强制重新处理（即使输出文件已存在）
    
    返回:
        tuple: (处理的图片数量, 跳过的图片数量, 失败的图片数量)
    """
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # 收集所有图片文件
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(root, input_dir)
                image_files.append((root, file, rel_path))
    
    if len(image_files) == 0:
        print(f"在 {input_dir} 中未找到图片文件")
        return 0, 0, 0
    
    print(f"找到 {len(image_files)} 张图片，开始颜色分割预处理...")
    
    # 处理每张图片
    for root, file, rel_path in tqdm(image_files, desc="处理图片"):
        input_path = os.path.join(root, file)
        
        # 构建输出路径，保持目录结构
        if rel_path == '.':
            output_subdir = output_dir
        else:
            output_subdir = os.path.join(output_dir, rel_path)
        
        os.makedirs(output_subdir, exist_ok=True)
        
        # 输出文件名（保持原格式，或统一为PNG）
        output_filename = file
        output_path = os.path.join(output_subdir, output_filename)
        
        # 检查是否已处理
        if not force_reprocess and os.path.exists(output_path):
            skipped_count += 1
            continue
        
        # 处理图片
        try:
            # 读取原始图片
            original_image = Image.open(input_path).convert('RGB')
            
            # 应用颜色分割
            processed_image = extract_green_plant(original_image)
            
            # 保存处理后的图片
            processed_image.save(output_path, quality=95)
            
            processed_count += 1
        except Exception as e:
            print(f"处理图片 {input_path} 时出错: {type(e).__name__}: {str(e)}")
            failed_count += 1
    
    return processed_count, skipped_count, failed_count


def preprocess_train_data(force_reprocess=False):
    """
    预处理训练数据
    
    参数:
        force_reprocess: 是否强制重新处理
    
    返回:
        tuple: (处理的图片数量, 跳过的图片数量, 失败的图片数量)
    """
    input_dir = cfg.RAW_DATA_DIR
    output_dir = os.path.join(cfg.COLOR_SEGMENTATION_DIR, "train")
    
    print(f"\n开始颜色分割预处理（训练集）...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    processed, skipped, failed = preprocess_directory(
        input_dir, output_dir, force_reprocess
    )
    
    print(f"\n训练数据预处理完成:")
    print(f"  处理: {processed} 张")
    print(f"  跳过: {skipped} 张（已存在）")
    print(f"  失败: {failed} 张")
    
    if failed > 0:
        print(f"\n警告: {failed} 张图片处理失败，请检查错误信息")
    
    return processed, skipped, failed


def preprocess_test_data(force_reprocess=False):
    """
    预处理测试数据
    
    参数:
        force_reprocess: 是否强制重新处理
    
    返回:
        tuple: (处理的图片数量, 跳过的图片数量, 失败的图片数量)
    """
    input_dir = cfg.TEST_DATA_DIR
    output_dir = os.path.join(cfg.COLOR_SEGMENTATION_DIR, "test")
    
    print(f"\n开始颜色分割预处理（测试集）...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    processed, skipped, failed = preprocess_directory(
        input_dir, output_dir, force_reprocess
    )
    
    print(f"\n测试数据预处理完成:")
    print(f"  处理: {processed} 张")
    print(f"  跳过: {skipped} 张（已存在）")
    print(f"  失败: {failed} 张")
    
    if failed > 0:
        print(f"\n警告: {failed} 张图片处理失败，请检查错误信息")
    
    return processed, skipped, failed


def check_preprocessing_status():
    """
    检查颜色分割预处理的完成状态
    
    返回:
        dict: 包含训练集和测试集的预处理状态
    """
    train_output = os.path.join(cfg.COLOR_SEGMENTATION_DIR, "train")
    test_output = os.path.join(cfg.COLOR_SEGMENTATION_DIR, "test")
    
    train_exists = os.path.exists(train_output)
    test_exists = os.path.exists(test_output)
    
    # 检查是否有文件
    train_has_files = False
    test_has_files = False
    
    if train_exists:
        train_files = [f for r, d, files in os.walk(train_output) for f in files 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_has_files = len(train_files) > 0
    
    if test_exists:
        test_files = [f for r, d, files in os.walk(test_output) for f in files 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        test_has_files = len(test_files) > 0
    
    return {
        'train': {'exists': train_exists, 'has_files': train_has_files},
        'test': {'exists': test_exists, 'has_files': test_has_files}
    }


