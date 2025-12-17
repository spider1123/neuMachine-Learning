# dataset_split.py

import os
from config import cfg


def get_class_to_idx():
    """
    扫描 RAW_DATA_DIR (train/) 目录，获取所有类别信息
    返回类别到索引的映射字典
    不再进行文件复制，只获取类别信息
    """
    raw_dir = cfg.RAW_DATA_DIR
    
    # 获取所有类别（文件夹名）
    classes = [d for d in sorted(os.listdir(raw_dir)) 
               if os.path.isdir(os.path.join(raw_dir, d))]
    
    # 构建类别到索引的映射
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"从 {raw_dir} 中找到 {len(classes)} 个类别: {classes}")
    
    # 统计每个类别的图片数量
    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  类别 {cls}: {len(images)} 张图片")
    
    return class_to_idx


if __name__ == "__main__":
    get_class_to_idx()
