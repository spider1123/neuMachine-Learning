import os
import numpy as np
from pathlib import Path
from typing import List,Tuple,Dict

def load_dataset(train_dir:str = 'train') -> Tuple[List[str],List[int],Dict[str,int]]:#读取train文件夹，返回图片路径以及对应标签
    train_path = Path(train_dir)#加载训练集路径
    image_paths = []#图片路径列表
    labels = []#标签的数字映射
    label_map = {}#标签名称对应的数字对应映射
    for label_idx,class_dir in enumerate(sorted(train_path.iterdir())):
        if class_dir.is_dir():
            class_name = class_dir.name#记录标签名称
            label_map[class_name] = label_idx#记录标签数字映射
            for image_path in class_dir.glob('*.png'):
                image_paths.append(str(image_path))#记录图片路径
                labels.append(label_idx)#记录图片标签
    return image_paths,labels,label_map

def load_prediction(prediction_dir:str = 'test') -> List[str]:#读取test文件夹，返回图片文件路径
    prediction_path = Path(prediction_dir)
    image_paths = []
    for image_path in prediction_path.glob('*.png'):
        image_paths.append(str(image_path))
    return image_paths
    
if __name__ == "__main__":
    train_paths, train_labels, label_map = load_dataset()
    test_paths = load_prediction()
    print(f"训练图片数量: {len(train_paths)}")
    print(f"类别映射: {label_map}")
    print(f"测试图片数量: {len(test_paths)}")
    