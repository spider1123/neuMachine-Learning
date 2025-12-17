"""
测试脚本：加载最佳模型，对test集进行预测，生成submission.csv
"""
import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import create_resnet50
from dataset import get_test_dataset


def main():
    config = Config()
    
    print(f'Using device: {config.device}')
    
    # 加载最佳模型权重
    if not os.path.exists(config.best_model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {config.best_model_path}')
    
    checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
    
    # 从checkpoint中读取配置（如果保存了的话），否则使用当前config
    # 确保模型结构与训练时一致
    use_landmark = checkpoint.get('use_landmark', getattr(config, 'use_landmark', True))
    num_landmarks = checkpoint.get('num_landmarks', getattr(config, 'num_landmarks', 98))
    
    # 创建模型（必须与训练时的结构一致）
    model = create_resnet50(
        num_classes=config.num_classes,
        num_landmarks=num_landmarks,
        use_landmark=use_landmark  # 必须与训练时一致
    )
    model = model.to(config.device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f'Loaded model from {config.best_model_path}')
    print(f'Model was trained for {checkpoint.get("epoch", "unknown")} epochs')
    print(f'Val Acc: {checkpoint.get("val_acc", "unknown"):.2f}%')
    
    model.eval()
    
    # 获取测试数据
    test_paths, test_dataset = get_test_dataset('.', config.test_dir)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f'Found {len(test_paths)} test images')
    
    # 预测
    predictions = []
    filenames = []
    
    pbar = tqdm(test_loader, desc='Predicting')
    with torch.no_grad():
        for images, _ in pbar:
            images = images.to(config.device)
            outputs = model(images)
            # 如果模型返回tuple（分类+热图），只取分类结果
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    
    # 提取文件名（不含路径）
    for img_path in test_paths:
        filename = Path(img_path).name
        filenames.append(filename)
    
    # 创建DataFrame并保存
    df = pd.DataFrame({
        'ID': filenames,
        'Emotion': predictions
    })
    
    # 按ID排序（如果需要）
    df = df.sort_values('ID')
    
    # 保存到CSV
    df.to_csv(config.output_csv, index=False)
    print(f'\nResults saved to {config.output_csv}')
    print(f'Total predictions: {len(predictions)}')
    print(f'Class distribution:')
    print(df['Emotion'].value_counts().sort_index())
    
    # 显示前几行
    print('\nFirst 10 predictions:')
    print(df.head(10))


if __name__ == '__main__':
    main()





