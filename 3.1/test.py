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
    
    # 创建模型
    model = create_resnet50(num_classes=config.num_classes, pretrained=False)
    model = model.to(config.device)
    
    # 加载最佳模型权重
    if not os.path.exists(config.best_model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {config.best_model_path}')
    
    checkpoint = torch.load(config.best_model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
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


