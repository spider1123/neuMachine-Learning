# tester.py

import torch
import pandas as pd
import os
from tqdm import tqdm
from config import cfg


def test_and_save_csv(model, test_loader, idx_to_class):
    """
    对测试集进行预测并保存为 CSV 文件
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_filenames = []
    all_preds = []
    
    print("\n开始测试集预测...")
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            
            # 收集文件名和预测结果
            all_filenames.extend(filenames)
            all_preds.extend(preds)
    
    # 将类别索引转换为类别名
    all_labels = [idx_to_class[pred] for pred in all_preds]
    
    # 创建 DataFrame
    df = pd.DataFrame({
        "ID": all_filenames,
        "Category": all_labels
    })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(cfg.PREDICTION_CSV) if os.path.dirname(cfg.PREDICTION_CSV) else ".", exist_ok=True)
    
    # 保存 CSV
    df.to_csv(cfg.PREDICTION_CSV, index=False, encoding='utf-8')
    print(f"\n预测结果已保存至: {cfg.PREDICTION_CSV}")
    print(f"共预测 {len(df)} 张图片")
    
    return df

