# main.py

import os
import torch
from config import cfg
from dataset_split import get_class_to_idx
from dataloader import get_full_train_dataset, get_test_loader, get_class_mappings
from model import get_model
from trainer import train_model
from tester import test_and_save_csv


def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    print("=" * 60)
    print("植物图像分类实验")
    print("=" * 60)
    
    # 1. 设置随机种子
    set_seed(cfg.SEED)
    print(f"随机种子: {cfg.SEED}")
    
    # 2. 扫描 train/ 目录获取类别信息
    print("\n步骤 1: 扫描数据集...")
    class_to_idx = get_class_to_idx()
    num_classes, idx_to_class = get_class_mappings(class_to_idx)
    
    # 3. 加载完整的训练数据集（不划分，只在内存中）
    print("\n步骤 2: 加载完整训练数据集到内存...")
    full_train_dataset = get_full_train_dataset(class_to_idx)
    
    # 4. 加载测试集
    print("\n步骤 3: 加载测试集...")
    test_loader = get_test_loader()
    
    # 5. 定义模型
    print("\n步骤 4: 构建模型...")
    model = get_model(num_classes)
    
    # 6. 训练（每轮训练前会重新划分 train/val）
    print("\n步骤 5: 开始训练...")
    train_model(model, full_train_dataset, class_to_idx, num_classes, idx_to_class)
    
    # 7. 测试并保存结果
    print("\n步骤 6: 加载最佳模型进行测试...")
    checkpoint_path = os.path.join(cfg.MODEL_SAVE_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 使用 checkpoint 中保存的 idx_to_class（更可靠）
        idx_to_class = checkpoint.get('idx_to_class', idx_to_class)
        print(f"已加载最佳模型 (验证集准确率: {checkpoint.get('acc', 0):.4f})")
    else:
        print("警告: 未找到保存的模型，使用当前模型进行测试")
    
    test_and_save_csv(model, test_loader, idx_to_class)
    
    print("\n" + "=" * 60)
    print("全部实验完成！")
    print("=" * 60)
