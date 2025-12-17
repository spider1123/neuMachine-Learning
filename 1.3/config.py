"""
配置文件
"""
import torch

class Config:
    # 数据路径
    train_dir = '../train'
    test_dir = '../test'
    output_csv = '../submission.csv'
    checkpoint_dir = 'checkpoints'
    best_model_path = 'checkpoints/best_model.pth'
    
    # 模型参数
    num_classes = 6
    input_size = 224  # ResNet50标准输入尺寸
    pretrained = True
    
    # 训练参数
    batch_size = 64
    num_workers = 4
    epochs = 200
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # 数据划分
    train_ratio = 0.8
    val_ratio = 0.2
    base_seed = 3407  # 基础随机种子，每轮使用 base_seed + epoch
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 其他
    save_best_only = True
    verbose = True


