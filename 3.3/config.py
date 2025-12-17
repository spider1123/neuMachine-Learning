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
    
    # 关键点热图辅助监督参数
    use_landmark = True  # 是否使用关键点辅助监督
    num_landmarks = 68   # 关键点数量（当前实现对68点支持最稳定，建议保持一致）
    landmark_weight = 0.3  # 关键点损失权重（总损失 = 0.7 * CE + 0.3 * MSE）
    landmark_dir = None  # 关键点标注目录（None则自动生成，或指定.npy/.json文件路径）
    
    # 训练参数
    batch_size = 128  # 增大batch size以提升稳定性
    num_workers = 4
    epochs = 350  # 更长的训练轮数
    learning_rate = 3e-4  # 使用AdamW推荐的学习率
    weight_decay = 0.05  # AdamW推荐权重衰减
    label_smoothing = 0.1  # 标签平滑（+1~2%精度）
    
    # 学习率调度器参数
    use_onecycle = True  # 使用OneCycleLR（推荐）
    max_lr = 3e-4  # OneCycleLR最大学习率
    pct_start = 0.1  # 前10%用于warmup
    
    # 数据划分
    train_ratio = 0.8
    val_ratio = 0.2
    base_seed = 3407  # 随机种子，用于固定数据划分
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 其他
    save_best_only = True
    verbose = True



