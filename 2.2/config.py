# config.py

import os


class Config:
    # 数据集路径（请修改为你的实际路径）
    RAW_DATA_DIR = "train"           # 原始训练数据集，按类别文件夹组织
    TEST_DATA_DIR = "test"           # 测试集，无标签图片
    PROCESSED_DATA_DIR = "data/processed"  # 划分后的 train/val
    
    # 划分比例（从 train 中划分）
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    
    # 数据增强参数
    NUM_AUGMENTATIONS = 4  # 每张原始图片生成的增强图片数量（默认4）
    
    # 训练超参数
    BATCH_SIZE = 32  # 单 GPU 推荐值，显存充足可调大
    NUM_WORKERS = 4  # Windows 下建议较小值，避免多进程问题
    EPOCHS = 80      # 追求高精度，可适当增加
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    
    # 图像尺寸与归一化参数（ImageNet 统计值）
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # 模型选择：resnet50, efficientnet_b0, convnext_tiny, swin_t 等
    BACKBONE = "resnet50"
    
    # 保存路径
    MODEL_SAVE_DIR = "runs"
    PREDICTION_CSV = "runs/submission.csv"
    
    SEED = 42


cfg = Config()

