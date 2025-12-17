IMG_DIR_TRAIN = "train"
IMG_DIR_TEST = "test"
CSV_GT = "fovea_localization_train_GT.csv"
XML_DIR = "train_location"

# 统一预处理尺寸（长边）
IMG_SIZE = 1024

# 二阶段 patch 与热力图尺寸
PATCH_SIZE = 384
HM_SIZE_STAGE1 = 64
HM_SIZE_STAGE2 = 96  # 提高 Stage2 热力图分辨率以提升精度

# 训练相关
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 3e-4  # 降低初始学习率，配合 Cosine 调度器
NUM_WORKERS = 4
NUM_FOLDS = 5

# 可见性阈值
VIS_THRESH = 0.5

# 训练策略
EARLY_STOP_PATIENCE = 10  # Early stopping 耐心值
GRAD_ACCUM_STEPS = 2  # 梯度累积步数（模拟更大 batch size）
USE_AMP = True  # 是否使用混合精度训练
LAMBDA_COORD = 0.5  # 坐标回归辅助损失权重




