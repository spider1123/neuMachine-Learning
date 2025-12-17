IMG_DIR_TRAIN = "train"
IMG_DIR_TEST = "test"
CSV_GT = "fovea_localization_train_GT.csv"
XML_DIR = "train_location"

# 统一预处理尺寸（长边）
IMG_SIZE = 1024

# 二阶段 patch 与热力图尺寸
PATCH_SIZE = 384
HM_SIZE_STAGE1 = 64
HM_SIZE_STAGE2 = 64

# 训练相关
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-3
NUM_WORKERS = 4
NUM_FOLDS = 5

# 可见性阈值
VIS_THRESH = 0.5




