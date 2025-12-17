import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from typing import List, Dict, Any, Tuple
import pickle
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial


#植物分类器
class PlantClassifier:
    #分类器初始化
    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: int = 8,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 **kwargs):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=114514,
            n_jobs=-1,
            use_label_encoder=False,    # 消除警告
            eval_metric='mlogloss',     # 消除警告
            **kwargs
        )
        self.label_map = None
        self.reverse_label_map = None
    #分类器训练
    def fit(self, image_paths:List[str], labels:List[int], label_map:Dict[str,int], 
            val_image_paths:List[str] = None, val_labels:List[int] = None,
            use_augmentation: bool = True, early_stopping_rounds: int = 10, verbose: bool = True):
        """
        训练分类器
        
        参数:
            image_paths: 训练图像路径列表
            labels: 训练标签列表
            label_map: 标签映射字典
            val_image_paths: 验证集图像路径列表（用于 early stopping）
            val_labels: 验证集标签列表
            use_augmentation: 是否使用数据增强
            early_stopping_rounds: early stopping 轮数（如果验证集性能连续N轮不提升则停止）
            verbose: 是否显示训练过程
        """
        self.label_map = label_map
        self.reverse_label_map = {v:k for k,v in label_map.items()} #建立反向映射
        
        # 数据扩充策略：保留原图 + 增强版
        # 1. 先提取一遍"原图特征"（不增强）
        if verbose:
            print("提取原始图像特征...")
        features_orig = self.feature_extractor.extract_batch(image_paths, is_training=False)
        
        # 2. 如果需要增强，再额外提取一遍"增强后的特征"
        if use_augmentation:
            if verbose:
                print("提取增强后的图像特征...")
            features_aug = self.feature_extractor.extract_batch(image_paths, is_training=True)
            # 拼接特征：原图在前，增强在后
            features = np.vstack([features_orig, features_aug])
            # 标签也复制一份
            labels_all = list(labels) + list(labels)
            if verbose:
                print(f"数据扩充完成：原始 {len(labels)} 张 → 扩充后 {len(labels_all)} 张（2倍）")
        else:
            features = features_orig
            labels_all = labels
            if verbose:
                print(f"使用原始数据：{len(labels)} 张（未使用增强）")
        
        # 3. 标准化用"扩充后的训练特征"
        features_scaled = self.scaler.fit_transform(features)
        
        # 准备验证集（如果提供）
        eval_set = None
        if val_image_paths is not None and val_labels is not None:
            val_features = self.feature_extractor.extract_batch(val_image_paths, is_training=False)
            val_features_scaled = self.scaler.transform(val_features)  # 使用训练集的 scaler
            eval_set = [(val_features_scaled, val_labels)]
            if verbose:
                print(f"使用验证集进行 early stopping（轮数={early_stopping_rounds}）...")
        
        # 训练XGBoost模型（带 early stopping）
        # 对于旧版本 XGBoost (1.1.0-1.2.0)，可能不支持 early_stopping_rounds 参数
        # 只使用 eval_set 来监控验证集性能，XGBoost 会自动显示训练进度
        if eval_set is not None:
            # 只传递 eval_set，不传递 early_stopping_rounds（旧版本不支持）
            self.classifier.fit(features_scaled, labels_all, eval_set=eval_set)
            if verbose:
                n_est = getattr(self.classifier, 'n_estimators', '未知')
                print(f"注意：当前 XGBoost 版本可能不支持自动 early stopping，将训练 {n_est} 轮")
                print("训练过程中会显示验证集性能，可用于手动判断是否收敛")
        else:
            self.classifier.fit(features_scaled, labels_all)
        
        if verbose and eval_set is not None:
            best_iteration = getattr(self.classifier, 'best_iteration', None)
            best_score = getattr(self.classifier, 'best_score', None)
            if best_iteration is not None and best_score is not None:
                print(f"训练完成：最佳迭代轮数 = {best_iteration}, 最佳验证集损失 = {best_score:.4f}")
            elif verbose:
                print("训练完成（无法获取最佳迭代信息，可能是 XGBoost 版本限制）")
    #分类器预测
    def predict(self,image_paths: List[str]) -> List[str]:
        features = self.feature_extractor.extract_batch(image_paths)#提取训练特征
        features_scaled = self.scaler.transform(features)           #标准化特征
        prediction_labels = self.classifier.predict(features_scaled)#预测数字标签
        predictions = [self.reverse_label_map[label] for label in prediction_labels]#数字标签转换为标签名称
        return predictions

    # 保存模型
    def save(self, filepath: str):
        """保存训练好的模型"""
        model_data = {
            'scaler': self.scaler,
            'classifier': self.classifier,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'feature_extractor_target_size': self.feature_extractor.target_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ 模型已保存到：{filepath}")

    # 加载模型
    @classmethod
    def load(cls, filepath: str) -> 'PlantClassifier':
        """加载已保存的模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # 创建实例
        instance = cls()
        instance.scaler = model_data['scaler']
        instance.classifier = model_data['classifier']
        instance.label_map = model_data['label_map']
        instance.reverse_label_map = model_data['reverse_label_map']
        instance.feature_extractor = FeatureExtractor(target_size=model_data['feature_extractor_target_size'])

        # === 关键修复：重建 HOG 描述符 ===
        win_size = instance.feature_extractor.target_size  # 使用 target_size 而不是固定的 (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        instance.feature_extractor.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        
        # 计算并设置 HOG 特征向量长度
        # 对于方形图像，两个维度相同，取第一个维度即可
        blocks_per_dim = int(((win_size[0] - block_size[0]) // block_stride[0]) + 1)
        instance.feature_extractor.hog_feature_length = int(blocks_per_dim * blocks_per_dim * 4 * nbins)
        # ==================================

        print(f"✓ 模型已从 {filepath} 加载")
        return instance

#特征提取器
class FeatureExtractor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        # 预创建 HOG 描述符 - win_size 必须与 target_size 一致
        win_size = target_size  # 修复：使用 target_size 而不是固定的 (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        
        # 计算 HOG 特征向量长度（用于错误处理）
        # blocks_per_dim = ((target_size - block_size) / block_stride) + 1
        # 每个 block 有 4 个 cells (2x2)，每个 cell 有 nbins 个 bins
        # 对于方形图像，两个维度相同，取第一个维度即可
        blocks_per_dim = int(((target_size[0] - block_size[0]) // block_stride[0]) + 1)
        self.hog_feature_length = int(blocks_per_dim * blocks_per_dim * 4 * nbins)

    def extract_single(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.target_size)

        features = []

        # HOG
        hog_features = self.hog.compute(gray)
        if hog_features is not None:
            features.extend(hog_features.flatten().tolist())

        # 颜色直方图
        features.extend(self._extract_color_histogram(image))

        # 纹理特征
        features.extend(self._extract_texture_features(gray))

        return np.array(features, dtype=np.float32)

    def _extract_color_histogram(self, image: np.ndarray) -> List[float]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [4], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        return np.concatenate([hist_h, hist_s, hist_v]).tolist()

    def _extract_texture_features(self, gray_image: np.ndarray) -> List[float]:
        glcm = self._compute_glcm(gray_image)
        contrast = self._glcm_contrast(glcm)
        homogeneity = self._glcm_homogeneity(glcm)
        energy = self._glcm_energy(glcm)
        correlation = self._glcm_correlation(glcm)
        return [contrast, homogeneity, energy, correlation]

    def _compute_glcm(self, gray_image: np.ndarray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
        gray_quantized = (gray_image / 32).astype(np.uint8)
        h, w = gray_quantized.shape
        glcm_accum = np.zeros((8, 8))
        count = 0

        for d in distances:
            for angle in angles:
                dx = int(np.round(d * np.cos(angle)))
                dy = int(np.round(d * np.sin(angle)))
                glcm = np.zeros((8, 8))

                for i in range(h):
                    for j in range(w):
                        ni, nj = i + dy, j + dx
                        if 0 <= ni < h and 0 <= nj < w:
                            glcm[gray_quantized[i, j], gray_quantized[ni, nj]] += 1

                if glcm.sum() > 0:
                    glcm /= glcm.sum()
                    glcm_accum += glcm
                    count += 1

        return glcm_accum / count if count > 0 else glcm_accum

    def _glcm_contrast(self, glcm): return float(np.sum(glcm * np.square(np.arange(8)[:, None] - np.arange(8))))
    def _glcm_homogeneity(self, glcm): return float(np.sum(glcm / (1 + np.square(np.arange(8)[:, None] - np.arange(8)))))
    def _glcm_energy(self, glcm): return float(np.sum(glcm ** 2))
    def _glcm_correlation(self, glcm):
        i, j = np.indices(glcm.shape)
        mean_i = np.sum(i * glcm.sum(axis=1, keepdims=True))
        mean_j = np.sum(j * glcm.sum(axis=0, keepdims=True))
        std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm.sum(axis=1, keepdims=True)))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm.sum(axis=0, keepdims=True)))
        if std_i == 0 or std_j == 0:
            return 0.0
        return float(np.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j))

    # 关键修复：提供可 pickle 的批量提取函数
    def extract_batch(self, image_paths: List[str], is_training: bool = False) -> np.ndarray:
        """安全的并行批量特征提取（不序列化 self.hog）"""
        print(f"正在并行提取 {len(image_paths)} 张图片的特征...")
        # 创建一个轻量级函数，只传递必要参数，不包含 cv2 对象
        extract_func = partial(_extract_features_safe, target_size=self.target_size, is_training=is_training)
        features_list = Parallel(n_jobs=-1, backend="loky")(  # loky 最稳定
            delayed(extract_func)(path) for path in image_paths
        )
        return np.array(features_list, dtype=np.float32)


# 独立的纯函数，不包含任何不可 pickle 的对象
def _extract_features_safe(image_path: str, target_size: Tuple[int, int], is_training: bool = False) -> np.ndarray:
    """完全独立的可 pickle 函数，用于 joblib 并行"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 应用数据增强（如果是训练模式）
    # 使用轻量版增强，避免训练集和验证集分布差异过大
    augmentation_applied = False
    if is_training:
        try:
            from augmentation import augment_image_light
            image = augment_image_light(image, target_size=target_size, is_training=True)
            augmentation_applied = True
        except ImportError as e:
            # 如果 augmentation 模块不存在，跳过增强并记录警告
            import warnings
            warnings.warn(f"数据增强模块导入失败，跳过增强: {e}", UserWarning)
        except Exception as e:
            # 其他错误也记录
            import warnings
            warnings.warn(f"数据增强应用失败: {e}", UserWarning)
    
    # 如果未应用增强，需要确保图像尺寸正确
    if not augmentation_applied:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG（与训练时完全一致）- 修复：win_size 必须与 target_size 一致
    # 计算 HOG 特征向量长度
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    # 对于方形图像，两个维度相同，取第一个维度即可
    blocks_per_dim = int(((target_size[0] - block_size[0]) // block_stride[0]) + 1)
    hog_feature_length = int(blocks_per_dim * blocks_per_dim * 4 * nbins)
    
    hog = cv2.HOGDescriptor(target_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(gray)
    hog_feat = h.flatten() if h is not None else np.zeros(hog_feature_length, dtype=np.float32)

    # 颜色直方图（与原版一致）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [4], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    color_feat = np.concatenate([hist_h, hist_s, hist_v])

    # 纹理特征（完全复制原版简化实现）
    gray_quantized = (gray / 32).astype(np.uint8)
    glcm_accum = np.zeros((8, 8))
    count = 0
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    for d in distances:
        for angle in angles:
            dx = int(round(d * np.cos(angle)))
            dy = int(round(d * np.sin(angle)))
            glcm = np.zeros((8, 8))
            h, w = gray_quantized.shape

            for i in range(h):
                for j in range(w):
                    ni, nj = i + dy, j + dx
                    if 0 <= ni < h and 0 <= nj < w:
                        glcm[gray_quantized[i, j], gray_quantized[ni, nj]] += 1

            if glcm.sum() > 0:
                glcm /= glcm.sum()
                glcm_accum += glcm
                count += 1

    if count > 0:
        glcm = glcm_accum / count
    else:
        glcm = glcm_accum

    # 四个统计量（与训练时完全一致）
    i, j = np.indices((8, 8))
    contrast = float(np.sum(glcm * (i - j) ** 2))
    homogeneity = float(np.sum(glcm / (1 + (i - j) ** 2)))
    energy = float(np.sum(glcm ** 2))

    # correlation（与原版一致）
    mean_i = np.sum(i * glcm.sum(axis=1, keepdims=True))
    mean_j = np.sum(j * glcm.sum(axis=0, keepdims=True))
    std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm.sum(axis=1, keepdims=True)) + 1e-8)
    std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm.sum(axis=0, keepdims=True)) + 1e-8)
    correlation = float(np.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j)) if std_i > 0 and std_j > 0 else 0.0

    texture_feat = np.array([contrast, homogeneity, energy, correlation], dtype=np.float32)

    return np.concatenate([hog_feat, color_feat, texture_feat])