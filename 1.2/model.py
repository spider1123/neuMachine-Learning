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
    def fit(self, image_paths:List[str], labels:List[int], label_map:Dict[str,int]):
        self.label_map = label_map
        self.reverse_label_map = {v:k for k,v in label_map.items()} #建立反向映射
        features = self.feature_extractor.extract_batch(image_paths)#提取训练特征
        features_scaled = self.scaler.fit_transform(features)               #标准化特征
        self.classifier.fit(features_scaled,labels)                        #训练XGBoost模型
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
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        instance.feature_extractor.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        # ==================================

        print(f"✓ 模型已从 {filepath} 加载")
        return instance

#特征提取器
class FeatureExtractor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        # 预创建 HOG 描述符
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

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
    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        """安全的并行批量特征提取（不序列化 self.hog）"""
        print(f"正在并行提取 {len(image_paths)} 张图片的特征...")
        # 创建一个轻量级函数，只传递必要参数，不包含 cv2 对象
        extract_func = partial(_extract_features_safe, target_size=self.target_size)
        features_list = Parallel(n_jobs=-1, backend="loky")(  # loky 最稳定
            delayed(extract_func)(path) for path in image_paths
        )
        return np.array(features_list, dtype=np.float32)


# 独立的纯函数，不包含任何不可 pickle 的对象
def _extract_features_safe(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """完全独立的可 pickle 函数，用于 joblib 并行"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)

    # HOG（与训练时完全一致）
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
    h = hog.compute(gray)
    hog_feat = h.flatten() if h is not None else np.zeros(3780, dtype=np.float32)

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