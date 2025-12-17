import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from typing import List, Dict, Tuple
import pickle
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
import time


# ==================== 核心预处理函数 ====================
def _preprocess_plant_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """输入BGR原图 → 返回增强灰度图 + 背景置0的彩色图"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    r, g, b = cv2.split(image)

    # Excess Green Index
    exg = 2 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
    exg = np.clip((exg - exg.min()) / (np.ptp(exg) + 1e-6) * 255, 0, 255).astype(np.uint8)

    # 多通道绿色掩码融合
    mask1 = cv2.inRange(hsv, (30, 30, 30), (90, 255, 255))
    mask2 = cv2.inRange(hsv, (20, 20, 20), (100, 255, 255))
    mask3 = cv2.inRange(lab[:, :, 1], 120, 160)  # a通道偏绿
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)
    mask = cv2.bitwise_or(mask, exg)

    # 形态学清理 + 保留最大连通域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        sizes = np.bincount(labels.ravel())[1:]
        largest = np.argmax(sizes) + 1
        mask = (labels == largest).astype(np.uint8) * 255

    # 轻微膨胀防止边缘被切掉
    mask = cv2.dilate(mask, kernel, iterations=3)

    # 背景置纯黑
    clean_bgr = image.copy()
    clean_bgr[mask == 0] = 0
    gray_enhanced = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)

    return gray_enhanced, clean_bgr


# ==================== 安全并行特征提取函数 ====================
def _extract_features_safe(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # ============ 1. 关键预处理：去除背景 ============
    gray, clean_bgr = _preprocess_plant_image(image)
    gray = cv2.resize(gray, target_size)
    clean_bgr = cv2.resize(clean_bgr, target_size)
    hsv = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2HSV)

    features = []

    # ============ 2. HOG ============
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
    h = hog.compute(gray)
    features.extend(h.flatten() if h is not None else np.zeros(3780, dtype=np.float32))

    # ============ 3. 颜色直方图 + 统计量 ============
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    features.extend(np.concatenate([hist_h, hist_s, hist_v]))

    # 颜色通道统计量
    h, s, v = cv2.split(hsv)
    for chan in [h.astype(float), s.astype(float), v.astype(float)]:
        features.extend([chan.mean(), chan.std(), skew(chan.ravel()), kurtosis(chan.ravel())])

    # ============ 4. LBP ============
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    features.extend(lbp_hist.astype(np.float32))

    # ============ 5. 形状特征（轮廓 + Hu矩）===========
    mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        convexity = area / hull_area if hull_area > 0 else 0
        moments = cv2.moments(mask)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-8)
        shape_feat = np.array([area, perimeter, circularity, convexity, *hu], dtype=np.float32)
    else:
        shape_feat = np.zeros(11, dtype=np.float32)
    features.extend(shape_feat)

    # ============ 6. GLCM纹理 ============
    gray_q = (gray // 32).astype(np.uint8)
    glcm = np.zeros((8, 8))
    offsets = [(1,0), (0,1), (1,1), (-1,1)]
    for dx, dy in offsets:
        temp = np.zeros((8, 8))
        for i in range(gray_q.shape[0]):
            for j in range(gray_q.shape[1]):
                ni, nj = i + dy, j + dx
                if 0 <= ni < gray_q.shape[0] and 0 <= nj < gray_q.shape[1]:
                    temp[gray_q[i,j], gray_q[ni,nj]] += 1
        if temp.sum() > 0:
            glcm += temp / temp.sum()
    glcm /= len(offsets)

    i, j = np.indices((8,8))
    contrast = np.sum(glcm * (i - j) ** 2)
    homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
    energy = np.sum(glcm ** 2)
    corr_num = np.sum(glcm * (i - np.sum(i*glcm)) * (j - np.sum(j*glcm)))
    corr_den = np.sqrt(np.sum(glcm * (i - np.sum(i*glcm))**2) * np.sum(glcm * (j - np.sum(j*glcm))**2)) + 1e-8
    correlation = corr_num / corr_den if corr_den > 0 else 0
    features.extend([contrast, homogeneity, energy, correlation])

    return np.array(features, dtype=np.float32)


# ==================== 特征提取器 ====================
class FeatureExtractor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size

    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        print(f"      并行提取 {len(image_paths)} 张特征...", end="")
        start = time.time()
        extract_func = partial(_extract_features_safe, target_size=self.target_size)
        feats = Parallel(n_jobs=-1, backend="loky")(
            delayed(extract_func)(p) for p in image_paths
        )
        arr = np.array(feats, dtype=np.float32)
        print(f" 完成 ({arr.shape[1]}维, {time.time()-start:.1f}s)")
        return arr


# ==================== 主分类器 ====================
class PlantClassifier:
    def __init__(self, n_estimators=500, max_depth=8, learning_rate=0.05, **kwargs):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.9, colsample_bytree=0.9, random_state=114514,
            n_jobs=-1, eval_metric='mlogloss', **kwargs
        )
        self.label_map = None
        self.reverse_label_map = None
        self.selected_features = None

    def fit(self, image_paths: List[str], labels: List[int], label_map: Dict[str, int]):
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}

        features = self.feature_extractor.extract_batch(image_paths)
        features_scaled = self.scaler.fit_transform(features)

        # === 特征选择：保留累计重要性前98% ===
        print("    - 初步训练用于特征选择...")
        temp_clf = XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=114514)
        temp_clf.fit(features_scaled, labels)
        importances = temp_clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        cumsum = np.cumsum(importances[sorted_idx])
        thresh = np.where(cumsum > 0.98)[0][0]
        self.selected_features = sorted_idx[:thresh + 1]
        print(f"    - 特征选择：{features_scaled.shape[1]} → {len(self.selected_features)} 维")

        features_final = features_scaled[:, self.selected_features]
        print(f"    - 正式训练最终模型（{features_final.shape[1]}维特征）...")
        self.classifier.fit(features_final, labels)
        print("    - 训练完成！")

    def predict(self, image_paths: List[str]) -> List[str]:
        features = self.feature_extractor.extract_batch(image_paths)
        features_scaled = self.scaler.transform(features)
        if self.selected_features is not None:
            features_scaled = features_scaled[:, self.selected_features]
        pred = self.classifier.predict(features_scaled)
        return [self.reverse_label_map[p] for p in pred]

    def save(self, filepath: str):
        data = {
            'scaler': self.scaler,
            'classifier': self.classifier,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'selected_features': self.selected_features,
            'target_size': self.feature_extractor.target_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"模型已保存 → {filepath}")

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls()
        instance.scaler = data['scaler']
        instance.classifier = data['classifier']
        instance.label_map = data['label_map']
        instance.reverse_label_map = data['reverse_label_map']
        instance.selected_features = data.get('selected_features')
        instance.feature_extractor = FeatureExtractor(target_size=data.get('target_size', (128,128)))
        print(f"模型已加载 ← {filepath}")
        return instance