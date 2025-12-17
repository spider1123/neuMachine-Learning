import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from typing import List, Dict, Any, Tuple
import pickle
from pathlib import Path

#植物分类器
class PlantClassifier:
    #分类器初始化
    def __init__(self,n_estimators:int = 100,max_depth:int =6,learning_rate:float = 0.01):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = XGBClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            learning_rate = learning_rate,
            random_state = 114514,
            n_jobs = -1
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
        features_scaled = self.scaler.fit_transform(features)       #标准化特征
        prediction_labels = self.classifier.predict(features_scaled)#预测数字标签
        predictions = [self.reverse_label_map[label] for label in prediction_labels]#数字标签转换为标签名称 
        return predictions
#特征提取器
class FeatureExtractor:
    #提取器初始化
    def __init__(self, target_size:Tuple[int,int]=(128,128)):
        self.target_size = target_size
    #从单张图片提取特征
    def extract_single(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)#读取图片
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#转换为灰度图
        gray = cv2.resize(gray,self.target_size)#调整尺寸
        features = []
        #hog特征
        hog_features = self._extract_hog(gray)
        features.extend(hog_features)
        #HSV空间
        color_features = self._extract_color_histogram(image)
        features.extend(color_features)
        #纹理特征
        texture_features = self._extract_texture_features(gray)
        features.extend(texture_features)

        return np.array(features)
    #批量提取图片特征
    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        features_list = []
        for i, path in enumerate(image_paths):
            features = self.extract_single(path)
            features_list.append(features)
            if (i + 1) % 10 == 0:  
                print(f"已处理 {i + 1}/{len(image_paths)} 张图片")
        return np.array(features_list)
    #提取HOG特征
    def _extract_hog(self, gray_image: np.ndarray) -> List[float]:
        # HOG参数
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        # 创建HOG描述符
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        # 计算HOG特征
        hog_features = hog.compute(gray_image)
        return hog_features.flatten().tolist()
    #HSV空间
    def _extract_color_histogram(self, image: np.ndarray) -> List[float]:
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 计算颜色直方图
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])  # 色调 Hue
        hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256])  # 饱和度 Saturation
        hist_v = cv2.calcHist([hsv], [2], None, [4], [0, 256])  # 亮度 Value
        # 归一化
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        return np.concatenate([hist_h, hist_s, hist_v]).tolist()
    #提取纹理特征
    def _extract_texture_features(self, gray_image: np.ndarray) -> List[float]:
        # 计算GLCM (Gray Level Co-occurrence Matrix，灰度共生矩阵)
        glcm = self._compute_glcm(gray_image)
        # 提取纹理统计量
        contrast = self._glcm_contrast(glcm) #对比度
        homogeneity = self._glcm_homogeneity(glcm) #同质性
        energy = self._glcm_energy(glcm)#能量
        correlation = self._glcm_correlation(glcm)#相关性
        return [contrast, homogeneity, energy, correlation]
    #计算灰度级
    def _compute_glcm(self, gray_image: np.ndarray, distances: List[int] = [1], angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
        gray_quantized = (gray_image / 32).astype(np.uint8)
        glcm = np.zeros((8, 8, len(distances), len(angles)))
        for d_idx, d in enumerate(distances):
            for a_idx, a in enumerate(angles):
                glcm[:, :, d_idx, a_idx] = cv2.calcHist(
                    [gray_quantized, gray_quantized], 
                    [0, 1], 
                    None, 
                    [8, 8], 
                    [0, 8, 0, 8]
                ).flatten().reshape(8, 8)
        return glcm.mean(axis=(2, 3))  # 对距离和角度取平均
    #计算对比度
    def _glcm_contrast(self, glcm: np.ndarray) -> float:
        i, j = np.ogrid[:glcm.shape[0], :glcm.shape[1]]
        return np.sum(glcm * (i - j) ** 2)
    #计算同质性
    def _glcm_homogeneity(self, glcm: np.ndarray) -> float:
        i, j = np.ogrid[:glcm.shape[0], :glcm.shape[1]]
        return np.sum(glcm / (1 + (i - j) ** 2))
    #计算能量
    def _glcm_energy(self, glcm: np.ndarray) -> float:
        return np.sum(glcm ** 2)
    #计算相关性
    def _glcm_correlation(self, glcm: np.ndarray) -> float:
        i, j = np.ogrid[:glcm.shape[0], :glcm.shape[1]]
        mean_i = np.sum(i * glcm.sum(axis=1, keepdims=True))
        mean_j = np.sum(j * glcm.sum(axis=0, keepdims=True))
        std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm.sum(axis=1, keepdims=True)))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm.sum(axis=0, keepdims=True)))
        if std_i == 0 or std_j == 0:
            return 0
        return np.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j)