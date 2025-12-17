"""
关键点标注工具
支持从文件加载或自动生成关键点标注
"""
import os
import numpy as np
import json
from pathlib import Path
import cv2


def load_landmarks_from_npy(npy_path):
    """
    从.npy文件加载关键点标注
    
    Args:
        npy_path: .npy文件路径，格式为字典 {image_path: landmarks_array}
    
    Returns:
        landmark_dict: {image_path: landmarks} 字典
    """
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def load_landmarks_from_json(json_path):
    """
    从.json文件加载关键点标注
    
    Args:
        json_path: .json文件路径，格式为 {"image_path": [x1, y1, x2, y2, ...]}
    
    Returns:
        landmark_dict: {image_path: landmarks} 字典
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def auto_detect_landmarks(image_path, num_landmarks=68):
    """
    使用dlib或insightface自动检测关键点（可选功能）
    
    Args:
        image_path: 图像路径
        num_landmarks: 关键点数量（68或98）
    
    Returns:
        landmarks: 关键点坐标数组 [x1, y1, x2, y2, ...] 或 None（如果检测失败）
    """
    try:
        # 尝试使用dlib
        import dlib
        
        # 加载预训练的人脸检测器和关键点预测器
        detector_path = 'shape_predictor_68_face_landmarks.dat'
        if not os.path.exists(detector_path):
            # 如果没有dlib模型，返回None
            return None
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(detector_path)
        
        # 读取图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # 检测人脸
        faces = detector(img)
        if len(faces) == 0:
            return None
        
        # 检测关键点
        face = faces[0]  # 使用第一个检测到的人脸
        landmarks = predictor(img, face)
        
        # 提取关键点坐标
        points = []
        for i in range(landmarks.num_parts):
            point = landmarks.part(i)
            points.extend([point.x, point.y])
        
        # dlib默认68个关键点，如果需要98个，需要额外处理
        if num_landmarks != 68:
            # 明确提示：当前只返回68点，num_landmarks 需与之匹配
            print(f"Warning: dlib predictor provides 68 landmarks, but num_landmarks={num_landmarks}. "
                  f"Returning 68-point landmarks; please set num_landmarks=68 or use a 98-point model.")
        
        return np.array(points, dtype=np.float32)
    
    except ImportError:
        # dlib未安装，尝试使用insightface
        try:
            from insightface.app import FaceAnalysis
            
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            faces = app.get(img)
            if len(faces) == 0:
                return None
            
            # insightface返回106个关键点，取前98个
            landmarks = faces[0].landmark_2d_106[:num_landmarks]
            points = []
            for point in landmarks:
                points.extend([point[0], point[1]])
            
            return np.array(points, dtype=np.float32)
        
        except ImportError:
            # 两个库都未安装，返回None
            return None


def generate_landmarks_for_dataset(image_paths, landmark_dir=None, num_landmarks=68, auto_detect=False):
    """
    为数据集生成关键点标注
    
    Args:
        image_paths: 图像路径列表
        landmark_dir: 关键点标注目录（如果提供，从此目录加载）
        num_landmarks: 关键点数量
        auto_detect: 如果标注文件不存在，是否自动检测
    
    Returns:
        landmark_dict: {image_path: landmarks} 字典
    """
    landmark_dict = {}
    
    # 如果提供了标注目录，尝试加载
    if landmark_dir:
        landmark_path = Path(landmark_dir)
        
        # 尝试加载.npy文件
        npy_files = list(landmark_path.glob('*.npy'))
        if npy_files:
            print(f"Loading landmarks from {npy_files[0]}")
            landmark_dict = load_landmarks_from_npy(str(npy_files[0]))
            return landmark_dict
        
        # 尝试加载.json文件
        json_files = list(landmark_path.glob('*.json'))
        if json_files:
            print(f"Loading landmarks from {json_files[0]}")
            landmark_dict = load_landmarks_from_json(str(json_files[0]))
            return landmark_dict
    
    # 如果没有提供标注或加载失败，且启用自动检测
    if auto_detect:
        print("Auto-detecting landmarks (this may take a while)...")
        for img_path in image_paths:
            landmarks = auto_detect_landmarks(img_path, num_landmarks)
            if landmarks is not None:
                # 强校验：长度必须为 num_landmarks * 2，否则跳过/警告
                if landmarks.shape[0] != num_landmarks * 2:
                    print(
                        f"Warning: landmarks for {img_path} have length {landmarks.shape[0]}, "
                        f"expected {num_landmarks * 2}. This sample will be skipped for landmarks."
                    )
                    continue
                landmark_dict[img_path] = landmarks
            else:
                # 如果检测失败，使用默认值（图像中心）
                # 实际应用中可能需要更智能的处理
                default_landmarks = np.zeros(num_landmarks * 2, dtype=np.float32)
                default_landmarks[0::2] = 24.0  # x坐标（图像中心）
                default_landmarks[1::2] = 24.0  # y坐标（图像中心）
                landmark_dict[img_path] = default_landmarks
        
        print(f"Auto-detected landmarks for {len(landmark_dict)}/{len(image_paths)} images")
    
    return landmark_dict


if __name__ == '__main__':
    # 测试关键点工具
    print("Landmark utilities loaded successfully")
    print("Note: Auto-detection requires dlib or insightface to be installed")





