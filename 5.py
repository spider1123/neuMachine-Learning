import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
import shutil

# ======================== 超参数配置 ========================
class Config:
    # 数据集路径 - 按照你的真实结构
    TRAIN_LABELS_PATH = "./train/label"      # 训练集标签路径（二值掩码图像）
    TRAIN_IMAGES_PATH = "./train/image"      # 训练集图像路径
    TEST_IMAGES_PATH = "./test/image"        # 测试集图像路径
    
    # 转换后的YOLO标签路径
    YOLO_LABELS_PATH = "./train/labels_yolo"  # YOLO格式标签（txt文件）
    
    # 模型保存路径
    WEIGHTS_SAVE_PATH = "./weights/best.pt"            # 分割权重保存路径
    CSV_SAVE_PATH = "./submission.csv"                 # CSV保存路径
    
    # 训练参数
    MODEL_NAME = "yolov8n-seg"
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 512
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
    
    # 推理参数
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # 输出路径
    PRED_IMAGE_DIR = "./image"


# ======================== 二值掩码转YOLO格式 ========================
class MaskToYOLOConverter:
    """将二值掩码图像转换为YOLO分割格式"""
    
    @staticmethod
    def mask_to_polygon(mask, class_id=0, simplification_factor=0.002):
        """将二值掩码转换为多边形坐标"""
        if mask.dtype != np.uint8:
            mask = (mask > 127).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        height, width = mask.shape
        polygon_coords = []
        
        for contour in contours:
            epsilon = simplification_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:
                for point in approx:
                    x, y = point[0]
                    norm_x = x / width
                    norm_y = y / height
                    polygon_coords.extend([norm_x, norm_y])
        
        if len(polygon_coords) < 6:
            return None
        
        polygon_str = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in polygon_coords])
        return polygon_str
    
    @staticmethod
    def convert_mask_to_yolo(mask_path, output_txt_path, class_id=0):
        """转换单个掩码文件为YOLO格式"""
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"  ✗ 无法读取掩码文件: {mask_path}")
                return False
            
            polygon_str = MaskToYOLOConverter.mask_to_polygon(mask, class_id)
            
            if polygon_str is None:
                # 创建空标签文件
                with open(output_txt_path, 'w') as f:
                    f.write("")
                return True
            
            # 保存为YOLO标签文件
            with open(output_txt_path, 'w') as f:
                f.write(polygon_str)
            
            return True
        
        except Exception as e:
            print(f"  ✗ 转换失败 {os.path.basename(mask_path)}: {e}")
            return False
    
    @staticmethod
    def batch_convert(mask_dir, image_dir, yolo_label_dir, class_id=0):
        """
        批量转换所有掩码文件
        从mask_dir读取二值掩码 -> 转换为txt文件放入yolo_label_dir
        同时将对应的图像复制到image_dir
        """
        os.makedirs(yolo_label_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # 获取所有掩码文件
        mask_files = []
        for filename in os.listdir(mask_dir):
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.gif')):
                mask_files.append(filename)
        
        if not mask_files:
            print(f"  ⚠ 警告：未找到掩码文件 {mask_dir}")
            return False
        
        print(f"  开始转换 {len(mask_files)} 个掩码文件...")
        
        success_count = 0
        for idx, filename in enumerate(mask_files):
            mask_path = os.path.join(mask_dir, filename)
            
            # 转换为txt文件
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_txt_path = os.path.join(yolo_label_dir, output_filename)
            
            # 复制图像到train/images目录
            dest_image = os.path.join(image_dir, filename)
            try:
                shutil.copy(mask_path, dest_image)
            except:
                pass
            
            if MaskToYOLOConverter.convert_mask_to_yolo(mask_path, output_txt_path, class_id):
                success_count += 1
            
            if (idx + 1) % 10 == 0:
                print(f"    已处理 {idx + 1}/{len(mask_files)}")
        
        print(f"  ✓ 转换完成：{success_count}/{len(mask_files)} 个文件成功")
        return success_count > 0


# ======================== 数据集准备 ========================
class DatasetPreparer:
    """准备YOLO格式的数据集"""
    
    @staticmethod
    def create_dataset_yaml(config):
        """创建dataset.yaml文件"""
        yaml_content = f"""path: {os.path.abspath('./')}
train: train/images
val: train/images

nc: 1
names: ['vessel']
"""
        
        os.makedirs("./", exist_ok=True)
        yaml_path = "./dataset.yaml"
        
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        
        print(f"  ✓ dataset.yaml 已创建")
        return yaml_path


# ======================== 模型训练 ========================
class VesselSegmentationTrainer:
    """眼底血管分割模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device(f"cuda:{config.DEVICE}" 
                                   if torch.cuda.is_available() 
                                   else "cpu")
        
    def load_model(self):
        """加载YOLOv8预训练模型"""
        print(f"  ✓ 加载YOLOv8模型: {self.config.MODEL_NAME}")
        self.model = YOLO(f"{self.config.MODEL_NAME}.pt")
        return self.model
    
    def train(self, yaml_path):
        """训练模型"""
        if self.model is None:
            self.load_model()
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"dataset.yaml 不存在: {yaml_path}")
        
        print("  开始训练...")
        
        results = self.model.train(
            data=yaml_path,
            epochs=self.config.EPOCHS,
            imgsz=self.config.IMG_SIZE,
            batch=self.config.BATCH_SIZE,
            device=self.config.DEVICE,
            patience=20,
            save=True,
            project="./runs/segment",
            name="vessel_seg"
        )
        
        # 保存最佳权重
        os.makedirs(os.path.dirname(self.config.WEIGHTS_SAVE_PATH), exist_ok=True)
        best_weights = "./runs/segment/vessel_seg/weights/best.pt"
        
        if os.path.exists(best_weights):
            shutil.copy(best_weights, self.config.WEIGHTS_SAVE_PATH)
            print(f"  ✓ 权重已保存到 {self.config.WEIGHTS_SAVE_PATH}")
        
        return results


# ======================== 推理与分割 ========================
class SegmentationInference:
    """分割推理器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def process_segmentation_mask(self, mask):
        """处理分割掩码：血管=0，其他=255"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        if mask.max() > 1:
            mask = mask / 255.0
        
        output_mask = np.zeros_like(mask, dtype=np.uint8)
        output_mask[mask < 0.5] = 0
        output_mask[mask >= 0.5] = 255
        
        return output_mask
    
    def predict_and_save(self, image_path, output_dir):
        """预测单个图像并保存分割结果"""
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.config.CONF_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                imgsz=self.config.IMG_SIZE,
                device=self.config.DEVICE,
                verbose=False
            )
            
            result = results[0]
            
            if result.masks is not None:
                masks = result.masks.data
                combined_mask = torch.zeros(masks.shape[1:], device=masks.device)
                for mask in masks:
                    combined_mask = torch.logical_or(combined_mask, mask).float()
                
                combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)
                combined_mask = torch.nn.functional.interpolate(
                    combined_mask,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            else:
                combined_mask = torch.zeros((512, 512))
            
            output_mask = self.process_segmentation_mask(combined_mask)
            
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(output_mask).save(output_path)
            
            return output_path
        
        except Exception as e:
            print(f"  ✗ 推理失败 {os.path.basename(image_path)}: {e}")
            return None
    
    def batch_predict(self, test_images_dir):
        """批量预测"""
        if not os.path.exists(test_images_dir):
            print(f"  ⚠ 测试图像目录不存在")
            return []
        
        image_list = []
        for filename in os.listdir(test_images_dir):
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.gif')):
                image_list.append(os.path.join(test_images_dir, filename))
        
        print(f"  找到 {len(image_list)} 个测试图像")
        
        for idx, image_path in enumerate(image_list):
            print(f"    处理 {idx+1}/{len(image_list)}: {os.path.basename(image_path)}")
            self.predict_and_save(image_path, self.config.PRED_IMAGE_DIR)
        
        return image_list


# ======================== CSV转换 ========================
class CSVConverter:
    """将分割结果转换为RLE编码的CSV文件"""
    
    @staticmethod
    def get_img_file(image_dir):
        """获取图像文件列表"""
        imagelist = []
        namelist = []
        if not os.path.exists(image_dir):
            return imagelist, namelist
            
        for parent, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                if filename.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.gif')):
                    imagelist.append(os.path.join(parent, filename))
                    namelist.append(filename)
            return imagelist, namelist
    
    @staticmethod
    def turn_to_str(image_list):
        """将分割图像转换为RLE编码字符串"""
        outputs = []
        for image_path in image_list:
            try:
                image = Image.open(image_path).convert('L')
                transform = torchvision.transforms.ToTensor()
                image = image.resize((512, 512), Image.Resampling.BILINEAR)
                image = transform(image)
                image[image > 0] = 1
                dots = np.where(image.flatten() == 1)[0]
                run_lengths = []
                prev = -2
                for b in dots:
                    if (b > prev + 1):
                        run_lengths.extend((b + 1, 0))
                    run_lengths[-1] += 1
                    prev = b
                output = ' '.join([str(r) for r in run_lengths])
                outputs.append(output)
            except Exception as e:
                print(f"  ✗ 处理失败 {os.path.basename(image_path)}: {e}")
                outputs.append("")
        
        return outputs
    
    @staticmethod
    def save_to_csv(name_list, str_list, csv_path):
        """保存为CSV文件"""
        df = pd.DataFrame(columns=['Id', 'Predicted'])
        df['Id'] = [i.split('.')[0] for i in name_list]
        df['Predicted'] = str_list
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        df.to_csv(csv_path, index=None)
        print(f"  ✓ CSV已保存到 {csv_path}")


# ======================== 主程序 ========================
def main():
    """主程序"""
    config = Config()
    
    print("\n" + "=" * 70)
    print("眼底血管分割系统 - 完整流程")
    print("=" * 70)
    
    # 创建必要的目录
    for directory in [config.TRAIN_IMAGES_PATH, config.TRAIN_LABELS_PATH, 
                      config.TEST_IMAGES_PATH, config.PRED_IMAGE_DIR, config.YOLO_LABELS_PATH]:
        os.makedirs(directory, exist_ok=True)
    
    # 第一步：转换二值掩码为YOLO格式
    print("\n[步骤1] 将二值掩码转换为YOLO格式")
    print("-" * 70)
    
    if not os.listdir(config.TRAIN_LABELS_PATH):
        print(f"✗ 错误：{config.TRAIN_LABELS_PATH} 目录为空")
        print("请将二值掩码放入该目录")
        return
    
    converter = MaskToYOLOConverter()
    if not converter.batch_convert(
        config.TRAIN_LABELS_PATH,
        config.TRAIN_IMAGES_PATH,
        config.YOLO_LABELS_PATH,
        class_id=0
    ):
        print("✗ 转换失败")
        return
    
    # 验证数据
    train_images_count = len([f for f in os.listdir(config.TRAIN_IMAGES_PATH) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    train_labels_count = len([f for f in os.listdir(config.YOLO_LABELS_PATH) 
                              if f.endswith('.txt')])
    
    print(f"\n数据统计:")
    print(f"  训练图像数: {train_images_count}")
    print(f"  训练标签数: {train_labels_count}")
    
    if train_images_count == 0 or train_labels_count == 0:
        print("✗ 错误：训练数据不完整")
        return
    
    # 第二步：准备数据集
    print("\n[步骤2] 准备YOLO数据集")
    print("-" * 70)
    yaml_path = DatasetPreparer.create_dataset_yaml(config)
    
    # 第三步：训练模型
    print("\n[步骤3] 训练分割模型")
    print("-" * 70)
    try:
        trainer = VesselSegmentationTrainer(config)
        trainer.load_model()
        trainer.train(yaml_path)
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return
    
    # 第四步：推理
    print("\n[步骤4] 加载权重进行推理")
    print("-" * 70)
    
    if os.path.exists(config.TEST_IMAGES_PATH) and os.listdir(config.TEST_IMAGES_PATH):
        try:
            inference_model = YOLO(config.WEIGHTS_SAVE_PATH)
            segmenter = SegmentationInference(inference_model, config)
            segmenter.batch_predict(config.TEST_IMAGES_PATH)
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            return
    else:
        print(f"⚠ 提示：将测试图像放在 {config.TEST_IMAGES_PATH} 进行推理")
    
    # 第五步：生成CSV文件
    print("\n[步骤5] 生成RLE编码CSV文件")
    print("-" * 70)
    
    csv_converter = CSVConverter()
    image_list, name_list = csv_converter.get_img_file(config.PRED_IMAGE_DIR)
    
    if image_list:
        str_list = csv_converter.turn_to_str(image_list)
        csv_converter.save_to_csv(name_list, str_list, config.CSV_SAVE_PATH)
    else:
        print(f"⚠ 预测结果目录为空")
    
    # 完成
    print("\n" + "=" * 70)
    print("✓ 处理完成！")
    print("=" * 70)
    print(f"分割权重: {config.WEIGHTS_SAVE_PATH}")
    print(f"YOLO标签: {config.YOLO_LABELS_PATH}")
    print(f"预测结果: {config.PRED_IMAGE_DIR}")
    print(f"CSV文件: {config.CSV_SAVE_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
