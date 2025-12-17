"""
ResNet50模型：适配灰度图输入，使用随机初始化
支持多任务学习：情绪分类 + 关键点热图回归
"""
import torch
import torch.nn as nn
import torchvision.models as models
from landmark_head import LandmarkHead


class ResNet50MultiTask(nn.Module):
    """
    ResNet50多任务模型
    主任务：情绪分类
    辅助任务：关键点热图回归
    """
    def __init__(self, num_classes=6, num_landmarks=98, use_landmark=True):
        """
        Args:
            num_classes: 分类数量
            num_landmarks: 关键点数量（68或98）
            use_landmark: 是否使用关键点辅助监督
        """
        super().__init__()
        
        # ResNet50骨干网络
        backbone = models.resnet50(pretrained=False)
        
        # 提取特征提取部分（去掉avgpool和fc）
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        
        # 分类头
        num_features = backbone.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)
        
        # 关键点热图回归头（可选）
        self.use_landmark = use_landmark
        if use_landmark:
            # ResNet50 layer4输出为 (B, 2048, 7, 7)，上采样到56x56
            self.landmark_head = LandmarkHead(
                in_channels=2048,
                num_landmarks=num_landmarks,
                heatmap_size=56  # 224 / 4 = 56
            )
        
        # 注意：保持模型第一层为3通道输入
        # 灰度图会复制为3通道
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: 输入图像 (B, 3, 224, 224)
            return_features: 是否返回中间特征（用于关键点预测）
        
        Returns:
            - 如果 use_landmark=False: 只返回分类logits
            - 如果 use_landmark=True: 返回 (分类logits, 关键点热图)
            - 如果 return_features=True: 额外返回layer4特征
        """
        # 特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, 7, 7)
        
        # 分类分支
        pooled = self.avgpool(x)  # (B, 2048, 1, 1)
        pooled = torch.flatten(pooled, 1)  # (B, 2048)
        cls_logits = self.fc(pooled)  # (B, num_classes)
        
        # 关键点分支（如果启用）
        if self.use_landmark:
            heatmap = self.landmark_head(x)  # (B, num_landmarks, 56, 56)
            if return_features:
                return cls_logits, heatmap, x
            return cls_logits, heatmap
        else:
            if return_features:
                return cls_logits, x
            return cls_logits


def create_resnet50(num_classes=6, num_landmarks=98, use_landmark=True):
    """
    创建ResNet50多任务模型
    
    Args:
        num_classes: 分类数量
        num_landmarks: 关键点数量（68或98，默认98）
        use_landmark: 是否使用关键点辅助监督（默认True）
    
    Returns:
        ResNet50MultiTask模型
    """
    model = ResNet50MultiTask(
        num_classes=num_classes,
        num_landmarks=num_landmarks,
        use_landmark=use_landmark
    )
    return model


if __name__ == '__main__':
    # 测试模型（带关键点）
    model = create_resnet50(num_classes=6, num_landmarks=98, use_landmark=True)
    print(f"Model created: {type(model)}")
    
    # 测试前向传播（3通道输入，224×224）
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cls_logits, heatmap = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Classification logits shape: {cls_logits.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    
    # 测试模型（不带关键点）
    model_no_landmark = create_resnet50(num_classes=6, use_landmark=False)
    with torch.no_grad():
        cls_logits_only = model_no_landmark(x)
    print(f"\nWithout landmark:")
    print(f"Classification logits shape: {cls_logits_only.shape}")



