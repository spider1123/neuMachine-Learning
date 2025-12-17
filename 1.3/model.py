"""
ResNet50模型：适配灰度图输入
"""
import torch
import torch.nn as nn
import torchvision.models as models


def create_resnet50(num_classes=6, pretrained=True):
    """
    创建ResNet50模型，适配灰度图输入
    
    Args:
        num_classes: 分类数量
        pretrained: 是否使用ImageNet预训练权重
    
    Returns:
        ResNet50模型
    """
    # 加载预训练的ResNet50
    model = models.resnet50(pretrained=pretrained)
    
    # 修改分类头
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # 注意：保持模型第一层为3通道输入
    # 灰度图会复制为3通道，这样可以完全利用预训练权重
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_resnet50(num_classes=6, pretrained=True)
    print(f"Model created: {type(model)}")
    
    # 测试前向传播（3通道输入，224×224）
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")


