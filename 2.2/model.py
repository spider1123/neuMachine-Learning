# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, convnext_tiny, swin_t
from config import cfg


def get_model(num_classes):
    """
    根据配置的 BACKBONE 创建模型
    支持: resnet50, efficientnet_b0, convnext_tiny, swin_t
    """
    if cfg.BACKBONE == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif cfg.BACKBONE == "efficientnet_b0":
        model = efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif cfg.BACKBONE == "convnext_tiny":
        model = convnext_tiny(weights="IMAGENET1K_V1")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    elif cfg.BACKBONE == "swin_t":
        model = swin_t(weights="IMAGENET1K_V1")
        model.head = nn.Linear(model.head.in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported backbone: {cfg.BACKBONE}")
    
    print(f"使用模型: {cfg.BACKBONE}")
    print(f"输出类别数: {num_classes}")
    
    return model

