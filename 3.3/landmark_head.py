"""
人脸关键点热图回归头
用于多任务学习，辅助情绪分类
"""
import torch
import torch.nn as nn


class LandmarkHead(nn.Module):
    """
    关键点热图回归头
    从ResNet50的layer4输出（2048通道）生成关键点热图
    """
    def __init__(self, in_channels=2048, num_landmarks=68, heatmap_size=56):
        """
        Args:
            in_channels: 输入特征图通道数（ResNet50 layer4输出为2048）
            num_landmarks: 关键点数量（68或98）
            heatmap_size: 热图尺寸（通常为输入尺寸的1/4，224/4=56）
        """
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        
        # 热图生成头
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_landmarks, kernel_size=1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 特征图 (B, 2048, H, W)，通常为 (B, 2048, 7, 7) 或 (B, 2048, 14, 14)
        Returns:
            heatmap: 关键点热图 (B, num_landmarks, heatmap_size, heatmap_size)
        """
        # 如果输入尺寸不是目标尺寸，需要上采样
        if x.shape[2] != self.heatmap_size or x.shape[3] != self.heatmap_size:
            x = nn.functional.interpolate(
                x, 
                size=(self.heatmap_size, self.heatmap_size),
                mode='bilinear',
                align_corners=False
            )
        
        # 生成热图
        heatmap = self.heatmap_head(x)  # (B, num_landmarks, heatmap_size, heatmap_size)
        
        return heatmap


if __name__ == '__main__':
    # 测试LandmarkHead
    head = LandmarkHead(in_channels=2048, num_landmarks=68, heatmap_size=56)
    
    # 模拟ResNet50 layer4输出（batch_size=2, channels=2048, spatial=7x7）
    x = torch.randn(2, 2048, 7, 7)
    
    with torch.no_grad():
        heatmap = head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output heatmap shape: {heatmap.shape}")
    print(f"Expected: (2, 68, 56, 56)")





