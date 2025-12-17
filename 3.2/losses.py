"""
多任务损失函数
支持情绪分类 + 关键点热图回归
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    总损失 = (1 - landmark_weight) * 分类损失 + landmark_weight * 热图回归损失
    """
    def __init__(self, num_classes=6, num_landmarks=98, label_smoothing=0.1, landmark_weight=0.3):
        """
        Args:
            num_classes: 分类数量
            num_landmarks: 关键点数量
            label_smoothing: 标签平滑系数
            landmark_weight: 关键点损失权重（默认0.3，即总损失 = 0.7 * CE + 0.3 * MSE）
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_landmarks = num_landmarks
        self.label_smoothing = label_smoothing
        self.landmark_weight = landmark_weight
        
        # 分类损失（带标签平滑的交叉熵）
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 热图回归损失（MSE）
        self.heatmap_criterion = nn.MSELoss()
    
    def forward(self, cls_logits, cls_labels, heatmap_pred=None, heatmap_target=None):
        """
        计算多任务损失
        
        Args:
            cls_logits: 分类logits (B, num_classes)
            cls_labels: 分类标签 (B,)
            heatmap_pred: 预测的关键点热图 (B, num_landmarks, H, W)，可选
            heatmap_target: 真实的关键点热图 (B, num_landmarks, H, W)，可选
        
        Returns:
            total_loss: 总损失
            cls_loss: 分类损失
            heatmap_loss: 热图损失（如果提供热图）
        """
        # 分类损失
        cls_loss = self.cls_criterion(cls_logits, cls_labels)
        
        # 热图损失（如果提供）
        if heatmap_pred is not None and heatmap_target is not None:
            heatmap_loss = self.heatmap_criterion(heatmap_pred, heatmap_target)
            total_loss = (1 - self.landmark_weight) * cls_loss + self.landmark_weight * heatmap_loss
            return total_loss, cls_loss, heatmap_loss
        else:
            # 如果没有热图，只返回分类损失
            return cls_loss, cls_loss, torch.tensor(0.0, device=cls_logits.device)


def generate_heatmap_from_landmarks(landmarks, heatmap_size=56, sigma=2.0):
    """
    从关键点坐标生成高斯热图
    
    Args:
        landmarks: 关键点坐标 (B, num_landmarks, 2)，每个关键点为 (x, y)，坐标范围 [0, 224]
        heatmap_size: 热图尺寸（默认56，即224/4）
        sigma: 高斯核标准差
    
    Returns:
        heatmaps: 热图 (B, num_landmarks, heatmap_size, heatmap_size)
    """
    batch_size, num_landmarks, _ = landmarks.shape
    
    # 将坐标从 [0, 224] 缩放到 [0, heatmap_size]
    scale = heatmap_size / 224.0
    landmarks_scaled = landmarks * scale  # (B, num_landmarks, 2)
    
    # 创建坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(heatmap_size, device=landmarks.device, dtype=torch.float32),
        torch.arange(heatmap_size, device=landmarks.device, dtype=torch.float32),
        indexing='ij'
    )
    coords = torch.stack([x_coords, y_coords], dim=-1)  # (heatmap_size, heatmap_size, 2)
    
    # 生成热图
    heatmaps = []
    for b in range(batch_size):
        batch_heatmaps = []
        for l in range(num_landmarks):
            landmark = landmarks_scaled[b, l]  # (2,)
            if landmark[0] < 0 or landmark[1] < 0:  # 无效关键点
                heatmap = torch.zeros(heatmap_size, heatmap_size, device=landmarks.device)
            else:
                # 计算每个像素到关键点的距离
                dist_sq = torch.sum((coords - landmark) ** 2, dim=-1)  # (heatmap_size, heatmap_size)
                # 生成高斯热图
                heatmap = torch.exp(-dist_sq / (2 * sigma ** 2))
            batch_heatmaps.append(heatmap)
        heatmaps.append(torch.stack(batch_heatmaps, dim=0))  # (num_landmarks, heatmap_size, heatmap_size)
    
    heatmaps = torch.stack(heatmaps, dim=0)  # (B, num_landmarks, heatmap_size, heatmap_size)
    return heatmaps


if __name__ == '__main__':
    # 测试损失函数
    criterion = MultiTaskLoss(num_classes=6, label_smoothing=0.1, landmark_weight=0.3)
    
    # 模拟数据
    batch_size = 4
    cls_logits = torch.randn(batch_size, 6)
    cls_labels = torch.randint(0, 6, (batch_size,))
    heatmap_pred = torch.randn(batch_size, 98, 56, 56)
    heatmap_target = torch.randn(batch_size, 98, 56, 56)
    
    # 计算损失
    total_loss, cls_loss, heatmap_loss = criterion(
        cls_logits, cls_labels, heatmap_pred, heatmap_target
    )
    
    print(f"Classification loss: {cls_loss.item():.4f}")
    print(f"Heatmap loss: {heatmap_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Expected: {0.7 * cls_loss.item() + 0.3 * heatmap_loss.item():.4f}")
    
    # 测试热图生成
    landmarks = torch.rand(batch_size, 98, 2) * 224  # 随机关键点坐标
    heatmaps = generate_heatmap_from_landmarks(landmarks, heatmap_size=56, sigma=2.0)
    print(f"\nGenerated heatmaps shape: {heatmaps.shape}")

