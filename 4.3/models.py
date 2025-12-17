import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import HM_SIZE_STAGE1, HM_SIZE_STAGE2, PATCH_SIZE


class Stage1Net(nn.Module):
    """阶段一：整图 -> 低分辨率热力图 + 可见性分类。"""

    def __init__(self, hm_size=HM_SIZE_STAGE1, pretrained=True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        # 去掉 fc 与 avgpool，保留 C5 特征图
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        in_channels = 2048

        self.hm_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Upsample(size=(hm_size, hm_size), mode="bilinear", align_corners=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.vis_fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        hm = self.hm_head(feat)
        pooled = self.avgpool(feat).view(feat.size(0), -1)
        vis_logit = self.vis_fc(pooled)
        return hm, vis_logit


class Stage2Net(nn.Module):
    """阶段二：patch -> 高分辨率热力图（使用 ResNet18 encoder）。"""

    def __init__(self, hm_size=HM_SIZE_STAGE2, patch_size=PATCH_SIZE, pretrained=True):
        super().__init__()
        # 使用预训练 ResNet18 作为 encoder
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # C5 特征图
        in_channels = 512
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Upsample(size=(hm_size, hm_size), mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        feat = self.encoder(x)
        hm = self.decoder(feat)
        return hm


def heatmap_to_coord(hm, img_w, img_h, eps=1e-6):
    """Soft-Argmax：从热力图解析亚像素坐标，返回 (x, y)（浮点），在给定 img_w/img_h 尺度下。"""
    # hm: [B,1,Hm,Wm]
    b, _, h, w = hm.shape
    hm_flat = hm.view(b, -1)
    # 使用 softmax 得到概率分布
    prob = torch.softmax(hm_flat, dim=1)  # [B, Hm*Wm]

    # 构建坐标网格
    ys = torch.arange(h, device=hm.device).view(1, h, 1).expand(b, h, w)
    xs = torch.arange(w, device=hm.device).view(1, 1, w).expand(b, h, w)
    xs = xs.contiguous().view(b, -1).float()
    ys = ys.contiguous().view(b, -1).float()

    # 加权平均得到亚像素坐标
    x = (prob * xs).sum(dim=1)
    y = (prob * ys).sum(dim=1)

    # 缩放到目标图像尺寸
    x = x * (float(img_w) / float(w))
    y = y * (float(img_h) / float(h))
    coords = torch.stack([x, y], dim=1)  # [B,2]
    return coords




