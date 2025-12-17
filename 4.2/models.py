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
    """阶段二：patch -> 高分辨率热力图。"""

    def __init__(self, hm_size=HM_SIZE_STAGE2, patch_size=PATCH_SIZE):
        super().__init__()
        # 简单 encoder-decoder 结构，可后续替换为更强 backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Upsample(size=(hm_size, hm_size), mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        feat = self.encoder(x)
        hm = self.decoder(feat)
        return hm


def heatmap_to_coord(hm, img_w, img_h):
    """从单通道热力图中解析坐标，返回 (x, y)（浮点），在给定 img_w/img_h 尺度下。"""
    # hm: [B,1,Hm,Wm]
    b, _, h, w = hm.shape
    hm_reshaped = hm.view(b, -1)
    idx = hm_reshaped.argmax(dim=1)  # [B]
    ys = (idx // w).float()
    xs = (idx % w).float()
    xs = xs * (float(img_w) / float(w))
    ys = ys * (float(img_h) / float(h))
    coords = torch.stack([xs, ys], dim=1)  # [B,2]
    return coords




