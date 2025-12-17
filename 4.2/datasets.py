import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

from config import IMG_SIZE, PATCH_SIZE, HM_SIZE_STAGE1, HM_SIZE_STAGE2


def resize_keep_ratio(img, target_long=IMG_SIZE):
    """按长边等比例缩放，返回缩放后图像及原/新尺寸（不填充）。"""
    h, w = img.shape[:2]
    scale = float(target_long) / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized, (w, h), (new_w, new_h)


def make_gaussian_heatmap(x, y, w, h, hm_size, sigma=2.0):
    """在 hm_size×hm_size 网格上，以 (x,y) 为中心生成 2D 高斯热力图（x,y 为缩放后坐标）。"""
    hm = np.zeros((hm_size, hm_size), dtype=np.float32)
    if x <= 0 and y <= 0 or w <= 0 or h <= 0:
        return hm

    sx = hm_size / float(w)
    sy = hm_size / float(h)
    cx = x * sx
    cy = y * sy

    tmp_size = sigma * 3
    ul = [int(cx - tmp_size), int(cy - tmp_size)]
    br = [int(cx + tmp_size + 1), int(cy + tmp_size + 1)]

    if ul[0] >= hm_size or ul[1] >= hm_size or br[0] < 0 or br[1] < 0:
        return hm

    size = int(2 * tmp_size + 1)
    x_vec = np.arange(0, size, 1, np.float32)
    y_vec = x_vec[:, None]
    x0 = y0 = size // 2
    g = np.exp(-((x_vec - x0) ** 2 + (y_vec - y0) ** 2) / (2 * sigma**2))

    g_x = max(0, -ul[0]), min(br[0], hm_size) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], hm_size) - ul[1]
    hm_x = max(0, ul[0]), min(br[0], hm_size)
    hm_y = max(0, ul[1]), min(br[1], hm_size)

    hm[hm_y[0] : hm_y[1], hm_x[0] : hm_x[1]] = np.maximum(
        hm[hm_y[0] : hm_y[1], hm_x[0] : hm_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return hm


class FoveaStage1Dataset(Dataset):
    """阶段一：整图 + 热力图 + 可见性。"""

    def __init__(
        self,
        csv_path,
        img_dir,
        indices=None,
        img_size=IMG_SIZE,
        hm_size=HM_SIZE_STAGE1,
        sigma=2.0,
        transforms=None,
    ):
        self.df = pd.read_csv(csv_path)
        if indices is not None:
            self.df = self.df[self.df["data"].isin(indices)].reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = int(row["data"])
        img_path = os.path.join(self.img_dir, f"{idx:04d}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = float(row["Fovea_X"])
        y = float(row["Fovea_Y"])
        visible = not (x == 0 and y == 0)

        img_resized, (w0, h0), (w1, h1) = resize_keep_ratio(img, self.img_size)
        sx = w1 / float(w0)
        sy = h1 / float(h0)
        x_res = x * sx
        y_res = y * sy

        # padding 到固定尺寸，方便 DataLoader stack
        pad_h = self.img_size - img_resized.shape[0]
        pad_w = self.img_size - img_resized.shape[1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Unexpected resized size larger than target IMG_SIZE.")
        if pad_h > 0 or pad_w > 0:
            img_resized = np.pad(
                img_resized,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if self.transforms is not None:
            aug = self.transforms(image=img_resized)
            img_resized = aug["image"]

        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        hm = make_gaussian_heatmap(x_res, y_res, w1, h1, self.hm_size, self.sigma)
        hm_t = torch.from_numpy(hm).unsqueeze(0)

        target = {
            "coords_resized": torch.tensor([x_res, y_res], dtype=torch.float32),
            "heatmap": hm_t,
            "visible": torch.tensor([1.0 if visible else 0.0]),
            "resized_size": torch.tensor([w1, h1], dtype=torch.float32),
            "orig_size": torch.tensor([w0, h0], dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.int64),
        }
        return img_t, target


class FoveaStage2Dataset(Dataset):
    """阶段二：用 GT 坐标裁剪 patch，训练局部热力图。"""

    def __init__(
        self,
        csv_path,
        img_dir,
        indices=None,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        hm_size=HM_SIZE_STAGE2,
        sigma=2.0,
        transforms=None,
    ):
        self.df = pd.read_csv(csv_path)
        if indices is not None:
            self.df = self.df[self.df["data"].isin(indices)].reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.patch_size = patch_size
        self.hm_size = hm_size
        self.sigma = sigma
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def _crop_patch(self, img_resized, x_res, y_res):
        h1, w1 = img_resized.shape[:2]
        ps = self.patch_size
        cx, cy = int(x_res), int(y_res)
        x1 = max(0, cx - ps // 2)
        y1 = max(0, cy - ps // 2)
        if w1 > ps:
            x1 = min(x1, w1 - ps)
        else:
            x1 = 0
        if h1 > ps:
            y1 = min(y1, h1 - ps)
        else:
            y1 = 0
        patch = img_resized[y1 : y1 + ps, x1 : x1 + ps]
        x_local = x_res - x1
        y_local = y_res - y1
        return patch, x_local, y_local, (x1, y1), (w1, h1)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = int(row["data"])
        img_path = os.path.join(self.img_dir, f"{idx:04d}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = float(row["Fovea_X"])
        y = float(row["Fovea_Y"])
        visible = not (x == 0 and y == 0)

        img_resized, (w0, h0), (w1, h1) = resize_keep_ratio(img, self.img_size)
        # padding 到固定尺寸，方便 DataLoader stack
        pad_h = self.img_size - img_resized.shape[0]
        pad_w = self.img_size - img_resized.shape[1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Unexpected resized size larger than target IMG_SIZE.")
        if pad_h > 0 or pad_w > 0:
            img_resized = np.pad(
                img_resized,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        sx = w1 / float(w0)
        sy = h1 / float(h0)
        x_res = x * sx
        y_res = y * sy

        patch, x_local, y_local, (x1, y1), (w1, h1) = self._crop_patch(
            img_resized, x_res, y_res
        )

        if self.transforms is not None:
            aug = self.transforms(image=patch)
            patch = aug["image"]

        patch_t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0

        hm = make_gaussian_heatmap(
            x_local, y_local, self.patch_size, self.patch_size, self.hm_size, self.sigma
        )
        hm_t = torch.from_numpy(hm).unsqueeze(0)

        target = {
            "heatmap": hm_t,
            "coords_local": torch.tensor([x_local, y_local], dtype=torch.float32),
            "visible": torch.tensor([1.0 if visible else 0.0]),
            "patch_origin": torch.tensor([x1, y1], dtype=torch.float32),
            "resized_size": torch.tensor([w1, h1], dtype=torch.float32),
            "orig_size": torch.tensor([w0, h0], dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.int64),
        }
        return patch_t, target


class FoveaTestDataset(Dataset):
    """测试集整图 Dataset，用于推理。"""

    def __init__(self, img_dir, img_size=IMG_SIZE, transforms=None):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        name = self.img_names[i]
        path = os.path.join(self.img_dir, name)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized, (w0, h0), (w1, h1) = resize_keep_ratio(img, self.img_size)
        if self.transforms is not None:
            aug = self.transforms(image=img_resized)
            img_resized = aug["image"]
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_id = int(os.path.splitext(name)[0])
        meta = {
            "orig_size": torch.tensor([w0, h0], dtype=torch.float32),
            "resized_size": torch.tensor([w1, h1], dtype=torch.float32),
            "img_id": torch.tensor(img_id, dtype=torch.int64),
        }
        return img_t, meta


