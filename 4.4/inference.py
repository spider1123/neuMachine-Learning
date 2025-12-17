import os
import csv
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    IMG_DIR_TEST,
    VIS_THRESH,
    BATCH_SIZE,
    NUM_WORKERS,
    HM_SIZE_STAGE1,
    HM_SIZE_STAGE2,
    PATCH_SIZE,
    IMG_SIZE,
)
from datasets import FoveaTestDataset, resize_keep_ratio, IMAGENET_MEAN, IMAGENET_STD
from models import Stage1Net, Stage2Net, heatmap_to_coord


# 多尺度 TTA 配置
TTA_SCALES = [0.9, 1.0, 1.1]  # 多尺度测试时增强


def load_models(device, ckpt_dir="checkpoints", num_folds=5):
    models1 = []
    models2 = []
    fold_weights = []  # 用于加权集成（可以根据验证集表现调整）
    for fold in range(1, num_folds + 1):
        m1 = Stage1Net().to(device)
        m2 = Stage2Net().to(device)
        p1 = os.path.join(ckpt_dir, f"stage1_fold{fold}.pth")
        p2 = os.path.join(ckpt_dir, f"stage2_fold{fold}.pth")
        if os.path.isfile(p1):
            m1.load_state_dict(torch.load(p1, map_location=device))
            m1.eval()
            models1.append(m1)
            # 默认等权重，可以根据验证集表现调整
            fold_weights.append(1.0 / num_folds)
        if os.path.isfile(p2):
            m2.load_state_dict(torch.load(p2, map_location=device))
            m2.eval()
            models2.append(m2)
    # 归一化权重
    if fold_weights:
        total_weight = sum(fold_weights)
        fold_weights = [w / total_weight for w in fold_weights]
    return models1, models2, fold_weights


def crop_patch_from_coord(img_resized, x_res, y_res):
    """根据阶段一预测坐标在 resize 图上裁剪 patch，并返回 patch 及偏移。"""
    h1, w1 = img_resized.shape[:2]
    ps = PATCH_SIZE
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
    return patch, x1, y1, w1, h1


def resize_with_scale(img_np, scale, target_size=IMG_SIZE):
    """按 scale 缩放图像，然后 resize 到 target_size（保持长边）。"""
    h, w = img_np.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    img_scaled = cv2.resize(img_np, (new_w, new_h))
    img_resized, _, _ = resize_keep_ratio(img_scaled, target_size)
    return img_resized


@torch.no_grad()
def run_inference(device, out_path="submission.csv", num_folds=5, 
                  tta_flip=False, tta_scale=False):
    models1, models2, fold_weights = load_models(device, num_folds=num_folds)
    if len(models1) == 0 or len(models2) == 0:
        raise RuntimeError("No stage1/stage2 checkpoints found in checkpoints/.")

    test_ds = FoveaTestDataset(IMG_DIR_TEST, img_size=IMG_SIZE)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    rows = [("ImageID", "value")]

    for img_t, meta in test_loader:
        img_t = img_t.to(device)
        w0, h0 = meta["orig_size"][0].tolist()
        w1, h1 = meta["resized_size"][0].tolist()
        img_id = int(meta["img_id"][0].item())

        # 阶段一集成：热力图 + 可见性（多尺度 + HFlip TTA）
        hm_list_all = []
        vis_list_all = []
        
        scales_to_use = TTA_SCALES if tta_scale else [1.0]
        
        for scale in scales_to_use:
            # 多尺度预处理
            if scale != 1.0 and tta_scale:
                # 读取原图
                img_name = f"{img_id:04d}.jpg"
                img_path = os.path.join(IMG_DIR_TEST, img_name)
                img_raw = cv2.imread(img_path)
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                # 按 scale 缩放
                img_resized_scale, _, (w1_scale, h1_scale) = resize_keep_ratio(
                    img_raw, int(IMG_SIZE * scale)
                )
                pad_h = IMG_SIZE - img_resized_scale.shape[0]
                pad_w = IMG_SIZE - img_resized_scale.shape[1]
                if pad_h > 0 or pad_w > 0:
                    img_resized_scale = np.pad(
                        img_resized_scale,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                img_scale = img_resized_scale.astype(np.float32) / 255.0
                img_scale = (img_scale - IMAGENET_MEAN) / IMAGENET_STD
                img_t_scale = torch.from_numpy(img_scale).permute(2, 0, 1).float().unsqueeze(0).to(device)
            else:
                img_t_scale = img_t
                w1_scale, h1_scale = w1, h1
            
            for model_idx, m1 in enumerate(models1):
                weight = fold_weights[model_idx] if fold_weights else 1.0 / len(models1)
                
                # 原始预测
                hm_pred, vis_logit = m1(img_t_scale)
                vis_val = torch.sigmoid(vis_logit).item()
                
                # 水平翻转 TTA
                if tta_flip:
                    img_flip = torch.flip(img_t_scale, dims=[3])
                    hm_flip, vis_logit_flip = m1(img_flip)
                    hm_flip = torch.flip(hm_flip, dims=[3])
                    hm_pred = (hm_pred + hm_flip) / 2.0
                    vis_val = 0.5 * (vis_val + torch.sigmoid(vis_logit_flip).item())
                
                # 如果使用了多尺度，需要将热力图缩放到原始尺寸
                if scale != 1.0 and tta_scale:
                    # 简化：直接缩放热力图到原始尺寸
                    hm_pred = torch.nn.functional.interpolate(
                        hm_pred, size=(HM_SIZE_STAGE1, HM_SIZE_STAGE1),
                        mode="bilinear", align_corners=False
                    )
                
                hm_list_all.append(hm_pred * weight)
                vis_list_all.append(vis_val * weight)
        
        # 加权平均
        hm_mean = torch.stack(hm_list_all, dim=0).sum(dim=0)
        p_vis = float(sum(vis_list_all))

        if p_vis < VIS_THRESH:
            x_final, y_final = 0.0, 0.0
        else:
            # 解析阶段一坐标（在 resize 后坐标系）
            coord_stage1 = heatmap_to_coord(hm_mean, w1, h1)[0]
            x_c, y_c = coord_stage1.tolist()

            # 从原图重新读取并做与训练一致的预处理（避免 tensor 逆变换误差）
            img_name = f"{img_id:04d}.jpg"
            img_path = os.path.join(IMG_DIR_TEST, img_name)
            img_raw = cv2.imread(img_path)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_resized_np, (w0_raw, h0_raw), (w1_raw, h1_raw) = resize_keep_ratio(
                img_raw, IMG_SIZE
            )
            # padding（与训练一致）
            pad_h = IMG_SIZE - img_resized_np.shape[0]
            pad_w = IMG_SIZE - img_resized_np.shape[1]
            if pad_h < 0 or pad_w < 0:
                raise ValueError("Unexpected resized size larger than target IMG_SIZE.")
            if pad_h > 0 or pad_w > 0:
                img_resized_np = np.pad(
                    img_resized_np,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            # 用预处理后的图像裁剪 patch
            patch, x1, y1, w1p, h1p = crop_patch_from_coord(img_resized_np, x_c, y_c)

            # 归一化（与训练一致）
            patch = patch.astype(np.float32) / 255.0
            patch = (patch - IMAGENET_MEAN) / IMAGENET_STD
            patch_t = (
                torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0)
            ).to(device)

            # 阶段二集成：局部热力图（可选 HFlip TTA）
            hm2_list = []
            for model_idx, m2 in enumerate(models2):
                weight = fold_weights[model_idx] if fold_weights else 1.0 / len(models2)
                
                hm2 = m2(patch_t)
                if tta_flip:
                    patch_flip = torch.flip(patch_t, dims=[3])
                    hm2_flip = m2(patch_flip)
                    hm2_flip = torch.flip(hm2_flip, dims=[3])
                    hm2 = (hm2 + hm2_flip) / 2.0
                hm2_list.append(hm2 * weight)
            hm2_mean = torch.stack(hm2_list, dim=0).sum(dim=0)

            # 在 patch 尺度下解析坐标
            coord_local = heatmap_to_coord(hm2_mean, PATCH_SIZE, PATCH_SIZE)[0]
            x_local, y_local = coord_local.tolist()

            # 还原到 resize 图坐标
            x_res = x_local + x1
            y_res = y_local + y1

            # 反缩放到原图坐标
            sx = w1 / float(w0)
            sy = h1 / float(h0)
            x_final = x_res / sx
            y_final = y_res / sy

        rows.append((f"{img_id}_Fovea_X", x_final))
        rows.append((f"{img_id}_Fovea_Y", y_final))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved submission to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument(
        "--tta_flip",
        action="store_true",
        help="是否在推理时使用水平翻转 TTA",
    )
    parser.add_argument(
        "--tta_scale",
        action="store_true",
        help="是否在推理时使用多尺度 TTA",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_inference(
        device,
        out_path=args.out,
        num_folds=args.folds,
        tta_flip=args.tta_flip,
        tta_scale=args.tta_scale,
    )


if __name__ == "__main__":
    main()


