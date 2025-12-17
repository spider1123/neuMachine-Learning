import os
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from config import (
    IMG_DIR_TRAIN,
    CSV_GT,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    NUM_WORKERS,
    NUM_FOLDS,
    EARLY_STOP_PATIENCE,
)
from datasets import FoveaStage1Dataset, FoveaStage2Dataset
from models import Stage1Net, Stage2Net, heatmap_to_coord


def stage1_train_one_epoch(model, loader, optimizer, device, lambda_hm=1.0, lambda_cls=1.0):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        # DataLoader 会把 dict 自动 collate 成 dict[str -> Tensor]
        hm_gt = targets["heatmap"].to(device)
        visible = targets["visible"].to(device)

        optimizer.zero_grad()
        hm_pred, vis_logit = model(imgs)

        mask = visible.view(-1, 1, 1, 1)
        # 加权 MSE：对热力图高值区域（峰值附近）给予更大权重
        alpha = 4.0  # 可调参数，范围 2-10
        w = 1.0 + alpha * hm_gt
        hm_loss = ((hm_pred - hm_gt) ** 2 * w * mask).mean()
        vis_loss = F.binary_cross_entropy_with_logits(vis_logit, visible)
        loss = lambda_hm * hm_loss + lambda_cls * vis_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def stage1_eval(model, loader, device):
    model.eval()
    total_se = 0.0
    total_n = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        coords_gt = targets["coords_resized"].to(device)
        sizes = targets["resized_size"].to(device)

        hm_pred, _ = model(imgs)
        # 解析热力图坐标
        batch_coords = []
        for b in range(hm_pred.size(0)):
            w1, h1 = sizes[b]
            coord = heatmap_to_coord(hm_pred[b : b + 1], w1.item(), h1.item())
            batch_coords.append(coord[0])
        coords_pred = torch.stack(batch_coords, dim=0)

        se = F.mse_loss(coords_pred, coords_gt, reduction="sum").item()
        total_se += se
        total_n += coords_gt.numel()
    return total_se / total_n


def stage2_train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        hm_gt = targets["heatmap"].to(device)
        visible = targets["visible"].to(device)

        optimizer.zero_grad()
        hm_pred = model(imgs)

        mask = visible.view(-1, 1, 1, 1)
        # 加权 MSE：对热力图高值区域给予更大权重
        alpha = 4.0
        w = 1.0 + alpha * hm_gt
        hm_loss = ((hm_pred - hm_gt) ** 2 * w * mask).mean()

        hm_loss.backward()
        optimizer.step()
        total_loss += hm_loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def run_training(device):
    df = pd.read_csv(CSV_GT)
    all_ids = df["data"].values  # 1~80

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # 数据增强配置（仅用于训练集）
    train_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.HorizontalFlip(p=0.5),  # 水平翻转（如果解剖学允许）
    ])

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_ids), 1):
        print(f"=== Fold {fold}/{NUM_FOLDS} ===")
        train_ids = all_ids[train_idx]
        val_ids = all_ids[val_idx]

        # 阶段一
        train_ds1 = FoveaStage1Dataset(CSV_GT, IMG_DIR_TRAIN, indices=train_ids,
                                       transforms=train_transform)
        val_ds1 = FoveaStage1Dataset(CSV_GT, IMG_DIR_TRAIN, indices=val_ids)

        train_loader1 = DataLoader(
            train_ds1,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        val_loader1 = DataLoader(
            val_ds1,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        model1 = Stage1Net().to(device)
        opt1 = torch.optim.AdamW(model1.parameters(), lr=LR)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=NUM_EPOCHS)

        best_mse = float("inf")
        no_improve = 0
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = stage1_train_one_epoch(model1, train_loader1, opt1, device)
            val_mse = stage1_eval(model1, val_loader1, device)
            scheduler1.step()
            current_lr = scheduler1.get_last_lr()[0]
            print(
                f"[Fold {fold}] Epoch {epoch} Stage1 train_loss={train_loss:.4f} "
                f"val_mse={val_mse:.4f} lr={current_lr:.6f}"
            )
            if val_mse < best_mse:
                best_mse = val_mse
                no_improve = 0
                torch.save(
                    model1.state_dict(),
                    os.path.join("checkpoints", f"stage1_fold{fold}.pth"),
                )
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"[Fold {fold}] Stage1 Early stopping at epoch {epoch}")
                    break

        # 阶段二：使用 GT 裁剪 patch 训练精细热力图
        train_ds2 = FoveaStage2Dataset(CSV_GT, IMG_DIR_TRAIN, indices=train_ids,
                                      transforms=train_transform)
        val_ds2 = FoveaStage2Dataset(CSV_GT, IMG_DIR_TRAIN, indices=val_ids)

        train_loader2 = DataLoader(
            train_ds2,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        val_loader2 = DataLoader(
            val_ds2,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        model2 = Stage2Net().to(device)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=LR)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=NUM_EPOCHS)

        best_val_loss2 = float("inf")
        no_improve2 = 0
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss2 = stage2_train_one_epoch(model2, train_loader2, opt2, device)
            # 这里简单用验证集热力图 MSE 作为指标
            model2.eval()
            total_loss2 = 0.0
            total_n2 = 0
            with torch.no_grad():
                for imgs, targets in val_loader2:
                    imgs = imgs.to(device)
                    hm_gt = targets["heatmap"].to(device)
                    visible = targets["visible"].to(device)
                    hm_pred = model2(imgs)
                    mask = visible.view(-1, 1, 1, 1)
                    loss2 = F.mse_loss(hm_pred * mask, hm_gt * mask, reduction="sum")
                    total_loss2 += loss2.item()
                    total_n2 += imgs.size(0)
            val_loss2 = total_loss2 / max(total_n2, 1)
            scheduler2.step()
            current_lr2 = scheduler2.get_last_lr()[0]
            print(
                f"[Fold {fold}] Epoch {epoch} Stage2 train_loss={train_loss2:.4f} "
                f"val_loss={val_loss2:.4f} lr={current_lr2:.6f}"
            )
            if val_loss2 < best_val_loss2:
                best_val_loss2 = val_loss2
                no_improve2 = 0
                torch.save(
                    model2.state_dict(),
                    os.path.join("checkpoints", f"stage2_fold{fold}.pth"),
                )
            else:
                no_improve2 += 1
                if no_improve2 >= EARLY_STOP_PATIENCE:
                    print(f"[Fold {fold}] Stage2 Early stopping at epoch {epoch}")
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_training(device)


if __name__ == "__main__":
    main()


