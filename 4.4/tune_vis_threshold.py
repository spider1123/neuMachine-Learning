"""
可见性阈值优化脚本：在验证集上扫描最优阈值。
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from config import (
    IMG_DIR_TRAIN,
    CSV_GT,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_FOLDS,
    IMG_SIZE,
)
from datasets import FoveaStage1Dataset
from models import Stage1Net, heatmap_to_coord
from eval_mse import read_submission, main as eval_main


def evaluate_with_threshold(model, val_loader, device, threshold, out_path="temp_pred.csv"):
    """使用指定阈值评估模型，返回 MSE。"""
    import csv
    
    model.eval()
    rows = [("ImageID", "value")]
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            sizes = targets["resized_size"].to(device)
            visible = targets["visible"].to(device)
            idxs = targets["idx"]
            orig_sizes = targets["orig_size"].to(device)
            
            hm_pred, vis_logit = model(imgs)
            p_vis = torch.sigmoid(vis_logit)
            
            for b in range(hm_pred.size(0)):
                idx = int(idxs[b].item())
                w0, h0 = orig_sizes[b].tolist()
                w1, h1 = sizes[b].tolist()
                
                if p_vis[b].item() < threshold:
                    x_final, y_final = 0.0, 0.0
                else:
                    coord = heatmap_to_coord(hm_pred[b : b + 1], w1, h1)[0]
                    x_res, y_res = coord.tolist()
                    sx = w1 / float(w0)
                    sy = h1 / float(h0)
                    x_final = x_res / sx
                    y_final = y_res / sy
                
                rows.append((f"{idx}_Fovea_X", x_final))
                rows.append((f"{idx}_Fovea_Y", y_final))
    
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--thresh_min", type=float, default=0.3)
    parser.add_argument("--thresh_max", type=float, default=0.7)
    parser.add_argument("--thresh_step", type=float, default=0.05)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(CSV_GT)
    all_ids = df["data"].values
    
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    best_thresh_overall = 0.5
    best_mse_overall = float("inf")
    
    # 生成 GT submission 文件
    from make_train_gt_submission import main as make_gt
    gt_path = "train_gt_submission.csv"
    make_gt(gt_path)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_ids), 1):
        print(f"\n=== Fold {fold}/{args.folds} ===")
        val_ids = all_ids[val_idx]
        
        val_ds = FoveaStage1Dataset(CSV_GT, IMG_DIR_TRAIN, indices=val_ids)
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        
        # 加载模型
        model = Stage1Net().to(device)
        ckpt_path = os.path.join("checkpoints", f"stage1_fold{fold}.pth")
        if not os.path.isfile(ckpt_path):
            print(f"Warning: {ckpt_path} not found, skipping fold {fold}")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        best_thresh = 0.5
        best_mse = float("inf")
        
        # 扫描阈值
        for thresh in np.arange(args.thresh_min, args.thresh_max + args.thresh_step, args.thresh_step):
            pred_path = f"temp_pred_fold{fold}_thresh{thresh:.2f}.csv"
            evaluate_with_threshold(model, val_loader, device, thresh, pred_path)
            
            # 计算 MSE（只对验证集的样本）
            # 简化：直接读取预测文件并计算
            ids_pred, vals_pred = read_submission(pred_path)
            ids_gt, vals_gt = read_submission(gt_path)
            
            # 只保留验证集的样本
            val_ids_set = set([f"{idx}_Fovea_X" for idx in val_ids] + 
                            [f"{idx}_Fovea_Y" for idx in val_ids])
            mask = [id in val_ids_set for id in ids_pred]
            if sum(mask) == 0:
                continue
            
            vals_pred_filtered = np.array(vals_pred)[mask]
            vals_gt_filtered = np.array(vals_gt)[mask]
            mse = float(np.mean((vals_pred_filtered - vals_gt_filtered) ** 2))
            
            print(f"  Threshold {thresh:.2f}: MSE = {mse:.6f}")
            
            if mse < best_mse:
                best_mse = mse
                best_thresh = thresh
        
        print(f"Fold {fold} best threshold: {best_thresh:.2f}, MSE: {best_mse:.6f}")
        
        if best_mse < best_mse_overall:
            best_mse_overall = best_mse
            best_thresh_overall = best_thresh
    
    print(f"\n=== Overall Best Threshold ===")
    print(f"Best threshold: {best_thresh_overall:.2f}")
    print(f"Best MSE: {best_mse_overall:.6f}")
    print(f"\n建议在 config.py 中设置: VIS_THRESH = {best_thresh_overall}")


if __name__ == "__main__":
    main()

