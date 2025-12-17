"""
训练脚本：固定划分训练集和验证集，确保验证集固定以可靠评估模型进步
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import Config
from model import create_resnet50
from dataset import get_train_val_split, GrayImageDataset, get_train_transforms, get_val_transforms
from losses import MultiTaskLoss, generate_heatmap_from_landmarks
from landmark_utils import generate_landmarks_for_dataset


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, use_landmark=False, num_landmarks=98):
    """训练一个epoch（支持多任务学习）"""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_heatmap_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        if use_landmark and len(batch) == 3:
            images, labels, landmarks = batch
            images = images.to(device)
            labels = labels.to(device)
            landmarks = landmarks.to(device)  # (B, num_landmarks*2)
            
            # 生成热图目标
            landmarks_reshaped = landmarks.view(-1, num_landmarks, 2)
            heatmap_target = generate_heatmap_from_landmarks(
                landmarks_reshaped, 
                heatmap_size=56, 
                sigma=2.0
            ).to(device)  # (B, num_landmarks, 56, 56)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            heatmap_target = None
        
        # 前向传播
        optimizer.zero_grad()
        if use_landmark and heatmap_target is not None:
            cls_logits, heatmap_pred = model(images)
            total_loss, cls_loss, heatmap_loss = criterion(
                cls_logits, labels, heatmap_pred, heatmap_target
            )
        else:
            cls_logits = model(images)
            if isinstance(cls_logits, tuple):
                cls_logits = cls_logits[0]  # 如果模型返回tuple，取第一个
            total_loss, cls_loss, heatmap_loss = criterion(cls_logits, labels)
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 更新学习率（OneCycleLR需要在每个batch后更新）
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # 统计
        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        if heatmap_loss.item() > 0:
            running_heatmap_loss += heatmap_loss.item()
        _, predicted = cls_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        current_lr = optimizer.param_groups[0]['lr']
        postfix = {
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'cls': f'{running_cls_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%',
            'lr': f'{current_lr:.6f}'
        }
        if use_landmark and running_heatmap_loss > 0:
            postfix['heatmap'] = f'{running_heatmap_loss / (pbar.n + 1):.4f}'
        pbar.set_postfix(postfix)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_cls_loss = running_cls_loss / len(train_loader)
    epoch_heatmap_loss = running_heatmap_loss / len(train_loader) if running_heatmap_loss > 0 else 0.0
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, epoch_cls_loss, epoch_heatmap_loss


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_landmark=False, num_landmarks=98):
    """验证（支持多任务学习）"""
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_heatmap_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc='Validating')
    for batch in pbar:
        if use_landmark and len(batch) == 3:
            images, labels, landmarks = batch
            images = images.to(device)
            labels = labels.to(device)
            landmarks = landmarks.to(device)
            
            # 生成热图目标
            landmarks_reshaped = landmarks.view(-1, num_landmarks, 2)
            heatmap_target = generate_heatmap_from_landmarks(
                landmarks_reshaped, 
                heatmap_size=56, 
                sigma=2.0
            ).to(device)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            heatmap_target = None
        
        if use_landmark and heatmap_target is not None:
            cls_logits, heatmap_pred = model(images)
            total_loss, cls_loss, heatmap_loss = criterion(
                cls_logits, labels, heatmap_pred, heatmap_target
            )
        else:
            cls_logits = model(images)
            if isinstance(cls_logits, tuple):
                cls_logits = cls_logits[0]
            total_loss, cls_loss, heatmap_loss = criterion(cls_logits, labels)
        
        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        if heatmap_loss.item() > 0:
            running_heatmap_loss += heatmap_loss.item()
        _, predicted = cls_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        postfix = {
            'loss': f'{running_loss / len(val_loader):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        }
        if use_landmark and running_heatmap_loss > 0:
            postfix['heatmap'] = f'{running_heatmap_loss / len(val_loader):.4f}'
        pbar.set_postfix(postfix)
    
    epoch_loss = running_loss / len(val_loader)
    epoch_cls_loss = running_cls_loss / len(val_loader)
    epoch_heatmap_loss = running_heatmap_loss / len(val_loader) if running_heatmap_loss > 0 else 0.0
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, epoch_cls_loss, epoch_heatmap_loss


def main():
    config = Config()
    
    # 设置随机种子（用于模型初始化等）
    torch.manual_seed(config.base_seed)
    np.random.seed(config.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.base_seed)
    
    # 启用cudnn benchmark以加速训练（输入尺寸固定224×224）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print('CUDNN benchmark enabled for faster training')
    
    print(f'Using device: {config.device}')
    print(f'Base seed: {config.base_seed}')
    print('Using random initialization (no pretrained weights)')
    print(f'Model weights will be initialized with seed: {config.base_seed}')
    
    # 创建输出目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 创建模型（使用base_seed进行随机初始化）
    # 注意：模型创建前已经设置了随机种子，确保权重初始化可复现
    model = create_resnet50(
        num_classes=config.num_classes,
        num_landmarks=config.num_landmarks,
        use_landmark=config.use_landmark
    )
    model = model.to(config.device)
    
    # 损失函数（多任务损失：分类 + 关键点热图）
    if config.use_landmark:
        criterion = MultiTaskLoss(
            num_classes=config.num_classes,
            num_landmarks=config.num_landmarks,
            label_smoothing=config.label_smoothing,
            landmark_weight=config.landmark_weight
        )
        print(f'Using MultiTaskLoss (cls_weight={1-config.landmark_weight:.1f}, heatmap_weight={config.landmark_weight:.1f})')
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        print(f'Using CrossEntropyLoss with label_smoothing={config.label_smoothing}')
    
    # 优化器：AdamW（推荐用于深度学习）
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    print(f'Using AdamW optimizer (lr={config.learning_rate}, weight_decay={config.weight_decay})')
    
    # 固定划分一次（在训练开始前）
    # 使用固定的随机种子，确保验证集固定，可以可靠评估模型进步
    print(f'Using seed: {config.base_seed} for data split (fixed)')
    train_paths, train_labels, val_paths, val_labels = get_train_val_split(
        '.', config.train_dir, 
        config.train_ratio, config.val_ratio, 
        seed=config.base_seed  # 固定随机种子
    )
    
    # 加载关键点标注（如果使用）
    landmark_data = {}
    if config.use_landmark:
        print("Loading landmark annotations...")
        all_paths = train_paths + val_paths
        landmark_data = generate_landmarks_for_dataset(
            all_paths,
            landmark_dir=config.landmark_dir,
            num_landmarks=config.num_landmarks,
            auto_detect=False  # 如果标注不存在，不自动检测（需要手动准备）
        )
        if len(landmark_data) > 0:
            print(f"Loaded landmarks for {len(landmark_data)}/{len(all_paths)} images")
        else:
            print("⚠ Warning: No landmark data found. Set landmark_dir or provide .npy/.json files.")
            print("  Continuing without landmark supervision...")
            config.use_landmark = False
    
    # 创建数据集（固定划分，不每轮重新创建）
    train_dataset = GrayImageDataset(
        train_paths, train_labels,
        transform=get_train_transforms(),
        landmark_data=landmark_data,
        use_landmark=config.use_landmark
    )
    
    val_dataset = GrayImageDataset(
        val_paths, val_labels,
        transform=get_val_transforms(),
        landmark_data=landmark_data,
        use_landmark=config.use_landmark
    )
    
    # 创建数据加载器（固定划分）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 训练时shuffle，但划分是固定的
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 学习率调度器：OneCycleLR（推荐）或CosineAnnealingLR
    if config.use_onecycle:
        total_steps = config.epochs * len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            total_steps=total_steps,
            pct_start=config.pct_start,
            anneal_strategy='cos'
        )
        print(f'Using OneCycleLR scheduler (max_lr={config.max_lr}, total_steps={total_steps}, pct_start={config.pct_start})')
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
        print(f'Using CosineAnnealingLR scheduler (T_max={config.epochs}, eta_min=1e-6)')
    
    # 训练循环
    best_val_acc = 0.0
    patience = 30  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch+1}/{config.epochs}')
        print('-' * 50)
        
        # 每个epoch创建新的DataLoader，打乱训练集顺序
        # 注意：数据划分是固定的，但每个epoch的顺序会打乱，避免记忆时序
        train_loader_epoch = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,  # 每个epoch打乱训练集顺序
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader_epoch = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,  # 验证集不需要shuffle
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 训练（OneCycleLR在train_epoch内部更新，其他scheduler在外部更新）
        train_loss, train_acc, train_cls_loss, train_heatmap_loss = train_epoch(
            model, train_loader_epoch, criterion, optimizer, config.device,
            scheduler if config.use_onecycle else None,
            use_landmark=config.use_landmark,
            num_landmarks=config.num_landmarks if config.use_landmark else 98
        )
        
        # 验证
        val_loss, val_acc, val_cls_loss, val_heatmap_loss = validate(
            model, val_loader_epoch, criterion, config.device,
            use_landmark=config.use_landmark,
            num_landmarks=config.num_landmarks if config.use_landmark else 98
        )
        
        # 更新学习率调度器（非OneCycleLR的情况）
        if not config.use_onecycle:
            scheduler.step()
        
        print(f'Train - Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}', end='')
        if config.use_landmark and train_heatmap_loss > 0:
            print(f', heatmap: {train_heatmap_loss:.4f})', end='')
        else:
            print(')', end='')
        print(f', Acc: {train_acc:.2f}%')
        
        print(f'Val   - Loss: {val_loss:.4f} (cls: {val_cls_loss:.4f}', end='')
        if config.use_landmark and val_heatmap_loss > 0:
            print(f', heatmap: {val_heatmap_loss:.4f})', end='')
        else:
            print(')', end='')
        print(f', Acc: {val_acc:.2f}%')
        print(f'LR    - {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'use_landmark': config.use_landmark,  # 保存配置信息
                'num_landmarks': config.num_landmarks if config.use_landmark else 98,
                'num_classes': config.num_classes,
            }, config.best_model_path)
            print(f'✓ Saved best model (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {patience} epochs without improvement')
                break
    
    print(f'\nTraining completed!')
    print(f'Best Val Accuracy: {best_val_acc:.2f}%')


if __name__ == '__main__':
    main()



