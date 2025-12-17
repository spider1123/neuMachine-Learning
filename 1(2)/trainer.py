# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm
from config import cfg
from dataloader import create_epoch_loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, full_dataset, class_to_idx, num_classes, idx_to_class):
    """
    训练模型，每轮训练前重新划分训练集和验证集
    
    参数:
        model: 要训练的模型
        full_dataset: 完整的训练数据集（ImageFolder 对象）
        class_to_idx: 类别到索引的映射字典
        num_classes: 类别数量
        idx_to_class: 索引到类别的映射字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.LEARNING_RATE, 
        weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    
    best_acc = 0.0
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    
    print(f"\n开始训练，共 {cfg.EPOCHS} 个 epoch...")
    print(f"每轮训练前将按 {cfg.TRAIN_RATIO:.0%}/{cfg.VAL_RATIO:.0%} 重新划分训练集和验证集")
    print("-" * 60)
    
    for epoch in range(1, cfg.EPOCHS + 1):
        # 每轮训练前重新划分数据集
        print(f"\nEpoch {epoch:03d}/{cfg.EPOCHS}: 重新划分数据集...")
        train_loader, val_loader = create_epoch_loaders(
            full_dataset, epoch, class_to_idx
        )
        # 计算原始训练样本数（增强前的数量）
        original_train_size = len(train_loader.dataset) // (1 + cfg.NUM_AUGMENTATIONS)
        print(f"  原始训练集: {original_train_size} 张")
        print(f"  增强后训练集: {len(train_loader.dataset)} 张 (每张原始图片生成 {1 + cfg.NUM_AUGMENTATIONS} 个样本)")
        print(f"  验证集: {len(val_loader.dataset)} 张")
        
        # 训练和验证
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch:03d}/{cfg.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'idx_to_class': idx_to_class,
                'acc': best_acc,
                'epoch': epoch
            }, os.path.join(cfg.MODEL_SAVE_DIR, "best_model.pth"))
            print(f"  -> 保存最佳模型 (验证集准确率: {best_acc:.4f})")
    
    print("-" * 60)
    print(f"训练完成！最佳验证集准确率: {best_acc:.4f}")
    return model
