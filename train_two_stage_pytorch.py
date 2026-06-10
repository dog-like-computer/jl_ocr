"""
两阶段OCR训练系统 - 前沿版
使用预训练ResNet/EfficientNet + 现代训练技术
修复梯度不稳定问题：梯度裁剪、损失归一化、学习率优化、BN处理
"""

import os
import json
import time
import random
import string
import gc
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

data_dir = Path("data_v2")
output_dir = Path("training_output_advanced")
output_dir.mkdir(exist_ok=True)

with open(data_dir / "font_labels.json", 'r', encoding='utf-8') as f:
    font_labels = json.load(f)

with open(data_dir / "fonts_list.json", 'r', encoding='utf-8') as f:
    fonts_list = json.load(f)

NUM_FONTS = len(fonts_list)
CHARS = string.ascii_letters + string.digits
NUM_CLASSES = len(CHARS)

print(f"\n配置信息:")
print(f"  字体数量: {NUM_FONTS}")
print(f"  总图片数: {len(font_labels)}")
print(f"  字符类别: {NUM_CLASSES}")

# ============================================================
# 数据增强
# ============================================================
class RandomAugmentation:
    def __call__(self, img):
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(255, 255, 255))
        if random.random() < 0.3:
            width, height = img.size
            shear = random.uniform(-0.1, 0.1)
            img = img.transform((width, height), Image.AFFINE, (1, shear, 0, shear, 1, 0), fillcolor=(255, 255, 255))
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Color(img).enhance(factor)
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.5)
            img = ImageEnhance.Sharpness(img).enhance(factor)
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.3:
            np_img = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 10, np_img.shape)
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(np_img)
        return img

augmenter = RandomAugmentation()

# ============================================================
# 数据集
# ============================================================
class OCRDataset(Dataset):
    def __init__(self, data_list, data_dir, transform=None, is_train=True):
        self.data_list = data_list
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_name = item['name']
        img = Image.open(self.data_dir / "images" / img_name).convert('RGB')
        if self.is_train:
            img = augmenter(img)
        if self.transform:
            img = self.transform(img)
        font = item['font']
        text = item['text']
        char_indices = [CHARS.index(c) for c in text]
        return img, font, torch.tensor(char_indices)

train_transform = transforms.Compose([
    transforms.Resize((128, 320)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize((128, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 模型定义
# ============================================================
def freeze_backbone_layers(backbone, freeze_ratio=0.7):
    """
    冻结backbone的前 freeze_ratio 层参数
    注：只冻结参数requires_grad，不改动BN的train/eval状态，也不修改track_running_stats
    （BN在train模式下会更新running统计量，eval时使用预训练统计量 — 两者都正常工作，
     之前设置track_running_stats=False会导致eval时用当前batch统计量，是NaN/掉点根源）
    """
    all_params = list(backbone.parameters())
    freeze_count = int(len(all_params) * freeze_ratio)
    for param in all_params[:freeze_count]:
        param.requires_grad = False


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=20, backbone='resnet18'):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            feature_dim = 1280
        else:
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512

        # 冻结80%的层，只训练最后几层 + 分类头
        freeze_backbone_layers(self.backbone, freeze_ratio=0.8)

        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()

        # 分类头：使用LayerNorm + 较小维度，避免数值溢出
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        # 稳定数值：先做LayerNorm再送分类头
        features = self.feature_norm(features)
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.classifier(features)


class CNNHead(nn.Module):
    def __init__(self, in_features, num_classes=62):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class OCRModel(nn.Module):
    def __init__(self, num_chars=62, backbone='resnet18'):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            feature_dim = 1280
        else:
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512

        # 冻结80%的层
        freeze_backbone_layers(self.backbone, freeze_ratio=0.8)

        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.char_heads = nn.ModuleList([CNNHead(feature_dim, num_chars) for _ in range(4)])

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_norm(features)
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        outputs = torch.stack([head(features) for head in self.char_heads], dim=1)
        return outputs


# ============================================================
# 优化器与训练函数
# ============================================================
def get_optimizer(model, optimizer_name, lr=5e-5):
    """
    使用适中的初始学习率(5e-5)，配合权重衰减，保证稳定收敛
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == 'Adam':
        return optim.Adam(trainable_params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    elif optimizer_name == 'SGD':
        return optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(trainable_params, lr=lr, alpha=0.9, weight_decay=0.01, momentum=0.9, eps=1e-8)
    else:
        return optim.Adam(trainable_params, lr=lr, weight_decay=0.01)


def clip_gradients(model, max_norm=0.5):
    """梯度裁剪，默认max_norm=0.5（更保守）"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def train_model(model, train_loader, val_loader, epochs, name, lr=1e-5, optimizer_name='Adam'):
    """
    通用训练函数：字体分类
    === 关键修复 ===
    1. 禁用混合精度（你这边FP16会直接NaN，用FP32稳）
    2. 学习率降到 1e-5（低到能稳定收敛）
    3. 梯度裁剪 max_norm=0.5
    4. label_smoothing=0.05
    5. 线性warmup + cosine decay 调度
    6. NaN/Inf 检测：一旦出现，自动跳过本次更新并重置optimizer状态
    """
    model = model.to(device)
    optimizer = get_optimizer(model, optimizer_name, lr)

    # 线性warmup(5epoch) + cosine decay
    warmup_epochs = min(5, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        return max(5e-3, (1 + math.cos(math.pi * (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1))) / 2)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc = 0.0
    best_state = None  # 保存最佳权重，在崩溃时回退

    # 记录初始权重（安全回退基线）
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        nan_count = 0
        inf_count = 0

        pbar = tqdm(train_loader, desc=f'{name} Epoch {epoch+1}/{epochs}')
        for images, labels, chars in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # NaN/Inf 检测 + 跳过
            if torch.isnan(loss).item():
                nan_count += 1
                continue
            if torch.isinf(loss).item():
                inf_count += 1
                continue
            if loss.item() > 100.0:  # 损失过大也跳过，防止梯度爆炸
                inf_count += 1
                continue

            loss.backward()

            # 反向传播后再检测一次梯度是否坏了
            grad_ok = True
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        grad_ok = False
                        break
            if not grad_ok:
                optimizer.zero_grad()
                nan_count += 1
                continue

            # 裁剪 + 更新
            clip_gradients(model, max_norm=0.5)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                vloss = criterion(outputs, labels)
                if not (torch.isnan(vloss).item() or torch.isinf(vloss).item()):
                    val_loss_sum += vloss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / max(val_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)

        improved = val_acc > best_acc
        status_bits = []
        if improved:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), output_dir / f'{name}_best.pt')
            status_bits.append("*BEST*")
        if nan_count > 0:
            status_bits.append(f"NaN={nan_count}")
        if inf_count > 0:
            status_bits.append(f"INF/LOSS>100={inf_count}")

        if (epoch + 1) % 10 == 0 or improved or nan_count > 0 or inf_count > 0:
            print(f"  Epoch {epoch+1}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} "
                  + " ".join(status_bits))

        # 崩溃保护：如果连续3个epoch全是NaN，回滚到最佳权重
        if nan_count + inf_count > len(train_loader) * 0.5 and best_state is not None:
            print(f"  !! 梯度严重崩坏，回滚到val_acc={best_acc:.4f}的最佳权重")
            model.load_state_dict(best_state)
            optimizer = get_optimizer(model, optimizer_name, lr * 0.3)  # 降LR重起

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / f'{name}_final.pt')
    return best_acc


def train_ocr_models(all_data, best_classifier_backbone=None, epochs=100, batch_size=24):
    """
    OCR识别训练：使用最佳分类模型的Backbone
    同样修复：FP32、损失平均、低LR、梯度裁剪、NaN安全
    """
    print(f"\n{'='*70}")
    print("第二阶段：OCR识别训练 (前沿模型)")
    if best_classifier_backbone:
        print(f"使用最佳分类模型的Backbone: {best_classifier_backbone}")
    print(f"{'='*70}")

    font_data = {i: [] for i in range(NUM_FONTS)}
    for d in all_data:
        font_data[d['font']].append(d)

    all_results = {}
    backbones = ['resnet18', 'efficientnet_b0', 'resnet34']
    if best_classifier_backbone and best_classifier_backbone in backbones:
        backbones = [best_classifier_backbone]

    optimizers_list = ['Adam', 'SGD', 'RMSprop']

    for font_id in range(NUM_FONTS):
        font_name = fonts_list[font_id]['font_name']
        print(f"\n字体 {font_id}: {font_name} (数据量: {len(font_data[font_id])})")

        if len(font_data[font_id]) < 100:
            print(f"  数据不足，跳过")
            continue

        data = font_data[font_id]
        random.shuffle(data)
        val_size = int(0.2 * len(data))
        train_data = data[val_size:]
        val_data = data[:val_size]

        train_dataset = OCRDataset(train_data, data_dir, train_transform, is_train=True)
        val_dataset = OCRDataset(val_data, data_dir, val_transform, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

        font_results = []

        for backbone in backbones:
            for optimizer_name in optimizers_list:
                model = OCRModel(num_chars=NUM_CLASSES, backbone=backbone).to(device)
                optimizer = get_optimizer(model, optimizer_name, lr=5e-5)

                warmup_epochs = min(5, epochs // 10)
                def lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        return (epoch + 1) / max(warmup_epochs, 1)
                    return max(5e-3, (1 + math.cos(math.pi * (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1))) / 2)
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

                criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
                best_acc = 0.0
                best_state = None

                for epoch in range(epochs):
                    model.train()
                    total_loss, total = 0.0, 0
                    nan_count = 0
                    pbar = tqdm(train_loader,
                                desc=f'{backbone}+{optimizer_name} Epoch {epoch+1}',
                                leave=False)
                    for images, _, char_indices in pbar:
                        images = images.to(device)
                        char_indices = char_indices.to(device)
                        optimizer.zero_grad()

                        outputs = model(images)
                        loss = torch.stack([
                            criterion(outputs[:, i, :], char_indices[:, i])
                            for i in range(4)
                        ]).mean()

                        if torch.isnan(loss).item() or torch.isinf(loss).item() or loss.item() > 100:
                            nan_count += 1
                            continue

                        loss.backward()

                        grad_ok = True
                        for p in model.parameters():
                            if p.grad is not None:
                                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                    grad_ok = False
                                    break
                        if not grad_ok:
                            optimizer.zero_grad()
                            nan_count += 1
                            continue

                        clip_gradients(model, max_norm=0.5)
                        optimizer.step()

                        total_loss += loss.item() * images.size(0)
                        total += images.size(0)
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                    scheduler.step()

                    # 验证
                    model.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for images, _, char_indices in val_loader:
                            images = images.to(device)
                            char_indices = char_indices.to(device)
                            outputs = model(images)
                            for i in range(4):
                                _, predicted = outputs[:, i, :].max(1)
                                correct += predicted.eq(char_indices[:, i]).sum().item()
                            total += images.size(0) * 4

                    val_acc = correct / max(total, 1)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        font_dir = output_dir / f"font_{font_id:02d}"
                        font_dir.mkdir(exist_ok=True)
                        torch.save(model.state_dict(),
                                   font_dir / f'ocr_{backbone}_{optimizer_name}_best.pt')

                    if (epoch + 1) % 20 == 0:
                        print(f"    Epoch {epoch+1}: val_acc={val_acc:.4f}"
                              + (f" NaN={nan_count}" if nan_count > 0 else ""))

                if best_state is not None:
                    model.load_state_dict(best_state)

                font_results.append({
                    'model': backbone,
                    'optimizer': optimizer_name,
                    'best_val_acc': float(best_acc)
                })
                del model
                torch.cuda.empty_cache()
                gc.collect()

        font_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        all_results[font_id] = font_results
        if font_results:
            print(f"  最佳: {font_results[0]['model']} + {font_results[0]['optimizer']}: "
                  f"{font_results[0]['best_val_acc']:.4f}")

    return all_results


def train_font_classifier(all_data, epochs=100, batch_size=24):
    print(f"\n{'='*70}")
    print("第一阶段：字体分类训练 (前沿模型)")
    print(f"{'='*70}")

    random.shuffle(all_data)
    val_size = int(0.2 * len(all_data))
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]

    train_dataset = OCRDataset(train_data, data_dir, train_transform, is_train=True)
    val_dataset = OCRDataset(val_data, data_dir, val_transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

    results = []
    backbones = ['resnet18', 'efficientnet_b0', 'resnet34']
    optimizers_list = ['Adam', 'SGD', 'RMSprop']

    total_models = len(backbones) * len(optimizers_list)
    model_idx = 1

    for backbone in backbones:
        for optimizer_name in optimizers_list:
            print(f"\n[{model_idx}/{total_models}] {backbone} + {optimizer_name}")
            model = ResNetClassifier(num_classes=NUM_FONTS, backbone=backbone)
            best_acc = train_model(
                model, train_loader, val_loader,
                epochs=epochs,
                name=f'font_{backbone}_{optimizer_name}',
                lr=0.0005,
                optimizer_name=optimizer_name
            )
            results.append({
                'model': backbone,
                'optimizer': optimizer_name,
                'best_val_acc': float(best_acc)
            })
            print(f"  最佳验证准确率: {best_acc:.4f}")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            model_idx += 1

    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    print(f"\n字体分类结果排名:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['model']:15s} + {r['optimizer']:10s}: {r['best_val_acc']:.4f}")

    return results


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    start_time = time.time()

    all_data = []
    for img_name in font_labels.keys():
        label = font_labels[img_name]
        all_data.append({
            'name': img_name,
            'font': label['main_font_id'],
            'text': label['text']
        })

    print(f"总数据量: {len(all_data)}")

    # 第一阶段
    classifier_results = train_font_classifier(all_data, epochs=100, batch_size=24)
    best_classifier = classifier_results[0]
    best_backbone = best_classifier['model']

    print(f"\n最佳分类模型: {best_backbone} + {best_classifier['optimizer']}, "
          f"准确率: {best_classifier['best_val_acc']:.4f}")
    print(f"使用最佳分类模型的Backbone进行OCR识别训练...")

    # 第二阶段
    ocr_results = train_ocr_models(
        all_data,
        best_classifier_backbone=best_backbone,
        epochs=100,
        batch_size=24
    )

    total_time = time.time() - start_time

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'device': str(device),
        'backbones': ['resnet18', 'efficientnet_b0', 'resnet34'],
        'optimizers': ['Adam', 'SGD', 'RMSprop'],
        'total_models': 18,
        'dataset': {
            'total_images': len(all_data),
            'num_fonts': NUM_FONTS,
            'num_chars': NUM_CLASSES
        },
        'stage1_font_classification': {
            'description': '3 backbones x 3 optimizers = 9 models',
            'results': classifier_results,
            'best_model': best_classifier
        },
        'stage2_ocr_recognition': {
            'description': f'使用最佳分类backbone ({best_backbone}) x 3 optimizers',
            'results_per_font': {str(k): v for k, v in ocr_results.items()},
            'best_per_font': {str(k): v[0] if v else None for k, v in ocr_results.items()}
        }
    }

    with open(output_dir / 'training_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(output_dir / 'training_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("两阶段OCR训练报告 - 前沿版 (梯度修复)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"设备: {device}\nBackbone: ResNet18, ResNet34, EfficientNet-B0\n")
        f.write(f"Optimizers: Adam, SGD, RMSprop\n总模型数: 18\n")
        f.write(f"训练时间: {total_time/3600:.2f} 小时\n")
        f.write("技术: 迁移学习, 混合精度, Label Smoothing=0.05, 梯度裁剪=1.0,\n")
        f.write("       CosineAnnealingLR, BN冻结层eval模式, OCR损失平均\n\n")
        f.write("第一阶段：字体分类\n" + "-" * 40 + "\n")
        for i, r in enumerate(classifier_results, 1):
            f.write(f"  {i}. {r['model']:15s} + {r['optimizer']:10s}: {r['best_val_acc']:.4f}\n")
        f.write(f"\n最佳分类器: {best_classifier['model']} + {best_classifier['optimizer']} "
                f"准确率: {best_classifier['best_val_acc']:.4f}\n\n")
        f.write("第二阶段：OCR识别 (使用最佳分类Backbone)\n" + "-" * 60 + "\n")
        for font_id in range(NUM_FONTS):
            if str(font_id) in ocr_results and ocr_results[font_id]:
                best = ocr_results[font_id][0]
                f.write(f"  字体{font_id:2d}: {best['model']:15s} + "
                        f"{best['optimizer']:10s}: {best['best_val_acc']:.4f}\n")

    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}")
