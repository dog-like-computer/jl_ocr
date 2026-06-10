"""
统一OCR训练系统 - 端到端多字体识别
直接用一个模型识别所有字体，不依赖字体分类阶段
核心思路：预训练模型具备字体无关的特征提取能力，
         只需足够多样化的训练数据，模型能自动学会忽略字体差异
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
        text = item['text']
        char_indices = torch.tensor([CHARS.index(c) for c in text])
        return img, char_indices

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
# 模型定义 - 统一OCR（单阶段）
# ============================================================
class UnifiedOCRModel(nn.Module):
    """
    端到端OCR：用一个模型识别所有字体
    Backbone: ResNet/EfficientNet 提取字体无关的字符特征
    Heads: 4个字符位置各自的分类头
    """
    def __init__(self, num_chars=62, backbone='resnet18'):
        super().__init__()
        if backbone == 'resnet18':
            base = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'resnet34':
            base = models.resnet34(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='IMAGENET1K_V1')
            feature_dim = 1280
        else:
            base = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512

        # 冻结前70%层，微调后30% + 分类头
        all_params = list(base.parameters())
        freeze_count = int(len(all_params) * 0.7)
        for param in all_params[:freeze_count]:
            param.requires_grad = False

        if hasattr(base, 'fc'):
            base.fc = nn.Identity()
        elif hasattr(base, 'classifier'):
            base.classifier = nn.Identity()
        self.backbone = base

        # 全局特征归一化
        self.feature_norm = nn.LayerNorm(feature_dim)

        # 4个字符头，使用残差连接增强特征
        self.char_heads = nn.ModuleList()
        for _ in range(4):
            self.char_heads.append(nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_chars)
            ))

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_norm(features)
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        outputs = torch.stack([head(features) for head in self.char_heads], dim=1)
        return outputs  # shape: (batch, 4, 62)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                params_of_module = list(module.parameters(recurse=False))
                if params_of_module and all(not p.requires_grad for p in params_of_module):
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        module.eval()


# ============================================================
# 训练函数
# ============================================================
def get_optimizer(model, optimizer_name, lr):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == 'Adam':
        return optim.Adam(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    elif optimizer_name == 'SGD':
        return optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(trainable, lr=lr, alpha=0.9, weight_decay=0.01, momentum=0.9, eps=1e-8)
    return optim.Adam(trainable, lr=lr, weight_decay=0.01)


def clip_gradients(model, max_norm=0.5):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def train_unified_ocr(all_data, epochs=100, batch_size=32):
    """
    训练统一的端到端OCR模型（跳过字体分类）
    数据：使用全部20000张混合字体图片
    目标：直接预测4个字符的位置
    """
    print(f"\n{'='*70}")
    print("端到端OCR训练 (统一模型)")
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
    lrs = {'Adam': 3e-5, 'SGD': 1e-4, 'RMSprop': 3e-5}

    total_models = len(backbones) * len(optimizers_list)
    model_idx = 1

    for backbone in backbones:
        for optimizer_name in optimizers_list:
            lr = lrs[optimizer_name]
            print(f"\n[{model_idx}/{total_models}] {backbone} + {optimizer_name} (lr={lr})")
            model = UnifiedOCRModel(num_chars=NUM_CLASSES, backbone=backbone).to(device)
            optimizer = get_optimizer(model, optimizer_name, lr)

            warmup_epochs = min(5, epochs // 10)
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / max(warmup_epochs, 1)
                return max(1e-3, (1 + math.cos(math.pi * (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1))) / 2)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            best_acc = 0.0
            best_state = None

            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                char_correct = [0] * 4
                char_total = [0] * 4
                nan_count = 0

                pbar = tqdm(train_loader, desc=f'{backbone}+{optimizer_name} Epoch {epoch+1}/{epochs}')
                for images, char_indices in pbar:
                    images = images.to(device)
                    char_indices = char_indices.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)  # (batch, 4, 62)

                    # 计算4个位置的损失并平均
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
                        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                            grad_ok = False
                            break
                    if not grad_ok:
                        optimizer.zero_grad()
                        nan_count += 1
                        continue

                    clip_gradients(model, max_norm=0.5)
                    optimizer.step()

                    total_loss += loss.item() * images.size(0)
                    for i in range(4):
                        _, predicted = outputs[:, i, :].max(1)
                        char_correct[i] += predicted.eq(char_indices[:, i]).sum().item()
                        char_total[i] += images.size(0)

                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                scheduler.step()
                train_loss = total_loss / max(sum(char_total), 1)

                # 验证
                model.eval()
                val_correct = [0] * 4
                val_total = [0] * 4
                val_loss_sum = 0.0
                with torch.no_grad():
                    for images, char_indices in val_loader:
                        images = images.to(device)
                        char_indices = char_indices.to(device)
                        outputs = model(images)
                        vloss = torch.stack([
                            criterion(outputs[:, i, :], char_indices[:, i])
                            for i in range(4)
                        ]).mean()
                        if not (torch.isnan(vloss).item() or torch.isinf(vloss).item()):
                            val_loss_sum += vloss.item() * images.size(0)
                        for i in range(4):
                            _, predicted = outputs[:, i, :].max(1)
                            val_correct[i] += predicted.eq(char_indices[:, i]).sum().item()
                            val_total[i] += images.size(0)

                val_loss = val_loss_sum / max(sum(val_total), 1)
                total_correct = sum(val_correct)
                total_val = sum(val_total)
                val_acc = total_correct / max(total_val, 1)

                improved = val_acc > best_acc
                if improved:
                    best_acc = val_acc
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    torch.save(model.state_dict(), output_dir / f'unified_{backbone}_{optimizer_name}_best.pt')

                # 每个字符位置的准确率
                pos_accs = [f"pos{i+1}={val_correct[i]/max(val_total[i],1):.3f}" for i in range(4)]

                if (epoch + 1) % 10 == 0 or improved or nan_count > 0:
                    flag = " *BEST*" if improved else ""
                    nan_str = f" NaN={nan_count}" if nan_count > 0 else ""
                    print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, val_acc={val_acc:.4f} ({' '.join(pos_accs)}){flag}{nan_str}")

            if best_state is not None:
                model.load_state_dict(best_state)
            torch.save(model.state_dict(), output_dir / f'unified_{backbone}_{optimizer_name}_final.pt')

            results.append({
                'backbone': backbone,
                'optimizer': optimizer_name,
                'best_val_acc': float(best_acc)
            })
            print(f"  最佳验证准确率: {best_acc:.4f}")

            del model
            torch.cuda.empty_cache()
            gc.collect()
            model_idx += 1

    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    print(f"\n统一OCR模型结果排名:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['backbone']:15s} + {r['optimizer']:10s}: {r['best_val_acc']:.4f}")

    return results


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    start_time = time.time()

    all_data = []
    for img_name in font_labels.keys():
        label = font_labels[img_name]
        all_data.append({'name': img_name, 'text': label['text']})

    print(f"总数据量: {len(all_data)}")

    ocr_results = train_unified_ocr(all_data, epochs=100, batch_size=32)
    best_model = ocr_results[0]

    total_time = time.time() - start_time

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'device': str(device),
        'backbones': ['resnet18', 'efficientnet_b0', 'resnet34'],
        'optimizers': ['Adam', 'SGD', 'RMSprop'],
        'total_models': 9,
        'dataset': {
            'total_images': len(all_data),
            'num_fonts': NUM_FONTS,
            'num_chars': NUM_CLASSES
        },
        'unified_ocr_results': {
            'description': '端到端OCR，无字体分类阶段',
            'results': ocr_results,
            'best_model': best_model
        }
    }

    with open(output_dir / 'unified_ocr_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(output_dir / 'unified_ocr_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("统一OCR训练报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"设备: {device}\n")
        f.write(f"训练时间: {total_time/3600:.2f} 小时\n\n")
        f.write("端到端OCR模型排名\n" + "-" * 40 + "\n")
        for i, r in enumerate(ocr_results, 1):
            f.write(f"  {i}. {r['backbone']:15s} + {r['optimizer']:10s}: {r['best_val_acc']:.4f}\n")
        f.write(f"\n最佳模型: {best_model['backbone']} + {best_model['optimizer']} 准确率: {best_model['best_val_acc']:.4f}\n")

    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"最佳模型: {best_model['backbone']} + {best_model['optimizer']} 准确率: {best_model['best_val_acc']:.4f}")
    print(f"{'='*70}")
