"""
两阶段OCR训练系统 - 前沿版
使用预训练ResNet/EfficientNet + 现代训练技术
"""

import os
import json
import time
import random
import string
import gc
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
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
])

val_transform = transforms.Compose([
    transforms.Resize((128, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class CNNHead(nn.Module):
    def __init__(self, in_features, num_classes=62):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)

class OCRModel(nn.Module):
    def __init__(self, num_chars=62, backbone='resnet18'):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            feature_dim = 1280
        else:
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        
        self.char_heads = nn.ModuleList([CNNHead(feature_dim, num_chars) for _ in range(4)])
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = torch.stack([head(features) for head in self.char_heads], dim=1)
        return outputs

def train_model(model, train_loader, val_loader, epochs, name, lr=0.001):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'{name} Epoch {epoch+1}/{epochs}')
        for images, labels, chars in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total
        
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with autocast():
                    outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += images.size(0)
        
        val_acc = val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / f'{name}_best.pt')
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    torch.save(model.state_dict(), output_dir / f'{name}_final.pt')
    return best_acc

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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    results = []
    backbones = ['resnet18', 'efficientnet_b0', 'resnet34']
    
    for idx, backbone in enumerate(backbones, 1):
        print(f"\n[{idx}/3] {backbone}")
        model = ResNetClassifier(num_classes=NUM_FONTS, backbone=backbone)
        best_acc = train_model(model, train_loader, val_loader, epochs=epochs, name=f'font_{backbone}', lr=0.001)
        results.append({'model': backbone, 'optimizer': 'AdamW', 'best_val_acc': float(best_acc)})
        print(f"  最佳验证准确率: {best_acc:.4f}")
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    print(f"\n字体分类结果排名:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['model']:15s} + {r['optimizer']:10s}: {r['best_val_acc']:.4f}")
    
    return results

def train_ocr_models(all_data, epochs=100, batch_size=24):
    print(f"\n{'='*70}")
    print("第二阶段：OCR识别训练 (前沿模型)")
    print(f"{'='*70}")
    
    font_data = {i: [] for i in range(NUM_FONTS)}
    for d in all_data:
        font_data[d['font']].append(d)
    
    all_results = {}
    backbones = ['resnet18', 'efficientnet_b0', 'resnet34']
    
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        font_results = []
        
        for backbone in backbones:
            model = OCRModel(num_chars=NUM_CLASSES, backbone=backbone).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            best_acc = 0
            
            for epoch in range(epochs):
                model.train()
                pbar = tqdm(train_loader, desc=f'{backbone} Epoch {epoch+1}', leave=False)
                for images, _, char_indices in pbar:
                    images = images.to(device)
                    char_indices = char_indices.to(device)
                    optimizer.zero_grad()
                    with autocast():
                        outputs = model(images)
                        loss = sum(nn.CrossEntropyLoss(label_smoothing=0.1)(outputs[:, i, :], char_indices[:, i]) for i in range(4))
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                scheduler.step()
                
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for images, _, char_indices in val_loader:
                        images = images.to(device)
                        char_indices = char_indices.to(device)
                        with autocast():
                            outputs = model(images)
                        for i in range(4):
                            _, predicted = outputs[:, i, :].max(1)
                            correct += predicted.eq(char_indices[:, i]).sum().item()
                        total += images.size(0) * 4
                
                val_acc = correct / total
                if val_acc > best_acc:
                    best_acc = val_acc
                    font_dir = output_dir / f"font_{font_id:02d}"
                    font_dir.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), font_dir / f'ocr_{backbone}_best.pt')
            
            font_results.append({'model': backbone, 'optimizer': 'AdamW', 'best_val_acc': float(best_acc)})
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        font_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        all_results[font_id] = font_results
        print(f"  最佳: {font_results[0]['model']} + {font_results[0]['optimizer']}: {font_results[0]['best_val_acc']:.4f}")
    
    return all_results

if __name__ == "__main__":
    start_time = time.time()
    
    all_data = []
    for img_name in font_labels.keys():
        label = font_labels[img_name]
        all_data.append({'name': img_name, 'font': label['main_font_id'], 'text': label['text']})
    
    print(f"总数据量: {len(all_data)}")
    
    classifier_results = train_font_classifier(all_data, epochs=100, batch_size=24)
    best_classifier = classifier_results[0]
    ocr_results = train_ocr_models(all_data, epochs=100, batch_size=24)
    
    total_time = time.time() - start_time
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'device': str(device),
        'backbone': ['resnet18', 'efficientnet_b0', 'resnet34'],
        'dataset': {'total_images': len(all_data), 'num_fonts': NUM_FONTS, 'num_chars': NUM_CLASSES},
        'stage1_font_classification': {'description': '3 pretrained backbones', 'results': classifier_results, 'best_model': best_classifier},
        'stage2_ocr_recognition': {
            'description': '20 fonts x 3 pretrained backbones',
            'results_per_font': {str(k): v for k, v in ocr_results.items()},
            'best_per_font': {str(k): v[0] if v else None for k, v in ocr_results.items()}
        }
    }
    
    with open(output_dir / 'training_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'training_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("两阶段OCR训练报告 - 前沿版\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"设备: {device}\nBackbone: ResNet18, ResNet34, EfficientNet-B0\n训练时间: {total_time/3600:.2f} 小时\n技术: 迁移学习, 混合精度, Label Smoothing, AdamW\n\n")
        f.write("第一阶段：字体分类\n" + "-" * 40 + "\n")
        for i, r in enumerate(classifier_results, 1):
            f.write(f"  {i}. {r['model']:15s}: {r['best_val_acc']:.4f}\n")
        f.write(f"\n最佳分类器: {best_classifier['model']} 准确率: {best_classifier['best_val_acc']:.4f}\n\n")
        f.write("第二阶段：OCR识别\n" + "-" * 40 + "\n")
        for font_id in range(NUM_FONTS):
            if str(font_id) in ocr_results and ocr_results[font_id]:
                best = ocr_results[font_id][0]
                f.write(f"  字体{font_id:2d}: {best['model']:15s}: {best['best_val_acc']:.4f}\n")
    
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}")
