
import os
import json
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import string
from models import OCRModel, OCRLoss
from data_loader import get_data_loaders


class Trainer:
    def __init__(self, model_dir: str = "model", process_dir: str = "process", device: str = None):
        self.model_dir = Path(model_dir)
        self.process_dir = Path(process_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.process_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        self.chars = string.ascii_letters + string.digits
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        
    def train(self, num_epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        train_loader, val_loader, num_classes = get_data_loaders(batch_size=batch_size)
        
        model = OCRModel(num_classes=num_classes).to(self.device)
        criterion = OCRLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            model.train()
            total_train_loss = 0.0
            correct = 0
            total = 0
            
            for imgs, targets in tqdm(train_loader, desc="训练"):
                imgs = imgs.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                batch_correct, batch_total = self._calculate_accuracy(outputs, targets)
                correct += batch_correct
                total += batch_total
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_acc = correct / total if total > 0 else 0.0
            
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model(model, optimizer, epoch, val_acc, "best_model.pth")
                print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
            
            self._save_model(model, optimizer, epoch, val_acc, f"checkpoint_epoch_{epoch+1}.pth")
        
        self._save_metrics(train_losses, val_losses, train_accs, val_accs)
        self._plot_metrics(train_losses, val_losses, train_accs, val_accs)
        
        return model
    
    def _validate(self, model, val_loader, criterion):
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc="验证"):
                imgs = imgs.to(self.device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                
                total_val_loss += loss.item()
                
                batch_correct, batch_total = self._calculate_accuracy(outputs, targets)
                correct += batch_correct
                total += batch_total
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        
        return avg_val_loss, val_acc
    
    def _calculate_accuracy(self, outputs, targets):
        correct = 0
        total = 0
        
        for i in range(len(targets)):
            pred_logits = outputs['char_logits'][i]
            target_chars = targets[i]['chars']
            
            num_chars = min(len(target_chars), pred_logits.size(0))
            
            for j in range(num_chars):
                pred_idx = torch.argmax(pred_logits[j]).item()
                target_idx = target_chars[j].item()
                
                if pred_idx == target_idx:
                    correct += 1
                total += 1
        
        return correct, total
    
    def _save_model(self, model, optimizer, epoch, val_acc, filename):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, self.model_dir / filename)
    
    def _save_metrics(self, train_losses, val_losses, train_accs, val_accs):
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        
        with open(self.process_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        with open(self.process_dir / 'evaluation.txt', 'w', encoding='utf-8') as f:
            f.write(f"最终训练损失: {train_losses[-1]:.4f}\n")
            f.write(f"最终验证损失: {val_losses[-1]:.4f}\n")
            f.write(f"最终训练准确率: {train_accs[-1]:.4f}\n")
            f.write(f"最终验证准确率: {val_accs[-1]:.4f}\n")
            f.write(f"最佳验证准确率: {max(val_accs):.4f}\n")
    
    def _plot_metrics(self, train_losses, val_losses, train_accs, val_accs):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(train_losses, label='训练损失')
        axes[0].plot(val_losses, label='验证损失')
        axes[0].set_title('损失变化')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(train_accs, label='训练准确率')
        axes[1].plot(val_accs, label='验证准确率')
        axes[1].set_title('准确率变化')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.process_dir / 'training_curves.png')
        plt.close()
    
    def load_model(self, model_path: str):
        model = OCRModel(num_classes=len(self.chars)).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(num_epochs=10, batch_size=16)

