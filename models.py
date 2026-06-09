
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, max_boxes: int = 4):
        super().__init__()
        self.max_boxes = max_boxes
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        
        self.bbox_head = nn.Sequential(
            nn.Linear(256 * 8 * 20, 512),
            nn.ReLU(),
            nn.Linear(512, max_boxes * 4)
        )
        
        self.conf_head = nn.Sequential(
            nn.Linear(256 * 8 * 20, 512),
            nn.ReLU(),
            nn.Linear(512, max_boxes)
        )
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        
        bboxes = self.bbox_head(x)
        bboxes = bboxes.view(-1, self.max_boxes, 4)
        bboxes = torch.sigmoid(bboxes)
        
        conf = self.conf_head(x)
        conf = torch.sigmoid(conf)
        
        return bboxes, conf


class RecognitionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, max_boxes: int = 4):
        super().__init__()
        self.max_boxes = max_boxes
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        
        self.char_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 * 8 * 20, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            ) for _ in range(max_boxes)
        ])
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        
        char_logits = []
        for head in self.char_heads:
            logits = head(x)
            char_logits.append(logits)
        
        char_logits = torch.stack(char_logits, dim=1)
        return char_logits


class OCRModel(nn.Module):
    def __init__(self, num_classes: int, max_boxes: int = 4):
        super().__init__()
        self.backbone = Backbone()
        self.detection_head = DetectionHead(256, max_boxes)
        self.recognition_head = RecognitionHead(256, num_classes, max_boxes)
        self.max_boxes = max_boxes
        
    def forward(self, x):
        features = self.backbone(x)
        bboxes, conf = self.detection_head(features)
        char_logits = self.recognition_head(features)
        return {
            'bboxes': bboxes,
            'conf': conf,
            'char_logits': char_logits
        }


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_loss = nn.MSELoss()
        self.conf_loss = nn.BCELoss()
        
    def forward(self, pred_bboxes, pred_conf, target_bboxes):
        batch_size = pred_bboxes.size(0)
        num_boxes = pred_bboxes.size(1)
        
        target_conf = torch.ones(batch_size, num_boxes, device=pred_bboxes.device)
        
        pad_len = num_boxes - target_bboxes.size(1)
        if pad_len > 0:
            pad = torch.zeros(batch_size, pad_len, 4, device=target_bboxes.device)
            target_bboxes = torch.cat([target_bboxes, pad], dim=1)
        
        bbox_loss = self.bbox_loss(pred_bboxes, target_bboxes)
        conf_loss = self.conf_loss(pred_conf, target_conf)
        
        return bbox_loss + conf_loss


class RecognitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred_logits, target_chars):
        batch_size = pred_logits.size(0)
        num_boxes = pred_logits.size(1)
        
        pad_len = num_boxes - target_chars.size(1)
        if pad_len > 0:
            pad = torch.zeros(batch_size, pad_len, dtype=torch.long, device=target_chars.device)
            target_chars = torch.cat([target_chars, pad], dim=1)
        
        total_loss = 0
        for i in range(num_boxes):
            loss = self.ce_loss(pred_logits[:, i], target_chars[:, i])
            total_loss += loss
        
        return total_loss / num_boxes


class OCRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_loss = DetectionLoss()
        self.rec_loss = RecognitionLoss()
        
    def forward(self, predictions, targets):
        pred_bboxes = predictions['bboxes']
        pred_conf = predictions['conf']
        pred_logits = predictions['char_logits']
        
        total_det_loss = 0
        total_rec_loss = 0
        
        for i in range(len(targets)):
            target_bbox = targets[i]['bboxes'].unsqueeze(0)
            target_char = targets[i]['chars'].unsqueeze(0)
            
            det_loss = self.det_loss(
                pred_bboxes[i:i+1], 
                pred_conf[i:i+1], 
                target_bbox
            )
            
            rec_loss = self.rec_loss(
                pred_logits[i:i+1],
                target_char
            )
            
            total_det_loss += det_loss
            total_rec_loss += rec_loss
        
        return (total_det_loss + total_rec_loss) / len(targets)


if __name__ == "__main__":
    model = OCRModel(num_classes=62)
    print(model)
    x = torch.randn(2, 3, 60, 160)
    out = model(x)
    print(f"bboxes shape: {out['bboxes'].shape}")
    print(f"conf shape: {out['conf'].shape}")
    print(f"char_logits shape: {out['char_logits'].shape}")

