
import torch
from PIL import Image, ImageDraw
import numpy as np
import string
from torchvision import transforms
from pathlib import Path
from trainer import Trainer


class OCRInference:
    def __init__(self, model_path: str = "model/best_model.pth"):
        self.trainer = Trainer()
        self.model = self.trainer.load_model(model_path)
        self.device = self.trainer.device
        
        self.chars = string.ascii_letters + string.digits
        self.char_to_idx = self.trainer.char_to_idx
        self.idx_to_char = self.trainer.idx_to_char
        
        self.transform = transforms.Compose([
            transforms.Resize((60, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_size = (160, 60)
        
    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        original_size = image.size
        img = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img)
        
        bboxes = outputs['bboxes'][0].cpu().numpy()
        conf = outputs['conf'][0].cpu().numpy()
        char_logits = outputs['char_logits'][0].cpu().numpy()
        
        bboxes[:, [0, 2]] *= self.image_size[0]
        bboxes[:, [1, 3]] *= self.image_size[1]
        
        results = []
        for i in range(len(bboxes)):
            char_idx = np.argmax(char_logits[i])
            char = self.idx_to_char[char_idx]
            results.append({
                'bbox': bboxes[i].tolist(),
                'confidence': float(conf[i]),
                'char': char
            })
        
        results = sorted(results, key=lambda x: x['bbox'][0])
        
        text = ''.join([r['char'] for r in results])
        
        return {
            'text': text,
            'detections': results
        }
    
    def visualize(self, image, results, output_path=None):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        draw = ImageDraw.Draw(image)
        
        for det in results['detections']:
            bbox = det['bbox']
            char = det['char']
            conf = det['confidence']
            
            draw.rectangle(bbox, outline='red', width=2)
            draw.text((bbox[0], bbox[1] - 15), f"{char} ({conf:.2f})", fill='red')
        
        if output_path:
            image.save(output_path)
        
        return image


def main():
    import argparse
    parser = argparse.ArgumentParser(description='OCR推理')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--model', type=str, default='model/best_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='可视化输出路径')
    args = parser.parse_args()
    
    ocr = OCRInference(args.model)
    results = ocr.predict(args.image)
    
    print(f"识别结果: {results['text']}")
    print("检测详情:")
    for det in results['detections']:
        print(f"  字符: {det['char']}, 置信度: {det['confidence']:.3f}, 位置: {det['bbox']}")
    
    if args.output:
        ocr.visualize(args.image, results, args.output)
        print(f"可视化结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

