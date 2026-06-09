
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='OCR训练和推理系统')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    gen_parser = subparsers.add_parser('generate', help='生成数据集')
    gen_parser.add_argument('--data_dir', type=str, default='data', help='数据集保存目录')
    
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    train_parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    train_parser.add_argument('--model_dir', type=str, default='model', help='模型保存目录')
    train_parser.add_argument('--process_dir', type=str, default='process', help='过程保存目录')
    
    infer_parser = subparsers.add_parser('infer', help='推理')
    infer_parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    infer_parser.add_argument('--model', type=str, default='model/best_model.pth', help='模型路径')
    infer_parser.add_argument('--output', type=str, default=None, help='可视化输出路径')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        from dataset_generator import DatasetGenerator
        print("开始生成数据集...")
        generator = DatasetGenerator(data_dir=args.data_dir)
        generator.generate()
        print("数据集生成完成！")
    
    elif args.command == 'train':
        from trainer import Trainer
        print("开始训练...")
        trainer = Trainer(model_dir=args.model_dir, process_dir=args.process_dir)
        trainer.train(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print("训练完成！")
    
    elif args.command == 'infer':
        from inference import OCRInference
        print("开始推理...")
        ocr = OCRInference(args.model)
        results = ocr.predict(args.image)
        
        print(f"\n识别结果: {results['text']}")
        print("\n检测详情:")
        for det in results['detections']:
            print(f"  字符: {det['char']}, 置信度: {det['confidence']:.3f}, 位置: {det['bbox']}")
        
        if args.output:
            ocr.visualize(args.image, results, args.output)
            print(f"\n可视化结果已保存到: {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

