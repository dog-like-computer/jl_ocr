
# OCR 训练系统

这是一个完整的OCR（光学字符识别）训练和推理系统，支持从Windows字体库生成数据集，进行目标检测和字符识别训练。

## 项目结构

```
ocr_train/
├── dataset_generator.py  # 数据集生成模块
├── data_loader.py        # 数据加载模块
├── models.py             # 模型定义
├── trainer.py            # 训练模块
├── inference.py          # 推理模块
├── main.py               # 主入口脚本
├── requirements.txt      # 依赖包
├── README.md             # 项目说明
├── data/                 # 数据集目录（自动生成）
├── model/                # 模型保存目录（自动生成）
└── process/              # 训练过程记录（自动生成）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 生成数据集

从Windows字体库生成训练数据集：

```bash
python main.py generate --data_dir data
```

这将会：
- 扫描Windows系统字体库
- 每个字体生成350张图片
- 每张图片包含4个随机字符（a-z, A-Z, 0-9）
- 字符会有随机的倾斜、翻转和噪声

### 2. 训练模型

```bash
python main.py train --epochs 50 --batch_size 32 --lr 0.001
```

训练参数：
- `--epochs`: 训练轮数（默认50）
- `--batch_size`: 批次大小（默认32）
- `--lr`: 学习率（默认0.001）

训练过程中会：
- 保存最佳模型到 `model/best_model.pth`
- 保存每个epoch的checkpoint
- 记录训练指标到 `process/` 目录

### 3. 推理

使用训练好的模型进行OCR识别：

```bash
python main.py infer --image path/to/image.png --output result.png
```

推理参数：
- `--image`: 输入图片路径（必需）
- `--model`: 模型路径（默认 `model/best_model.pth`）
- `--output`: 可视化输出路径（可选）

## 模型架构

系统采用两阶段模型：
1. **目标检测**：检测图片中字符的位置
2. **字符识别**：识别每个检测框中的字符

最终将识别结果按位置顺序拼接输出。

## 输出说明

- `model/` 目录：包含训练好的模型文件
- `process/` 目录：
  - `metrics.json`: 训练指标JSON
  - `evaluation.txt`: 最终评估结果
  - `training_curves.png`: 训练曲线图
- `data/` 目录：生成的训练数据集

## 注意事项

- 确保系统有足够的字体文件（默认从C:\Windows\Fonts读取）
- 训练需要一定时间，建议使用GPU加速
- 如果字体目录不存在，会使用默认字体生成少量数据

