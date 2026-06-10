# OCR 训练系统 - 前沿版

这是一个完整的OCR（光学字符识别）训练和推理系统，支持从Windows字体库生成数据集，采用两阶段级联架构进行字体分类和字符识别。

## 项目结构

```
ocr_train/
├── train_two_stage_pytorch.py  # 主训练脚本（18个模型）
├── generate_dataset_v2.py      # 数据集生成模块（多字体混搭）
├── inference.py                # 推理模块
├── main.py                     # 主入口脚本
├── models.py                   # 模型定义
├── trainer.py                  # 训练器模块
├── requirements.txt            # 依赖包
├── README.md                   # 项目说明
├── data_v2/                    # 数据集目录（自动生成）
└── training_output_advanced/   # 训练输出目录（自动生成）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 生成数据集

从Windows字体库生成训练数据集（支持字体混搭）：

```bash
python generate_dataset_v2.py
```

这将会：
- 扫描Windows系统字体库（C:\Windows\Fonts）
- 生成20000+张图片，每张图片包含4个字符
- 支持字体混搭（单张图片可包含多个字体）
- 字符会有随机的倾斜、翻转、高斯模糊和椒盐噪声
- 图片存储在 `data_v2/images/`，标签存储在 `data_v2/labels/`

### 2. 训练模型（18个模型）

```bash
python train_two_stage_pytorch.py
```

**模型配置（共18个模型）：**

| 阶段 | Backbone | Optimizer | 模型数量 |
|------|----------|-----------|----------|
| 字体分类 | ResNet18, ResNet34, EfficientNet-B0 | Adam, SGD, RMSprop | 9 |
| OCR识别 | 最佳分类模型的Backbone | Adam, SGD, RMSprop | 9 |

**训练技术：**
- 迁移学习（冻结预训练层）
- 混合精度训练（AMP）
- Label Smoothing（0.1）
- CosineAnnealingWarmRestarts学习率调度
- 梯度累积优化

### 3. 推理

使用训练好的模型进行OCR识别：

```bash
python inference.py --image path/to/image.png
```

## 模型架构

系统采用两阶段级联识别架构：

**第一阶段：字体分类**
- 使用预训练模型（ResNet18/34、EfficientNet-B0）
- 输入：验证码图片
- 输出：字体类别概率

**第二阶段：OCR识别**
- 使用第一阶段表现最佳的Backbone
- 输入：验证码图片
- 输出：4个字符的识别结果

**流程：**
1. 输入图片 → 字体分类 → 选择最佳OCR模型 → 字符识别 → 结果拼接

## 输出说明

- `training_output_advanced/` 目录：
  - `training_report.json`: 完整训练报告
  - `training_summary.txt`: 训练摘要
  - `font_*.pt`: 字体分类模型
  - `font_XX/ocr_*.pt`: 各字体专用OCR模型

## 配置说明

- **学习率**: 0.001
- **训练轮数**: 100
- **批次大小**: 24
- **字符类别**: 62（a-z, A-Z, 0-9）
- **字体数量**: 20+

## 注意事项

- 确保系统有足够的字体文件（默认从C:\Windows\Fonts读取）
- 建议使用GPU加速训练（支持CUDA自动检测）
- CPU模式下会自动限制内存使用（num_workers=0）
- 训练完成后自动清理模型缓存，释放内存