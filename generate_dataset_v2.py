"""
两阶段OCR数据集生成器 - 混搭字体版
- 每张图片的4个字符可以来自不同字体
- 第一阶段：字体分类（识别图片中使用的字体组合）
- 第二阶段：OCR识别

数据量：20种字体，总共20000张图片
"""

import os
import random
import string
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

print("=" * 60)
print("两阶段OCR数据集生成器 - 混搭字体版")
print("=" * 60)

# 创建目录
data_dir = Path("data_v2")
images_dir = data_dir / "images"
labels_dir = data_dir / "labels"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(exist_ok=True)

# 字符集
CHARS = string.ascii_letters + string.digits
NUM_CHARS = 4  # 每张图片4个字符

# 获取Windows系统字体
def get_windows_fonts():
    """获取Windows系统字体文件列表"""
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path(os.environ.get('LOCALAPPDATA', '')) / "Microsoft/Windows/Fonts"
    ]
    
    fonts = []
    for font_dir in font_dirs:
        if font_dir.exists():
            fonts.extend(list(font_dir.glob("*.ttf")))
            fonts.extend(list(font_dir.glob("*.TTF")))
    
    # 过滤掉一些不适合的字体
    excluded_keywords = ['symbol', 'dingbat', 'emoji', 'math', 'music', 'wing', 'segui', 'holo']
    filtered_fonts = []
    for font in fonts:
        name_lower = font.name.lower()
        if not any(kw in name_lower for kw in excluded_keywords):
            filtered_fonts.append(font)
    
    return filtered_fonts

# 获取字体列表
all_fonts = get_windows_fonts()
print(f"找到 {len(all_fonts)} 种字体")

# 选择20种字体
SELECTED_FONT_COUNT = 20
if len(all_fonts) >= SELECTED_FONT_COUNT:
    step = len(all_fonts) // SELECTED_FONT_COUNT
    selected_fonts = [all_fonts[i * step] for i in range(SELECTED_FONT_COUNT)]
else:
    selected_fonts = all_fonts

print(f"选择 {len(selected_fonts)} 种字体")

# 总图片数
TOTAL_IMAGES = 20000
IMAGES_PER_FONT_CATEGORY = TOTAL_IMAGES // SELECTED_FONT_COUNT  # 每种字体类别1000张

print(f"\n数据集配置:")
print(f"  字体数量: {len(selected_fonts)} 种")
print(f"  总图片数: {TOTAL_IMAGES} 张")
print(f"  每张图片字符数: {NUM_CHARS} 个")
print(f"  混搭模式: 每个字符可使用不同字体")


def generate_captcha_mixed(texts, font_paths, width=160, height=60):
    """
    生成验证码图片（混搭字体）
    texts: 4个字符的列表
    font_paths: 4个字体路径的列表（每个字符对应一个字体）
    """
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    char_width = width // NUM_CHARS
    char_boxes = []
    
    for i, (char, font_path) in enumerate(zip(texts, font_paths)):
        # 加载字体
        try:
            font_size = random.randint(28, 36)
            font = ImageFont.truetype(str(font_path), font_size)
        except:
            font = ImageFont.load_default()
        
        # 随机位置偏移
        x_offset = random.randint(-5, 5)
        y_offset = random.randint(-8, 8)
        
        x = i * char_width + char_width // 4 + x_offset
        y = (height - font_size) // 2 + y_offset
        
        # 随机颜色（深色）
        color = (
            random.randint(0, 100),
            random.randint(0, 100),
            random.randint(0, 100)
        )
        
        # 绘制字符
        draw.text((x, y), char, font=font, fill=color)
        
        char_boxes.append({
            'char': char,
            'x': x,
            'y': y,
            'w': font_size,
            'h': font_size
        })
    
    # 添加干扰线
    for _ in range(random.randint(2, 5)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
    
    # 添加噪点
    for _ in range(random.randint(50, 150)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.point((x, y), fill=color)
    
    # 添加干扰圆圈
    for _ in range(random.randint(2, 5)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(5, 15)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=1)
    
    # 高斯噪声
    img_array = list(img.getdata())
    for i in range(len(img_array)):
        r, g, b = img_array[i]
        noise = random.randint(-10, 10)
        r = max(0, min(255, r + noise))
        g = max(0, min(255, g + noise))
        b = max(0, min(255, b + noise))
        img_array[i] = (r, g, b)
    img.putdata(img_array)
    
    # 随机模糊
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    
    return img, char_boxes


# 字体标签数据
font_labels = {}

print(f"\n开始生成数据集...")

total_generated = 0

# 生成策略：确保每种字体都有足够的使用次数
# 每种字体至少出现在 (TOTAL_IMAGES * 4 / SELECTED_FONT_COUNT) 个字符位置

font_usage_count = [0] * len(selected_fonts)  # 记录每种字体的使用次数

for img_idx in range(TOTAL_IMAGES):
    # 生成4个随机字符
    chars = [random.choice(CHARS) for _ in range(NUM_CHARS)]
    
    # 选择4个字体（可以是相同或不同的）
    # 策略：优先选择使用次数较少的字体，保证均衡
    fonts_for_chars = []
    for _ in range(NUM_CHARS):
        # 70%概率随机选择，30%概率选择使用次数最少的字体
        if random.random() < 0.7:
            font_idx = random.randint(0, len(selected_fonts) - 1)
        else:
            # 选择使用次数最少的字体
            min_usage = min(font_usage_count)
            candidates = [i for i, c in enumerate(font_usage_count) if c == min_usage]
            font_idx = random.choice(candidates)
        
        fonts_for_chars.append(selected_fonts[font_idx])
        font_usage_count[font_idx] += 1
    
    # 生成验证码
    img, char_boxes = generate_captcha_mixed(chars, fonts_for_chars)
    
    # 确定主要字体（使用次数最多的字体作为该图片的主字体标签）
    font_indices = [selected_fonts.index(f) for f in fonts_for_chars]
    main_font_idx = max(set(font_indices), key=font_indices.count)
    
    # 保存图片
    img_filename = f"img_{img_idx:05d}.png"
    img_path = images_dir / img_filename
    img.save(img_path)
    
    # 保存标签
    label_filename = f"img_{img_idx:05d}.txt"
    label_path = labels_dir / label_filename
    with open(label_path, 'w', encoding='utf-8') as f:
        for box, font_idx in zip(char_boxes, font_indices):
            f.write(f"{box['char']} {box['x']} {box['y']} {box['w']} {box['h']} font_{font_idx:04d}\n")
    
    # 记录字体标签
    font_labels[img_filename] = {
        'main_font_id': main_font_idx,  # 主要字体
        'font_ids': font_indices,        # 每个字符的字体
        'text': ''.join(chars)
    }
    
    total_generated += 1
    
    if (img_idx + 1) % 2000 == 0:
        print(f"  已生成: {img_idx + 1}/{TOTAL_IMAGES}")

# 保存字体标签文件
font_labels_path = data_dir / "font_labels.json"
with open(font_labels_path, 'w', encoding='utf-8') as f:
    json.dump(font_labels, f, indent=2, ensure_ascii=False)

# 保存字体列表
fonts_list_path = data_dir / "fonts_list.json"
fonts_info = [
    {
        'font_id': idx,
        'font_name': font_path.stem,
        'font_path': str(font_path),
        'usage_count': font_usage_count[idx]
    }
    for idx, font_path in enumerate(selected_fonts)
]
with open(fonts_list_path, 'w', encoding='utf-8') as f:
    json.dump(fonts_info, f, indent=2, ensure_ascii=False)

# 统计信息
print(f"\n{'='*60}")
print("数据集生成完成!")
print(f"{'='*60}")
print(f"\n统计信息:")
print(f"  总图片数: {total_generated}")
print(f"  字体数量: {len(selected_fonts)}")
print(f"  每张图片字符数: {NUM_CHARS}")
print(f"  字符集大小: {len(CHARS)}")
print(f"\n字体使用统计:")
for idx, count in enumerate(font_usage_count):
    print(f"  字体 {idx:2d} ({selected_fonts[idx].stem[:20]:20s}): {count:5d} 次")

print(f"\n文件结构:")
print(f"  {data_dir}/")
print(f"    ├── images/          # {total_generated} 张验证码图片")
print(f"    ├── labels/          # {total_generated} 个标签文件")
print(f"    ├── font_labels.json # 字体标签映射")
print(f"    └── fonts_list.json  # 字体列表信息")

print(f"\n两阶段训练计划:")
print(f"  第一阶段: 字体分类器 (识别图片主要字体)")
print(f"    - 20分类任务")
print(f"    - 训练集: {int(total_generated * 0.8)} 张")
print(f"    - 验证集: {int(total_generated * 0.2)} 张")
print(f"  第二阶段: 通用OCR模型")
print(f"    - 输入: 图片 + 字体类别")
print(f"    - 输出: 4个字符")
