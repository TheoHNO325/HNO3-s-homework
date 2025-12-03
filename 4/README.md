# YOLO VOC2007 目标检测作业

本项目使用YOLO11模型在VOC2007数据集上进行目标检测和分类任务。
目前进度：实现了对矿泉水（农夫山泉红瓶、怡泉绿瓶），可乐（百事、无糖可口、普通可口），阿萨姆原味，东方树叶，雪碧的分类与目标检测模型训练，使用手动数据增强扩充数据集进行训练，但是效果欠佳。结算画面见/inference

## 项目结构

```
.
├── main.py              # 主程序入口
├── preprocess_data.py   # 数据预处理脚本（VOC格式转YOLO格式）
├── train.py             # 模型训练脚本
├── train_config.py     # 训练配置（类别选择等）
├── inference_video.py   # 视频推理脚本
├── data_overview.py     # 数据概览脚本
├── show_classes.py      # 类别查看工具
├── requirements.txt     # Python依赖包
├── dataset.yaml         # YOLO数据集配置文件（预处理后自动生成）
├── dataset/             # 预处理后的数据集目录（预处理后自动生成）
│   ├── classes.txt      # 类别列表
│   ├── images/          # 图像文件
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/          # YOLO格式标注文件
│       ├── train/
│       ├── val/
│       └── test/
└── VOC2007/             # 原始VOC2007数据集
    └── VOC2007/
        ├── JPEGImages_old/
        ├── Annotations_old/
        ├── ImageSets/
        └── labels.txt
```

## 环境配置

1. 激活conda环境：
```bash
conda activate task4
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用步骤

### 1. 数据预处理

将VOC格式的数据集转换为YOLO格式：

```bash
python preprocess_data.py
```

或者使用main.py：

```bash
python main.py --mode preprocess
```

预处理完成后，会在项目根目录生成：
- `dataset/` 目录：包含转换后的YOLO格式数据集
- `dataset/classes.txt`：包含所有类别名称

### 2. 数据概览（可选）

生成数据概览报告，包括每个类别的样本数量和样例图片：

```bash
python data_overview.py
```

或者使用main.py：

```bash
python main.py --mode overview
```

这将生成：
- `data_overview/data_overview.txt`：文本格式的统计报告
- `data_overview/data_overview.html`：HTML格式的可视化报告
- `data_overview/class_samples/`：每个类别的样例图片（带标注框）

### 3. 配置训练类别

在 `train_config.py` 中配置要训练的类别：

**方式1：使用类别索引**
```python
SELECTED_CLASS_INDICES = [0, 1, 2, 3, 4]  # 使用前5个类别
```

**方式2：使用类别名称**
```python
SELECTED_CLASS_NAMES = ['Coca_cola_500', 'Pepsi_600', 'Sprite_500']
```

**使用所有类别**
```python
SELECTED_CLASS_INDICES = None  # 或 SELECTED_CLASS_NAMES = None
```

### 4. 训练模型

```bash
python train.py
```

或者使用main.py：

```bash
python main.py --mode train
```

训练参数可以在 `train_config.py` 中修改：
- `MODEL_NAME`: 模型选择（'yolov8n.pt', 'yolo11.pt'等）
- `EPOCHS`: 训练轮数（默认100）
- `IMG_SIZE`: 输入图像尺寸（默认640）
- `BATCH_SIZE`: 批次大小（默认16，根据GPU内存调整）
- `DEVICE`: 设备（0表示GPU 0，'cpu'表示使用CPU）

### 数据增强配置

在 `train_config.py` 中可以配置数据增强参数：
- `ENABLE_AUGMENTATION`: 是否启用数据增强（默认True）
- `AUGMENTATION_CONFIG`: 数据增强参数字典
  - `hsv_h/s/v`: HSV颜色空间增强
  - `degrees`: 旋转角度范围
  - `translate`: 平移范围
  - `scale`: 缩放范围
  - `fliplr`: 左右翻转概率
  - `mosaic`: 马赛克增强概率
  - `mixup`: 混合增强概率

### 类别平衡配置

在 `train_config.py` 中可以配置类别平衡：
- `ENABLE_CLASS_BALANCING`: 是否启用类别平衡（默认True）
- `CLASS_BALANCING_METHOD`: 平衡方法（'augment'推荐）
- `TARGET_SAMPLES_PER_CLASS`: 每个类别的目标样本数（None为自动）

也可以手动运行类别平衡脚本：
```bash
python balance_dataset.py --input dataset_filtered --output dataset_balanced
```

如果配置了类别选择，训练脚本会自动过滤数据集并重新映射类别索引。

训练完成后，模型会保存在 `runs/detect/yolo_voc2007/weights/best.pt`

### 5. 视频推理

对视频进行目标检测和分类：

```bash
python inference_video.py --model runs/detect/yolo_voc2007/weights/best.pt --video path/to/video.mp4 --output output_video.mp4
```

或者使用main.py：

```bash
python main.py --mode inference --model runs/detect/yolo_voc2007/weights/best.pt --video path/to/video.mp4 --output output_video.mp4
```

参数说明：
- `--model`: 训练好的模型权重路径
- `--video`: 输入视频路径
- `--output`: 输出视频路径（可选，默认为输入视频名_detected.mp4）
- `--conf`: 置信度阈值（默认0.25）

## 类别信息

数据集包含69个类别，主要是饮料和调味品类别。完整的类别列表保存在 `dataset/classes.txt` 文件中。

查看所有类别：
```bash
python show_classes.py
```

## 类别选择功能

通过 `train_config.py` 可以灵活控制训练时使用的类别：

1. **使用部分类别训练**：可以只选择部分类别进行训练，减少模型复杂度
2. **自动过滤数据集**：训练脚本会自动过滤标注，只保留选中类别的样本
3. **自动重新映射**：类别索引会自动重新映射为0, 1, 2, ...

示例：只训练前10个类别
```python
# 在 train_config.py 中设置
SELECTED_CLASS_INDICES = list(range(10))  # 使用前10个类别
```

## 数据增强和类别平衡

### 数据增强

数据增强可以提升模型的泛化能力，减少过拟合。本项目支持以下增强方式：

1. **颜色空间增强**：HSV色调、饱和度、明度调整
2. **几何变换**：旋转、平移、缩放、剪切、透视变换
3. **翻转增强**：水平翻转、垂直翻转
4. **高级增强**：马赛克增强、混合增强

在 `train_config.py` 中配置：
```python
ENABLE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,      # 色调增强
    'degrees': 10.0,     # 旋转角度
    'mosaic': 1.0,       # 马赛克增强
    'mixup': 0.1,        # 混合增强
    # ... 更多参数
}
```

### 类别平衡

类别不平衡会导致模型偏向多数类别。本项目通过以下方式平衡：

1. **过采样**：对少数类别进行数据增强生成更多样本
2. **自动平衡**：训练时自动分析类别分布并平衡

在 `train_config.py` 中配置：
```python
ENABLE_CLASS_BALANCING = True
CLASS_BALANCING_METHOD = 'augment'  # 使用数据增强平衡
TARGET_SAMPLES_PER_CLASS = None     # None表示自动计算
```

## 参考文档

- [Ultralytics YOLO官方文档](https://docs.ultralytics.com/)
- [YOLOv8训练指南](https://docs.ultralytics.com/modes/train/)

