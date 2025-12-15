#!/usr/bin/env python3
"""
训练配置：控制训练时使用哪些类别
"""
from pathlib import Path

# ============================================================================
# 类别选择配置
# ============================================================================
# 方式1: 使用类别索引列表（推荐）
# 例如: SELECTED_CLASS_INDICES = [0, 1, 2, 3, 4]  # 使用前5个类别
# 例如: SELECTED_CLASS_INDICES = None  # 使用所有类别

SELECTED_CLASS_INDICES = None  # 使用所有类别

# 方式2: 使用类别名称列表
# 例如: SELECTED_CLASS_NAMES = ['Coca_cola_500', 'Pepsi_600', 'Sprite_500']
# 如果设置了SELECTED_CLASS_NAMES，将优先使用类别名称

SELECTED_CLASS_NAMES = ['NongFu spring_Drink water_550ml','Cestbon_Drink water_555ml', 'Pepsi_600', 'Sprite_500','Assam milk tea_Original flavor_500ml','Assam milk tea_Fried tea milky green_450ml','C vitamin water_Citrus flavor_500ml','C100_Calamansi_445ml'] # 使用所有类别

# ============================================================================
# 其他训练配置
# ============================================================================

# 模型选择: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
MODEL_NAME = 'yolo11n.pt'

# 训练参数
EPOCHS = 200
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0  # 0表示GPU 0，'cpu'表示使用CPU
WORKERS = 8

# 项目配置
PROJECT_NAME = 'runs/detect'
EXPERIMENT_NAME = 'yolo_voc2007'

# ============================================================================
# 数据增强配置
# ============================================================================

# 启用数据增强
ENABLE_AUGMENTATION = True

# 数据增强参数（参考ultralytics文档）
AUGMENTATION_CONFIG = {
    # HSV颜色空间增强
    'hsv_h': 0.015,      # 色调增强幅度 (0-1)
    'hsv_s': 0.7,        # 饱和度增强幅度 (0-1)
    'hsv_v': 0.4,        # 明度增强幅度 (0-1)
    
    # 几何变换
    'degrees': 10.0,     # 旋转角度范围 (-degrees to +degrees)
    'translate': 0.1,    # 平移范围 (0-1, 相对于图像尺寸)
    'scale': 0.5,        # 缩放范围 (1-scale to 1+scale)
    'shear': 2.0,        # 剪切角度范围 (degrees)
    'perspective': 0.0005,  # 透视变换 (0-0.001)
    
    # 翻转
    'flipud': 0.3,       # 上下翻转概率 (0-1)
    'fliplr': 0.3,       # 左右翻转概率 (0-1)
    
    # 高级增强
    'mosaic': 0.5,       # 马赛克增强概率 (0-1)
    'mixup': 0.1,        # 混合增强概率 (0-1)
    'copy_paste': 0.3,   # 复制粘贴增强概率 (0-1)
}

# ============================================================================
# 类别平衡配置
# ============================================================================

# 启用类别平衡
ENABLE_CLASS_BALANCING = True

# 类别平衡方法: 'oversample', 'augment', 'weighted_loss', 'none'
CLASS_BALANCING_METHOD = 'augment'  # 推荐使用 'augment'

# 目标样本数（None表示使用最大类别的80%）
TARGET_SAMPLES_PER_CLASS = 200

# 加权损失（当使用weighted_loss时）
USE_WEIGHTED_LOSS = False

# ============================================================================
# 辅助函数
# ============================================================================

def load_selected_classes(classes_file):
    """加载选中的类别"""
    classes_file = Path(classes_file)
    
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")
    
    with open(classes_file, 'r', encoding='utf-8') as f:
        all_classes = [line.strip() for line in f if line.strip()]
    
    # 如果指定了类别名称，优先使用
    if SELECTED_CLASS_NAMES is not None:
        selected_classes = []
        selected_indices = []
        for name in SELECTED_CLASS_NAMES:
            if name in all_classes:
                idx = all_classes.index(name)
                selected_classes.append(name)
                selected_indices.append(idx)
            else:
                print(f"Warning: Class '{name}' not found in classes file")
        return selected_classes, selected_indices
    
    # 如果指定了类别索引
    if SELECTED_CLASS_INDICES is not None:
        selected_classes = [all_classes[i] for i in SELECTED_CLASS_INDICES if i < len(all_classes)]
        selected_indices = [i for i in SELECTED_CLASS_INDICES if i < len(all_classes)]
        return selected_classes, selected_indices
    
    # 使用所有类别
    return all_classes, list(range(len(all_classes)))

def get_config_info():
    """获取配置信息"""
    info = {
        'selected_class_indices': SELECTED_CLASS_INDICES,
        'selected_class_names': SELECTED_CLASS_NAMES,
        'model_name': MODEL_NAME,
        'epochs': EPOCHS,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'device': DEVICE,
    }
    return info


