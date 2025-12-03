#!/usr/bin/env python3
"""
YOLO模型训练脚本
使用ultralytics YOLO进行训练
支持类别选择功能
"""
from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil
from train_config import (
    SELECTED_CLASS_INDICES, 
    SELECTED_CLASS_NAMES,
    MODEL_NAME,
    EPOCHS,
    IMG_SIZE,
    BATCH_SIZE,
    DEVICE,
    WORKERS,
    PROJECT_NAME,
    EXPERIMENT_NAME,
    ENABLE_AUGMENTATION,
    AUGMENTATION_CONFIG,
    ENABLE_CLASS_BALANCING,
    CLASS_BALANCING_METHOD,
    TARGET_SAMPLES_PER_CLASS,
    load_selected_classes
)

def filter_dataset_by_classes(dataset_root, selected_indices, output_root):
    """根据选择的类别过滤数据集，重新映射类别索引"""
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 创建类别映射（旧索引 -> 新索引）
    class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
    
    # 处理每个数据集划分
    for split in ['train', 'val', 'test']:
        images_dir = dataset_root / 'images' / split
        labels_dir = dataset_root / 'labels' / split
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        print(f"Filtering {split} split...")
        
        label_files = list(labels_dir.glob('*.txt'))
        filtered_count = 0
        
        for label_file in label_files:
            image_id = label_file.stem
            
            # 查找对应的图像文件
            image_path = None
            for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                candidate = images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                continue
            
            # 读取并过滤标注
            filtered_labels = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class_idx = int(parts[0])
                        if old_class_idx in class_mapping:
                            new_class_idx = class_mapping[old_class_idx]
                            filtered_labels.append(f"{new_class_idx} {' '.join(parts[1:])}\n")
            
            # 如果过滤后有标注，保存文件
            if filtered_labels:
                # 复制图像
                output_image_path = output_root / 'images' / split / image_path.name
                shutil.copy2(image_path, output_image_path)
                
                # 保存过滤后的标注
                output_label_path = output_root / 'labels' / split / label_file.name
                with open(output_label_path, 'w') as f:
                    f.writelines(filtered_labels)
                
                filtered_count += 1
        
        print(f"  {split}: {filtered_count} images with selected classes")

def create_dataset_yaml(dataset_root, selected_classes, output_yaml):
    """创建YOLO数据集配置文件"""
    dataset_root = Path(dataset_root)
    
    # 创建YAML配置
    config = {
        'path': str(dataset_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(selected_classes),
        'names': selected_classes
    }
    
    # 保存YAML文件
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"Created dataset YAML file: {output_yaml}")
    print(f"Number of classes: {len(selected_classes)}")
    return config

def main():
    # 路径配置
    dataset_root = Path(__file__).parent / 'dataset'
    classes_file = dataset_root / 'classes.txt'
    dataset_yaml = Path(__file__).parent / 'dataset.yaml'
    
    # 检查数据集是否存在
    if not dataset_root.exists():
        print("Error: Dataset directory not found!")
        print("Please run preprocess_data.py first to create the dataset.")
        return
    
    # 加载选中的类别
    print("Loading selected classes...")
    try:
        selected_classes, selected_indices = load_selected_classes(classes_file)
        print(f"Selected {len(selected_classes)} classes out of total classes")
        
        if len(selected_classes) < len(Path(classes_file).read_text().strip().split('\n')):
            print("\nSelected classes:")
            for idx, cls_name in enumerate(selected_classes):
                print(f"  [{idx}] {cls_name}")
            
            # 如果选择了部分类别，需要过滤数据集
            print("\nFiltering dataset for selected classes...")
            filtered_dataset_root = Path(__file__).parent / 'dataset_filtered'
            filter_dataset_by_classes(dataset_root, selected_indices, filtered_dataset_root)
            dataset_root = filtered_dataset_root
            
            # 保存过滤后的类别文件
            filtered_classes_file = filtered_dataset_root / 'classes.txt'
            with open(filtered_classes_file, 'w', encoding='utf-8') as f:
                for cls_name in selected_classes:
                    f.write(f"{cls_name}\n")
        else:
            print("Using all classes")
    
    except Exception as e:
        print(f"Error loading selected classes: {e}")
        print("Using all classes from classes.txt")
        with open(classes_file, 'r', encoding='utf-8') as f:
            selected_classes = [line.strip() for line in f if line.strip()]
    
    # 类别平衡处理
    if ENABLE_CLASS_BALANCING and CLASS_BALANCING_METHOD == 'augment':
        print("\n" + "="*60)
        print("Class Balancing: Augmenting dataset...")
        print("="*60)
        from balance_dataset import balance_dataset
        
        balanced_dataset_root = Path(__file__).parent / 'dataset_balanced'
        print(f"Creating balanced dataset at: {balanced_dataset_root}")
        
        balance_dataset(
            dataset_root, 
            balanced_dataset_root, 
            TARGET_SAMPLES_PER_CLASS,
            max_augmentations=5
        )
        
        dataset_root = balanced_dataset_root
        print(f"Using balanced dataset: {dataset_root}")
    
    # 创建数据集YAML配置文件
    print("\nCreating dataset configuration...")
    create_dataset_yaml(dataset_root, selected_classes, dataset_yaml)
    
    # 加载预训练的YOLO模型
    print(f"\nLoading YOLO model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    # 训练参数
    train_args = {
        'data': str(dataset_yaml),
        'epochs': EPOCHS,
        'imgsz': IMG_SIZE,
        'batch': BATCH_SIZE,
        'device': DEVICE,
        'workers': WORKERS,
        'project': PROJECT_NAME,
        'name': EXPERIMENT_NAME,
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True,
    }
    
    # 添加数据增强参数
    if ENABLE_AUGMENTATION:
        print("\n" + "="*60)
        print("Data Augmentation: Enabled")
        print("="*60)
        for key, value in AUGMENTATION_CONFIG.items():
            train_args[key] = value
            print(f"  {key}: {value}")
    else:
        print("\nData Augmentation: Disabled")
    
    # 类别平衡信息
    if ENABLE_CLASS_BALANCING:
        print(f"\nClass Balancing: {CLASS_BALANCING_METHOD}")
        if CLASS_BALANCING_METHOD == 'augment':
            print(f"  Target samples per class: {TARGET_SAMPLES_PER_CLASS or 'Auto (80% of max)'}")
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print("Training parameters:")
    for key, value in train_args.items():
        if key not in AUGMENTATION_CONFIG or ENABLE_AUGMENTATION:
            print(f"  {key}: {value}")
    
    # 开始训练
    results = model.train(**train_args)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print("="*50)

if __name__ == '__main__':
    main()

