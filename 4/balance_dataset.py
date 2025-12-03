#!/usr/bin/env python3
"""
数据集类别平衡脚本：通过过采样和数据增强平衡各类别样本数量
"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
import shutil
from PIL import Image, ImageEnhance, ImageFilter
import xml.etree.ElementTree as ET

def analyze_class_distribution(dataset_root):
    """分析数据集中各类别的样本分布"""
    dataset_root = Path(dataset_root)
    labels_dir = dataset_root / 'labels' / 'train'
    
    class_counts = defaultdict(int)
    image_class_map = defaultdict(list)  # 每个类别对应的图片列表
    
    print("Analyzing class distribution...")
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            classes_in_image = set()
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    classes_in_image.add(class_id)
                    class_counts[class_id] += 1
            
            # 记录包含每个类别的图片
            for class_id in classes_in_image:
                image_class_map[class_id].append(label_file.stem)
    
    return class_counts, image_class_map

def augment_image(image_path, output_path, augmentation_type='random'):
    """对图像进行数据增强"""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    
    if augmentation_type == 'random':
        augmentation_type = random.choice([
            'flip_horizontal', 'flip_vertical', 'rotate', 
            'brightness', 'contrast', 'noise', 'blur'
        ])
    
    if augmentation_type == 'flip_horizontal':
        img = cv2.flip(img, 1)
    elif augmentation_type == 'flip_vertical':
        img = cv2.flip(img, 0)
    elif augmentation_type == 'rotate':
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif augmentation_type == 'brightness':
        alpha = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    elif augmentation_type == 'contrast':
        alpha = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=random.randint(-10, 10))
    elif augmentation_type == 'noise':
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    elif augmentation_type == 'blur':
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    cv2.imwrite(str(output_path), img)
    return True

def augment_label(label_path, output_label_path, augmentation_type, img_width, img_height):
    """根据图像增强类型调整标注文件"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 对于水平翻转，需要调整x坐标
            if augmentation_type == 'flip_horizontal':
                x_center = 1.0 - x_center
            # 对于垂直翻转，需要调整y坐标
            elif augmentation_type == 'flip_vertical':
                y_center = 1.0 - y_center
            # 对于旋转，标注可能超出边界，这里简化处理（实际应该重新计算）
            # 其他增强类型不需要修改标注
            
            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    with open(output_label_path, 'w') as f:
        f.writelines(new_lines)

def balance_dataset(dataset_root, output_root, target_samples_per_class=None, max_augmentations=5):
    """平衡数据集，通过过采样少数类别"""
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 分析类别分布
    class_counts, image_class_map = analyze_class_distribution(dataset_root)
    
    if not class_counts:
        print("Error: No classes found in dataset")
        return
    
    # 确定目标样本数（默认为最大类别样本数的80%）
    if target_samples_per_class is None:
        max_count = max(class_counts.values())
        target_samples_per_class = int(max_count * 0.8)
    
    print(f"\nClass distribution analysis:")
    print(f"Target samples per class: {target_samples_per_class}")
    print(f"Current distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {count} samples")
    
    # 处理训练集
    print("\nBalancing training set...")
    images_dir = dataset_root / 'images' / 'train'
    labels_dir = dataset_root / 'labels' / 'train'
    
    augmented_count = 0
    
    for class_id, current_count in class_counts.items():
        if current_count >= target_samples_per_class:
            # 类别样本充足，直接复制
            for image_id in image_class_map[class_id]:
                # 查找图像文件
                image_path = None
                for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                    candidate = images_dir / f"{image_id}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                
                if image_path:
                    label_path = labels_dir / f"{image_id}.txt"
                    if label_path.exists():
                        shutil.copy2(image_path, output_root / 'images' / 'train' / image_path.name)
                        shutil.copy2(label_path, output_root / 'labels' / 'train' / label_path.name)
        else:
            # 类别样本不足，需要增强
            needed = target_samples_per_class - current_count
            print(f"  Class {class_id}: {current_count} -> {target_samples_per_class} (need {needed} more)")
            
            # 复制现有样本
            for image_id in image_class_map[class_id]:
                image_path = None
                for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                    candidate = images_dir / f"{image_id}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                
                if image_path:
                    label_path = labels_dir / f"{image_id}.txt"
                    if label_path.exists():
                        shutil.copy2(image_path, output_root / 'images' / 'train' / image_path.name)
                        shutil.copy2(label_path, output_root / 'labels' / 'train' / label_path.name)
            
            # 通过增强生成新样本
            augmentation_types = ['flip_horizontal', 'brightness', 'contrast', 'noise', 'blur']
            samples_to_augment = image_class_map[class_id] * ((needed // len(image_class_map[class_id])) + 1)
            samples_to_augment = samples_to_augment[:needed]
            
            for idx, image_id in enumerate(samples_to_augment):
                image_path = None
                for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                    candidate = images_dir / f"{image_id}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                
                if not image_path:
                    continue
                
                label_path = labels_dir / f"{image_id}.txt"
                if not label_path.exists():
                    continue
                
                # 选择增强类型
                aug_type = augmentation_types[idx % len(augmentation_types)]
                
                # 生成增强后的文件名
                output_image_name = f"{image_id}_aug{idx}_{aug_type}{image_path.suffix}"
                output_label_name = f"{image_id}_aug{idx}_{aug_type}.txt"
                
                output_image_path = output_root / 'images' / 'train' / output_image_name
                output_label_path = output_root / 'labels' / 'train' / output_label_name
                
                # 读取图像尺寸
                img = cv2.imread(str(image_path))
                if img is not None:
                    h, w = img.shape[:2]
                    
                    # 增强图像
                    if augment_image(image_path, output_image_path, aug_type):
                        # 调整标注（只对需要调整标注的增强类型进行处理）
                        if aug_type in ['flip_horizontal', 'flip_vertical']:
                            augment_label(label_path, output_label_path, aug_type, w, h)
                        else:
                            # 其他增强类型不需要修改标注，直接复制
                            shutil.copy2(label_path, output_label_path)
                        augmented_count += 1
    
    # 复制验证集和测试集（不进行增强）
    print("\nCopying validation and test sets...")
    for split in ['val', 'test']:
        src_images = dataset_root / 'images' / split
        src_labels = dataset_root / 'labels' / split
        dst_images = output_root / 'images' / split
        dst_labels = output_root / 'labels' / split
        
        if src_images.exists():
            for img_file in src_images.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img_file, dst_images / img_file.name)
        
        if src_labels.exists():
            for label_file in src_labels.glob('*.txt'):
                shutil.copy2(label_file, dst_labels / label_file.name)
    
    print(f"\nDataset balancing completed!")
    print(f"Generated {augmented_count} augmented samples")
    print(f"Balanced dataset saved to: {output_root}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance dataset by class')
    parser.add_argument('--input', type=str, default='dataset_filtered',
                        help='Input dataset directory')
    parser.add_argument('--output', type=str, default='dataset_balanced',
                        help='Output balanced dataset directory')
    parser.add_argument('--target', type=int, default=None,
                        help='Target samples per class (default: 80% of max)')
    parser.add_argument('--max-aug', type=int, default=5,
                        help='Maximum augmentations per image')
    
    args = parser.parse_args()
    
    input_dir = Path(__file__).parent / args.input
    output_dir = Path(__file__).parent / args.output
    
    if not input_dir.exists():
        print(f"Error: Input dataset not found: {input_dir}")
        return
    
    balance_dataset(input_dir, output_dir, args.target, args.max_aug)

if __name__ == '__main__':
    main()

