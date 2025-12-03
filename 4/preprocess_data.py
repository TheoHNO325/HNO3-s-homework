#!/usr/bin/env python3
"""
数据预处理脚本：将VOC格式转换为YOLO格式
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def parse_voc_xml(xml_path):
    """解析VOC格式的XML文件，返回图像信息和标注"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图像尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # 获取所有目标对象
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    
    return {
        'width': img_width,
        'height': img_height,
        'objects': objects
    }

def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    """将VOC格式的边界框转换为YOLO格式（归一化的中心点坐标和宽高）"""
    # 计算中心点
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    
    # 计算宽高
    width = xmax - xmin
    height = ymax - ymin
    
    # 归一化
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return center_x, center_y, width, height

def load_class_names(labels_file):
    """从labels.txt加载类别名称，返回类别到索引的映射"""
    class_names = []
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            class_name = line.strip()
            if class_name:
                class_names.append(class_name)
    
    # 创建类别名到索引的映射
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return class_names, class_to_idx

def process_dataset(voc_root, output_root, class_to_idx):
    """处理数据集，将VOC格式转换为YOLO格式"""
    voc_root = Path(voc_root)
    output_root = Path(output_root)
    
    # 创建输出目录
    (output_root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_root / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
    
    images_dir = voc_root / 'JPEGImages_old'
    annotations_dir = voc_root / 'Annotations_old'
    imagesets_dir = voc_root / 'ImageSets' / 'Main'
    
    # 处理train、val、test数据集
    for split in ['train', 'val', 'test']:
        split_file = imagesets_dir / f'{split}.txt'
        if not split_file.exists():
            print(f"Warning: {split_file} does not exist, skipping {split} split")
            continue
        
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {split} split: {len(image_ids)} images")
        
        for image_id in image_ids:
            # 查找图像文件（可能.jpg或.JPG）
            image_path = None
            for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                candidate = images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                print(f"Warning: Image {image_id} not found, skipping")
                continue
            
            # 查找对应的XML标注文件
            xml_path = annotations_dir / f"{image_id}.xml"
            if not xml_path.exists():
                print(f"Warning: Annotation {image_id}.xml not found, skipping")
                continue
            
            # 解析XML
            try:
                annotation = parse_voc_xml(xml_path)
            except Exception as e:
                print(f"Error parsing {xml_path}: {e}, skipping")
                continue
            
            # 复制图像文件
            output_image_path = output_root / 'images' / split / image_path.name
            shutil.copy2(image_path, output_image_path)
            
            # 转换标注为YOLO格式并保存
            output_label_path = output_root / 'labels' / split / f"{image_id}.txt"
            with open(output_label_path, 'w') as f:
                for obj in annotation['objects']:
                    class_name = obj['name']
                    
                    # 检查类别是否在映射中
                    if class_name not in class_to_idx:
                        print(f"Warning: Unknown class '{class_name}' in {image_id}, skipping")
                        continue
                    
                    class_idx = class_to_idx[class_name]
                    center_x, center_y, width, height = convert_to_yolo_format(
                        obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'],
                        annotation['width'], annotation['height']
                    )
                    
                    f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Completed {split} split")

def main():
    # 路径配置
    voc_root = Path(__file__).parent / 'VOC2007' / 'VOC2007'
    output_root = Path(__file__).parent / 'dataset'
    labels_file = voc_root / 'labels.txt'
    
    print("Loading class names...")
    class_names, class_to_idx = load_class_names(labels_file)
    print(f"Loaded {len(class_names)} classes")
    
    # 保存类别名称到文件
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / 'classes.txt', 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print("\nProcessing dataset...")
    process_dataset(voc_root, output_root, class_to_idx)
    
    print("\n" + "="*50)
    print("Data preprocessing completed!")
    print(f"Output directory: {output_root}")
    print(f"Total classes: {len(class_names)}")
    print("="*50)

if __name__ == '__main__':
    main()

