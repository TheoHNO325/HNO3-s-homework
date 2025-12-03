#!/usr/bin/env python3
"""
数据概览脚本：统计每个类别的样本数量，并保存样例图片
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import cv2
import shutil
from PIL import Image, ImageDraw, ImageFont
import os

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

def draw_boxes_on_image(image_path, annotation, output_path):
    """在图像上绘制边界框并保存"""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    
    # 绘制边界框
    for obj in annotation['objects']:
        xmin = obj['xmin']
        ymin = obj['ymin']
        xmax = obj['xmax']
        ymax = obj['ymax']
        class_name = obj['name']
        
        # 绘制矩形框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        
        # 添加类别标签
        label = class_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制文本背景
        cv2.rectangle(img, 
                     (xmin, ymin - text_height - baseline - 5),
                     (xmin + text_width, ymin),
                     (0, 255, 0), -1)
        
        # 绘制文本
        cv2.putText(img, label,
                   (xmin, ymin - baseline - 2),
                   font, font_scale, (0, 0, 0), thickness)
    
    # 保存图像
    cv2.imwrite(str(output_path), img)
    return True

def generate_overview(voc_root, output_dir, classes_file=None):
    """生成数据概览"""
    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建样例图片目录
    samples_dir = output_dir / 'class_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = voc_root / 'JPEGImages_old'
    annotations_dir = voc_root / 'Annotations_old'
    
    # 统计每个类别的样本数量
    class_counts = defaultdict(int)
    class_samples = {}  # 存储每个类别的第一个样例图片路径
    
    # 遍历所有标注文件
    print("Scanning annotations...")
    xml_files = list(annotations_dir.glob('*.xml'))
    print(f"Found {len(xml_files)} annotation files")
    
    for xml_path in xml_files:
        try:
            annotation = parse_voc_xml(xml_path)
            image_id = xml_path.stem
            
            # 查找对应的图像文件
            image_path = None
            for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                candidate = images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                continue
            
            # 统计每个类别
            for obj in annotation['objects']:
                class_name = obj['name']
                class_counts[class_name] += 1
                
                # 保存第一个样例图片
                if class_name not in class_samples:
                    class_samples[class_name] = {
                        'image_path': image_path,
                        'annotation': annotation,
                        'image_id': image_id
                    }
        
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            continue
    
    # 如果有classes_file，按顺序排序
    if classes_file and Path(classes_file).exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            ordered_classes = [line.strip() for line in f if line.strip()]
    else:
        ordered_classes = sorted(class_counts.keys())
    
    # 生成统计报告
    report_path = output_dir / 'data_overview.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("数据概览报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"总类别数: {len(class_counts)}\n")
        f.write(f"总样本数: {sum(class_counts.values())}\n\n")
        f.write("-"*80 + "\n")
        f.write(f"{'类别索引':<10} {'类别名称':<50} {'样本数量':<10}\n")
        f.write("-"*80 + "\n")
        
        for idx, class_name in enumerate(ordered_classes):
            count = class_counts.get(class_name, 0)
            f.write(f"{idx:<10} {class_name:<50} {count:<10}\n")
        
        f.write("-"*80 + "\n")
    
    print(f"\n统计报告已保存到: {report_path}")
    
    # 保存样例图片
    print("\n保存样例图片...")
    saved_count = 0
    for idx, class_name in enumerate(ordered_classes):
        if class_name in class_samples:
            sample_info = class_samples[class_name]
            image_path = sample_info['image_path']
            annotation = sample_info['annotation']
            
            # 创建输出文件名（使用类别索引和名称）
            safe_name = class_name.replace('/', '_').replace('\\', '_')
            output_image_path = samples_dir / f"{idx:02d}_{safe_name}.jpg"
            
            # 绘制边界框并保存
            if draw_boxes_on_image(image_path, annotation, output_image_path):
                saved_count += 1
                if saved_count % 10 == 0:
                    print(f"已保存 {saved_count}/{len(class_samples)} 个样例图片...")
    
    print(f"样例图片已保存到: {samples_dir}")
    print(f"共保存 {saved_count} 个类别的样例图片")
   
    
    return ordered_classes, class_counts



def main():
    voc_root = Path(__file__).parent / 'VOC2007' / 'VOC2007'
    output_dir = Path(__file__).parent / 'data_overview'
    classes_file = voc_root / 'labels.txt'
    
    if not voc_root.exists():
        print(f"Error: VOC dataset directory not found: {voc_root}")
        return
    
    print("="*80)
    print("数据概览生成")
    print("="*80)
    
    ordered_classes, class_counts = generate_overview(voc_root, output_dir, classes_file)
    
    print("\n" + "="*80)
    print("数据概览完成！")
    print(f"输出目录: {output_dir}")
    print("="*80)
    
    # 打印简要统计
    print("\n类别统计（前10个）:")
    for idx, class_name in enumerate(ordered_classes):
        count = class_counts.get(class_name, 0)
        print(f"  [{idx}] {class_name}: {count} 个样本")

if __name__ == '__main__':
    main()

