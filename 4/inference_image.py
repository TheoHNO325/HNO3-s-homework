#!/usr/bin/env python3
"""
图片推理脚本：对图片中的物体进行目标检测和分类
支持单张图片和批量图片处理
"""
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import numpy as np
from collections import defaultdict

def load_class_names(classes_file):
    """加载类别名称"""
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def draw_boxes(image, results, class_names, conf_threshold=0.25, line_thickness=2, font_scale=0.6):
    """在图像上绘制检测框和标签"""
    detected_objects = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # 只绘制置信度大于阈值的检测框
            if confidence < conf_threshold:
                continue
            
            # 获取类别名称
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            
            # 记录检测结果
            detected_objects.append({
                'class': class_name,
                'confidence': float(confidence),
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
            
            # 绘制边界框（使用不同颜色区分不同类别）
            color = get_color_for_class(class_id)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
            
            # 准备标签文本
            label = f"{class_name}: {confidence:.2f}"
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            
            # 绘制标签背景
            cv2.rectangle(
                image,
                (int(x1), int(y1) - text_height - baseline - 5),
                (int(x1) + text_width, int(y1)),
                color,
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                image,
                label,
                (int(x1), int(y1) - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness
            )
    
    return image, detected_objects

def get_color_for_class(class_id):
    """为不同类别生成不同的颜色"""
    colors = [
        (0, 255, 0),      # 绿色
        (255, 0, 0),      # 蓝色
        (0, 0, 255),      # 红色
        (255, 255, 0),    # 青色
        (255, 0, 255),    # 洋红色
        (0, 255, 255),    # 黄色
        (128, 0, 128),    # 紫色
        (255, 165, 0),    # 橙色
    ]
    return colors[class_id % len(colors)]

def inference_single_image(model, image_path, output_path, class_names, conf_threshold=0.25, save_result=True):
    """对单张图片进行推理"""
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Cannot read image: {image_path}")
        return None
    
    print(f"Processing: {image_path.name}")
    
    # 运行推理
    results = model(image, conf=conf_threshold, verbose=False)
    
    # 绘制检测框
    annotated_image, detected_objects = draw_boxes(image.copy(), results, class_names, conf_threshold)
    
    # 保存结果
    if save_result:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated_image)
        print(f"  Saved to: {output_path}")
    
    # 打印检测结果
    if detected_objects:
        print(f"  Detected {len(detected_objects)} objects:")
        for obj in detected_objects:
            print(f"    - {obj['class']}: {obj['confidence']:.2f}")
    else:
        print(f"  No objects detected (conf >= {conf_threshold})")
    
    return {
        'image_path': str(image_path),
        'output_path': str(output_path) if save_result else None,
        'detections': detected_objects,
        'count': len(detected_objects)
    }

def inference_batch_images(model, image_dir, output_dir, class_names, conf_threshold=0.25, 
                           image_extensions=None):
    """批量处理图片"""
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP']
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'**/*{ext}'))
    
    if not image_files:
        print(f"No images found in: {image_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    class_statistics = defaultdict(int)
    
    for idx, image_path in enumerate(image_files, 1):
        # 生成输出路径
        relative_path = image_path.relative_to(image_dir)
        output_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
        
        print(f"\n[{idx}/{len(image_files)}] ", end="")
        result = inference_single_image(
            model, image_path, output_path, class_names, conf_threshold, save_result=True
        )
        
        if result:
            results.append(result)
            # 统计每个类别的检测次数
            for det in result['detections']:
                class_statistics[det['class']] += 1
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Detection Statistics:")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {sum(r['count'] for r in results)}")
    print("\nDetections per class:")
    for class_name, count in sorted(class_statistics.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    print("="*60)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLO Image Inference')
    parser.add_argument('--model', type=str, default='runs/detect/yolo_voc20072/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image file or directory containing images')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output image/directory (default: source_detected/)')
    parser.add_argument('--classes', type=str, default='dataset_filtered/classes.txt',
                        help='Path to classes.txt file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--batch', action='store_true',
                        help='Process directory of images in batch mode')
    
    args = parser.parse_args()
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # 检查类别文件
    classes_path = Path(args.classes)
    if not classes_path.exists():
        # 尝试从dataset_filtered目录查找
        filtered_classes = Path(__file__).parent / 'dataset_filtered' / 'classes.txt'
        if filtered_classes.exists():
            classes_path = filtered_classes
            print(f"Using classes from: {classes_path}")
        else:
            print(f"Error: Classes file not found: {args.classes}")
            return
    
    # 加载模型和类别
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Loading class names from: {classes_path}")
    class_names = load_class_names(classes_path)
    print(f"Number of classes: {len(class_names)}")
    
    # 处理输入
    source_path = Path(args.source)
    
    if not source_path.exists():
        print(f"Error: Source path not found: {source_path}")
        return
    
    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        if source_path.is_file():
            output_path = source_path.parent / f"{source_path.stem}_detected{source_path.suffix}"
        else:
            output_path = source_path.parent / f"{source_path.name}_detected"
    
    # 判断是单张图片还是批量处理
    if source_path.is_file() or (not args.batch and source_path.is_file()):
        # 单张图片处理
        print("\n" + "="*60)
        print("Single Image Inference")
        print("="*60)
        inference_single_image(
            model, source_path, output_path, class_names, args.conf, save_result=True
        )
        print(f"\nResult saved to: {output_path}")
    else:
        # 批量处理
        print("\n" + "="*60)
        print("Batch Image Inference")
        print("="*60)
        inference_batch_images(
            model, source_path, output_path, class_names, args.conf
        )
        print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()

