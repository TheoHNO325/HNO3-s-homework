#!/usr/bin/env python3
"""
视频推理脚本：对视频中的物体进行目标检测和分类
"""
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse

def load_class_names(classes_file):
    """加载类别名称"""
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def draw_boxes(frame, results, class_names, conf_threshold=0.25):
    """在图像上绘制检测框和标签"""
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
            
            # 绘制边界框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 准备标签文本
            label = f"{class_name}: {confidence:.2f}"
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制标签背景
            cv2.rectangle(
                frame,
                (int(x1), int(y1) - text_height - baseline - 5),
                (int(x1) + text_width, int(y1)),
                (0, 255, 0),
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
    
    return frame

def inference_video(model_path, video_path, output_path, classes_file, conf_threshold=0.25):
    """对视频进行推理"""
    # 加载模型
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # 加载类别名称
    print(f"Loading class names from: {classes_file}")
    class_names = load_class_names(classes_file)
    print(f"Number of classes: {len(class_names)}")
    
    # 打开视频文件
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("\nProcessing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 运行推理
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # 绘制检测框
        frame = draw_boxes(frame, results, class_names, conf_threshold)
        
        # 显示进度
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # 写入输出视频
        out.write(frame)
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo processing completed!")
    print(f"Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Video Inference')
    parser.add_argument('--model', type=str, default="runs/detect/yolo_voc20072/weights/best.pt",
                        help='Path to trained model weights (e.g., runs/detect/yolo_voc2007/weights/best.pt)')
    parser.add_argument('--video', type=str, default='inference/572996846.mp4',
                        help='input video file')
    parser.add_argument('--output', type=str, default='output_video/input_video_detected.mp4',
                        help='Path to output video file (default: input_video_detected.mp4)')
    parser.add_argument('--classes', type=str, default='dataset_filtered/classes.txt',
                        help='Path to classes.txt file (default: dataset_filtered/classes.txt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # 设置默认输出路径
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_detected.mp4")
    
    # 检查文件是否存在
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not Path(args.classes).exists():
        print(f"Error: Classes file not found: {args.classes}")
        return
    
    # 执行推理
    inference_video(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        classes_file=args.classes,
        conf_threshold=args.conf
    )

if __name__ == '__main__':
    main()

