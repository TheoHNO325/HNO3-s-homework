#!/usr/bin/env python3
"""
视频推理脚本：对视频中的物体进行目标检测和分类
"""
from curses import color_content
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import random

def load_class_names(classes_file):
    """加载类别名称"""
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

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

def draw_boxes(frame, results, class_names, conf_threshold=0.25, line_thickness=5, font_scale=3.0):
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
            color = get_color_for_class(class_id)
            
            # 绘制边界框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
            
            # 准备标签文本
            label = f"{class_name}: {confidence:.2f}"
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            
            # 绘制标签背景
            cv2.rectangle(
                frame,
                (int(x1), int(y1) - text_height - baseline - 5),
                (int(x1) + text_width, int(y1)),
                color,
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                line_thickness
            )
    
    return frame

def inference_video(model_path, video_path, output_path, classes_file, conf_threshold=0.25, save_random_frame=True, random_frame_output=None):
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
    
    # 随机选择一帧用于保存
    random_frame_number = None
    if save_random_frame and total_frames > 0:
        random_frame_number = random.randint(1, total_frames)
        print(f"\nWill save random frame #{random_frame_number} for label inspection")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    saved_random_frame = False
    
    print("\nProcessing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 运行推理
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # 绘制检测框
        frame_with_boxes = draw_boxes(frame.copy(), results, class_names, conf_threshold)
        
        # 如果当前帧是随机选择的帧，保存它
        if save_random_frame and random_frame_number is not None and frame_count == random_frame_number and not saved_random_frame:
            if random_frame_output is None:
                video_path_obj = Path(video_path)
                random_frame_output = str(video_path_obj.parent / f"{video_path_obj.stem}_random_frame_{random_frame_number}.jpg")
            
            cv2.imwrite(random_frame_output, frame_with_boxes)
            print(f"\nSaved random frame #{random_frame_number} to: {random_frame_output}")
            saved_random_frame = True
        
        # 显示进度
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # 写入输出视频
        out.write(frame_with_boxes)
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo processing completed!")
    print(f"Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")
    if save_random_frame and saved_random_frame:
        print(f"Random frame saved to: {random_frame_output}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Video Inference')
    parser.add_argument('--model', type=str, default="runs/detect/yolo_voc20072/weights/best.pt",
                        help='Path to trained model weights (e.g., runs/detect/yolo_voc2007/weights/best.pt)')
    parser.add_argument('--video', type=str, default='inference/572996846.mp4',
                        help='input video file')
    parser.add_argument('--output', type=str, default='/home/hno3/Documents/GithubFiles/HNO3‘s  homework/4/inference/input_video_detected.mp4',
                        help='Path to output video file (default: input_video_detected.mp4)')
    parser.add_argument('--classes', type=str, default='dataset_filtered/classes.txt',
                        help='Path to classes.txt file (default: dataset_filtered/classes.txt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--save-random-frame', action='store_true', default=True,
                        help='Save a random frame from video for label inspection (default: True)')
    parser.add_argument('--no-save-random-frame', dest='save_random_frame', action='store_false',
                        help='Disable saving random frame')
    parser.add_argument('--random-frame-output', type=str, default=None,
                        help='Path to save random frame image (default: auto-generated)')
    
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
        conf_threshold=args.conf,
        save_random_frame=args.save_random_frame,
        random_frame_output=args.random_frame_output
    )

if __name__ == '__main__':
    main()

