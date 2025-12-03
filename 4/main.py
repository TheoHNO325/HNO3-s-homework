#!/usr/bin/env python3
"""
主程序入口
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='YOLO VOC2007 Project Main Entry')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'train', 'inference', 'overview'],
                        help='Mode: preprocess (数据预处理), train (训练模型), inference (视频推理), overview (数据概览)')
    
    # 预处理参数
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing')
    
    # 训练参数
    parser.add_argument('--train', action='store_true',
                        help='Start training')
    
    # 推理参数
    parser.add_argument('--model', type=str,
                        help='Path to model for inference')
    parser.add_argument('--video', type=str,
                        help='Path to video for inference')
    parser.add_argument('--output', type=str,
                        help='Output video path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for inference')
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        print("Running data preprocessing...")
        from preprocess_data import main as preprocess_main
        preprocess_main()
    
    elif args.mode == 'train':
        print("Starting training...")
        from train import main as train_main
        train_main()
    
    elif args.mode == 'overview':
        print("Generating data overview...")
        from data_overview import main as overview_main
        overview_main()
    
    elif args.mode == 'inference':
        if not args.model or not args.video:
            print("Error: --model and --video are required for inference mode")
            print("Usage: python main.py --mode inference --model <model_path> --video <video_path> [--output <output_path>] [--conf <threshold>]")
            return
        
        print("Running video inference...")
        from inference_video import inference_video, load_class_names
        from pathlib import Path
        
        classes_file = Path(__file__).parent / 'dataset_filtered' / 'classes.txt'
        if not classes_file.exists():
            print(f"Error: Classes file not found: {classes_file}")
            return
        
        output_path = args.output or str(Path(args.video).parent / f"{Path(args.video).stem}_detected.mp4")
        
        inference_video(
            model_path=args.model,
            video_path=args.video,
            output_path=output_path,
            classes_file=str(classes_file),
            conf_threshold=args.conf
        )

if __name__ == '__main__':
    main()

