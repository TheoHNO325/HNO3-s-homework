#!/usr/bin/env python3
"""
显示所有类别信息
"""
from pathlib import Path

def show_classes(labels_file):
    """显示所有类别"""
    labels_file = Path(labels_file)
    
    if not labels_file.exists():
        print(f"Error: Labels file not found: {labels_file}")
        return
    
    print("="*60)
    print("所有类别列表")
    print("="*60)
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    for idx, class_name in enumerate(classes, 1):
        print(f"{idx:3d}. {class_name}")
    
    print("="*60)
    print(f"总共 {len(classes)} 个类别")
    print("="*60)

if __name__ == '__main__':
    # 尝试从VOC2007目录读取
    labels_file = Path(__file__).parent / 'VOC2007' / 'VOC2007' / 'labels.txt'
    
    if not labels_file.exists():
        # 如果不存在，尝试从dataset目录读取
        labels_file = Path(__file__).parent / 'dataset' / 'classes.txt'
    
    show_classes(labels_file)


