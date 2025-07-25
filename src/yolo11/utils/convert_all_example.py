#!/usr/bin/env python3
"""
Example usage of the KITTI to YOLO converter
"""

from kitti_all_converter import KittiToYoloConverter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define paths - UPDATE THESE TO YOUR ACTUAL PATHS
    kitti_root = "./data/kitti"  # Update this path
    yolo_root = "./data/yolo_kitti_all"  # Update this path
    
    print("KITTI to YOLO Dataset Converter")
    print("=" * 40)
    print(f"Input (KITTI): {kitti_root}")
    print(f"Output (YOLO): {yolo_root}")
    print()
    
    # Create converter
    converter = KittiToYoloConverter(kitti_root, yolo_root)
    
    # Run conversion
    converter.convert()
    
    print("\nConversion completed!")
    print(f"YOLO dataset created at: {yolo_root}")
    print("Dataset structure:")
    print("  yolo_kitti_all/")
    print("  ├── train/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  ├── val/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  ├── test/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  └── dataset.yaml")

if __name__ == "__main__":
    main()
