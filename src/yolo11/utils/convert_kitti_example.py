#!/usr/bin/env python3
"""
Example usage of the KITTI to YOLO converter with benchmark class remapping

This example demonstrates how to use the KITTI converter with the standard
benchmark class mapping (3 scored classes + ignored classes).
"""

from kitti_converter import KittiToYoloConverter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define paths - UPDATE THESE TO YOUR ACTUAL PATHS
    kitti_root = "./data/kitti"  # Update this path
    yolo_root = "./data/yolo_kitti"  # Update this path
    
    print("KITTI to YOLO Dataset Converter (Benchmark Class Mapping)")
    print("=" * 60)
    print(f"Input (KITTI): {kitti_root}")
    print(f"Output (YOLO): {yolo_root}")
    print()
    print("Class Mapping:")
    print("  Car + Van → Class 0 (Car)")
    print("  Pedestrian + Person_sitting → Class 1 (Pedestrian)")
    print("  Cyclist → Class 2 (Cyclist)")
    print("  Truck, Tram, Misc, DontCare → Class -1 (Ignored)")
    print()
    
    # Create converter
    converter = KittiToYoloConverter(kitti_root, yolo_root)
    
    # Run conversion
    converter.convert()
    
    print("\nConversion completed!")
    print(f"YOLO dataset created at: {yolo_root}")
    print("Dataset structure:")
    print("  yolo_kitti/")
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
    print()
    print("Note: Ignored boxes are saved with class_id = -1 in label files")
    print("YOLO will automatically handle these during training (zero loss)")
    print("Only 3 scored classes (Car, Pedestrian, Cyclist) appear in dataset.yaml")

if __name__ == "__main__":
    main()
