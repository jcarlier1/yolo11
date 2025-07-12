#!/usr/bin/env python3
"""
Example usage of the KITTI to YOLO converter for Car detection only
"""

from kitti_car_converter import KittiCarToYoloConverter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define paths
    kitti_root = "/home/carlier1/data/kitti"
    yolo_root = "/home/carlier1/data/yolo_kitti_cars"
    
    print("KITTI to YOLO Dataset Converter (Car Detection Only)")
    print("=" * 55)
    print(f"Input (KITTI): {kitti_root}")
    print(f"Output (YOLO): {yolo_root}")
    print()
    print("Features:")
    print("- Filters only Car bounding boxes from KITTI dataset")
    print("- Ignores all other classes (Van, Truck, Pedestrian, etc.)")
    print("- Converts Car class to YOLO class ID 0")
    print("- Creates single-class car detection dataset")
    print()
    
    # Create converter
    converter = KittiCarToYoloConverter(kitti_root, yolo_root)
    
    # Run conversion
    converter.convert()
    
    print("\nConversion completed!")
    print(f"YOLO car detection dataset created at: {yolo_root}")
    print("Dataset structure:")
    print("  yolo_kitti_cars/")
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
    print("Notes:")
    print("- Only images with Car annotations will have corresponding label files")
    print("- Images without Cars will have empty or no label files")
    print("- All Car instances are labeled with class ID 0")
    print("- Dataset is ready for YOLO training with 1 class (Car)")

if __name__ == "__main__":
    main()
