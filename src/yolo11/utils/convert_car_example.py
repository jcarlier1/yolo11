#!/usr/bin/env python3
"""
Example usage of the KITTI to YOLO converter for Car detection only
"""

from kitti_car_converter import KittiCarToYoloConverter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define paths - UPDATE THESE TO YOUR ACTUAL PATHS
    kitti_root = "./data/kitti"  # Update this path
    yolo_root = "./data/yolo_kitti_cars"  # Update this path
    
    # Define train/validation split ratio (80% train, 20% validation)
    train_split = 0.8
    
    print("KITTI to YOLO Dataset Converter (Car Detection Only)")
    print("=" * 55)
    print(f"Input (KITTI): {kitti_root}")
    print(f"Output (YOLO): {yolo_root}")
    print(f"Train/Val Split: {train_split:.1%} train, {1-train_split:.1%} validation")
    print()
    print("Features:")
    print("- Filters only Car bounding boxes from KITTI dataset")
    print("- Ignores all other classes (Van, Truck, Pedestrian, etc.)")
    print("- Converts Car class to YOLO class ID 0")
    print("- Creates single-class car detection dataset")
    print("- Creates custom train/validation splits from training data")
    print("- Uses testing data for test set")
    print()
    
    # Create converter with custom train_split
    converter = KittiCarToYoloConverter(kitti_root, yolo_root, train_split)
    
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
    print("- Train/validation split is created from KITTI training data")
    print("- Test set uses KITTI testing data (no labels available)")
    print("- Split ratio is configurable (default: 80% train, 20% validation)")
    print("- Only images with Car annotations will have corresponding label files")
    print("- Images without Cars will have empty or no label files")
    print("- All Car instances are labeled with class ID 0")
    print("- Dataset is ready for YOLO training with 1 class (Car)")
    print()
    print("Alternative usage with different split ratios:")
    print("- converter = KittiCarToYoloConverter(kitti_root, yolo_root, 0.7)  # 70/30 split")
    print("- converter = KittiCarToYoloConverter(kitti_root, yolo_root, 0.85) # 85/15 split")

if __name__ == "__main__":
    main()
