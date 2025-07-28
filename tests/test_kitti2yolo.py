#!/usr/bin/env python3
"""
Test script for KITTI to YOLO converter.

This script creates sample KITTI data and tests the conversion functionality.
"""

import tempfile
import shutil
from pathlib import Path
import sys
import os

try:
    from src.yolo11.utils.kitti2yolo import KittiToYoloConverter
except ImportError as e:
    print(f"Error importing kitti2yolo: {e}")
    print("Make sure PyYAML, NumPy, and tqdm are installed:")
    print("pip install pyyaml numpy tqdm")
    sys.exit(1)

def create_sample_kitti_data(kitti_root: Path):
    """Create sample KITTI dataset for testing."""
    # Create directory structure
    image_dir = kitti_root / "training" / "image_2"
    label_dir = kitti_root / "training" / "label_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images (empty files for testing)
    sample_images = ["000001.png", "000002.png", "000003.png", "000004.png", "000005.png"]
    for img_name in sample_images:
        (image_dir / img_name).write_bytes(b"fake_image_data")
    
    # Create sample labels with various classes
    labels = {
        "000001.txt": [
            "Car 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 1.63 8.34 2.57 1.57 69.44 -1.56",
            "Pedestrian 0.00 0 -0.20 423.17 173.67 433.72 224.03 1.89 0.48 1.20 1.84 1.47 8.41 0.01"
        ],
        "000002.txt": [
            "Van 0.00 0 1.85 387.63 181.54 423.81 203.12 2.20 1.47 5.63 4.59 1.32 45.84 1.84",
            "Cyclist 0.00 0 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 4.59 1.32 45.84 -1.55"
        ],
        "000003.txt": [
            "Car 0.00 0 -1.57 318.19 157.00 403.24 220.58 1.56 1.58 3.48 -2.15 1.65 9.04 -1.59",
            "Person_sitting 0.00 1 -1.22 696.27 143.35 711.73 164.13 1.56 1.58 3.48 -2.15 1.65 9.04 -1.22"
        ],
        "000004.txt": [
            "Truck 0.00 0 -1.63 511.35 140.16 527.81 179.61 2.85 1.63 8.34 2.57 1.57 69.44 -1.56",
            "DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10"
        ],
        "000005.txt": [
            # Empty file (no annotations)
        ]
    }
    
    for label_name, lines in labels.items():
        label_file = label_dir / label_name
        label_file.write_text('\n'.join(lines) + '\n' if lines else '')
    
    print(f"Created sample KITTI data in {kitti_root}")

def test_conversion():
    """Test the KITTI to YOLO conversion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        kitti_path = temp_path / "kitti"
        yolo_path = temp_path / "yolo"
        
        # Create sample data
        create_sample_kitti_data(kitti_path)
        
        # Test conversion with default 3-class mapping
        converter = KittiToYoloConverter()
        
        try:
            converter.convert_kitti_to_yolo(
                kitti_path=str(kitti_path),
                output_path=str(yolo_path),
                train_ratio=0.8,
                kfold=3,
                seed=42,
                stratify=True
            )
            
            print("✅ Conversion completed successfully!")
            
            # Verify output structure
            expected_dirs = [
                "images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test",
                "splits"
            ]
            
            for dir_path in expected_dirs:
                full_path = yolo_path / dir_path
                if not full_path.exists():
                    print(f"❌ Missing directory: {full_path}")
                    return False
                else:
                    print(f"✅ Found directory: {dir_path}")
            
            # Check dataset.yaml
            dataset_yaml = yolo_path / "dataset.yaml"
            if dataset_yaml.exists():
                print("✅ Found dataset.yaml")
                print(f"Content:\n{dataset_yaml.read_text()}")
            else:
                print("❌ Missing dataset.yaml")
                return False
            
            # Check k-fold splits
            splits_dir = yolo_path / "splits"
            kfold_files = list(splits_dir.glob("kfold_*.txt"))
            if len(kfold_files) == 3:
                print(f"✅ Found {len(kfold_files)} k-fold split files")
            else:
                print(f"❌ Expected 3 k-fold files, found {len(kfold_files)}")
                return False
            
            # Check some label files
            train_labels = list((yolo_path / "labels" / "train").glob("*.txt"))
            val_labels = list((yolo_path / "labels" / "val").glob("*.txt"))
            
            print(f"✅ Train labels: {len(train_labels)}")
            print(f"✅ Val labels: {len(val_labels)}")
            
            # Show content of a sample label file
            if train_labels:
                sample_label = train_labels[0]
                content = sample_label.read_text().strip()
                if content:
                    print(f"✅ Sample label content ({sample_label.name}):")
                    for line in content.split('\n'):
                        print(f"    {line}")
                else:
                    print(f"✅ Sample label ({sample_label.name}) is empty (no annotations)")
            
            # Print conversion statistics
            print("\n📊 Final Statistics:")
            converter.print_statistics()
            
            return True
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_different_mappings():
    """Test different class mapping configurations."""
    print("\n" + "="*60)
    print("Testing different class mappings...")
    print("="*60)
    
    # Test mapping 1: All classes
    all_classes_remap = {
        "Car": 0, "Van": 1, "Truck": 2,
        "Pedestrian": 3, "Person_sitting": 4, "Cyclist": 5,
        "Tram": 6, "Misc": 7
    }
    
    # Test mapping 2: Car detection only
    car_only_remap = {
        "Car": 0, "Van": 0
    }
    
    # Test mapping 3: With ignore classes
    ignore_remap = {
        "Car": 0, "Van": 0,
        "Pedestrian": 1, "Person_sitting": 1, "Cyclist": 2,
        "Truck": -1, "Misc": -1  # Ignore these
    }
    
    test_mappings = [
        ("All 8 classes", all_classes_remap),
        ("Car detection only", car_only_remap),
        ("With ignore classes", ignore_remap)
    ]
    
    for name, remap in test_mappings:
        print(f"\nTesting: {name}")
        print(f"Mapping: {remap}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            kitti_path = temp_path / "kitti"
            yolo_path = temp_path / "yolo"
            
            create_sample_kitti_data(kitti_path)
            
            converter = KittiToYoloConverter(remap=remap)
            
            try:
                converter.convert_kitti_to_yolo(
                    kitti_path=str(kitti_path),
                    output_path=str(yolo_path),
                    train_ratio=0.8,
                    kfold=2,
                    seed=42,
                    stratify=False
                )
                print(f"✅ {name} mapping test passed")
                converter.print_statistics()
                
            except Exception as e:
                print(f"❌ {name} mapping test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing KITTI to YOLO Converter")
    print("="*60)
    
    # Test basic conversion
    success = test_conversion()
    
    if success:
        # Test different mappings
        test_different_mappings()
        print("\n🎉 All tests completed!")
    else:
        print("\n❌ Basic test failed. Please check the implementation.")
        sys.exit(1)
