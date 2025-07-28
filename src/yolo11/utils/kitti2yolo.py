#!/usr/bin/env python3
"""
KITTI to YOLO Dataset Converter

Converts KITTI 2D object detection data to YOLO format with configurable class mapping,
train/val splits, and k-fold cross-validation support.

USAGE EXAMPLES:
    # Basic conversion with default 3-class mapping
    python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset
    
    # Custom train/val ratio with stratified sampling and 10-fold CV
    python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset \
                         --train_ratio 0.7 --kfold 10 --stratify --seed 123

LABEL REMAPPING:
    Edit the REMAP dictionary below to control class mapping:
    - Map multiple KITTI classes to one YOLO class: "Car": 0, "Van": 0
    - Keep classes as-is: "Car": 0, "Pedestrian": 1, etc.
    - Drop classes: omit them from the dict (lines will be skipped)
    - Mark as ignore: use -1 (though YOLO validation may skip these images)
    
    Example remappings:
    # 3-class (default):
    REMAP = {"Car": 0, "Van": 0, "Pedestrian": 1, "Person_sitting": 1, "Cyclist": 2}
    
    # All 8 classes:
    REMAP = {"Car": 0, "Van": 1, "Truck": 2, "Pedestrian": 3, 
             "Person_sitting": 4, "Cyclist": 5, "Tram": 6, "Misc": 7}
    
    # Car detection only:
    REMAP = {"Car": 0, "Van": 0}
"""

import argparse
import logging
import os
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import json

import numpy as np
from tqdm import tqdm

# =============================================================================
# LABEL REMAPPING CONFIGURATION
# =============================================================================
# Edit this dictionary to control how KITTI classes map to YOLO classes:
# - Key: KITTI class name (from label files)
# - Value: YOLO class ID (0-based integer, or -1 for ignore)
# - Classes not in this dict will be dropped/ignored

REMAP = {
    "Car": 0,
    "Van": 0,
    "Pedestrian": 1, 
    "Person_sitting": 1,
    "Cyclist": 2,
    # Dropped classes: "Truck", "Tram", "Misc", "DontCare"
}

# Alternative mappings (uncomment to use):
# 
# # All 8 real classes (no DontCare):
# REMAP = {
#     "Car": 0,
#     "Van": 1, 
#     "Truck": 2,
#     "Pedestrian": 3,
#     "Person_sitting": 4,
#     "Cyclist": 5,
#     "Tram": 6,
#     "Misc": 7
# }
#
# # Car detection only:
# REMAP = {
#     "Car": 0,
#     "Van": 0
# }
#
# # Mark some as ignore (class -1):
# REMAP = {
#     "Car": 0,
#     "Van": 0,
#     "Pedestrian": 1,
#     "Person_sitting": 1,
#     "Cyclist": 2,
#     "Truck": -1,  # Ignore during training/validation
#     "Tram": -1,
#     "Misc": -1
# }

# =============================================================================

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class KittiToYoloConverter:
    """Converts KITTI 2D object detection dataset to YOLO format."""
    
    def __init__(self, remap: Dict[str, int] = None):
        """
        Initialize converter.
        
        Args:
            remap: Dictionary mapping KITTI class names to YOLO class IDs
        """
        self.remap = remap or REMAP
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'converted_labels': 0,
            'dropped_labels': 0,
            'invalid_boxes': 0,
            'class_counts': Counter(),
            'images_per_class': defaultdict(set)
        }
        
    def parse_kitti_label(self, label_line: str) -> Optional[Tuple[str, float, float, float, float]]:
        """
        Parse a single line from KITTI label file.
        
        Args:
            label_line: Single line from KITTI label file
            
        Returns:
            Tuple of (class_name, left, top, right, bottom) or None if invalid
        """
        parts = label_line.strip().split()
        if len(parts) < 15:
            return None
            
        class_name = parts[0]
        truncated = float(parts[1])
        occluded = int(parts[2])
        alpha = float(parts[3])
        
        # 2D bounding box coordinates
        left = float(parts[4])
        top = float(parts[5]) 
        right = float(parts[6])
        bottom = float(parts[7])
        
        # Skip if box is invalid
        if right <= left or bottom <= top:
            return None
            
        return class_name, left, top, right, bottom
    
    def kitti_to_yolo_bbox(self, left: float, top: float, right: float, bottom: float, 
                          img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert KITTI bbox format to YOLO format.
        
        Args:
            left, top, right, bottom: KITTI bbox coordinates
            img_width, img_height: Image dimensions
            
        Returns:
            YOLO format (x_center, y_center, width, height) normalized to [0,1]
        """
        # Convert to center coordinates and normalize
        x_center = (left + right) / 2.0 / img_width
        y_center = (top + bottom) / 2.0 / img_height
        width = (right - left) / img_width
        height = (bottom - top) / img_height
        
        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return x_center, y_center, width, height
    
    def convert_single_image(self, image_path: Path, label_path: Path, 
                           output_label_path: Path) -> bool:
        """
        Convert a single image's labels from KITTI to YOLO format.
        
        Args:
            image_path: Path to image file
            label_path: Path to KITTI label file
            output_label_path: Path where YOLO label should be saved
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Get image dimensions (assuming standard KITTI image size)
            # In practice, you might want to load the image to get actual dimensions
            img_width, img_height = 1242, 375  # Standard KITTI image size
            
            converted_lines = []
            image_classes = set()
            
            if not label_path.exists():
                # Create empty label file for images without annotations
                output_label_path.write_text("")
                return True
                
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    self.stats['total_labels'] += 1
                    
                    parsed = self.parse_kitti_label(line)
                    if parsed is None:
                        self.stats['invalid_boxes'] += 1
                        self.logger.warning(f"Invalid label in {label_path}:{line_num}: {line.strip()}")
                        continue
                        
                    class_name, left, top, right, bottom = parsed
                    
                    # Check if class should be remapped
                    if class_name not in self.remap:
                        self.stats['dropped_labels'] += 1
                        continue
                        
                    yolo_class_id = self.remap[class_name]
                    
                    # Convert bbox format
                    x_center, y_center, width, height = self.kitti_to_yolo_bbox(
                        left, top, right, bottom, img_width, img_height
                    )
                    
                    # Skip zero-area boxes
                    if width <= 0 or height <= 0:
                        self.stats['invalid_boxes'] += 1
                        continue
                        
                    # Add to output
                    converted_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    self.stats['converted_labels'] += 1
                    self.stats['class_counts'][yolo_class_id] += 1
                    image_classes.add(yolo_class_id)
            
            # Update images_per_class statistics
            image_name = image_path.stem
            for cls_id in image_classes:
                self.stats['images_per_class'][cls_id].add(image_name)
            
            # Write YOLO label file
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(converted_lines))
                if converted_lines:  # Add final newline if file not empty
                    f.write('\n')
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting {image_path}: {e}")
            return False
    
    def create_stratified_splits(self, image_names: List[str], train_ratio: float, 
                               seed: int) -> Tuple[List[str], List[str]]:
        """
        Create stratified train/val split based on class distribution.
        
        Args:
            image_names: List of image names (without extension)
            train_ratio: Fraction of data for training
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_names, val_names)
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Group images by their class combinations
        class_combinations = defaultdict(list)
        
        for img_name in image_names:
            # Find which classes this image contains
            img_classes = []
            for cls_id in self.stats['images_per_class']:
                if img_name in self.stats['images_per_class'][cls_id]:
                    img_classes.append(cls_id)
            
            # Use sorted tuple as key to group similar images
            class_key = tuple(sorted(img_classes)) if img_classes else tuple()
            class_combinations[class_key].append(img_name)
        
        train_names = []
        val_names = []
        
        # Split each group according to train_ratio
        for class_key, img_list in class_combinations.items():
            random.shuffle(img_list)
            n_train = int(len(img_list) * train_ratio)
            train_names.extend(img_list[:n_train])
            val_names.extend(img_list[n_train:])
        
        return train_names, val_names
    
    def create_kfold_splits(self, image_names: List[str], k: int, seed: int, 
                          stratify: bool = False) -> List[List[str]]:
        """
        Create k-fold cross-validation splits.
        
        Args:
            image_names: List of image names
            k: Number of folds
            seed: Random seed
            stratify: Whether to use stratified sampling
            
        Returns:
            List of k folds, each containing image names
        """
        random.seed(seed)
        np.random.seed(seed)
        
        if not stratify:
            # Simple random split
            shuffled = image_names.copy()
            random.shuffle(shuffled)
            fold_size = len(shuffled) // k
            
            folds = []
            for i in range(k):
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < k - 1 else len(shuffled)
                folds.append(shuffled[start_idx:end_idx])
            return folds
        
        # Stratified k-fold
        class_combinations = defaultdict(list)
        
        for img_name in image_names:
            img_classes = []
            for cls_id in self.stats['images_per_class']:
                if img_name in self.stats['images_per_class'][cls_id]:
                    img_classes.append(cls_id)
            
            class_key = tuple(sorted(img_classes)) if img_classes else tuple()
            class_combinations[class_key].append(img_name)
        
        # Initialize folds
        folds = [[] for _ in range(k)]
        
        # Distribute each class combination across folds
        for class_key, img_list in class_combinations.items():
            random.shuffle(img_list)
            for i, img_name in enumerate(img_list):
                fold_idx = i % k
                folds[fold_idx].append(img_name)
        
        return folds
    
    def convert_kitti_to_yolo(self, kitti_path: str, output_path: str, 
                            train_ratio: float = 0.8, kfold: int = 5, 
                            seed: int = 42, stratify: bool = False) -> None:
        """
        Main conversion function.
        
        Args:
            kitti_path: Path to KITTI dataset root
            output_path: Path where YOLO dataset should be created
            train_ratio: Fraction of data for training (rest for validation)
            kfold: Number of k-fold splits to generate
            seed: Random seed for reproducibility
            stratify: Whether to use stratified sampling
        """
        kitti_root = Path(kitti_path)
        output_root = Path(output_path)
        
        # Validate input paths
        image_dir = kitti_root / "training" / "image_2"
        label_dir = kitti_root / "training" / "label_2"
        
        if not image_dir.exists():
            raise FileNotFoundError(f"KITTI image directory not found: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"KITTI label directory not found: {label_dir}")
        
        # Get all image files
        image_files = list(image_dir.glob("*.png"))
        if not image_files:
            image_files = list(image_dir.glob("*.jpg"))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {image_dir}")
        
        self.logger.info(f"Found {len(image_files)} images")
        self.logger.info(f"Using class mapping: {self.remap}")
        
        # Create output directories
        for split in ["train", "val", "test"]:
            (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        (output_root / "splits").mkdir(parents=True, exist_ok=True)
        
        # Convert all labels first
        self.logger.info("Converting labels...")
        valid_images = []
        
        for image_path in tqdm(image_files, desc="Converting labels"):
            image_name = image_path.stem
            label_path = label_dir / f"{image_name}.txt"
            
            # Convert and save to temporary location
            temp_label_path = output_root / "temp_labels" / f"{image_name}.txt"
            temp_label_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.convert_single_image(image_path, label_path, temp_label_path):
                valid_images.append(image_path)
                self.stats['total_images'] += 1
        
        if not valid_images:
            raise RuntimeError("No valid images found after conversion")
        
        self.logger.info(f"Successfully converted {len(valid_images)} images")
        
        # Create train/val split
        image_names = [img.stem for img in valid_images]
        
        if stratify:
            train_names, val_names = self.create_stratified_splits(image_names, train_ratio, seed)
            self.logger.info("Using stratified train/val split")
        else:
            random.seed(seed)
            random.shuffle(image_names)
            n_train = int(len(image_names) * train_ratio)
            train_names = image_names[:n_train]
            val_names = image_names[n_train:]
            self.logger.info("Using random train/val split")
        
        self.logger.info(f"Train: {len(train_names)}, Val: {len(val_names)}")
        
        # Copy files to train/val directories
        def copy_files(names: List[str], split: str):
            for name in tqdm(names, desc=f"Copying {split} files"):
                # Copy image
                src_img = next(img for img in valid_images if img.stem == name)
                dst_img = output_root / "images" / split / src_img.name
                dst_img.write_bytes(src_img.read_bytes())
                
                # Copy label
                src_label = output_root / "temp_labels" / f"{name}.txt"
                dst_label = output_root / "labels" / split / f"{name}.txt"
                if src_label.exists():
                    dst_label.write_text(src_label.read_text())
                else:
                    dst_label.write_text("")  # Empty label file
        
        copy_files(train_names, "train")
        copy_files(val_names, "val")
        
        # Generate k-fold splits
        if kfold > 1:
            self.logger.info(f"Generating {kfold}-fold cross-validation splits...")
            folds = self.create_kfold_splits(image_names, kfold, seed, stratify)
            
            for i, fold in enumerate(folds):
                fold_file = output_root / "splits" / f"kfold_{i}.txt"
                with open(fold_file, 'w') as f:
                    for name in fold:
                        f.write(f"{name}\n")
            
            self.logger.info(f"Saved k-fold splits to {output_root / 'splits'}")
        
        # Clean up temporary directory
        import shutil
        temp_dir = output_root / "temp_labels"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Create dataset.yaml for YOLO
        class_names = {}
        for kitti_class, yolo_id in self.remap.items():
            if yolo_id >= 0:  # Skip ignore classes
                if yolo_id not in class_names:
                    class_names[yolo_id] = []
                class_names[yolo_id].append(kitti_class)
        
        # Create readable class names
        yolo_class_names = {}
        for yolo_id in sorted(class_names.keys()):
            kitti_classes = class_names[yolo_id]
            if len(kitti_classes) == 1:
                yolo_class_names[yolo_id] = kitti_classes[0]
            else:
                yolo_class_names[yolo_id] = f"{'_'.join(kitti_classes)}"
        
        dataset_config = {
            'path': str(output_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(yolo_class_names),
            'names': [yolo_class_names[i] for i in sorted(yolo_class_names.keys())]
        }
        
        with open(output_root / "dataset.yaml", 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        # Print statistics
        self.print_statistics()
        
        self.logger.info(f"✅ Conversion completed successfully!")
        self.logger.info(f"Dataset saved to: {output_root}")
        self.logger.info(f"Use dataset.yaml for YOLO training")
    
    def print_statistics(self) -> None:
        """Print conversion statistics."""
        self.logger.info("\n" + "="*50)
        self.logger.info("CONVERSION STATISTICS")
        self.logger.info("="*50)
        self.logger.info(f"Total images processed: {self.stats['total_images']}")
        self.logger.info(f"Total labels found: {self.stats['total_labels']}")
        self.logger.info(f"Labels converted: {self.stats['converted_labels']}")
        self.logger.info(f"Labels dropped: {self.stats['dropped_labels']}")
        self.logger.info(f"Invalid boxes: {self.stats['invalid_boxes']}")
        
        if self.stats['class_counts']:
            self.logger.info("\nClass distribution:")
            for class_id in sorted(self.stats['class_counts'].keys()):
                count = self.stats['class_counts'][class_id]
                img_count = len(self.stats['images_per_class'][class_id])
                self.logger.info(f"  Class {class_id}: {count} objects in {img_count} images")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert KITTI 2D object detection dataset to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with default 3-class mapping
  python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset
  
  # Custom split ratio with stratified sampling and 10-fold CV
  python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset \\
                       --train_ratio 0.7 --kfold 10 --stratify --seed 123
  
  # Verbose output
  python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset --verbose

Note: Edit the REMAP dictionary in the script to change class mappings.
        """
    )
    
    parser.add_argument("--kitti_path", required=True, type=str,
                       help="Path to KITTI dataset root directory")
    parser.add_argument("--out_path", required=True, type=str,
                       help="Output path for YOLO dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--kfold", type=int, default=5,
                       help="Number of k-fold splits to generate (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--stratify", action="store_true",
                       help="Use stratified sampling for balanced splits")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Create converter and run
    converter = KittiToYoloConverter(remap=REMAP)
    
    try:
        converter.convert_kitti_to_yolo(
            kitti_path=args.kitti_path,
            output_path=args.out_path,
            train_ratio=args.train_ratio,
            kfold=args.kfold,
            seed=args.seed,
            stratify=args.stratify
        )
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
