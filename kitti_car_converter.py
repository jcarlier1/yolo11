#!/usr/bin/env python3
"""
KITTI to YOLO Dataset Converter - Car Detection Only

This script converts KITTI dataset format to YOLO format for object detection,
specifically filtering for only Car bounding boxes. All other classes are ignored.

KITTI format:
- Labels: class truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y
- Bounding boxes: x1, y1, x2, y2 (top-left, bottom-right corners)

YOLO format:
- Labels: class_id center_x center_y width height (normalized 0-1)
- class_id: 0 (Car only)
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KittiCarToYoloConverter:
    """Converts KITTI dataset format to YOLO format for Car detection only."""
    
    def __init__(self, kitti_root: str, yolo_root: str, train_split: float = 0.8):
        """
        Initialize the converter.
        
        Args:
            kitti_root: Path to KITTI dataset root directory
            yolo_root: Path where YOLO dataset will be created
            train_split: Percentage of training data to use for training (rest goes to validation)
        """
        self.kitti_root = Path(kitti_root)
        self.yolo_root = Path(yolo_root)
        self.train_split = train_split
        
        # Only Car class mapping - all cars get class ID 0
        self.class_mapping = {
            'Car': 0,
        }
        
        # Create YOLO directory structure
        self._create_yolo_structure()
    
    def _create_yolo_structure(self):
        """Create YOLO dataset directory structure."""
        # Create main directories
        for split in ['train', 'val', 'test']:
            (self.yolo_root / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_root / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created YOLO directory structure at {self.yolo_root}")
    
    def _create_splits(self) -> Dict[str, List[str]]:
        """
        Create train/val/test splits from available data.
        
        Returns:
            Dictionary mapping split names to list of image IDs
        """
        splits = {'train': [], 'val': [], 'test': []}
        
        # Get training data IDs from training/image_2
        training_images = self.kitti_root / 'training' / 'image_2'
        if training_images.exists():
            training_ids = [f.stem for f in training_images.glob('*.png')]
            training_ids.sort()  # Sort for consistent splits
            
            # Split training data based on train_split parameter
            split_idx = int(self.train_split * len(training_ids))
            splits['train'] = training_ids[:split_idx]
            splits['val'] = training_ids[split_idx:]
            
            logger.info(f"Training data split: {len(splits['train'])} train, {len(splits['val'])} validation")
        else:
            logger.warning(f"Training images directory not found at {training_images}")
        
        # Get testing data IDs from testing/image_2
        testing_images = self.kitti_root / 'testing' / 'image_2'
        if testing_images.exists():
            splits['test'] = [f.stem for f in testing_images.glob('*.png')]
            logger.info(f"Testing data: {len(splits['test'])} samples")
        else:
            logger.warning(f"Testing images directory not found at {testing_images}")
        
        return splits
    
    def _parse_kitti_annotation(self, annotation_path: Path, img_width: int, img_height: int) -> List[str]:
        """
        Parse KITTI annotation file and convert to YOLO format.
        Only processes Car bounding boxes, ignoring all other classes.
        
        Args:
            annotation_path: Path to KITTI annotation file
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of YOLO format annotation strings for Car objects only
        """
        yolo_annotations = []
        car_count = 0
        
        if not annotation_path.exists():
            return yolo_annotations
        
        with open(annotation_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                class_name = parts[0]
                
                # Only process Car class - skip all others
                if class_name != 'Car':
                    continue
                
                # Parse bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(float, parts[4:8])
                
                # Validate bounding box
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bounding box in {annotation_path}: {x1}, {y1}, {x2}, {y2}")
                    continue
                
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                center_x = (x1 + x2) / 2.0 / img_width
                center_y = (y1 + y2) / 2.0 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Ensure coordinates are within bounds
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # Skip very small bounding boxes (likely annotation errors)
                #if width < 0.01 or height < 0.01:
                #    logger.warning(f"Skipping very small bounding box: {width:.4f} x {height:.4f}")
                #    continue

                # Car class ID is always 0 in our single-class system
                class_id = 0
                yolo_annotation = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                yolo_annotations.append(yolo_annotation)
                car_count += 1
        
        if car_count > 0:
            logger.debug(f"Found {car_count} car(s) in {annotation_path}")
        
        return yolo_annotations
    
    def _get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """
        Get image dimensions without loading the full image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (width, height)
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except ImportError:
            logger.warning("PIL not available, using default dimensions (1242, 375)")
            return 1242, 375  # KITTI default dimensions
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            return 1242, 375
    
    def convert_split(self, split_name: str, image_ids: List[str], is_test: bool = False):
        """
        Convert a specific split of the dataset.
        
        Args:
            split_name: Name of the split (train/val/test)
            image_ids: List of image IDs for this split
            is_test: Whether this is test set (no labels available)
        """
        logger.info(f"Converting {split_name} split with {len(image_ids)} samples")
        
        images_src = self.kitti_root / ('testing' if is_test else 'training') / 'image_2'
        labels_src = self.kitti_root / 'training' / 'label' if not is_test else None
        
        images_dst = self.yolo_root / split_name / 'images'
        labels_dst = self.yolo_root / split_name / 'labels'
        
        converted_count = 0
        images_with_cars = 0
        total_cars = 0
        
        for image_id in image_ids:
            # Copy image
            image_src_path = images_src / f"{image_id}.png"
            if not image_src_path.exists():
                logger.warning(f"Image not found: {image_src_path}")
                continue
            
            image_dst_path = images_dst / f"{image_id}.jpg"  # Convert to jpg for YOLO
            
            try:
                # Copy and optionally convert image format
                if image_src_path.suffix.lower() == '.png':
                    # Convert PNG to JPG for smaller file size
                    from PIL import Image
                    with Image.open(image_src_path) as img:
                        rgb_img = img.convert('RGB')
                        rgb_img.save(image_dst_path, 'JPEG', quality=95)
                else:
                    shutil.copy2(image_src_path, image_dst_path)
            except ImportError:
                # Fallback: just copy the file as-is
                shutil.copy2(image_src_path, image_dst_path.with_suffix('.png'))
                image_dst_path = image_dst_path.with_suffix('.png')
            
            # Convert annotations (only for training data)
            if not is_test and labels_src:
                annotation_src_path = labels_src / f"{image_id}.txt"
                annotation_dst_path = labels_dst / f"{image_id}.txt"
                
                # Get image dimensions
                img_width, img_height = self._get_image_dimensions(image_dst_path)
                
                # Convert annotations (Car only)
                yolo_annotations = self._parse_kitti_annotation(
                    annotation_src_path, img_width, img_height
                )
                
                # Write YOLO annotations
                with open(annotation_dst_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                # Track statistics
                if yolo_annotations:
                    images_with_cars += 1
                    total_cars += len(yolo_annotations)
            
            converted_count += 1
            
            if converted_count % 100 == 0:
                logger.info(f"Converted {converted_count}/{len(image_ids)} samples")
        
        logger.info(f"Completed {split_name} split: {converted_count} samples converted")
        if not is_test:
            logger.info(f"Car statistics for {split_name}: {images_with_cars} images with cars, {total_cars} total car instances")
    
    def create_yaml_config(self):
        """Create YOLO dataset configuration file for Car detection."""
        yaml_content = f"""# YOLO dataset configuration - Car Detection Only
# Converted from KITTI dataset

path: {self.yolo_root.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path') 
test: test/images    # test images (relative to 'path')

# Classes (Car only)
names:
  0: Car

# Number of classes
nc: 1
"""
        
        yaml_path = self.yolo_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created dataset configuration: {yaml_path}")
    
    def convert(self):
        """Convert the entire KITTI dataset to YOLO format (Car detection only)."""
        logger.info("Starting KITTI to YOLO conversion (Car detection only)")
        logger.info(f"Train/validation split ratio: {self.train_split:.1%} train, {1-self.train_split:.1%} validation")
        
        # Create splits from available data
        splits = self._create_splits()
        
        # Convert each split
        for split_name, image_ids in splits.items():
            if not image_ids:
                continue
                
            is_test = split_name == 'test'
            self.convert_split(split_name, image_ids, is_test)
        
        # Create YAML configuration
        self.create_yaml_config()
        
        logger.info("KITTI to YOLO conversion (Car detection only) completed successfully!")


def main():
    """Main function to run the converter."""
    parser = argparse.ArgumentParser(description='Convert KITTI dataset to YOLO format (Car detection only)')
    parser.add_argument('--kitti_root', type=str, default='/home/carlier1/data/kitti',
                       help='Path to KITTI dataset root directory')
    parser.add_argument('--yolo_root', type=str, default='/home/carlier1/data/yolo_kitti_cars',
                       help='Path where YOLO dataset will be created')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Percentage of training data to use for training (default: 0.8)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not os.path.exists(args.kitti_root):
        logger.error(f"KITTI dataset directory not found: {args.kitti_root}")
        return
    
    # Validate train_split parameter
    if not 0.1 <= args.train_split <= 0.9:
        logger.error(f"train_split must be between 0.1 and 0.9, got {args.train_split}")
        return
    
    # Create converter and run conversion
    converter = KittiCarToYoloConverter(args.kitti_root, args.yolo_root, args.train_split)
    converter.convert()


if __name__ == "__main__":
    main()
