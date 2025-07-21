#!/usr/bin/env python
"""
K-Fold Cross-Validation for YOLO Object Detection
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from collections import defaultdict, Counter
import json

class YOLOCrossValidator:
    """
    K-Fold Cross-Validation for YOLO object detection models.
    Handles data splitting, training, and evaluation across multiple folds.
    """
    
    def __init__(self, 
                 data_root: str = "./data/yolo_kitti",
                 output_dir: str = "runs/cross_validation",
                 k_folds: int = 5,
                 seed: int = 42):
        """
        Initialize the cross-validator.
        
        Args:
            data_root: Root directory containing the YOLO dataset
            output_dir: Directory to save cross-validation results
            k_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.k_folds = k_folds
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Dataset paths
        self.train_images = self.data_root / "train" / "images"
        self.train_labels = self.data_root / "train" / "labels"
        self.val_images = self.data_root / "val" / "images"
        self.val_labels = self.data_root / "val" / "labels"
        self.test_images = self.data_root / "test" / "images"
        self.test_labels = self.data_root / "test" / "labels"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store original dataset.yaml
        self.original_dataset_yaml = self.data_root / "dataset.yaml"
        
        # Results storage
        self.fold_results = []
        
    def get_image_class_distribution(self, image_files: List[Path]) -> Dict[str, List[int]]:
        """
        Analyze class distribution across images for stratified splitting.
        
        Args:
            image_files: List of image file paths
            
        Returns:
            Dictionary mapping image stems to class counts
        """
        class_distribution = {}
        
        for img_file in image_files:
            # Find corresponding label file
            label_file = self.train_labels / f"{img_file.stem}.txt"
            
            if label_file.exists():
                classes = []
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            classes.append(int(parts[0]))  # First value is class ID
                
                class_distribution[img_file.stem] = classes
            else:
                # No annotations (background image)
                class_distribution[img_file.stem] = []
        
        return class_distribution
    
    def create_stratified_folds(self) -> List[Tuple[List[str], List[str]]]:
        """
        Create stratified k-fold splits based on class distribution.
        
        Returns:
            List of (train_stems, val_stems) tuples for each fold
        """
        print("Creating stratified k-fold splits...")
        
        # Get all training images (combine train and val from original split)
        train_images = list(self.train_images.glob("*"))
        val_images = list(self.val_images.glob("*"))
        all_images = train_images + val_images
        
        print(f"Total images for cross-validation: {len(all_images)}")
        
        # Get class distribution
        class_dist = self.get_image_class_distribution(all_images)
        
        # Create a simple stratification based on dominant class per image
        image_stems = []
        dominant_classes = []
        
        for img_file in all_images:
            stem = img_file.stem
            classes = class_dist.get(stem, [])
            
            # Use most frequent class as dominant class for stratification
            if classes:
                dominant_class = Counter(classes).most_common(1)[0][0]
            else:
                dominant_class = -1  # Background/no objects
            
            image_stems.append(stem)
            dominant_classes.append(dominant_class)
        
        # Create stratified folds
        if len(set(dominant_classes)) > 1:
            # Use StratifiedKFold if we have multiple classes
            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            fold_splits = list(skf.split(image_stems, dominant_classes))
        else:
            # Use regular KFold if only one class
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            fold_splits = list(kf.split(image_stems))
        
        # Convert indices to actual stems
        folds = []
        for train_idx, val_idx in fold_splits:
            train_stems = [image_stems[i] for i in train_idx]
            val_stems = [image_stems[i] for i in val_idx]
            folds.append((train_stems, val_stems))
        
        print(f"Created {len(folds)} folds")
        for i, (train_stems, val_stems) in enumerate(folds):
            print(f"  Fold {i+1}: {len(train_stems)} train, {len(val_stems)} val")
        
        return folds
    
    def setup_fold_data(self, fold_idx: int, train_stems: List[str], val_stems: List[str]):
        """
        Set up data structure for a specific fold.
        
        Args:
            fold_idx: Current fold index
            train_stems: List of training image stems
            val_stems: List of validation image stems
        """
        fold_dir = self.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Create fold-specific directories
        fold_train_images = fold_dir / "train" / "images"
        fold_train_labels = fold_dir / "train" / "labels"
        fold_val_images = fold_dir / "val" / "images"
        fold_val_labels = fold_dir / "val" / "labels"
        
        for dir_path in [fold_train_images, fold_train_labels, fold_val_images, fold_val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy or symlink files for training set
        for stem in train_stems:
            # Find image file (could be in train or val)
            src_img = None
            for img_dir in [self.train_images, self.val_images]:
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    candidate = img_dir / f"{stem}{ext}"
                    if candidate.exists():
                        src_img = candidate
                        break
                if src_img:
                    break
            
            if src_img:
                # Copy image
                dst_img = fold_train_images / src_img.name
                shutil.copy2(src_img, dst_img)
                
                # Copy label if it exists
                for label_dir in [self.train_labels, self.val_labels]:
                    src_label = label_dir / f"{stem}.txt"
                    if src_label.exists():
                        dst_label = fold_train_labels / f"{stem}.txt"
                        shutil.copy2(src_label, dst_label)
                        break
        
        # Copy or symlink files for validation set
        for stem in val_stems:
            # Find image file (could be in train or val)
            src_img = None
            for img_dir in [self.train_images, self.val_images]:
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    candidate = img_dir / f"{stem}{ext}"
                    if candidate.exists():
                        src_img = candidate
                        break
                if src_img:
                    break
            
            if src_img:
                # Copy image
                dst_img = fold_val_images / src_img.name
                shutil.copy2(src_img, dst_img)
                
                # Copy label if it exists
                for label_dir in [self.train_labels, self.val_labels]:
                    src_label = label_dir / f"{stem}.txt"
                    if src_label.exists():
                        dst_label = fold_val_labels / f"{stem}.txt"
                        shutil.copy2(src_label, dst_label)
                        break
        
        # Create fold-specific dataset.yaml
        self.create_fold_dataset_yaml(fold_dir)
        
        print(f"Fold {fold_idx} data setup complete:")
        print(f"  Train: {len(list(fold_train_images.glob('*')))} images")
        print(f"  Val: {len(list(fold_val_images.glob('*')))} images")
    
    def create_fold_dataset_yaml(self, fold_dir: Path):
        """
        Create dataset.yaml for a specific fold.
        
        Args:
            fold_dir: Directory for the current fold
        """
        # Load original dataset.yaml or create default
        if self.original_dataset_yaml.exists():
            with open(self.original_dataset_yaml, 'r') as f:
                dataset_config = yaml.safe_load(f)
        else:
            # Create default config for KITTI
            dataset_config = {
                'nc': 8,  # Common KITTI classes
                'names': {
                    0: 'Car',
                    1: 'Van',
                    2: 'Truck',
                    3: 'Pedestrian',
                    4: 'Person_sitting',
                    5: 'Cyclist',
                    6: 'Tram',
                    7: 'Misc'
                }
            }
        
        # Update paths for this fold
        dataset_config['path'] = str(fold_dir.absolute())
        dataset_config['train'] = 'train/images'
        dataset_config['val'] = 'val/images'
        
        # Save fold-specific dataset.yaml
        fold_yaml = fold_dir / "dataset.yaml"
        with open(fold_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
    
    def train_fold(self, fold_idx: int, **train_kwargs) -> dict:
        """
        Train YOLO model for a specific fold.
        
        Args:
            fold_idx: Current fold index
            **train_kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        from ultralytics import YOLO
        
        print(f"\n=== Training Fold {fold_idx} ===")
        
        fold_dir = self.output_dir / f"fold_{fold_idx}"
        dataset_yaml = fold_dir / "dataset.yaml"
        
        # Load model
        model = YOLO("yolo11x.pt")
        
        # Default training parameters
        default_params = {
            'data': str(dataset_yaml),
            'epochs': 100,  # Reduced for cross-validation
            'patience': 20,
            'imgsz': 640,
            'batch': 32,  # Reduced for multiple folds
            'device': 0,
            'name': f'fold_{fold_idx}',
            'save_period': 10,
            'val': True,
            'plots': True,
            'verbose': False,  # Reduce verbosity for multiple folds
            'exist_ok': True,
            'lr0': 0.001,
            'optimizer': 'AdamW',
            'cos_lr': True,
            'warmup_epochs': 3,
            'amp': True,
            'cache': 'ram',
            'workers': 4,  # Reduced for multiple folds
            'seed': self.seed,
            'project': str(self.output_dir),
        }
        
        # Override with user parameters
        default_params.update(train_kwargs)
        
        # Train model
        results = model.train(**default_params)
        
        # Extract key metrics
        metrics = {
            'fold': fold_idx,
            'best_fitness': float(results.best_fitness) if hasattr(results, 'best_fitness') else None,
            'best_epoch': int(results.best_epoch) if hasattr(results, 'best_epoch') else None,
        }
        
        # Try to extract validation metrics
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            metrics.update({
                'mAP50': float(results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results_dict.get('metrics/recall(B)', 0)),
                'box_loss': float(results_dict.get('train/box_loss', 0)),
                'cls_loss': float(results_dict.get('train/cls_loss', 0)),
                'dfl_loss': float(results_dict.get('train/dfl_loss', 0)),
            })
        
        print(f"Fold {fold_idx} completed:")
        print(f"  Best mAP@0.5: {metrics.get('mAP50', 'N/A'):.4f}")
        print(f"  Best mAP@0.5:0.95: {metrics.get('mAP50_95', 'N/A'):.4f}")
        
        return metrics
    
    def run_cross_validation(self, **train_kwargs) -> Dict:
        """
        Run complete k-fold cross-validation.
        
        Args:
            **train_kwargs: Training parameters to pass to each fold
            
        Returns:
            Dictionary containing cross-validation results
        """
        print("=== Starting K-Fold Cross-Validation ===")
        print(f"Number of folds: {self.k_folds}")
        print(f"Output directory: {self.output_dir}")
        
        # Create stratified folds
        folds = self.create_stratified_folds()
        
        # Train each fold
        fold_results = []
        
        for fold_idx, (train_stems, val_stems) in enumerate(folds):
            print(f"\n=== Processing Fold {fold_idx + 1}/{self.k_folds} ===")
            
            # Setup data for this fold
            self.setup_fold_data(fold_idx, train_stems, val_stems)
            
            # Train model for this fold
            fold_metrics = self.train_fold(fold_idx, **train_kwargs)
            fold_results.append(fold_metrics)
        
        # Aggregate results
        cv_results = self.aggregate_results(fold_results)
        
        # Save results
        self.save_results(cv_results)
        
        return cv_results
    
    def aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate results across all folds.
        
        Args:
            fold_results: List of fold result dictionaries
            
        Returns:
            Aggregated cross-validation results
        """
        print("\n=== Aggregating Cross-Validation Results ===")
        
        # Metrics to aggregate
        metrics_to_aggregate = ['mAP50', 'mAP50_95', 'precision', 'recall', 
                               'box_loss', 'cls_loss', 'dfl_loss']
        
        aggregated = {
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'mean_metrics': {},
            'std_metrics': {},
            'best_fold': None,
            'worst_fold': None
        }
        
        # Calculate mean and std for each metric
        for metric in metrics_to_aggregate:
            values = [result.get(metric, 0) for result in fold_results if result.get(metric) is not None]
            if values:
                aggregated['mean_metrics'][metric] = np.mean(values)
                aggregated['std_metrics'][metric] = np.std(values)
        
        # Find best and worst performing folds (based on mAP50)
        if aggregated['mean_metrics'].get('mAP50'):
            mAP50_values = [result.get('mAP50', 0) for result in fold_results]
            best_idx = np.argmax(mAP50_values)
            worst_idx = np.argmin(mAP50_values)
            aggregated['best_fold'] = {'fold': best_idx, 'mAP50': mAP50_values[best_idx]}
            aggregated['worst_fold'] = {'fold': worst_idx, 'mAP50': mAP50_values[worst_idx]}
        
        # Print summary
        print("\nCross-Validation Summary:")
        print(f"Number of folds: {aggregated['n_folds']}")
        for metric, mean_val in aggregated['mean_metrics'].items():
            std_val = aggregated['std_metrics'].get(metric, 0)
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        if aggregated['best_fold']:
            print(f"  Best fold: {aggregated['best_fold']['fold']} (mAP50: {aggregated['best_fold']['mAP50']:.4f})")
            print(f"  Worst fold: {aggregated['worst_fold']['fold']} (mAP50: {aggregated['worst_fold']['mAP50']:.4f})")
        
        return aggregated
    
    def save_results(self, cv_results: Dict):
        """
        Save cross-validation results to files.
        
        Args:
            cv_results: Aggregated cross-validation results
        """
        results_file = self.output_dir / "cross_validation_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert results
        json_results = json.loads(json.dumps(cv_results, default=convert_numpy_types))
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save to CSV for easy analysis
        csv_file = self.output_dir / "fold_results.csv"
        df = pd.DataFrame(cv_results['fold_results'])
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {results_file}")
        print(f"  CSV: {csv_file}")


def main():
    """
    Example usage of the cross-validation system.
    """
    # Initialize cross-validator
    cv = YOLOCrossValidator(
        data_root="./data/yolo_kitti",
        output_dir="runs/cross_validation",
        k_folds=5,
        seed=42
    )
    
    # Run cross-validation with custom parameters
    cv_results = cv.run_cross_validation(
        epochs=50,          # Reduced for faster CV
        patience=10,        # Reduced patience
        batch=32,           # Adjust based on GPU memory
        imgsz=640,
        lr0=0.001,
        optimizer='AdamW',
        cos_lr=True,
        amp=True,
        cache='ram',
        workers=4,
    )
    
    print("\n=== Cross-Validation Complete ===")
    print(f"Mean mAP@0.5: {cv_results['mean_metrics'].get('mAP50', 0):.4f} ± {cv_results['std_metrics'].get('mAP50', 0):.4f}")
    print(f"Mean mAP@0.5:0.95: {cv_results['mean_metrics'].get('mAP50_95', 0):.4f} ± {cv_results['std_metrics'].get('mAP50_95', 0):.4f}")


if __name__ == "__main__":
    main()
