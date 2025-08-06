#!/usr/bin/env python
"""
Configuration utilities for YOLO11 training and testing
Provides a centralized way to manage paths and settings with local overrides
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager with local override support."""
    
    # Default configuration values
    DEFAULTS = {
        # Dataset paths
        'dataset_root': './data/yolo_kitti',
        'all_dataset_root': './data/yolo_kitti_all',
        'car_dataset_root': './data/yolo_kitti_cars',
        
        # Model paths
        'models_dir': 'models',
        'weights_dir': 'weights',
        
        # Output directories
        'results_dir': 'results',
        'runs_dir': 'runs',
        'cv_results_dir': 'runs/cross_validation',
        
        # Default model settings
        'default_model': 'yolo11s.pt',
        'default_epochs': 500,
        'default_imgsz': 640,
        'default_batch_size': -1,
        
        # Training settings
        'project_name': 'runs/detect',
        'experiment_name': 'kitti_yolo11s',
        
        # Training hyperparameters
        'training_patience': 50,
        'training_device': 0,
        'enable_validation': True,
        'generate_plots': True,
        'verbose_output': True,
        'allow_overwrite': True,
        
        # Optimizer settings
        'learning_rate': 0.001,
        'optimizer': 'AdamW',
        'cosine_lr_scheduler': True,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights
        'box_loss_weight': 7.5,
        'cls_loss_weight': 0.5,
        'dfl_loss_weight': 1.5,
        
        # Performance optimizations
        'mixed_precision': True,
        'cache_mode': 'ram',
        'num_workers': 8,
        'close_mosaic_epochs': 20,
        'rectangular_training': True,
        'single_class': False,
        'deterministic': False,
        'random_seed': 42,

        # Testing settings (used by test.py)
        'default_test_weights': "models/11n.pt",  # Default weights file for testing
        'default_conf_threshold': 0.25,  # Default confidence threshold
        'default_iou_threshold': 0.7,     # Default IoU threshold for NMS
        'default_test_imgsz': 640,        # Default image size for testing
        'save_images_default': 'false',    # Default setting for saving prediction images
        'save_txt_default': 'true',       # Default setting for saving text predictions
        'save_conf_default': 'true'      # Default setting for saving confidence scores
    }
    
    def __init__(self, local_config_path: str = 'local_config.yaml'):
        """Initialize configuration with optional local overrides."""
        self.config = self.DEFAULTS.copy()
        self.local_config_path = local_config_path
        self._load_local_config()
        self._setup_derived_paths()
    
    def _load_local_config(self):
        """Load local configuration if it exists."""
        if Path(self.local_config_path).exists():
            try:
                with open(self.local_config_path, 'r') as f:
                    local_config = yaml.safe_load(f) or {}
                self.config.update(local_config)
                print(f"✓ Loaded local configuration from {self.local_config_path}")
            except Exception as e:
                print(f"Warning: Could not load local config: {e}")
        else:
            print(f"No local config found at {self.local_config_path}, using defaults")
    
    def _setup_derived_paths(self):
        """Setup derived paths based on configuration."""
        # Main dataset paths (updated to match kitti2yolo.py output structure)
        yolo_root = Path(self.config['dataset_root'])
        self.config.update({
            'yolo_root': yolo_root,
            'data_yaml': yolo_root / 'dataset.yaml',
            'train_dir': yolo_root,  # No longer a subfolder, root is used
            'val_dir': yolo_root,
            'test_dir': yolo_root,
            'train_images': yolo_root / 'images' / 'train',
            'train_labels': yolo_root / 'labels' / 'train',
            'val_images': yolo_root / 'images' / 'val',
            'val_labels': yolo_root / 'labels' / 'val',
            'test_images': yolo_root / 'images' / 'test',
            'test_labels': yolo_root / 'labels' / 'test',
        })
        
        # Car dataset paths (updated to match kitti2yolo.py output structure)
        car_yolo_root = Path(self.config['car_dataset_root'])
        self.config.update({
            'car_yolo_root': car_yolo_root,
            'car_data_yaml': car_yolo_root / 'dataset.yaml',
            'car_train_dir': car_yolo_root,
            'car_val_dir': car_yolo_root,
            'car_test_dir': car_yolo_root,
            'car_train_images': car_yolo_root / 'images' / 'train',
            'car_train_labels': car_yolo_root / 'labels' / 'train',
            'car_val_images': car_yolo_root / 'images' / 'val',
            'car_val_labels': car_yolo_root / 'labels' / 'val',
            'car_test_images': car_yolo_root / 'images' / 'test',
            'car_test_labels': car_yolo_root / 'labels' / 'test',
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def get_path(self, key: str) -> Path:
        """Get configuration value as Path object."""
        value = self.config.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found")
        return Path(value)
    
    def print_config(self):
        """Print current configuration for debugging."""
        print("\n=== Current Configuration ===")
        for key, value in sorted(self.config.items()):
            if isinstance(value, Path):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        print("=" * 30)

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def verify_paths(required_paths: list, check_exists: bool = True) -> bool:
    """
    Verify that required paths exist.
    
    Args:
        required_paths: List of path keys to check
        check_exists: Whether to check if paths actually exist
    
    Returns:
        True if all paths are valid
    """
    cfg = get_config()
    missing_paths = []
    
    for path_key in required_paths:
        try:
            path = cfg.get_path(path_key)
            if check_exists and not path.exists():
                missing_paths.append(f"{path_key}: {path}")
            else:
                print(f"✓ {path_key}: {path}")
        except KeyError:
            missing_paths.append(f"{path_key}: Configuration key not found")
    
    if missing_paths:
        print("\n❌ Missing or invalid paths:")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    
    return True

def get_dataset_config(dataset_type: str = 'default') -> Dict[str, Path]:
    """
    Get dataset configuration for a specific dataset type.
    
    Args:
        dataset_type: 'default' for main dataset, 'car' for car dataset
        
    Returns:
        Dictionary with dataset paths
    """
    cfg = get_config()
    
    if dataset_type == 'car':
        return {
            'root': cfg.get_path('car_yolo_root'),
            'data_yaml': cfg.get_path('car_data_yaml'),
            'train_dir': cfg.get_path('car_train_dir'),
            'val_dir': cfg.get_path('car_val_dir'),
            'test_dir': cfg.get_path('car_test_dir'),
            'train_images': cfg.get_path('car_train_images'),
            'train_labels': cfg.get_path('car_train_labels'),
            'val_images': cfg.get_path('car_val_images'),
            'val_labels': cfg.get_path('car_val_labels'),
            'test_images': cfg.get_path('car_test_images'),
            'test_labels': cfg.get_path('car_test_labels'),
        }
    elif dataset_type == 'default':
        return {
            'root': cfg.get_path('yolo_root'),
            'data_yaml': cfg.get_path('data_yaml'),
            'train_dir': cfg.get_path('train_dir'),
            'val_dir': cfg.get_path('val_dir'),
            'test_dir': cfg.get_path('test_dir'),
            'train_images': cfg.get_path('train_images'),
            'train_labels': cfg.get_path('train_labels'),
            'val_images': cfg.get_path('val_images'),
            'val_labels': cfg.get_path('val_labels'),
            'test_images': cfg.get_path('test_images'),
            'test_labels': cfg.get_path('test_labels'),
        }
