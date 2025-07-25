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

        # Testing settings (used by test.py)
        'experiment_name': "kitti_test",  # Name for experiment output directories
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
        # Main dataset paths
        yolo_root = Path(self.config['dataset_root'])
        self.config.update({
            'yolo_root': yolo_root,
            'data_yaml': yolo_root / 'dataset.yaml',
            'train_dir': yolo_root / 'train',
            'val_dir': yolo_root / 'val',
            'test_dir': yolo_root / 'test',
            'train_images': yolo_root / 'train' / 'images',
            'train_labels': yolo_root / 'train' / 'labels',
            'val_images': yolo_root / 'val' / 'images',
            'val_labels': yolo_root / 'val' / 'labels',
            'test_images': yolo_root / 'test' / 'images',
            'test_labels': yolo_root / 'test' / 'labels',
        })
        
        # Car dataset paths
        car_yolo_root = Path(self.config['car_dataset_root'])
        self.config.update({
            'car_yolo_root': car_yolo_root,
            'car_data_yaml': car_yolo_root / 'dataset.yaml',
            'car_train_dir': car_yolo_root / 'train',
            'car_val_dir': car_yolo_root / 'val',
            'car_test_dir': car_yolo_root / 'test',
            'car_train_images': car_yolo_root / 'train' / 'images',
            'car_train_labels': car_yolo_root / 'train' / 'labels',
            'car_val_images': car_yolo_root / 'val' / 'images',
            'car_val_labels': car_yolo_root / 'val' / 'labels',
            'car_test_images': car_yolo_root / 'test' / 'images',
            'car_test_labels': car_yolo_root / 'test' / 'labels',
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
    else:
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
