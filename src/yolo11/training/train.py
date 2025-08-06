#!/usr/bin/env python
"""
Fine-tune YOLO on KITTI dataset converted to YOLO format

This script uses configuration from local_config.yaml to set all training parameters.
To customize training, modify the values in local_config.yaml or create one from the template
in docs/local_config.yaml.template

All training hyperparameters are configurable including:
- Model type and training epochs
- Batch size and image size
- Learning rate and optimizer settings
- Loss weights and augmentation parameters
- Performance optimizations
"""

from pathlib import Path
import shutil
import random
from PIL import Image
import yaml
from src.yolo11.utils.config_utils import get_config, get_dataset_config, verify_paths

# Get configuration
config = get_config()
dataset_paths = get_dataset_config('default')

def verify_dataset_structure():
    """Verify that the YOLO dataset structure exists and is valid."""
    print("Verifying dataset structure...")
    
    # Check main directories using config
    required_paths = ['yolo_root', 'train_dir', 'val_dir', 'test_dir', 
                     'train_images', 'train_labels', 'val_images', 'val_labels', 
                     'test_images', 'test_labels']
    
    if not verify_paths(required_paths):
        raise FileNotFoundError("Required directories not found")
    
    # Check dataset.yaml
    data_yaml = dataset_paths['data_yaml']
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    print(f"✓ {data_yaml}")
    
    # Count files in each split
    train_images = len(list(dataset_paths['train_images'].glob("*")))
    train_labels = len(list(dataset_paths['train_labels'].glob("*.txt")))
    val_images = len(list(dataset_paths['val_images'].glob("*")))
    val_labels = len(list(dataset_paths['val_labels'].glob("*.txt")))
    test_images = len(list(dataset_paths['test_images'].glob("*")))
    test_labels = len(list(dataset_paths['test_labels'].glob("*.txt")))
    
    print(f"\nDataset Summary:")
    print(f"  Train: {train_images} images, {train_labels} labels")
    print(f"  Val:   {val_images} images, {val_labels} labels")
    print(f"  Test:  {test_images} images, {test_labels} labels")
    
    # Verify dataset.yaml content
    with open(data_yaml, 'r') as f:
        config_data = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Path: {config_data.get('path', 'Not specified')}")
    print(f"  Classes: {len(config_data.get('names', {}))}")
    print(f"  Class names: {list(config_data.get('names', []))}")
    
    return True

def train_model():
    """Train YOLO model with configuration."""
    from ultralytics import YOLO

    print(f"Starting YOLO training with dataset: {dataset_paths['data_yaml']}")
    
    # Load pretrained model
    model = YOLO(config.get('default_model', 'yolo11n.pt'))
    
    # Get training configuration from config with fallbacks
    training_config = {
        'data': str(dataset_paths['data_yaml']),
        'epochs': config.get('default_epochs', 500),
        'patience': config.get('training_patience', 50),
        'imgsz': config.get('default_imgsz', 640),
        'batch': config.get('default_batch_size', -1),
        'device': config.get('training_device', 0),
        'name': config.get('experiment_name', 'kitti_default'),
        'val': config.get('enable_validation', True),
        'plots': config.get('generate_plots', True),
        'verbose': config.get('verbose_output', True),
        'exist_ok': config.get('allow_overwrite', True),
        # Optimizer settings
        'lr0': config.get('learning_rate', 0.001),
        'optimizer': config.get('optimizer', 'AdamW'),
        'cos_lr': config.get('cosine_lr_scheduler', True),
        'warmup_epochs': config.get('warmup_epochs', 3),
        'warmup_momentum': config.get('warmup_momentum', 0.8),
        'warmup_bias_lr': config.get('warmup_bias_lr', 0.1),
        # Loss weights
        'box': config.get('box_loss_weight', 7.5),
        'cls': config.get('cls_loss_weight', 0.5),
        'dfl': config.get('dfl_loss_weight', 1.5),
        # Performance optimizations
        'amp': config.get('mixed_precision', True),
        'cache': config.get('cache_mode', 'ram'),
        'workers': config.get('num_workers', 8),
        'close_mosaic': config.get('close_mosaic_epochs', 20),
        'rect': config.get('rectangular_training', False),
        'single_cls': config.get('single_class', False),
        'deterministic': config.get('deterministic', False),
        'seed': config.get('random_seed', 42),
    }
    
    # Print training configuration
    print(f"\nTraining Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    print()
    
    results = model.train(**training_config)
    
    print(f"Training completed successfully!")
    
    # Get experiment name for output paths
    experiment_name = config.get('experiment_name', 'kitti_default')
    project_name = config.get('project_name', 'runs/detect')
    
    print(f"Best weights saved at: {project_name}/{experiment_name}/weights/best.pt")
    print(f"Last weights saved at: {project_name}/{experiment_name}/weights/last.pt")
    
    # Print training summary
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\nTraining Summary:")
        for metric_key, display_name in [
            ('metrics/mAP50(B)', 'Best mAP@0.5'),
            ('metrics/mAP50-95(B)', 'Best mAP@0.5:0.95'),
            ('metrics/precision(B)', 'Precision'),
            ('metrics/recall(B)', 'Recall')
        ]:
            value = metrics.get(metric_key, 'N/A')
            print(f"  - {display_name}: {value}")
    
    return results

# ------------------------------------------------------------------------------------
# 5. Run everything
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== YOLO Training on Converted KITTI Dataset ===")
    print()
    
    try:
        # Step 1: Verify dataset structure
        print("Step 1: Verifying dataset structure...")
        verify_dataset_structure()
        print("✓ Dataset structure verified!")
        print()
        
        # Step 2: Train model
        print("Step 2: Starting YOLO training...")
        results = train_model()
        print("✓ Training completed!")
        print()
        
        # Step 3: Final summary
        print("=== All steps completed successfully! ===")
        project_name = config.get('project_name', 'runs/detect')
        print(f"Check the '{project_name}/' directory for results, plots, and weights.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise