#!/usr/bin/env python
"""
Fine-tune YOLO v11x on KITTI dataset converted to YOLO format
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
    print(f"  Class names: {list(config_data.get('names', {}).values())}")
    
    return True

def train_model():
    """Train YOLO model with configuration."""
    from ultralytics import YOLO

    print(f"Starting YOLO training with dataset: {dataset_paths['data_yaml']}")
    
    # Load pretrained model (using YOLOv11s)
    model = YOLO(config.get('default_model', 'yolo11s.pt'))
    
    # Training configuration optimized for single NVIDIA H200 GPU (144GB VRAM)
    results = model.train(
        data=str(dataset_paths['data_yaml']),        # Dataset configuration
        epochs=config.get('default_epochs', 500),   # Maximum epochs
        patience=50,                # Early stopping patience
        imgsz=config.get('default_imgsz', 640),     # Image size
        batch=config.get('default_batch_size', 64), # Large batch size for H200 GPU
        device=0,                   # Use first GPU only
        name="kitti_yolo11s",       # Experiment name
        save_period=10,             # Save checkpoint every 10 epochs
        val=True,                   # Enable validation
        plots=True,                 # Generate training plots
        verbose=True,               # Verbose output
        exist_ok=True,              # Allow overwriting existing experiment
        # Optimized parameters for single GPU training
        lr0=0.001,                  # Standard learning rate for single GPU
        optimizer='AdamW',          # Use AdamW optimizer
        cos_lr=True,                # Cosine learning rate scheduler
        warmup_epochs=3,            # Warmup epochs
        warmup_momentum=0.8,        # Warmup momentum
        warmup_bias_lr=0.1,         # Warmup bias learning rate
        box=7.5,                    # Box loss weight
        cls=0.5,                    # Classification loss weight
        dfl=1.5,                    # DFL loss weight
        amp=True,                   # Automatic Mixed Precision (crucial for H200)
        cache='ram',                # Cache entire dataset in RAM (H200 has lots of memory)
        workers=8,                  # Reasonable number of workers for single GPU
        close_mosaic=20,            # Close mosaic augmentation in last 20 epochs
        # Additional optimizations for single GPU training
        rect=True,                  # Rectangular training for efficiency
        single_cls=False,           # Multi-class training
        deterministic=False,        # Allow non-deterministic for speed
        seed=42,                    # Set seed for reproducibility
    )
    
    print(f"Training completed successfully!")
    print(f"Best weights saved at: runs/detect/kitti_yolo11s/weights/best.pt")
    print(f"Last weights saved at: runs/detect/kitti_yolo11s/weights/last.pt")
    
    # Print training summary
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\nTraining Summary:")
        print(f"  - Best mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  - Best mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  - Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"  - Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
    
    return results

def validate_model(weights_path="runs/detect/kitti_yolo11s/weights/best.pt"):
    """Validate the trained model."""
    from ultralytics import YOLO
    
    print(f"Validating model with weights: {weights_path}")
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run validation
    val_results = model.val(
        data=str(dataset_paths['data_yaml']),
        imgsz=640,
        batch=16,
        verbose=True,
        plots=True,
        save_json=True,
    )
    
    print("Validation completed!")
    return val_results

def test_model(weights_path="runs/detect/kitti_yolo11s/weights/best.pt"):
    """Test the trained model on test set."""
    from ultralytics import YOLO
    
    print(f"Testing model with weights: {weights_path}")
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run prediction on test set
    test_results = model.predict(
        source=str(dataset_paths['test_images']),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        save=True,
        save_txt=True,
        save_conf=True,
        project=config.get('runs_dir', 'runs') + "/detect",
        name="kitti_yolo11s_test",
        exist_ok=True,
    )
    
    print("Testing completed!")
    print(f"Results saved at: runs/detect/kitti_yolo11s_test/")
    return test_results

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
        
        # Step 3: Validate model
        print("Step 3: Validating trained model...")
        val_results = validate_model()
        print("✓ Validation completed!")
        print()
        
        # Step 4: Test model (optional)
        print("Step 4: Testing model on test set...")
        test_results = test_model()
        print("✓ Testing completed!")
        print()
        
        print("=== All steps completed successfully! ===")
        print("Check the 'runs/detect/' directory for results, plots, and weights.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise