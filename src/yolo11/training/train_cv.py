#!/usr/bin/env python
"""
Enhanced YOLO Training with Cross-Validation Support
"""

from pathlib import Path
import shutil
import random
from PIL import Image
import yaml
import argparse
import sys
from src.yolo11.utils.config_utils import get_config, get_dataset_config, verify_paths

# Import our cross-validation module
from src.yolo11.utils.cross_validation import YOLOCrossValidator

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

def train_model_single_split():
    """Train YOLO model with original train/val split."""
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
        batch=config.get('default_batch_size', 64), # Batch size
        device=0,                   # Use first GPU only
        name="kitti_yolo11s",       # Experiment name
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

def train_model_cross_validation(k_folds=5, epochs=100, batch=32):
    """Train YOLO model using k-fold cross-validation."""
    print(f"Starting {k_folds}-fold cross-validation...")
    
    # Initialize cross-validator
    cv = YOLOCrossValidator(
        data_root=str(dataset_paths['root']),
        output_dir=config.get('cv_results_dir', 'runs/cross_validation'),
        k_folds=k_folds,
        seed=42
    )
    
    # Run cross-validation with optimized parameters
    cv_results = cv.run_cross_validation(
        epochs=epochs,              # Reduced for faster CV
        patience=20,                # Reduced patience
        batch=batch,                # Adjust based on GPU memory
        imgsz=640,                  # Image size
        device=0,                   # Use first GPU
        lr0=0.001,                  # Learning rate
        optimizer='AdamW',          # Optimizer
        cos_lr=True,                # Cosine learning rate scheduler
        warmup_epochs=3,            # Warmup epochs
        amp=True,                   # Automatic Mixed Precision
        cache='ram',                # Cache dataset in RAM
        workers=4,                  # Reduced workers for multiple folds
        seed=42,                    # Reproducibility
        # Additional parameters
        box=7.5,                    # Box loss weight
        cls=0.5,                    # Classification loss weight
        dfl=1.5,                    # DFL loss weight
        warmup_momentum=0.8,        # Warmup momentum
        warmup_bias_lr=0.1,         # Warmup bias learning rate
        rect=True,                  # Rectangular training
        close_mosaic=20,            # Close mosaic augmentation
    )
    
    print(f"\n=== Cross-Validation Complete ===")
    print(f"Mean mAP@0.5: {cv_results['mean_metrics'].get('mAP50', 0):.4f} ± {cv_results['std_metrics'].get('mAP50', 0):.4f}")
    print(f"Mean mAP@0.5:0.95: {cv_results['mean_metrics'].get('mAP50_95', 0):.4f} ± {cv_results['std_metrics'].get('mAP50_95', 0):.4f}")
    print(f"Mean Precision: {cv_results['mean_metrics'].get('precision', 0):.4f} ± {cv_results['std_metrics'].get('precision', 0):.4f}")
    print(f"Mean Recall: {cv_results['mean_metrics'].get('recall', 0):.4f} ± {cv_results['std_metrics'].get('recall', 0):.4f}")
    
    return cv_results

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

def test_best_cv_model():
    """Test the best model from cross-validation."""
    import json
    
    cv_results_file = Path("runs/cross_validation/cross_validation_results.json")
    
    if not cv_results_file.exists():
        print("No cross-validation results found. Run cross-validation first.")
        return None
    
    # Load CV results
    with open(cv_results_file, 'r') as f:
        cv_results = json.load(f)
    
    # Find best fold
    best_fold = cv_results.get('best_fold', {})
    if not best_fold:
        print("No best fold information found.")
        return None
    
    best_fold_idx = best_fold['fold']
    best_weights = f"runs/cross_validation/fold_{best_fold_idx}/weights/best.pt"
    
    print(f"Testing best model from fold {best_fold_idx} (mAP50: {best_fold['mAP50']:.4f})")
    
    # Test the best model
    return test_model(best_weights)

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="YOLO Training with Cross-Validation")
    parser.add_argument("--mode", choices=["single", "cv", "test", "test-cv"], 
                       default="single", help="Training mode")
    parser.add_argument("--k-folds", type=int, default=5, 
                       help="Number of folds for cross-validation")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, 
                       help="Batch size")
    parser.add_argument("--weights", type=str, 
                       help="Path to weights for testing")
    
    args = parser.parse_args()
    
    print("=== YOLO Training on Converted KITTI Dataset ===")
    print(f"Mode: {args.mode}")
    print()
    
    try:
        # Step 1: Verify dataset structure
        print("Step 1: Verifying dataset structure...")
        verify_dataset_structure()
        print("✓ Dataset structure verified!")
        print()
        
        if args.mode == "single":
            # Single train/val split training
            print("Step 2: Starting single-split YOLO training...")
            results = train_model_single_split()
            print("✓ Training completed!")
            print()
            
            # Step 3: Validate model
            print("Step 3: Validating trained model...")
            val_results = validate_model()
            print("✓ Validation completed!")
            print()
            
            # Step 4: Test model
            print("Step 4: Testing model on test set...")
            test_results = test_model()
            print("✓ Testing completed!")
            
        elif args.mode == "cv":
            # Cross-validation training
            print("Step 2: Starting cross-validation training...")
            cv_results = train_model_cross_validation(
                k_folds=args.k_folds,
                epochs=args.epochs,
                batch=args.batch
            )
            print("✓ Cross-validation completed!")
            print()
            
            # Test best model
            print("Step 3: Testing best model from cross-validation...")
            test_results = test_best_cv_model()
            print("✓ Testing completed!")
            
        elif args.mode == "test":
            # Test only
            weights_path = args.weights or "runs/detect/kitti_yolo11s/weights/best.pt"
            test_results = test_model(weights_path)
            
        elif args.mode == "test-cv":
            # Test best CV model
            test_results = test_best_cv_model()
        
        print("\n=== All steps completed successfully! ===")
        print("Check the appropriate 'runs/' directory for results, plots, and weights.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
