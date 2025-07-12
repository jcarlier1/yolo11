#!/usr/bin/env python
"""
Fine-tune YOLO v11x on KITTI dataset converted to YOLO format
"""

from pathlib import Path
import shutil
import random
from PIL import Image
import yaml

# ------------------------------------------------------------------------------------
# 0. Fixed paths
# ------------------------------------------------------------------------------------

YOLO_ROOT    = Path("/home/carlier1/data/yolo_kitti")
DATA_YAML    = YOLO_ROOT / "dataset.yaml"  # dataset configuration file

# YOLO dataset structure
TRAIN_DIR    = YOLO_ROOT / "train"
VAL_DIR      = YOLO_ROOT / "val" 
TEST_DIR     = YOLO_ROOT / "test"

# Images and labels directories
TRAIN_IMAGES = TRAIN_DIR / "images"
TRAIN_LABELS = TRAIN_DIR / "labels"
VAL_IMAGES   = VAL_DIR / "images"
VAL_LABELS   = VAL_DIR / "labels"
TEST_IMAGES  = TEST_DIR / "images"
TEST_LABELS  = TEST_DIR / "labels"

def verify_dataset_structure():
    """Verify that the YOLO dataset structure exists and is valid."""
    print("Verifying dataset structure...")
    
    # Check main directories
    required_dirs = [YOLO_ROOT, TRAIN_DIR, VAL_DIR, TEST_DIR, 
                    TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, 
                    TEST_IMAGES, TEST_LABELS]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
        print(f"✓ {dir_path}")
    
    # Check dataset.yaml
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")
    print(f"✓ {DATA_YAML}")
    
    # Count files in each split
    train_images = len(list(TRAIN_IMAGES.glob("*")))
    train_labels = len(list(TRAIN_LABELS.glob("*.txt")))
    val_images = len(list(VAL_IMAGES.glob("*")))
    val_labels = len(list(VAL_LABELS.glob("*.txt")))
    test_images = len(list(TEST_IMAGES.glob("*")))
    test_labels = len(list(TEST_LABELS.glob("*.txt")))
    
    print(f"\nDataset Summary:")
    print(f"  Train: {train_images} images, {train_labels} labels")
    print(f"  Val:   {val_images} images, {val_labels} labels")
    print(f"  Test:  {test_images} images, {test_labels} labels")
    
    # Verify dataset.yaml content
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Path: {config.get('path', 'Not specified')}")
    print(f"  Classes: {len(config.get('names', {}))}")
    print(f"  Class names: {list(config.get('names', {}).values())}")
    
    return True

def train_model():
    """Train YOLO model with configuration."""
    from ultralytics import YOLO

    print(f"Starting YOLO training with dataset: {DATA_YAML}")
    
    # Load pretrained model (using YOLOv11x)
    model = YOLO("yolo11x.pt")
    
    # Training configuration
    results = model.train(
        data=str(DATA_YAML),        # Dataset configuration
        epochs=500,                 # Maximum epochs
        patience=50,                # Early stopping patience
        imgsz=640,                  # Image size (standard for YOLO)
        batch=-1,                   # Auto-batch size detection
        device=0,                   # First GPU (or CPU if no GPU)
        name="kitti_yolo11x",       # Experiment name
        save_period=10,             # Save checkpoint every 10 epochs
        val=True,                   # Enable validation
        plots=True,                 # Generate training plots
        verbose=True,               # Verbose output
        exist_ok=True,              # Allow overwriting existing experiment
        # Additional parameters for better training
        lr0=0.001,                  # Initial learning rate
        optimizer='AdamW',          # Use AdamW optimizer
        cos_lr=True,                # Cosine learning rate scheduler
        warmup_epochs=3,            # Warmup epochs
        warmup_momentum=0.8,        # Warmup momentum
        warmup_bias_lr=0.1,         # Warmup bias learning rate
        box=7.5,                    # Box loss weight
        cls=0.5,                    # Classification loss weight
        dfl=1.5,                    # DFL loss weight
        amp=True,                   # Automatic Mixed Precision
        cache=True,                 # Cache images for faster training
        workers=8,                  # Number of dataloader workers
    )
    
    print(f"Training completed successfully!")
    print(f"Best weights saved at: runs/detect/kitti_yolo11x/weights/best.pt")
    print(f"Last weights saved at: runs/detect/kitti_yolo11x/weights/last.pt")
    
    # Print training summary
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\nTraining Summary:")
        print(f"  - Best mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  - Best mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  - Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"  - Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
    
    return results

def validate_model(weights_path="runs/detect/kitti_yolo11x/weights/best.pt"):
    """Validate the trained model."""
    from ultralytics import YOLO
    
    print(f"Validating model with weights: {weights_path}")
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run validation
    val_results = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=16,
        verbose=True,
        plots=True,
        save_json=True,
    )
    
    print("Validation completed!")
    return val_results

def test_model(weights_path="runs/detect/kitti_yolo11x/weights/best.pt"):
    """Test the trained model on test set."""
    from ultralytics import YOLO
    
    print(f"Testing model with weights: {weights_path}")
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run prediction on test set
    test_results = model.predict(
        source=str(TEST_IMAGES),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/detect",
        name="kitti_yolo11x_test",
        exist_ok=True,
    )
    
    print("Testing completed!")
    print(f"Results saved at: runs/detect/kitti_yolo11x_test/")
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