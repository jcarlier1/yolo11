#!/usr/bin/env python
"""
Test YOLO v11x model on KITTI dataset
"""

from pathlib import Path
import argparse
import yaml
import sys

# ------------------------------------------------------------------------------------
# 0. Fixed paths
# ------------------------------------------------------------------------------------

YOLO_ROOT    = Path("/home/carlier1/data/yolo_kitti")
DATA_YAML    = YOLO_ROOT / "dataset.yaml"  # dataset configuration file

# YOLO dataset structure
TEST_DIR     = YOLO_ROOT / "test"
TEST_IMAGES  = TEST_DIR / "images"
TEST_LABELS  = TEST_DIR / "labels"

def verify_test_setup():
    """Verify that the test setup is valid."""
    print("Verifying test setup...")
    
    # Check test directories
    required_dirs = [YOLO_ROOT, TEST_DIR, TEST_IMAGES]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
        print(f"✓ {dir_path}")
    
    # Check dataset.yaml
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")
    print(f"✓ {DATA_YAML}")
    
    # Count test files
    test_images = len(list(TEST_IMAGES.glob("*")))
    test_labels = len(list(TEST_LABELS.glob("*.txt"))) if TEST_LABELS.exists() else 0
    
    print(f"\nTest Dataset Summary:")
    print(f"  Test: {test_images} images, {test_labels} labels")
    
    # Verify dataset.yaml content
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Path: {config.get('path', 'Not specified')}")
    print(f"  Classes: {len(config.get('names', {}))}")
    print(f"  Class names: {list(config.get('names', {}).values())}")
    
    return True

def test_model(weights_path, conf_threshold=0.25, iou_threshold=0.7, imgsz=640, 
               output_name="kitti_yolo11x_test", save_txt=True, save_conf=True):
    """
    Test the trained model on test set.
    
    Args:
        weights_path: Path to model weights
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
        imgsz: Image size for inference
        output_name: Name for output directory
        save_txt: Save predictions as text files
        save_conf: Save confidence scores
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)
    
    print(f"Testing model with weights: {weights_path}")
    print(f"Configuration:")
    print(f"  - Confidence threshold: {conf_threshold}")
    print(f"  - IoU threshold: {iou_threshold}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Output name: {output_name}")
    print(f"  - Save predictions: {save_txt}")
    print(f"  - Save confidence: {save_conf}")
    
    # Check if weights file exists
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run prediction on test set
    test_results = model.predict(
        source=str(TEST_IMAGES),
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        project="runs/detect",
        name=output_name,
        exist_ok=True,
        verbose=True,
    )
    
    print("Testing completed!")
    print(f"Results saved at: runs/detect/{output_name}/")
    
    # Print summary statistics
    if test_results:
        total_images = len(test_results)
        images_with_detections = sum(1 for result in test_results if len(result.boxes) > 0)
        total_detections = sum(len(result.boxes) for result in test_results)
        
        print(f"\nTest Summary:")
        print(f"  - Total images processed: {total_images}")
        print(f"  - Images with detections: {images_with_detections}")
        print(f"  - Total detections: {total_detections}")
        print(f"  - Average detections per image: {total_detections/total_images:.2f}")
        print(f"  - Detection rate: {images_with_detections/total_images:.1%}")
    
    return test_results

def validate_model(weights_path, imgsz=640, batch=16):
    """
    Validate the trained model (if validation data is available).
    
    Args:
        weights_path: Path to model weights
        imgsz: Image size for validation
        batch: Batch size for validation
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)
    
    print(f"Validating model with weights: {weights_path}")
    
    # Check if weights file exists
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run validation
    val_results = model.val(
        data=str(DATA_YAML),
        imgsz=imgsz,
        batch=batch,
        verbose=True,
        plots=True,
        save_json=True,
    )
    
    print("Validation completed!")
    return val_results

def main():
    """Main function to run the testing."""
    parser = argparse.ArgumentParser(description='Test YOLO model on KITTI dataset')
    parser.add_argument('--weights', type=str, 
                       default='runs/detect/kitti_yolo11x/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for inference')
    parser.add_argument('--name', type=str, default='kitti_yolo11x_test',
                       help='Name for output directory')
    parser.add_argument('--validate', action='store_true',
                       help='Also run validation on validation set')
    parser.add_argument('--no-save-txt', action='store_true',
                       help='Do not save predictions as text files')
    parser.add_argument('--no-save-conf', action='store_true',
                       help='Do not save confidence scores')
    
    args = parser.parse_args()
    
    print("=== YOLO Testing on KITTI Dataset ===")
    print()
    
    try:
        # Step 1: Verify test setup
        print("Step 1: Verifying test setup...")
        verify_test_setup()
        print("✓ Test setup verified!")
        print()
        
        # Step 2: Run validation (if requested)
        if args.validate:
            print("Step 2: Running validation...")
            val_results = validate_model(args.weights)
            print("✓ Validation completed!")
            print()
        
        # Step 3: Test model
        test_step = "Step 3" if args.validate else "Step 2"
        print(f"{test_step}: Testing model on test set...")
        test_results = test_model(
            weights_path=args.weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            output_name=args.name,
            save_txt=not args.no_save_txt,
            save_conf=not args.no_save_conf
        )
        print("✓ Testing completed!")
        print()
        
        print("=== Testing completed successfully! ===")
        print(f"Check the 'runs/detect/{args.name}/' directory for results.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
