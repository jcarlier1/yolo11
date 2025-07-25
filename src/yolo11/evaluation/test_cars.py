#!/usr/bin/env python
"""
Test YOLO11 model for Car detection on KITTI dataset
"""

from pathlib import Path
import argparse
import yaml
import sys
from src.yolo11.utils.config_utils import get_config, get_dataset_config, verify_paths

# Get configuration for car dataset
config = get_config()
dataset_paths = get_dataset_config('car')

def verify_test_setup():
    """Verify that the test setup is valid."""
    print("Verifying test setup...")
    
    # Check test directories using config
    required_paths = ['car_yolo_root', 'car_test_dir', 'car_test_images']
    
    if not verify_paths(required_paths):
        raise FileNotFoundError("Required directories not found")
    
    # Check dataset.yaml
    data_yaml = dataset_paths['data_yaml']
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    print(f"✓ {data_yaml}")
    
    # Count test files
    test_images = len(list(dataset_paths['test_images'].glob("*")))
    test_labels = len(list(dataset_paths['test_labels'].glob("*.txt"))) if dataset_paths['test_labels'].exists() else 0
    
    print(f"\nTest Dataset Summary:")
    print(f"  Test: {test_images} images, {test_labels} labels")
    
    # Verify dataset.yaml content
    with open(data_yaml, 'r') as f:
        config_data = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Path: {config_data.get('path', 'Not specified')}")
    print(f"  Classes: {len(config_data.get('names', {}))}")
    print(f"  Class names: {list(config_data.get('names', {}).values())}")
    
    print("✓ Test setup verified!")
    print()

    return True

def test_model(weights_path, conf_threshold=0.25, iou_threshold=0.7, imgsz=640, 
               save_images=False, save_txt=True, save_conf=True):
    """
    Test the trained car detection model on test set.
    
    Args:
        weights_path: Path to model weights
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
        imgsz: Image size for inference
        output_name: Name for output directory
        save_images: Save prediction images
        save_txt: Save predictions as text files
        save_conf: Save confidence scores
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)
    
    print(f"Testing car detection model with weights: {weights_path}")
    print(f"Configuration:")
    print(f"  - Confidence threshold: {conf_threshold}")
    print(f"  - IoU threshold: {iou_threshold}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Experiment name: {config.get('experiment_name', 'exp')}")
    print(f"  - Save images: {save_images}")
    print(f"  - Save predictions: {save_txt}")
    print(f"  - Save confidence: {save_conf}")
    
    # Check if weights file exists
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run prediction on test set
    test_results = model.predict(
        source=str(dataset_paths['test_images']),
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save_images,
        save_txt=save_txt,
        save_conf=save_conf,
        project=config.get('runs_dir', 'runs') + "/detect",
        name=Path(weights_path).stem + "_" + config.get('experiment_name', 'exp'),
        exist_ok=True,
        verbose=True,
        device=0,  # Use GPU if available
    )
    
    print("✓ Testing completed!")
    results_dir = f"{config.get('runs_dir', 'runs')}/detect/{Path(weights_path).stem}_{config.get('experiment_name', 'exp')}/"
    print(f"Results saved at: {results_dir}")
    print()
    
    
    # Print summary statistics
    if test_results:
        total_images = len(test_results)
        images_with_detections = sum(1 for result in test_results if len(result.boxes) > 0)
        total_detections = sum(len(result.boxes) for result in test_results)
        
        print(f"\nTest Summary:")
        print(f"  - Total images processed: {total_images}")
        print(f"  - Images with car detections: {images_with_detections}")
        print(f"  - Total car detections: {total_detections}")
        print(f"  - Average car detections per image: {total_detections/total_images:.2f}")
        print(f"  - Car detection rate: {images_with_detections/total_images:.1%}")
    
    return test_results

def validate_model(weights_path, imgsz=640, batch=16):
    """
    Validate the trained car detection model (if validation data is available).
    
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
    
    print(f"Validating car detection model with weights: {weights_path}")
    
    # Check if weights file exists
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run validation
    val_results = model.val(
        data=str(dataset_paths['data_yaml']),
        imgsz=imgsz,
        batch=batch,
        verbose=True,
        plots=True,
        save_json=True,
        device=0,  # Use GPU if available
        project=config.get('runs_dir', 'runs') + "/val",
        name=Path(weights_path).stem + "_" + config.get('experiment_name', 'exp'),
    )
    
    print("✓ Validation completed!")
    print()
    return val_results

def main():
    """Main function to run the car detection testing."""
    parser = argparse.ArgumentParser(description='Test YOLO11 car detection model on KITTI dataset')
    parser.add_argument('--weights', type=str, 
                       default='runs/detect/kitti_car_yolo11x/weights/best.pt',
                       help='Path to car detection model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for inference')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation on validation set')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip testing on test set')
    parser.add_argument('--save-images', action='store_true',
                       help='Save prediction images (default: do not save images)')
    parser.add_argument('--no-save-txt', action='store_true',
                       help='Do not save predictions as text files')
    parser.add_argument('--no-save-conf', action='store_true',
                       help='Do not save confidence scores')
    
    args = parser.parse_args()
    
    print("=== YOLO11 Car Detection Testing on KITTI Dataset ===")
    print()
    
    try:
        # Step 1: Verify test setup
        print("Step 1: Verifying test setup...")
        verify_test_setup()
        
        # Step 2: Run validation (if requested)
        if not args.no_validate:
            print("Step 2: Running validation...")
            val_results = validate_model(args.weights)
        
        # Step 3: Test model
        if not args.no_test:
            test_step = "Step 3" if not args.no_validate else "Step 2"
            print(f"{test_step}: Testing model on test set...")
            test_results = test_model(
                weights_path=args.weights,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                imgsz=args.imgsz,
                save_images=args.save_images,
                save_txt=not args.no_save_txt,
                save_conf=not args.no_save_conf
            )
        
        
        print("=== Testing script completed successfully! ===")
        print(f"Check the '{config.get('runs_dir', 'runs')}/detect/{Path(args.weights).stem}_{config.get('experiment_name', 'exp')}/' directory for results.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
