#!/usr/bin/env python
"""
Test model on KITTI dataset
"""

from pathlib import Path
import argparse
import yaml
import sys
from src.yolo11.utils.config_utils import get_config, get_dataset_config, verify_paths

# Get configuration
config = get_config()
dataset_paths = get_dataset_config('all')

def verify_test_setup():
    """Verify that the test setup is valid."""
    print("Verifying test setup...")
    
    # Check test directories using config
    required_paths = ['yolo_root', 'test_dir', 'test_images']
    
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
    Test the trained model on test set.
    
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
    
    print(f"Testing model with weights: {weights_path}")
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
        project=config.get('runs_dir', 'runs') + "/test",
        name=Path(weights_path).stem + "_" + config.get('experiment_name', 'exp'),
        exist_ok=True,
        verbose=True,
        device=0,  # Use GPU if available
    )

    # Save speed information to a text file
    results_dir = f"{config.get('runs_dir', 'runs')}/test/{Path(weights_path).stem}_{config.get('experiment_name', 'exp')}/"
    speed_file = Path(results_dir) / "speed_info.txt"
    speed_file.parent.mkdir(parents=True, exist_ok=True)

    if test_results and hasattr(test_results[0], 'speed'):
        # Use the speed from the first result (all should be similar)
        speed_info = test_results[0].speed
        with open(speed_file, 'w') as f:
            f.write("Speed Information (ms per image):\n")
            for k, v in speed_info.items():
                f.write(f"{k}: {v:.2f}\n")
        print(f"Speed information saved to: {speed_file}")
    
    print("✓ Testing completed!")
    results_dir = f"{config.get('runs_dir', 'runs')}/test/{Path(weights_path).stem}_{config.get('experiment_name', 'exp')}/"
    print(f"Results saved at: {results_dir}")
    print()
    
    
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

    # Save validation results to a text file
    results_dir = f"{config.get('runs_dir', 'runs')}/val/{Path(weights_path).stem}_{config.get('experiment_name', 'exp')}/"
    results_file = Path(results_dir) / "validation_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("Validation Results:\n")
        f.write(f"mAP50-95: {val_results.box.map:.4f}\n")
        f.write(f"mAP50: {val_results.box.map50:.4f}\n")
        f.write(f"mAP75: {val_results.box.map75:.4f}\n")
        f.write("Per-category mAP50-95:\n")
        for i, category_map in enumerate(val_results.box.maps):
            f.write(f"  Category {i}: {category_map:.4f}\n")
    
    print(f"Validation results saved to: {results_file}")
    
    print("✓ Validation completed!")
    print()
    return val_results

def main():
    """Main function to run the testing."""
    parser = argparse.ArgumentParser(description='Test YOLO model on KITTI dataset')
    parser.add_argument('--weights', type=str, 
                       default=config.get('default_test_weights', config.get('models_dir', 'models') + '/best.pt'),
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, 
                       default=config.get('default_conf_threshold', 0.25),
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, 
                       default=config.get('default_iou_threshold', 0.7),
                       help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, 
                       default=config.get('default_test_imgsz', 640),
                       help='Image size for inference')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation on validation set')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip testing on test set')
    parser.add_argument('--save-images', action='store_true',
                       default=config.get('save_images_default', False),
                       help='Save prediction images (default: do not save images)')
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
                save_txt=not args.no_save_txt if not args.no_save_txt else config.get('save_txt_default', True),
                save_conf=not args.no_save_conf if not args.no_save_conf else config.get('save_conf_default', True)
            )
        
        print("=== Testing script completed successfully! ===")
        print(f"Check the '{config.get('runs_dir', 'runs')}/test/{Path(args.weights).stem}_{config.get('experiment_name', 'exp')}/' directory for results.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
