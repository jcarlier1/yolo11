#!/usr/bin/env python3
"""
Test the trained YOLO11 Car detection model
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from src.yolo11.utils.config_utils import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

def test_car_detection(model_path, image_path, output_dir=None, conf_threshold=0.5):
    """
    Test car detection on a single image or directory of images.
    
    Args:
        model_path: Path to the trained YOLO model
        image_path: Path to image file or directory
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    """
    
    if output_dir is None:
        output_dir = config.get('results_dir', 'results')
    
    # Load the trained model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run inference
    logger.info(f"Running inference on: {image_path}")
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        project=output_dir,
        name="car_detection",
        conf=conf_threshold,
        max_det=50,
        device=0,  # Use GPU if available
    )
    
    # Process results
    for i, result in enumerate(results):
        # Get detection info
        boxes = result.boxes
        if boxes is not None:
            num_cars = len(boxes)
            confidences = boxes.conf.cpu().numpy()
            
            logger.info(f"Image {i+1}: Detected {num_cars} cars")
            for j, conf in enumerate(confidences):
                logger.info(f"  Car {j+1}: Confidence = {conf:.3f}")
        else:
            logger.info(f"Image {i+1}: No cars detected")
    
    print(f"\nResults saved to: {output_dir}/car_detection/")
    return results

def main():
    parser = argparse.ArgumentParser(description='Test YOLO11 Car Detection Model')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/kitti_car_yolo11n/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--source', type=str,
                       default='./data/yolo_kitti_cars/val/images',
                       help='Path to test image(s) or directory')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.error("Please train the model first using: python train_cars.py")
        return
    
    # Check if source exists
    if not Path(args.source).exists():
        logger.error(f"Source not found: {args.source}")
        logger.error("Please provide a valid image path or directory")
        return
    
    print("YOLO11 Car Detection Test")
    print("=" * 30)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Confidence threshold: {args.conf}")
    print()
    
    # Run test
    results = test_car_detection(
        model_path=args.model,
        image_path=args.source,
        output_dir=args.output,
        conf_threshold=args.conf
    )
    
    print("\nTest completed successfully!")
    print("Check the output directory for annotated images and detection files.")

if __name__ == "__main__":
    main()
