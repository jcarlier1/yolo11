#!/usr/bin/env python3
"""
Train YOLO11 model for Car detection using the converted KITTI dataset
"""

from ultralytics import YOLO
import logging
import os
from pathlib import Path
from src.yolo11.utils.config_utils import get_config, get_dataset_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()
dataset_paths = get_dataset_config('car')

def main():
    # Configuration
    dataset_yaml = str(dataset_paths['data_yaml'])
    model_size = config.get('default_model', 'yolo11s.pt')
    epochs = config.get('default_epochs', 500)
    imgsz = config.get('default_imgsz', 640)
    batch_size = config.get('default_batch_size', -1)
    project_name = config.get('project_name', 'runs/detect')
    experiment_name = "kitti_car_yolo11s"
    
    print("YOLO11 Car Detection Training")
    print("=" * 40)
    print(f"Dataset: {dataset_yaml}")
    print(f"Model: {model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Project: {project_name}")
    print(f"Experiment: {experiment_name}")
    print()
    
    # Check if dataset exists
    if not os.path.exists(dataset_yaml):
        logger.error(f"Dataset configuration not found: {dataset_yaml}")
        logger.error("Please run the KITTI car converter first:")
        logger.error("python convert_cars_example.py")
        return
    
    # Load YOLO model
    logger.info(f"Loading YOLO model: {model_size}")
    model = YOLO(model_size)
    
    # Display model info
    model.info()
    
    # Train the model
    logger.info("Starting training...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project_name,
        name=experiment_name,
        save=True,
        device=0,  # Use GPU if available, otherwise CPU
        workers=4,
        patience=20,  # Early stopping patience
        amp=True,  # Automatic Mixed Precision
        plots=True,  # Generate training plots
        val=True,  # Validate during training
    )
    
    # Model validation
    logger.info("Validating model...")
    metrics = model.val()
    
    # Display results
    print("\nTraining Results:")
    print("=" * 20)
    print(f"Best mAP50: {metrics.box.map50:.4f}")
    print(f"Best mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Export model
    logger.info("Exporting model...")
    model.export(format="onnx")  # Export to ONNX format
    
    # Test on test dataset
    logger.info("Running test dataset...")
    test_results = model.predict(
        source=str(dataset_paths['test_images']),
        save=True,
        save_txt=True,
        save_conf=True,
        project=project_name,
        name=f"{experiment_name}_test",
        conf=0.25,  # Lower confidence threshold for testing
        iou=0.7,    # IoU threshold for NMS
        verbose=True,
    )
    
    # Calculate test statistics
    total_test_images = len(test_results)
    images_with_detections = sum(1 for result in test_results if len(result.boxes) > 0)
    total_detections = sum(len(result.boxes) for result in test_results)
    
    logger.info("Test completed!")
    print(f"\nTest Results:")
    print("=" * 15)
    print(f"Total test images: {total_test_images}")
    print(f"Images with car detections: {images_with_detections}")
    print(f"Total car detections: {total_detections}")
    print(f"Average detections per image: {total_detections/total_test_images:.2f}")
    print(f"Detection rate: {images_with_detections/total_test_images:.1%}")
    
    # Sample inference on validation images for comparison
    logger.info("Running sample inference on validation images...")
    val_results = model.predict(
        source=str(dataset_paths['val_images']),
        save=True,
        save_txt=True,
        project=project_name,
        name=f"{experiment_name}_validation_inference",
        conf=0.5,  # Higher confidence threshold for validation samples
        max_det=50,  # Maximum detections per image
    )
    
    print(f"\nTraining completed!")
    print(f"Model weights saved in: {project_name}/{experiment_name}/weights/")
    print(f"Best weights: {project_name}/{experiment_name}/weights/best.pt")
    print(f"Last weights: {project_name}/{experiment_name}/weights/last.pt")
    print(f"Test results: {project_name}/{experiment_name}_test/")
    print(f"Validation inference results: {project_name}/{experiment_name}_validation_inference/")
    
    print("\nFiles generated:")
    print(f"- Training plots and metrics: {project_name}/{experiment_name}/")
    print(f"- Test predictions: {project_name}/{experiment_name}_test/")
    print(f"- Validation samples: {project_name}/{experiment_name}_validation_inference/")
    print(f"- ONNX model: {project_name}/{experiment_name}/weights/best.onnx")
    
    print("\nTo use the trained model:")
    print(f"from ultralytics import YOLO")
    print(f"model = YOLO('{project_name}/{experiment_name}/weights/best.pt')")
    print(f"results = model.predict('path/to/image.jpg')")
    
    print("\nTo test on new images:")
    print(f"python test.py --weights {project_name}/{experiment_name}/weights/best.pt")

if __name__ == "__main__":
    main()
