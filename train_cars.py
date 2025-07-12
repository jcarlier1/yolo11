#!/usr/bin/env python3
"""
Train YOLO11 model for Car detection using the converted KITTI dataset
"""

from ultralytics import YOLO
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Configuration
    dataset_yaml = "/home/carlier1/data/yolo_kitti_cars/dataset.yaml"
    model_size = "yolo11x.pt"  # Use x model for faster training
    epochs = 500
    imgsz = 640
    batch_size = -1
    project_name = "runs/detect"
    experiment_name = "kitti_car_yolo11x"
    
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
        save_period=10,  # Save checkpoint every 10 epochs
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
    
    # Test on sample images
    logger.info("Running inference on sample images...")
    results = model.predict(
        source=f"/home/carlier1/data/yolo_kitti_cars/val/images",
        save=True,
        save_txt=True,
        project=project_name,
        name=f"{experiment_name}_inference",
        conf=0.5,  # Confidence threshold
        max_det=50,  # Maximum detections per image
    )
    
    print(f"\nTraining completed!")
    print(f"Model weights saved in: {project_name}/{experiment_name}/weights/")
    print(f"Best weights: {project_name}/{experiment_name}/weights/best.pt")
    print(f"Last weights: {project_name}/{experiment_name}/weights/last.pt")
    print(f"Inference results: {project_name}/{experiment_name}_inference/")
    
    print("\nTo use the trained model:")
    print(f"from ultralytics import YOLO")
    print(f"model = YOLO('{project_name}/{experiment_name}/weights/best.pt')")
    print(f"results = model.predict('path/to/image.jpg')")

if __name__ == "__main__":
    main()
