# KITTI to YOLO Car Detection Converter

This repository contains tools to convert KITTI dataset to YOLO format specifically for Car detection, and train/test YOLO11 models for car detection.

## Features

- **Car-only filtering**: Converts KITTI dataset but only keeps Car bounding boxes
- **Single-class dataset**: All cars are mapped to class ID 0
- **YOLO11 compatible**: Ready for training with ultralytics YOLO11
- **Complete pipeline**: Convert → Train → Test

## Files Overview

### Core Converters
- `kitti_coco.py` - Original multi-class KITTI to YOLO converter
- `kitti_car_converter.py` - **New car-only converter**
- `convert_example.py` - Example for multi-class conversion
- `convert_cars_example.py` - **Example for car-only conversion**

### Training & Testing
- `train.py` - Original training script for multi-class
- `train_cars.py` - **Training script for car detection**
- `test_cars.py` - **Testing script for car detection**

## Quick Start

### 1. Convert KITTI Dataset (Car Detection Only)

```bash
# Run the car-only converter
python convert_cars_example.py
```

Or use the converter directly:
```bash
python kitti_car_converter.py --kitti_root /path/to/kitti --yolo_root /path/to/output
```

**What it does:**
- Scans KITTI annotations for Car objects only
- Ignores all other classes (Van, Truck, Pedestrian, etc.)
- Creates YOLO format labels with class ID 0 for all cars
- Generates `dataset.yaml` configured for single-class car detection

### 2. Train YOLO11 for Car Detection

```bash
python train_cars.py
```

**Training features:**
- Uses YOLO11 nano model for fast training
- Configured for single-class car detection
- Automatic mixed precision training
- Early stopping and checkpointing
- Validation during training

### 3. Test the Trained Model

```bash
# Test on validation images
python test_cars.py

# Test on specific image
python test_cars.py --source /path/to/image.jpg

# Test with custom confidence threshold
python test_cars.py --conf 0.7 --source /path/to/images/
```

## Dataset Structure

After conversion, your dataset will look like:
```
yolo_kitti_cars/
├── train/
│   ├── images/          # Training images (.jpg)
│   └── labels/          # Training labels (.txt)
├── val/
│   ├── images/          # Validation images (.jpg)
│   └── labels/          # Validation labels (.txt)
├── test/
│   ├── images/          # Test images (.jpg)
│   └── labels/          # Test labels (empty for test set)
└── dataset.yaml         # YOLO dataset configuration
```

## Label Format

Each label file contains one line per car:
```
0 center_x center_y width height
```
Where:
- `0` = Car class ID (always 0 for our single-class system)
- `center_x, center_y` = Normalized center coordinates (0-1)
- `width, height` = Normalized bounding box dimensions (0-1)

## Key Differences from Multi-Class Converter

| Feature | Multi-Class (`kitti_coco.py`) | Car-Only (`kitti_car_converter.py`) |
|---------|-------------------------------|-------------------------------------|
| Classes | 8 classes (Car, Van, Truck, etc.) | 1 class (Car only) |
| Class IDs | 0-7 | 0 only |
| Filtering | Processes all object types | Filters for Cars only |
| Dataset size | Larger (all objects) | Smaller (cars only) |
| Training speed | Slower | Faster |
| Model complexity | Multi-class detection | Binary detection (car/no-car) |

## Requirements

Install dependencies:
```bash
pip install ultralytics pillow opencv-python
```

Or use the existing requirements file:
```bash
pip install -r requirements.txt
```

## Configuration

### Default Paths
- KITTI dataset: `./data/kitti`
- Output dataset: `./data/yolo_kitti_cars`
- Model weights: `runs/detect/kitti_car_yolo11n/weights/best.pt`

### Customization
You can modify paths and parameters in the script headers or use command-line arguments:

```bash
# Custom paths
python kitti_car_converter.py \
    --kitti_root /custom/kitti/path \
    --yolo_root /custom/output/path

# Verbose logging
python kitti_car_converter.py --verbose
```

## Performance Tips

1. **GPU Training**: Ensure CUDA is available for faster training
2. **Batch Size**: Adjust batch size based on your GPU memory
3. **Model Size**: Use `yolo11s.pt` or `yolo11m.pt` for better accuracy
4. **Data Augmentation**: YOLO11 includes built-in augmentations
5. **Early Stopping**: Training will stop early if validation doesn't improve

## Output Files

### Training Output
- `runs/detect/kitti_car_yolo11n/weights/best.pt` - Best model weights
- `runs/detect/kitti_car_yolo11n/weights/last.pt` - Latest checkpoint
- Training plots and metrics in the experiment directory

### Testing Output  
- `test_results/car_detection/` - Annotated images with detections
- Detection confidence scores and bounding box coordinates

## Advanced Usage

### Custom Training Parameters

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='/path/to/dataset.yaml',
    epochs=200,
    imgsz=800,
    batch=32,
    lr0=0.001,
    # ... other parameters
)
```

### Inference on Custom Images

```python
from ultralytics import YOLO

model = YOLO('runs/detect/kitti_car_yolo11n/weights/best.pt')
results = model.predict('path/to/image.jpg', conf=0.5)

# Process results
for result in results:
    boxes = result.boxes
    if boxes is not None:
        print(f"Detected {len(boxes)} cars")
```

## Troubleshooting

1. **Import Error**: Make sure ultralytics is installed: `pip install ultralytics`
2. **CUDA Error**: Check GPU availability or set `device='cpu'` in training
3. **Path Error**: Verify KITTI dataset structure and paths
4. **Memory Error**: Reduce batch size or image size
5. **No Detections**: Lower confidence threshold or check model training

## License

This project follows the same license as the original YOLO implementation.
