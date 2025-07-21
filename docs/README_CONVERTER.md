# KITTI to YOLO Dataset Converter

This Python program converts KITTI dataset format to YOLO format for object detection training.

## Features

- Converts KITTI bounding box annotations to YOLO format
- Handles train/validation/test splits from ImageSets
- Creates proper YOLO directory structure
- Generates dataset.yaml configuration file
- Supports image format conversion (PNG to JPG)
- Comprehensive logging and error handling

## Dataset Structure

### Input (KITTI format)
```
kitti/
├── testing/
│   ├── calib/
│   ├── image_2/
│   └── ImageSets/
└── training/
    ├── calib/
    ├── image_2/
    ├── ImageSets/
    ├── label/
    ├── labels_car/
    └── planes/
```

### Output (YOLO format)
```
yolo_kitti/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line
```bash
python kitti_coco.py --kitti_root /path/to/kitti --yolo_root /path/to/output
```

### Python Script
```python
from kitti_coco import KittiToYoloConverter

converter = KittiToYoloConverter(
    kitti_root="/path/to/your/kitti/dataset",
    yolo_root="/path/to/your/yolo/dataset"
)
converter.convert()
```

### Example
```bash
python convert_example.py
```

## Class Mapping

The converter maps KITTI classes to YOLO class IDs:
- Car: 0
- Van: 1
- Truck: 2
- Pedestrian: 3
- Person_sitting: 4
- Cyclist: 5
- Tram: 6
- Misc: 7

## Format Conversion

### KITTI Label Format
```
class truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y
```

### YOLO Label Format
```
class_id center_x center_y width height
```
(All coordinates normalized to 0-1 range)

## Options

- `--kitti_root`: Path to KITTI dataset root directory
- `--yolo_root`: Path where YOLO dataset will be created
- `--verbose`: Enable verbose logging

## Notes

- Images are converted from PNG to JPG format for smaller file sizes
- DontCare class annotations are ignored
- If ImageSets are not available, the script creates default 80/20 train/val split
- Image dimensions are automatically detected using PIL
