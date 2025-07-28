# KITTI to YOLO Dataset Converter

A comprehensive Python utility that converts KITTI 2D object detection datasets to YOLO format with configurable class mapping, train/val splits, and k-fold cross-validation support.

## Features

- ✅ **Configurable class mapping** - Easy to modify in-code dictionary
- ✅ **Multiple mapping strategies** - Combine classes, drop classes, or mark as ignore
- ✅ **Stratified sampling** - Balanced class distribution across splits
- ✅ **K-fold cross-validation** - Generate non-overlapping folds
- ✅ **Data validation** - Skip invalid boxes and log warnings
- ✅ **Progress tracking** - Beautiful progress bars and detailed statistics
- ✅ **Reproducible** - Configurable random seeds
- ✅ **YOLO-ready output** - Compatible with Ultralytics YOLO

## Quick Start

### 1. Download KITTI Dataset

```bash
# Download the required KITTI files
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

# Extract the files
unzip data_object_image_2.zip
unzip data_object_label_2.zip
```

This will create a directory structure like:
```
kitti_data/
  training/
    image_2/           # Images (.png files)
    label_2/           # Labels (.txt files)
```

### 2. Install Dependencies

```bash
pip install ultralytics numpy pyyaml tqdm
```

### 3. Basic Usage

```bash
# Convert with default 3-class mapping (Car+Van, Pedestrian+Person_sitting, Cyclist)
python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset

# Custom train/val ratio with stratified sampling and 10-fold CV
python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset \
                     --train_ratio 0.7 --kfold 10 --stratify --seed 123

# Verbose output to see detailed conversion process
python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset --verbose
```

## Customizing Class Mappings

The most powerful feature is the configurable class mapping. Edit the `REMAP` dictionary at the top of `kitti2yolo.py`:

### Default 3-class mapping:
```python
REMAP = {
    "Car": 0,
    "Van": 0,                    # Combine Car and Van into class 0
    "Pedestrian": 1, 
    "Person_sitting": 1,         # Combine pedestrians into class 1
    "Cyclist": 2,
    # Dropped classes: "Truck", "Tram", "Misc", "DontCare"
}
```

### All 8 classes mapping:
```python
REMAP = {
    "Car": 0,
    "Van": 1, 
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7
}
```

### Car detection only:
```python
REMAP = {
    "Car": 0,
    "Van": 0
}
```

### With ignore classes:
```python
REMAP = {
    "Car": 0,
    "Van": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
    "Truck": -1,     # Mark as ignore (class -1)
    "Tram": -1,
    "Misc": -1
}
```

**Note:** YOLO validation may skip images with class -1 labels. For best results, omit unwanted classes from the mapping instead of using -1.

## Output Structure

The converter creates a YOLO-compatible dataset structure:

```
yolo_dataset/
├── dataset.yaml           # YOLO dataset configuration
├── images/
│   ├── train/            # Training images
│   ├── val/              # Validation images
│   └── test/             # Test images (empty, ready for your test data)
├── labels/
│   ├── train/            # Training labels in YOLO format
│   ├── val/              # Validation labels in YOLO format
│   └── test/             # Test labels (empty)
└── splits/
    ├── kfold_0.txt       # K-fold split files
    ├── kfold_1.txt
    └── ...
```

## Command Line Options

```bash
python kitti2yolo.py --help
```

- `--kitti_path`: Path to KITTI dataset root directory (required)
- `--out_path`: Output path for YOLO dataset (required)
- `--train_ratio`: Fraction of data for training (default: 0.8)
- `--kfold`: Number of k-fold splits to generate (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--stratify`: Use stratified sampling for balanced splits
- `--verbose`: Enable verbose logging

## Usage with YOLO Training

After conversion, you can train with Ultralytics YOLO:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.pt')  # or yolo11s.pt, yolo11m.pt, etc.

# Train the model
results = model.train(
    data='./yolo_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## Testing the Converter

Run the included test script to verify everything works:

```bash
python test_kitti2yolo.py
```

This will create sample KITTI data and test various mapping configurations.

## Advanced Features

### Stratified Sampling
Use `--stratify` to ensure balanced class distribution across train/val splits:

```bash
python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset --stratify
```

### K-fold Cross Validation
Generate k-fold splits for cross-validation experiments:

```bash
python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset --kfold 10
```

The splits are saved as text files in `yolo_dataset/splits/kfold_*.txt`, each containing image names for that fold.

### Reproducible Results
Set a specific seed for reproducible splits:

```bash
python kitti2yolo.py --kitti_path ./kitti_data --out_path ./yolo_dataset --seed 12345
```

## Statistics and Validation

The converter provides detailed statistics about the conversion process:

```
==================================================
CONVERSION STATISTICS
==================================================
Total images processed: 7481
Total labels found: 80256
Labels converted: 28742
Labels dropped: 51514
Invalid boxes: 0

Class distribution:
  Class 0: 28654 objects in 6733 images
  Class 1: 4487 objects in 4250 images  
  Class 2: 1627 objects in 1627 images
```

## Error Handling

The converter includes robust error handling:
- Validates input directory structure
- Skips invalid bounding boxes (zero area, out of bounds)
- Logs warnings for problematic labels
- Creates empty label files for images without annotations

## Integration with Existing Workflows

The converter is designed to be easily integrated into existing machine learning pipelines:

```python
from kitti2yolo import KittiToYoloConverter

# Custom class mapping
custom_remap = {"Car": 0, "Pedestrian": 1}

# Create converter with custom mapping
converter = KittiToYoloConverter(remap=custom_remap)

# Run conversion
converter.convert_kitti_to_yolo(
    kitti_path="./kitti_data",
    output_path="./custom_yolo_dataset",
    train_ratio=0.8,
    kfold=5,
    seed=42,
    stratify=True
)
```

## Troubleshooting

### Common Issues

1. **"KITTI image directory not found"**
   - Ensure you've extracted the KITTI zip files correctly
   - Check that the directory structure matches: `kitti_path/training/image_2/`

2. **"No image files found"**
   - KITTI images should be `.png` files in the `image_2` directory
   - If you have `.jpg` files, the converter will also look for those

3. **Validation warnings about class -1**
   - This is expected if you use ignore classes (-1)
   - Consider omitting unwanted classes from REMAP instead of using -1

4. **Empty dataset after conversion**
   - Check your REMAP dictionary - make sure it includes the classes you want
   - Use `--verbose` to see detailed conversion logs

### Getting Help

If you encounter issues:
1. Run with `--verbose` to see detailed logs
2. Check the conversion statistics for clues
3. Run `python test_kitti2yolo.py` to verify the converter works with sample data

## License

This utility is provided as-is for research and educational purposes. Please respect the KITTI dataset license terms when using this converter.
