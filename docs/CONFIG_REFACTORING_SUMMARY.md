## Usage Instructions

### For Local Development:
1. Edit `local_config.yaml` to match your local paths
2. Run any script normally - it will use your local configuration
3. Never commit `local_config.yaml` to git

### For HPCC Deployment:
1. Create a new `local_config.yaml` with HPCC-specific paths
2. All scripts will automatically use the HPCC configuration
3. No code changes needed

### For New Environments:
1. Copy and modify `local_config.yaml` for new paths
2. Verify configuration with: `python test_config.py`
3. Run any script to test the setup

## Configuration Reference

### Key Configuration Options:
```yaml
# Dataset paths
dataset_root: "/path/to/main/dataset"
car_dataset_root: "/path/to/car/dataset"

# Model settings  
default_model: "yolo11s.pt"
default_epochs: 500
default_imgsz: 640
default_batch_size: -1

# Output directories
results_dir: "results"
runs_dir: "runs"
cv_results_dir: "runs/cross_validation"
```

### Available Path Keys:
- Main dataset: `yolo_root`, `data_yaml`, `train_images`, `val_images`, `test_images`, etc.
- Car dataset: `car_yolo_root`, `car_data_yaml`, `car_train_images`, etc.
- Outputs: `results_dir`, `runs_dir`, `cv_results_dir`

## Verification
- ✅ All scripts compile without errors
- ✅ Configuration system tested and working
- ✅ Local config properly loaded and applied
- ✅ Git safety verified (local config ignored)
- ✅ Path verification utilities functional

This refactoring provides a robust, maintainable, and git-safe solution for managing environment-specific configurations across the YOLO11 project.
