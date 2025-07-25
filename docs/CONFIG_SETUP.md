# Configuration Setup Guide

This project uses a local configuration system to manage paths and settings across different environments (local, HPC, cloud, etc.).

## Quick Setup

1. **Copy the template configuration:**
   ```bash
   cp local_config.yaml.template local_config.yaml
   ```

2. **Edit `local_config.yaml` with your actual paths:**
   ```yaml
   # Dataset paths - UPDATE THESE TO YOUR ACTUAL PATHS
   dataset_root: "/path/to/your/yolo_kitti"
   all_dataset_root: "/path/to/your/yolo_kitti_all"
   car_dataset_root: "/path/to/your/yolo_kitti_cars"
   ```

3. **The file `local_config.yaml` is automatically ignored by git** (it's in `.gitignore`)

## Environment Examples

### Local Development
```yaml
dataset_root: "./data/yolo_kitti"
all_dataset_root: "./data/yolo_kitti_all"
car_dataset_root: "./data/yolo_kitti_cars"
```

### HPC Systems (like MSU HPCC)
```yaml
dataset_root: "/mnt/home/username/data/yolo_kitti"
all_dataset_root: "/mnt/home/username/data/yolo_kitti_all"
car_dataset_root: "/mnt/home/username/data/yolo_kitti_cars"
```

### Scratch Storage
```yaml
dataset_root: "/scratch/username/data/yolo_kitti"
all_dataset_root: "/scratch/username/data/yolo_kitti_all"
car_dataset_root: "/scratch/username/data/yolo_kitti_cars"
```

## SLURM Configuration

For SLURM jobs, update the email in the `.slurm` files:
```bash
#SBATCH --mail-user=your-email@example.com
```

## Benefits

- ✅ **No hardcoded paths** in the codebase
- ✅ **Environment-agnostic** code
- ✅ **Easy deployment** across different systems
- ✅ **Git-safe** (personal paths stay local)
- ✅ **Team-friendly** (each user has their own config)

## Configuration Files

- `local_config.yaml.template` - Template file (committed to git)
- `local_config.yaml` - Your personal config (NOT committed to git)
- `src/yolo11/utils/config_utils.py` - Configuration loading utilities
