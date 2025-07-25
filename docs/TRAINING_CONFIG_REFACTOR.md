# Training Script Configuration Refactor

## Overview
The training script (`src/yolo11/training/train.py`) has been refactored to use a configuration-driven approach, removing hardcoded values and making all training parameters configurable through the `local_config.yaml` file.

## Changes Made

### 1. Removed Hardcoded Training Parameters
Previously hardcoded values that are now configurable:

**Training Setup:**
- `experiment_name`: Now uses `config.get('experiment_name', 'kitti_yolo11s')`
- `patience`: Now configurable as `training_patience` (default: 50)
- `device`: Now configurable as `training_device` (default: 0)
- `save_period`: Now configurable (default: 10)
- `val`, `plots`, `verbose`, `exist_ok`: All now configurable

**Optimizer Settings:**
- `lr0`: Now `learning_rate` (default: 0.001)
- `optimizer`: Now configurable (default: 'AdamW')
- `cos_lr`: Now `cosine_lr_scheduler` (default: true)
- `warmup_epochs`: Now configurable (default: 3)
- `warmup_momentum`: Now configurable (default: 0.8)
- `warmup_bias_lr`: Now configurable (default: 0.1)

**Loss Weights:**
- `box`: Now `box_loss_weight` (default: 7.5)
- `cls`: Now `cls_loss_weight` (default: 0.5)
- `dfl`: Now `dfl_loss_weight` (default: 1.5)

**Performance Optimizations:**
- `amp`: Now `mixed_precision` (default: true)
- `cache`: Now `cache_mode` (default: 'ram')
- `workers`: Now `num_workers` (default: 8)
- `close_mosaic`: Now `close_mosaic_epochs` (default: 20)
- `rect`: Now `rectangular_training` (default: true)
- `single_cls`: Now `single_class` (default: false)
- `deterministic`: Now configurable (default: false)
- `seed`: Now `random_seed` (default: 42)

### 2. Updated Configuration Files

**config_utils.py:**
- Added all new training parameters to the `DEFAULTS` dictionary
- Maintained backward compatibility with existing configurations

**local_config.yaml.template:**
- Updated with comprehensive training parameter documentation
- Added comments explaining each parameter

**local_config.yaml:**
- Updated with new parameter examples (commented out to use defaults)
- Fixed `experiment_name` for consistency

### 3. Enhanced Training Script

**Dynamic Configuration Loading:**
- All training parameters are now loaded from configuration
- Configuration is printed before training starts for transparency
- Uses fallback defaults if parameters are not specified

**Configurable Output Paths:**
- Experiment names and project paths now use configuration
- Summary messages use configured paths

## Usage

### Using Default Configuration
The script will work out of the box with sensible defaults. No changes to `local_config.yaml` are required.

### Customizing Training Parameters
To customize training, add any of the following parameters to your `local_config.yaml`:

```yaml
# Example customizations
experiment_name: "my_custom_experiment"
default_epochs: 300
learning_rate: 0.002
mixed_precision: false  # Disable AMP if needed
cache_mode: "disk"      # Use disk cache instead of RAM
num_workers: 16         # More workers for faster data loading
```

### Parameter Categories

1. **Basic Training Settings:**
   - `default_epochs`, `training_patience`, `default_imgsz`, `default_batch_size`

2. **Optimizer Configuration:**
   - `learning_rate`, `optimizer`, `cosine_lr_scheduler`, `warmup_epochs`

3. **Loss Weights:**
   - `box_loss_weight`, `cls_loss_weight`, `dfl_loss_weight`

4. **Performance Tuning:**
   - `mixed_precision`, `cache_mode`, `num_workers`, `rectangular_training`

5. **Reproducibility:**
   - `random_seed`, `deterministic`

## Benefits

1. **Flexibility:** Easy to experiment with different hyperparameters
2. **Environment Adaptability:** Can optimize for different hardware setups
3. **Reproducibility:** All parameters are explicitly documented
4. **Maintainability:** Single source of truth for configuration
5. **Best Practices:** Follows configuration-driven development principles

## Migration Guide

Existing users need no changes - all defaults maintain the previous behavior. To customize parameters, simply add them to `local_config.yaml`.

## Testing

The refactored code has been syntax-checked and maintains the same external API. All paths and parameter resolution work through the existing configuration infrastructure.
