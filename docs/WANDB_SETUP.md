# Weights & Biases (wandb) Integration Guide

This guide explains how to use wandb logging with your YOLO training script.

## Setup

### 1. Install wandb
```bash
pip install wandb
```

### 2. Login to wandb
```bash
wandb login
```
This will open your browser to get your API key. You can also find your key at: https://wandb.ai/authorize

### 3. Configure wandb in local_config.yaml

The following wandb settings are available in `local_config.yaml`:

```yaml
# Weights & Biases logging
wandb_enabled: true                    # Enable/disable wandb logging
wandb_project: "yolo11-kitti"         # wandb project name
wandb_entity: null                     # wandb entity (team name), null for personal
wandb_run_name: null                   # custom run name, null for auto-generated
wandb_tags: ["yolo11", "kitti", "detection"]  # tags for the run
wandb_notes: "YOLO11 training on KITTI dataset"  # run description
wandb_save_code: true                  # save code to wandb
wandb_log_model: "best"                # log model artifacts: "all", "best", or false
```

## Configuration Options

### wandb_enabled
- `true`: Enable wandb logging
- `false`: Disable wandb logging (use local logging only)

### wandb_project
- Name of your wandb project
- Projects help organize related experiments

### wandb_entity
- Your wandb username or team name
- Leave as `null` for personal projects

### wandb_run_name
- Custom name for this specific run
- Leave as `null` for auto-generated names based on timestamp

### wandb_tags
- List of tags to help categorize your experiments
- Useful for filtering and searching runs

### wandb_log_model
- `"best"`: Log only the best model weights
- `"all"`: Log both best and last model weights
- `false`: Don't log model artifacts (saves storage)

### wandb_online_only
- `true`: Force online-only mode with no local wandb files. Script will fail if wandb can't connect online
- `false`: Allow offline mode if online connection fails (creates local wandb directory)

## Online-Only Mode (Default)

By default, this script is configured for **online-only** wandb logging:
- No local wandb directories are created
- All logs go directly to wandb cloud
- Script fails immediately if wandb can't connect online
- This prevents cluttering your workspace with local wandb files

If wandb can't connect online, the script will stop with a clear error message.

## What Gets Logged

### Automatically by YOLO
- Training and validation metrics (loss, mAP, precision, recall)
- Learning rate schedules
- Training plots and visualizations
- System information

### Additionally by This Script
- Configuration parameters
- Final metrics summary
- Model artifacts (if enabled)
- Training plots and results

## Usage

1. Set `wandb_enabled: true` in your `local_config.yaml`
2. Configure other wandb settings as needed
3. Run your training script normally: `python src/yolo11/training/train.py`

## Viewing Results

After training starts, you'll see a message like:
```
✓ wandb initialized: https://wandb.ai/your-username/project-name/runs/run-id
```

Click this URL to view your training progress in real-time.

## Troubleshooting

### wandb not installed
```
ERROR: wandb not installed. Install with 'pip install wandb'
```
Solution: Install wandb with `pip install wandb`

### Not logged in or connection failed
```
ERROR: wandb initialization failed: [connection error]
```
Solutions:
1. Run `wandb login` and enter your API key
2. Check your internet connection
3. Verify wandb service is available

### Online-only mode failures
When `wandb_online_only: true` (default), the script will fail if wandb can't connect online:
```
FATAL ERROR: wandb online initialization failed
Training cannot proceed without online wandb logging when wandb_enabled=true
```

**Solutions:**
1. **Fix connection**: Check internet and run `wandb login`
2. **Disable wandb**: Set `wandb_enabled: false` in config for offline training
3. **Allow offline**: Set `wandb_online_only: false` (not recommended - creates local files)

## Offline Mode

If you're training on a system without internet access, you can use wandb offline mode:

```bash
export WANDB_MODE=offline
python src/yolo11/training/train.py
```

After training, sync the results when you have internet access:
```bash
wandb sync wandb/offline-run-*
```

## Tips

1. Use descriptive experiment names and tags to organize your runs
2. Add meaningful notes to document what you're testing
3. Compare runs using wandb's built-in comparison tools
4. Share interesting runs with your team using wandb's sharing features
5. Use wandb sweeps for hyperparameter optimization (advanced)
