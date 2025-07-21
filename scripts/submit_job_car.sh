#!/bin/bash
# Check if dataset exists
if [ ! -f "$(python -c "from src.yolo11.utils.config_utils import get_dataset_config; print(get_dataset_config('car')['data_yaml'])")" ]; then
    echo "WARNING: Car dataset not found! Check your local_config.yaml file."
    echo "Make sure to run the KITTI car converter first and update your configuration!"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."submission script for YOLO car training

echo "Submitting YOLO car training job to SLURM..."
echo "Job configuration:"
echo "  - 1x GPU (general-long-gpu partition)"
echo "  - 8 CPU cores"
echo "  - 64GB RAM"
echo "  - 8 hour time limit"
echo ""


# Check if car dataset exists
if [ ! -f "$(python -c "from src.yolo11.utils.config_utils import get_dataset_config; print(get_dataset_config('car')['data_yaml'])" 2>/dev/null)" ]; then
    echo "WARNING: Car dataset not found! Check your local_config.yaml file."
    echo "Make sure to run the KITTI car converter first and update your configuration!"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 1
    fi
fi

# Submit the car training job
JOB_ID=$(sbatch train_yolo_car.slurm | grep -o '[0-9]*')

if [ ! -z "$JOB_ID" ]; then
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor your job with:"
    echo "  squeue -u $USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "Check job status:"
    echo "  scontrol show job $JOB_ID"
    echo ""
    echo "View logs (once job starts):"
    echo "  tail -f yolo11s_car_training_${JOB_ID}.out"
    echo "  tail -f yolo11s_car_training_${JOB_ID}.err"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $JOB_ID"
else
    echo "Job submission failed!"
    exit 1
fi
