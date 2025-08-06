#!/bin/bash
# Quick submission script for YOLO training

echo "Submitting YOLO training job to SLURM..."


# Check if dataset exists
export PYTHONPATH=$(pwd)
DATASET_YAML=$(python -c "from src.yolo11.utils.config_utils import get_dataset_config; print(get_dataset_config('default')['data_yaml'])" 2>/dev/null | tail -n 1 | tr -d ' \n')
if [ ! -f "$DATASET_YAML" ]; then
    echo "WARNING: Dataset not found! Check your local_config.yaml file."
    echo "Make sure to run the KITTI converter first and update your configuration!"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 1
    fi
fi

# Submit the training job
JOB_ID=$(sbatch scripts/train_yolo.slurm | grep -o '[0-9]*')

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
    echo "  tail -f yolo11s_training_${JOB_ID}.out"
    echo "  tail -f yolo11s_training_${JOB_ID}.err"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $JOB_ID"
else
    echo "Job submission failed!"
    exit 1
fi
