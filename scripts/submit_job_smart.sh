#!/bin/bash

# YOLO Training Job Submission Helper
# Automatically chooses the best GPU configuration based on queue status and requirements

echo "=== YOLO Training Job Submission Helper ==="
echo ""

# Check current queue status
total_queue=$(squeue | wc -l)
gpu_queue=$(squeue -p general-long-gpu | wc -l)

echo "Current queue status:"
echo "  Total jobs in queue: $((total_queue-1))"
echo "  GPU jobs in queue: $((gpu_queue-1))"

# Check GPU availability
echo ""
echo "Checking GPU availability..."

# Check for premium GPUs (A100, H200)
premium_available=$(sinfo -p general-long-gpu -t idle -h --format="%f %D" | grep -E "a100|h200" | awk '{sum+=$2} END {print sum+0}')

# Check for mid-tier GPUs (L40S, V100S, V100)
midtier_available=$(sinfo -p general-long-gpu -t idle -h --format="%f %D" | grep -E "l40s|v100s|v100" | awk '{sum+=$2} END {print sum+0}')

# Check for budget GPUs (K80)
budget_available=$(sinfo -p general-long-gpu -t idle -h --format="%f %D" | grep -E "k80" | awk '{sum+=$2} END {print sum+0}')

echo "  Premium GPUs (A100, H200) available: $premium_available"
echo "  Mid-tier GPUs (L40S, V100S, V100) available: $midtier_available"
echo "  Budget GPUs (K80) available: $budget_available"

echo ""
echo "Available submission options:"
echo "  1. Premium GPU (A100/H200) - Fastest training (~2-4 hours)"
echo "  2. Standard GPU (Auto-select) - Balanced performance (~4-8 hours)"
echo "  3. Budget GPU (K80) - Longer training but cheaper (~8-12 hours)"
echo "  4. Cancel"

echo ""
read -p "Choose your option (1-4): " choice

case $choice in
    1)
        if [ "$premium_available" -gt 0 ]; then
            echo "Submitting to premium GPU queue..."
            sbatch scripts/train_yolo_premium.slurm
        else
            echo "No premium GPUs available. Would you like to submit anyway and wait in queue? (y/n)"
            read -p "Choice: " wait_choice
            if [[ $wait_choice == "y" || $wait_choice == "Y" ]]; then
                sbatch scripts/train_yolo_premium.slurm
            else
                echo "Submission cancelled."
            fi
        fi
        ;;
    2)
        echo "Submitting to standard GPU queue with auto-selection..."
        sbatch scripts/train_yolo.slurm
        ;;
    3)
        echo "Submitting to budget GPU queue..."
        sbatch scripts/train_yolo_budget.slurm
        ;;
    4)
        echo "Submission cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "Job submitted! Check status with: squeue -u $USER"
echo "Monitor output with: tail -f slurm_outputs/\$JOBID/training.log"
