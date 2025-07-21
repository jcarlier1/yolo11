#!/bin/bash --login
#SBATCH --job-name=yolo11s_car_training
#SBATCH --partition=general-long-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"

# Load necessary modules
module purge
module load GCC/13.2.0
module load Python/3.11.5
module load CUDA/12.6.0

# Print GPU and CUDA information
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo "CUDA Version:"
nvcc --version

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH="$HOME/.local/bin:$PATH"

# Install PyTorch with CUDA 12.6 support
echo "Installing PyTorch and requirements..."
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install --user ultralytics opencv-python pyyaml pandas matplotlib seaborn

# Verify installation
echo "Verifying YOLO installation..."
python -c "from ultralytics import YOLO; print('YOLO imported successfully')"

# Print Python and PyTorch info
echo "Python version: $(python --version)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('Number of GPUs:', torch.cuda.device_count())"
python -c "import torch; print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Exit if CUDA is not available
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "❌ CUDA is not available - exiting"
    exit 1
fi

echo "✅ CUDA is working correctly"

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directory for this job
mkdir -p slurm_outputs/$SLURM_JOB_ID
export SLURM_OUTPUT_DIR="slurm_outputs/$SLURM_JOB_ID"

# Set up logging
exec > >(tee -a $SLURM_OUTPUT_DIR/training.log)
exec 2>&1

echo "Starting YOLO training..."
echo "Output directory: $SLURM_OUTPUT_DIR"


# Run the training script for car detection with YOLO11s
python train_cars.py

# Check training completion
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Copy results to output directory
    if [ -d "runs/detect/kitti_car_yolo11s" ]; then
        echo "Copying training results..."
        cp -r runs/detect/kitti_car_yolo11s $SLURM_OUTPUT_DIR/
        
        # Create summary file
        echo "Training Summary - Job ID: $SLURM_JOB_ID" > $SLURM_OUTPUT_DIR/summary.txt
        echo "Completed at: $(date)" >> $SLURM_OUTPUT_DIR/summary.txt
        echo "Node: $SLURM_NODELIST" >> $SLURM_OUTPUT_DIR/summary.txt
        echo "GPUs used: 1" >> $SLURM_OUTPUT_DIR/summary.txt
        echo "Training time: $((SECONDS/3600)) hours $((SECONDS%3600/60)) minutes" >> $SLURM_OUTPUT_DIR/summary.txt
        
        # Extract final metrics if available
        if [ -f "runs/detect/kitti_car_yolo11s/results.csv" ]; then
            echo "" >> $SLURM_OUTPUT_DIR/summary.txt
            echo "Final Metrics:" >> $SLURM_OUTPUT_DIR/summary.txt
            tail -1 runs/detect/kitti_car_yolo11s/results.csv >> $SLURM_OUTPUT_DIR/summary.txt
        fi
        
        echo "Results saved to: $SLURM_OUTPUT_DIR"
    fi
    
    # Send success notification
    echo "YOLO training completed successfully on job $SLURM_JOB_ID" | mail -s "YOLO Training Complete" your-email@example.com
    
else
    echo "Training failed with exit code $?"
    
    # Copy error logs
    cp $SLURM_OUTPUT_DIR/training.log $SLURM_OUTPUT_DIR/error_log.txt
    
    # Send failure notification
    echo "YOLO training failed on job $SLURM_JOB_ID. Check logs at: $SLURM_OUTPUT_DIR" | mail -s "YOLO Training Failed" your-email@example.com
    
    exit 1
fi

# Final GPU status
echo "Final GPU status:"
nvidia-smi

echo "Job completed at: $(date)"
echo "Total job time: $((SECONDS/3600)) hours $((SECONDS%3600/60)) minutes"

# Clean up temporary files if needed
# rm -rf /tmp/yolo_cache_*

echo "Job finished successfully!"
