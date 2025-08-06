#!/bin/bash --login
#SBATCH --job-name=50split_remap_11x
#SBATCH --partition=general-long-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlier1@msu.edu

# Explicitly request K80 for budget option (longer but cheaper)
#SBATCH --constraint="k80"

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"

# Load necessary modules for older GPUs
module purge
module load GCC/12.3.0
module load Python/3.11.3
module load CUDA/12.1.1  # More compatible with older GPUs

# Print GPU and CUDA information
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo "CUDA Version:"
nvcc --version

# Set environment variables optimized for older GPUs
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH="$HOME/.local/bin:$PATH"

# Budget GPU optimizations
export CUDA_LAUNCH_BLOCKING=1  # More stability for older GPUs
export TORCH_CUDA_ARCH_LIST="3.7;6.0;6.1;7.0"
export YOLO_CACHE_STRATEGY="disk"
export YOLO_BATCH_SIZE=4
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Install PyTorch with CUDA 12.1 support for older GPUs
echo "Installing PyTorch and requirements for budget GPUs..."
pip install --user torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
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

echo "✅ CUDA is working correctly with budget GPU"

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directory for this job
mkdir -p slurm_outputs/$SLURM_JOB_ID
export SLURM_OUTPUT_DIR="slurm_outputs/$SLURM_JOB_ID"

# Set up logging
exec > >(tee -a $SLURM_OUTPUT_DIR/training.log)
exec 2>&1

echo "Starting YOLO training on budget hardware..."
echo "Output directory: $SLURM_OUTPUT_DIR"

# Run the training script with budget GPU optimizations
python src/yolo11/training/train.py

# Check training completion
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Copy results to output directory
    if [ -d "runs/detect/kitti_yolo11s" ]; then
        echo "Copying training results..."
        cp -r runs/detect/kitti_yolo11s $SLURM_OUTPUT_DIR/
        
        # Create summary file
        echo "Budget Training Summary - Job ID: $SLURM_JOB_ID" > $SLURM_OUTPUT_DIR/summary.txt
        echo "Completed at: $(date)" >> $SLURM_OUTPUT_DIR/summary.txt
        echo "Node: $SLURM_NODELIST" >> $SLURM_OUTPUT_DIR/summary.txt
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $SLURM_OUTPUT_DIR/summary.txt
        echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)" >> $SLURM_OUTPUT_DIR/summary.txt
        echo "Training time: $((SECONDS/3600)) hours $((SECONDS%3600/60)) minutes" >> $SLURM_OUTPUT_DIR/summary.txt
        
        # Extract final metrics if available
        if [ -f "runs/detect/kitti_yolo11s/results.csv" ]; then
            echo "" >> $SLURM_OUTPUT_DIR/summary.txt
            echo "Final Metrics:" >> $SLURM_OUTPUT_DIR/summary.txt
            tail -1 runs/detect/kitti_yolo11s/results.csv >> $SLURM_OUTPUT_DIR/summary.txt
        fi
        
        echo "Results saved to: $SLURM_OUTPUT_DIR"
    fi
    
else
    echo "Training failed with exit code $?"
    exit 1
fi

# Final GPU status
echo "Final GPU status:"
nvidia-smi

echo "Job completed at: $(date)"
echo "Total job time: $((SECONDS/3600)) hours $((SECONDS%3600/60)) minutes"

echo "Budget GPU job finished successfully!"
