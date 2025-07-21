# YOLO11: Fine-Tuning YOLO for KITTI Object Detection

This repository provides a suite of tools and scripts for fine-tuning YOLO models on the KITTI object detection benchmark. It covers the full workflow from dataset conversion to model training, evaluation, and cross-validation.

## Features
- **Dataset Conversion**: Convert KITTI dataset annotations to YOLO format for seamless training.
- **Data Splitting**: Utilities for creating training, validation, and test splits tailored for KITTI.
- **Training Scripts**: Run training jobs for YOLO models, supporting both all-label and car-only modalities.
- **Evaluation Tools**: Assess model performance with provided scripts and visualization outputs.
- **K-Fold Cross Validation**: Built-in support for robust cross-validation experiments.

## Getting Started
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Convert KITTI Dataset**
   - Convert KITTI annotations to YOLO format using:
   ```bash
   python src/yolo11/utils/kitti_converter.py --kitti_root /path/to/kitti --yolo_root /path/to/output_yolo --train_split 0.8
   ```

3. **Configure Data Splits**
   - Data splits are handled automatically by the converter above. For custom splits or k-folds, use:
   ```bash
   python src/yolo11/utils/cross_validation.py --data_root /path/to/output_yolo --k_folds 5
   ```

4. **Train YOLO Models**
   - Submit a training job for all labels:
   ```bash
   bash scripts/submit_job.sh
   ```
   - Submit a training job for car label only:
   ```bash
   bash scripts/submit_job_car.sh
   ```

5. **Evaluate & Visualize Results**
   - After training, analyze results and visualize metrics:
   ```bash
   # Open and run the notebook for training analysis
   jupyter notebook src/yolo11/evaluation/training_csv_analysis.ipynb
   # Or view results in runs/detect/ and runs/cross_validation/
   ```

6. **Cross Validation**
   - Submit a k-fold cross-validation job:
   ```bash
   bash scripts/submit_job_cv.sh
   ```
   - Or run the SLURM script directly:
   ```bash
   sbatch scripts/train_yolo_cv.slurm
   ```

## Directory Structure
- `src/yolo11/` - Core source code for conversion, training, evaluation, and utilities.
- `scripts/` - Shell and SLURM scripts for hpcc job submission and training.
- `runs/` - Output results, including cross-validation and detection runs.
- `docs/` - Documentation and configuration templates.
- `tests/` - Unit tests for configuration and utilities.

## Notes
- See `README_CONVERTER.md` and other docs for detailed usage of specific tools.
- Example config files and templates are available in `docs/`.

## License
This project is maintained by JuanEstebanCarlier. See repository for license details.
