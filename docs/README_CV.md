# Cross-Validation for YOLO Object Detection

This implementation provides a comprehensive k-fold cross-validation system for YOLO object detection models, specifically designed for the KITTI dataset.

## Features

- **Stratified K-Fold**: Ensures balanced class distribution across folds
- **Automated Data Management**: Handles train/val splits for each fold
- **Comprehensive Metrics**: Tracks mAP@0.5, mAP@0.5:0.95, precision, recall, and loss metrics
- **Results Aggregation**: Provides mean and standard deviation across folds
- **Visualization**: Generates plots for easy analysis
- **Reproducibility**: Consistent results with fixed random seeds

## Files

- `cross_validation.py`: Main cross-validation implementation
- `train_cv.py`: Enhanced training script with CV support
- `analyze_cv.py`: Results analysis and visualization

## Usage

### 1. Basic Cross-Validation

```bash
# Run 5-fold cross-validation with default parameters
python train_cv.py --mode cv

# Run 10-fold cross-validation with custom epochs
python train_cv.py --mode cv --k-folds 10 --epochs 50

# Run with custom batch size
python train_cv.py --mode cv --batch 16 --epochs 100
```

### 2. Single Split Training (Original)

```bash
# Train with original train/val split
python train_cv.py --mode single
```

### 3. Testing

```bash
# Test specific weights
python train_cv.py --mode test --weights path/to/weights.pt

# Test best model from cross-validation
python train_cv.py --mode test-cv
```

### 4. Results Analysis

```bash
# Analyze cross-validation results
python analyze_cv.py --results-dir runs/cross_validation
```

## Implementation Details

### Cross-Validation Strategy

1. **Data Preparation**: Combines original train and validation sets for k-fold splitting
2. **Stratification**: Uses dominant class per image for balanced splits
3. **Fold Creation**: Creates separate directories for each fold with proper YOLO structure
4. **Training**: Trains independent models for each fold
5. **Aggregation**: Computes statistics across all folds

### Key Parameters

- `k_folds`: Number of folds (default: 5)
- `epochs`: Training epochs per fold (default: 100, reduced from 500)
- `batch`: Batch size (default: 32, reduced for multiple folds)
- `patience`: Early stopping patience (default: 20)
- `seed`: Random seed for reproducibility (default: 42)

### Output Structure

```
runs/cross_validation/
├── fold_0/
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   ├── dataset.yaml
│   └── weights/
├── fold_1/
│   └── ...
├── cross_validation_results.json
├── fold_results.csv
├── cv_metrics.png
└── metric_distributions.png
```

## Advantages of This Approach

1. **Robust Evaluation**: Better estimate of model performance
2. **Stability Assessment**: Understand model variance across different data splits
3. **Hyperparameter Tuning**: More reliable parameter selection
4. **Publication Ready**: Provides error bars and statistical significance
5. **Model Selection**: Identifies best-performing fold for final deployment

## Best Practices

1. **Resource Management**: 
   - Reduce epochs per fold (50-100 instead of 500)
   - Adjust batch size based on available GPU memory
   - Use fewer workers for parallel training

2. **Evaluation**:
   - Use mAP@0.5 as primary metric for object detection
   - Consider mAP@0.5:0.95 for more stringent evaluation
   - Monitor variance across folds for stability

3. **Final Model**:
   - Use best fold for final testing
   - Consider ensemble of top folds for better performance
   - Retrain on full dataset with best hyperparameters

## Expected Results

For a well-tuned model, you should expect:
- **Low variance** across folds (CV < 10%)
- **Consistent performance** across different data splits
- **Reliable metrics** for publication or deployment

## Troubleshooting

1. **Memory Issues**: Reduce batch size or use fewer workers
2. **Slow Training**: Reduce epochs or use smaller image size
3. **Unstable Results**: Check data quality and class balance
4. **File Not Found**: Ensure correct data paths in configuration

## Example Output

```
Cross-Validation Summary:
Number of folds: 5
  mAP50: 0.8245 ± 0.0156
  mAP50_95: 0.5123 ± 0.0234
  precision: 0.7892 ± 0.0198
  recall: 0.8156 ± 0.0145
  Best fold: 2 (mAP50: 0.8456)
  Worst fold: 4 (mAP50: 0.7998)
```

This provides a much more robust evaluation than a single train/val split!
