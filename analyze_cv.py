#!/usr/bin/env python
"""
Cross-Validation Results Analysis and Visualization
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from src.yolo11.utils.config_utils import get_config

# Get configuration
config = get_config()

def load_cv_results(results_dir=None):
    """Load cross-validation results from JSON file."""
    if results_dir is None:
        results_dir = config.get('cv_results_dir', 'runs/cross_validation')
    
    results_path = Path(results_dir) / "cross_validation_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results

def plot_cv_metrics(results, save_dir=None):
    """Plot cross-validation metrics across folds."""
    if save_dir is None:
        save_dir = config.get('cv_results_dir', 'runs/cross_validation')
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract fold results
    fold_results = results['fold_results']
    df = pd.DataFrame(fold_results)
    
    # Define metrics to plot
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
    loss_metrics = ['box_loss', 'cls_loss', 'dfl_loss']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cross-Validation Metrics Across Folds', fontsize=16)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        if metric in df.columns:
            values = df[metric].values
            folds = df['fold'].values + 1  # 1-indexed for display
            
            # Bar plot
            bars = ax.bar(folds, values, alpha=0.7, color=f'C{i}')
            
            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.4f}')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric.upper()}')
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cv_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot loss metrics separately
    if any(metric in df.columns for metric in loss_metrics):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Cross-Validation Loss Metrics Across Folds', fontsize=16)
        
        for i, metric in enumerate(loss_metrics):
            if metric in df.columns:
                values = df[metric].values
                folds = df['fold'].values + 1
                
                bars = axes[i].bar(folds, values, alpha=0.7, color=f'C{i+4}')
                
                mean_val = np.mean(values)
                axes[i].axhline(y=mean_val, color='red', linestyle='--',
                               label=f'Mean: {mean_val:.4f}')
                
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{val:.3f}', ha='center', va='bottom')
                
                axes[i].set_title(f'{metric.upper()}')
                axes[i].set_xlabel('Fold')
                axes[i].set_ylabel(metric)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'cv_losses.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_metric_distributions(results, save_dir="runs/cross_validation"):
    """Plot distributions of metrics across folds."""
    save_dir = Path(save_dir)
    
    fold_results = results['fold_results']
    df = pd.DataFrame(fold_results)
    
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Metric Distributions Across Folds', fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        if metric in df.columns:
            values = df[metric].values
            
            # Box plot
            bp = ax.boxplot(values, labels=[metric], patch_artist=True)
            bp['boxes'][0].set_facecolor(f'C{i}')
            bp['boxes'][0].set_alpha(0.7)
            
            # Add individual points
            ax.scatter([1] * len(values), values, alpha=0.7, color='red', s=50)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.text(1.2, mean_val, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                   verticalalignment='center')
            
            ax.set_title(f'{metric.upper()} Distribution')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(results):
    """Print a summary table of cross-validation results."""
    print("=" * 80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    
    # Basic info
    print(f"Number of folds: {results['n_folds']}")
    print(f"Random seed: 42")  # Assuming default seed
    print()
    
    # Fold-by-fold results
    print("FOLD-BY-FOLD RESULTS:")
    print("-" * 80)
    
    fold_results = results['fold_results']
    df = pd.DataFrame(fold_results)
    
    # Format the dataframe for display
    if not df.empty:
        display_df = df.copy()
        display_df['fold'] = display_df['fold'] + 1  # 1-indexed
        
        # Round numeric columns
        numeric_cols = ['mAP50', 'mAP50_95', 'precision', 'recall', 
                       'box_loss', 'cls_loss', 'dfl_loss']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        print(display_df.to_string(index=False))
    
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS:")
    print("-" * 80)
    
    mean_metrics = results['mean_metrics']
    std_metrics = results['std_metrics']
    
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    
    for metric in ['mAP50', 'mAP50_95', 'precision', 'recall']:
        if metric in mean_metrics:
            mean_val = mean_metrics[metric]
            std_val = std_metrics.get(metric, 0)
            
            # Calculate min/max from fold results
            values = [fold.get(metric, 0) for fold in fold_results]
            min_val = min(values)
            max_val = max(values)
            
            print(f"{metric:<15} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
    
    print()
    
    # Best and worst folds
    if results.get('best_fold'):
        best_fold = results['best_fold']
        worst_fold = results['worst_fold']
        
        print("BEST AND WORST PERFORMING FOLDS:")
        print("-" * 80)
        print(f"Best fold: {best_fold['fold'] + 1} (mAP50: {best_fold['mAP50']:.4f})")
        print(f"Worst fold: {worst_fold['fold'] + 1} (mAP50: {worst_fold['mAP50']:.4f})")
        print(f"Performance range: {best_fold['mAP50'] - worst_fold['mAP50']:.4f}")
    
    print()
    print("=" * 80)

def analyze_fold_variance(results, save_dir="runs/cross_validation"):
    """Analyze variance across folds to assess model stability."""
    save_dir = Path(save_dir)
    
    fold_results = results['fold_results']
    df = pd.DataFrame(fold_results)
    
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
    
    print("FOLD VARIANCE ANALYSIS:")
    print("-" * 50)
    
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv_val = (std_val / mean_val) * 100 if mean_val > 0 else 0
            
            print(f"{metric:<15} CV: {cv_val:.2f}%")
    
    print()
    
    # Stability assessment
    mAP50_cv = (results['std_metrics'].get('mAP50', 0) / 
                results['mean_metrics'].get('mAP50', 1)) * 100
    
    if mAP50_cv < 5:
        stability = "Very Stable"
    elif mAP50_cv < 10:
        stability = "Stable"
    elif mAP50_cv < 20:
        stability = "Moderately Stable"
    else:
        stability = "Unstable"
    
    print(f"Model Stability Assessment: {stability} (mAP50 CV: {mAP50_cv:.2f}%)")
    print()

def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze cross-validation results")
    parser.add_argument("--results-dir", default="runs/cross_validation",
                       help="Directory containing cross-validation results")
    
    args = parser.parse_args()
    
    try:
        # Load results
        results = load_cv_results(args.results_dir)
        
        # Print summary table
        print_summary_table(results)
        
        # Analyze variance
        analyze_fold_variance(results, args.results_dir)
        
        # Create visualizations
        plot_cv_metrics(results, args.results_dir)
        plot_metric_distributions(results, args.results_dir)
        
        print("Analysis complete! Check the results directory for plots.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
