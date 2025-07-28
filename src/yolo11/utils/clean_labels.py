#!/usr/bin/env python3
"""
Clean YOLO labels by removing lines with class -1.

This utility removes all lines with class ID -1 from YOLO label files,
which prevents YOLO validation from skipping images due to "negative class labels".

Usage:
    python clean_labels.py --labels_dir ./path/to/labels
    python clean_labels.py --labels_dir ./path/to/labels --backup
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def clean_label_file(label_path: Path, backup: bool = False) -> tuple[int, int]:
    """
    Clean a single label file by removing lines with class -1.
    
    Args:
        label_path: Path to the label file
        backup: Whether to create a backup before cleaning
        
    Returns:
        Tuple of (total_lines, lines_removed)
    """
    if not label_path.exists():
        return 0, 0
    
    # Read original content
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logging.error(f"Error reading {label_path}: {e}")
        return 0, 0
    
    if not lines:
        return 0, 0
    
    # Create backup if requested
    if backup:
        backup_path = label_path.with_suffix('.txt.bak')
        shutil.copy2(label_path, backup_path)
    
    # Filter out lines with class -1
    clean_lines = []
    removed_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 5:
            try:
                class_id = int(parts[0])
                if class_id == -1:
                    removed_count += 1
                    continue
                else:
                    clean_lines.append(line)
            except ValueError:
                # Keep lines that don't start with a valid integer
                clean_lines.append(line)
        else:
            # Keep lines that don't have enough parts (probably comments or empty)
            clean_lines.append(line)
    
    # Write cleaned content back
    try:
        with open(label_path, 'w') as f:
            for line in clean_lines:
                f.write(line + '\n')
    except Exception as e:
        logging.error(f"Error writing {label_path}: {e}")
        return len(lines), 0
    
    return len(lines), removed_count

def clean_labels_directory(labels_dir: Path, backup: bool = False) -> None:
    """
    Clean all label files in a directory.
    
    Args:
        labels_dir: Path to directory containing label files
        backup: Whether to create backups before cleaning
    """
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Find all .txt files
    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        logging.warning(f"No .txt files found in {labels_dir}")
        return
    
    logging.info(f"Found {len(label_files)} label files")
    if backup:
        logging.info("Creating backups (.txt.bak) before cleaning")
    
    total_lines = 0
    total_removed = 0
    files_modified = 0
    
    # Process each file
    for label_file in tqdm(label_files, desc="Cleaning labels"):
        lines_count, removed_count = clean_label_file(label_file, backup)
        total_lines += lines_count
        total_removed += removed_count
        
        if removed_count > 0:
            files_modified += 1
    
    # Print summary
    logging.info(f"\n{'='*50}")
    logging.info(f"CLEANING SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"Files processed: {len(label_files)}")
    logging.info(f"Files modified: {files_modified}")
    logging.info(f"Total lines processed: {total_lines}")
    logging.info(f"Lines with class -1 removed: {total_removed}")
    
    if total_removed > 0:
        logging.info(f"✅ Successfully cleaned {files_modified} files")
        if backup:
            logging.info(f"💾 Backups saved with .bak extension")
    else:
        logging.info(f"✅ No lines with class -1 found - files already clean")

def clean_dataset_recursively(dataset_dir: Path, backup: bool = False) -> None:
    """
    Clean all label files in a YOLO dataset recursively.
    
    Args:
        dataset_dir: Path to YOLO dataset root
        backup: Whether to create backups before cleaning
    """
    logging.info(f"Searching for label directories in {dataset_dir}")
    
    # Look for common YOLO label directory patterns
    label_dirs = []
    
    # Standard YOLO structure
    for split in ["train", "val", "test"]:
        labels_dir = dataset_dir / "labels" / split
        if labels_dir.exists():
            label_dirs.append(labels_dir)
    
    # Also check for labels directory at root level
    root_labels = dataset_dir / "labels"
    if root_labels.exists() and root_labels not in label_dirs:
        label_dirs.append(root_labels)
    
    # Look for any other directories containing .txt files
    for item in dataset_dir.rglob("*.txt"):
        parent_dir = item.parent
        if parent_dir not in label_dirs and "label" in parent_dir.name.lower():
            label_dirs.append(parent_dir)
    
    if not label_dirs:
        logging.warning(f"No label directories found in {dataset_dir}")
        return
    
    # Remove duplicates and sort
    label_dirs = sorted(list(set(label_dirs)))
    
    logging.info(f"Found {len(label_dirs)} label directories:")
    for labels_dir in label_dirs:
        logging.info(f"  - {labels_dir}")
    
    # Clean each directory
    for labels_dir in label_dirs:
        logging.info(f"\nCleaning directory: {labels_dir}")
        clean_labels_directory(labels_dir, backup)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean YOLO labels by removing lines with class -1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean a single labels directory
  python clean_labels.py --labels_dir ./yolo_dataset/labels/train
  
  # Clean entire YOLO dataset recursively
  python clean_labels.py --dataset_dir ./yolo_dataset --recursive
  
  # Create backups before cleaning
  python clean_labels.py --labels_dir ./labels --backup
  
  # Clean with verbose output
  python clean_labels.py --labels_dir ./labels --verbose
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--labels_dir", type=str,
                       help="Path to directory containing label files")
    group.add_argument("--dataset_dir", type=str,
                       help="Path to YOLO dataset root (will search recursively)")
    
    parser.add_argument("--backup", action="store_true",
                       help="Create backup files (.txt.bak) before cleaning")
    parser.add_argument("--recursive", action="store_true",
                       help="When using --dataset_dir, search recursively for label directories")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        if args.labels_dir:
            # Clean single directory
            labels_dir = Path(args.labels_dir)
            clean_labels_directory(labels_dir, args.backup)
        
        elif args.dataset_dir:
            # Clean dataset recursively
            dataset_dir = Path(args.dataset_dir)
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
            
            if args.recursive:
                clean_dataset_recursively(dataset_dir, args.backup)
            else:
                # Just look for standard labels directories
                labels_dir = dataset_dir / "labels"
                if labels_dir.exists():
                    clean_labels_directory(labels_dir, args.backup)
                else:
                    raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        logging.info("✅ Label cleaning completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Cleaning failed: {e}")
        raise

if __name__ == "__main__":
    main()
