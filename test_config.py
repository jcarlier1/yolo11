#!/usr/bin/env python
"""
Test script to verify the configuration system works properly
"""

from src.yolo11.utils.config_utils import get_config, get_dataset_config, verify_paths

def test_configuration():
    """Test the configuration system."""
    print("=== Testing Configuration System ===\n")
    
    # Get configuration
    config = get_config()
    
    # Print current configuration
    config.print_config()
    
    # Test dataset configurations
    print("\n=== Default Dataset Configuration ===")
    default_dataset = get_dataset_config('default')
    for key, path in default_dataset.items():
        print(f"  {key}: {path}")
    
    print("\n=== Car Dataset Configuration ===")
    car_dataset = get_dataset_config('car')
    for key, path in car_dataset.items():
        print(f"  {key}: {path}")
    
    # Test path verification (without checking if they exist)
    print("\n=== Path Verification Test (config only) ===")
    test_paths = ['yolo_root', 'data_yaml', 'train_images', 'val_images']
    verify_paths(test_paths, check_exists=False)
    
    print("\n=== Configuration Test Complete ===")

if __name__ == "__main__":
    test_configuration()
