#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def analyze_dataset_distribution(dataset_path, dataset_name="Dataset"):
    """Analyze class distribution in the dataset"""
    
    # Find image and label directories
    possible_image_dirs = ['images', 'Original', 'Image']
    possible_mask_dirs = ['masks', 'Ground Truth', 'GT', 'GroundTruth', 'labels']
    
    image_dir = None
    mask_dir = None
    
    for img_dir in possible_image_dirs:
        path = os.path.join(dataset_path, img_dir)
        if os.path.exists(path):
            image_dir = path
            break
    
    for msk_dir in possible_mask_dirs:
        path = os.path.join(dataset_path, msk_dir)
        if os.path.exists(path):
            mask_dir = path
            break
    
    if image_dir is None or mask_dir is None:
        print(f"Error: Cannot find image or label directory for {dataset_name}")
        print(f"   Dataset path: {dataset_path}")
        print(f"   Possible image directories: {possible_image_dirs}")
        print(f"   Possible label directories: {possible_mask_dirs}")
        return None
    
    print(f"Analyzing {dataset_name} dataset")
    print(f"   Image directory: {image_dir}")
    print(f"   Label directory: {mask_dir}")
    
    # Get all label files
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        mask_files.extend(glob(os.path.join(mask_dir, ext)))
    
    if len(mask_files) == 0:
        print(f"Error: No label files found in {mask_dir}")
        return None
    
    print(f"   Found {len(mask_files)} label files")
    
    # Count class pixels
    class_counts = defaultdict(int)
    total_pixels = 0
    polyp_areas = []
    image_sizes = []
    
    for i, mask_file in enumerate(mask_files):
        if i % 50 == 0:
            print(f"   Processing: {i+1}/{len(mask_files)}")
        
        # Read label
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        image_sizes.append(mask.shape)
        
        # Count pixels for each class
        unique, counts = np.unique(mask, return_counts=True)
        
        for val, count in zip(unique, counts):
            # Non-zero values as polyp (class 1), zero as background (class 0)
            if val > 0:
                class_counts[1] += count  # Polyp
                polyp_areas.append(count)
            else:
                class_counts[0] += count  # Background
            total_pixels += count
    
    # Calculate ratios
    background_pixels = class_counts[0]
    polyp_pixels = class_counts[1]
    
    background_ratio = background_pixels / total_pixels * 100
    polyp_ratio = polyp_pixels / total_pixels * 100
    
    # Calculate polyp region statistics
    polyp_areas = np.array(polyp_areas)
    avg_polyp_area = np.mean(polyp_areas) if len(polyp_areas) > 0 else 0
    median_polyp_area = np.median(polyp_areas) if len(polyp_areas) > 0 else 0
    
    # Calculate average image size
    heights = [size[0] for size in image_sizes]
    widths = [size[1] for size in image_sizes]
    avg_height = np.mean(heights)
    avg_width = np.mean(widths)
    
    results = {
        'dataset_name': dataset_name,
        'total_images': len(mask_files),
        'total_pixels': total_pixels,
        'background_pixels': background_pixels,
        'polyp_pixels': polyp_pixels,
        'background_ratio': background_ratio,
        'polyp_ratio': polyp_ratio,
        'avg_image_size': (avg_height, avg_width),
        'polyp_images': len(polyp_areas),
        'avg_polyp_area': avg_polyp_area,
        'median_polyp_area': median_polyp_area,
        'class_imbalance_ratio': background_pixels / polyp_pixels if polyp_pixels > 0 else float('inf')
    }
    
    return results

def print_analysis_results(results):
    """Print analysis results"""
    if results is None:
        return
    
    print(f"\n{results['dataset_name']} Class Distribution Analysis Results")
    print("=" * 60)
    print(f"Dataset Statistics:")
    print(f"   Total images: {results['total_images']:,}")
    print(f"   Total pixels: {results['total_pixels']:,}")
    print(f"   Average image size: {results['avg_image_size'][0]:.0f} x {results['avg_image_size'][1]:.0f}")
    
    print(f"\nClass Distribution:")
    print(f"   Background pixels: {results['background_pixels']:,} ({results['background_ratio']:.2f}%)")
    print(f"   Polyp pixels: {results['polyp_pixels']:,} ({results['polyp_ratio']:.2f}%)")
    print(f"   Class imbalance ratio: {results['class_imbalance_ratio']:.1f}:1 (background:polyp)")
    
    print(f"\nPolyp Region Statistics:")
    print(f"   Images with polyps: {results['polyp_images']:,}")
    print(f"   Average polyp region size: {results['avg_polyp_area']:.0f} pixels")
    print(f"   Median polyp region size: {results['median_polyp_area']:.0f} pixels")
    
    # Training recommendations
    print(f"\nTraining Recommendations:")
    if results['class_imbalance_ratio'] > 50:
        print(f"   WARNING: Severe class imbalance! Recommend using weighted loss")
        print(f"   Recommended polyp class weight: {results['class_imbalance_ratio']/10:.1f}")
    elif results['class_imbalance_ratio'] > 20:
        print(f"   WARNING: Moderate class imbalance, recommend using Focal Loss")
        print(f"   Recommended polyp class weight: {results['class_imbalance_ratio']/20:.1f}")
    else:
        print(f"   Class distribution is relatively balanced")
    
    if results['polyp_ratio'] < 5:
        print(f"   Polyp pixel ratio is very small ({results['polyp_ratio']:.2f}%), recommend:")
        print(f"      - Use Dice Loss")
        print(f"      - Increase data augmentation")
        print(f"      - Use hard example mining")

def analyze_phase3_datasets():
    """Analyze datasets used in Phase 3 training"""
    print("FD-MSP Phase 3 Dataset Class Distribution Analysis")
    print("=" * 70)
    
    datasets = [
        {
            'name': 'EndoTect (Source Domain)',
            'path': 'datasets/EndoTect'
        },
        {
            'name': 'CVC-ClinicDB (Target Domain)',
            'path': 'datasets/CVC-ClinicDB'
        }
    ]
    
    all_results = []
    
    for dataset in datasets:
        if os.path.exists(dataset['path']):
            print(f"\n{'='*50}")
            results = analyze_dataset_distribution(dataset['path'], dataset['name'])
            if results:
                print_analysis_results(results)
                all_results.append(results)
        else:
            print(f"\nError: Dataset path does not exist: {dataset['path']}")
    
    # Combined analysis
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("Domain Adaptation Analysis")
        print("=" * 70)
        
        source_results = all_results[0]
        target_results = all_results[1]
        
        print(f"Source vs Target Domain Comparison:")
        print(f"   Source domain polyp ratio: {source_results['polyp_ratio']:.2f}%")
        print(f"   Target domain polyp ratio: {target_results['polyp_ratio']:.2f}%")
        print(f"   Ratio difference: {abs(source_results['polyp_ratio'] - target_results['polyp_ratio']):.2f}%")
        
        ratio_diff = abs(source_results['class_imbalance_ratio'] - target_results['class_imbalance_ratio'])
        print(f"   Imbalance ratio difference: {ratio_diff:.1f}")
        
        if ratio_diff > 10:
            print("   WARNING: Large class distribution difference between domains")
            print("   Careful adjustment of domain adaptation strategy needed")
        else:
            print("   Class distribution is relatively consistent between domains")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Analyze class distribution in polyp segmentation datasets')
    parser.add_argument('--dataset_path', type=str, help='Dataset path')
    parser.add_argument('--phase3', action='store_true', help='Analyze all datasets used in Phase 3')
    
    args = parser.parse_args()
    
    if args.phase3:
        analyze_phase3_datasets()
    elif args.dataset_path:
        results = analyze_dataset_distribution(args.dataset_path)
        print_analysis_results(results)
    else:
        # Default: analyze Phase 3 datasets
        analyze_phase3_datasets()

if __name__ == '__main__':
    main()
