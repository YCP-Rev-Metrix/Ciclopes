#!/usr/bin/env python3
"""Visualize YOLO polygon annotations overlaid on images"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def load_yolo_label(label_path):
    """Load YOLO format label file. Returns list of (class_id, polygon_points)"""
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            # Reshape to Nx2
            points = np.array(coords).reshape(-1, 2)
            objects.append((class_id, points))
    return objects

def visualize_yolo_annotations(img_path, label_path, class_names, output_path=None):
    """Draw YOLO polygon annotations on image"""
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error loading image: {img_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Create overlay
    overlay = img.copy()
    
    # Colors for each class (BGR format)
    colors = {
        0: (0, 255, 0),      # Lane: Green
        1: (255, 0, 0),      # Ball: Blue
    }
    
    # Load labels
    if not label_path.exists():
        print(f"No label file found: {label_path}")
        return img
    
    objects = load_yolo_label(label_path)
    
    for class_id, norm_poly in objects:
        # Denormalize polygon coordinates
        poly = norm_poly.copy()
        poly[:, 0] *= w
        poly[:, 1] *= h
        poly = poly.astype(np.int32)
        
        # Draw filled polygon with transparency
        color = colors.get(class_id, (255, 255, 255))
        cv2.fillPoly(overlay, [poly], color)
        
        # Draw polygon outline
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=3)
        
        # Draw class label at centroid
        centroid = poly.mean(axis=0).astype(int)
        class_name = class_names.get(class_id, f"Class {class_id}")
        cv2.putText(img, class_name, tuple(centroid), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Blend overlay with original image
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"Saved visualization to: {output_path}")
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO polygon annotations')
    parser.add_argument('--dataset_dir', required=True, help='YOLO dataset directory')
    parser.add_argument('--split', default='train', choices=['train', 'val'], 
                        help='Dataset split to visualize')
    parser.add_argument('--num_samples', type=int, default=3, 
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', default='yolo_viz_checks', 
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    img_dir = dataset_dir / 'images' / args.split
    lbl_dir = dataset_dir / 'labels' / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    class_names = {
        0: 'lane',
        1: 'ball',
    }
    
    # Get all images
    images = sorted(img_dir.glob('*.jpg'))
    
    if not images:
        print(f"No images found in {img_dir}")
        return
    
    print(f"Found {len(images)} images in {args.split} split")
    print(f"Visualizing {min(args.num_samples, len(images))} samples...\n")
    
    for i, img_path in enumerate(images[:args.num_samples]):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        out_path = output_dir / f"{args.split}_{stem}_check.jpg"
        
        print(f"{i+1}. {stem}")
        
        # Check if label exists
        if not lbl_path.exists():
            print(f"   WARNING: No label file found!")
            continue
        
        # Load and show label info
        objects = load_yolo_label(lbl_path)
        print(f"   Objects: {len(objects)}")
        for cls_id, poly in objects:
            print(f"     - Class {cls_id} ({class_names.get(cls_id, 'unknown')}): {len(poly)} points")
        
        # Visualize
        visualize_yolo_annotations(img_path, lbl_path, class_names, out_path)
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("Please review the images to verify polygons match the objects!")

if __name__ == '__main__':
    main()

