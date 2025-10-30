"""
Visual verification script for segmentations.
Shows RGB image overlaid with instance/semantic masks to verify correct labeling.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def overlay_instance_mask(rgb, inst_mask, alpha=0.5):
    """Overlay instance mask with random colors per instance"""
    overlay = rgb.copy()
    unique_ids = np.unique(inst_mask)
    
    # Generate random colors for each instance
    np.random.seed(42)
    colors = {}
    for iid in unique_ids:
        if iid == 0 or iid == 1:  # Skip background/unlabelled
            continue
        colors[iid] = np.random.randint(0, 255, 3, dtype=np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(rgb)
    for iid, color in colors.items():
        mask = inst_mask == iid
        colored_mask[mask] = color
    
    # Blend
    result = cv2.addWeighted(rgb, 1-alpha, colored_mask, alpha, 0)
    
    # Add labels
    for iid, color in colors.items():
        ys, xs = np.where(inst_mask == iid)
        if len(ys) > 0:
            cy, cx = int(ys.mean()), int(xs.mean())
            cv2.putText(result, f"ID:{iid}", (cx-20, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result, colors

def overlay_semantic_mask(rgb, sem_mask, alpha=0.5):
    """Overlay semantic mask with predefined colors"""
    overlay = rgb.copy()
    
    # Define colors for known classes
    sem_colors = {
        0: (0, 0, 0),      # BACKGROUND - black
        1: (128, 128, 128), # UNLABELLED - gray
        2: (139, 69, 19),   # lane - brown
        3: (200, 200, 200), # ground - light gray
        4: (255, 0, 255),   # ball - magenta
    }
    
    colored_mask = np.zeros_like(rgb)
    for sid, color in sem_colors.items():
        mask = sem_mask == sid
        colored_mask[mask] = color
    
    result = cv2.addWeighted(rgb, 1-alpha, colored_mask, alpha, 0)
    return result

def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation masks")
    parser.add_argument("--data_dir", default="dataset_v1_debug", 
                       help="Directory containing rgb/instance/semantic files")
    parser.add_argument("--frame", type=int, default=0, 
                       help="Frame number to visualize")
    parser.add_argument("--save", action="store_true", 
                       help="Save visualizations instead of displaying")
    args = parser.parse_args()
    
    # Build file paths
    frame_id = f"{args.frame:04d}"
    rgb_path = os.path.join(args.data_dir, f"rgb_{frame_id}.png")
    inst_path = os.path.join(args.data_dir, f"instance_segmentation_{frame_id}.png")
    sem_path = os.path.join(args.data_dir, f"semantic_segmentation_{frame_id}.png")
    
    # Check files exist
    for path in [rgb_path, inst_path, sem_path]:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            return
    
    # Load images
    rgb = cv2.imread(rgb_path)
    inst = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
    sem = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
    
    if inst.ndim == 3:
        inst = inst[:, :, 0]  # Take first channel if multichannel
    if sem.ndim == 3:
        sem = sem[:, :, 0]
    
    print(f"Frame {frame_id}:")
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Instance IDs: {np.unique(inst)}")
    print(f"  Semantic IDs: {np.unique(sem)}")
    
    # Create overlays
    inst_overlay, inst_colors = overlay_instance_mask(rgb, inst, alpha=0.5)
    sem_overlay = overlay_semantic_mask(rgb, sem, alpha=0.5)
    
    # Create combined view
    h, w = rgb.shape[:2]
    combined = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    combined[0:h, 0:w] = rgb
    combined[0:h, w:2*w] = inst_overlay
    combined[h:2*h, 0:w] = sem_overlay
    
    # Add labels
    cv2.putText(combined, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Instance Segmentation", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Semantic Segmentation", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Legend for instance
    y_offset = h + 60
    cv2.putText(combined, "Instance Legend:", (w+10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for idx, (iid, color) in enumerate(inst_colors.items()):
        y_pos = y_offset + 30 + idx*25
        cv2.rectangle(combined, (w+10, y_pos-15), (w+40, y_pos+5), color.tolist(), -1)
        cv2.putText(combined, f"ID {iid}", (w+50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if args.save:
        out_path = f"segmentation_check_frame_{frame_id}.jpg"
        cv2.imwrite(out_path, combined)
        print(f"\nSaved visualization to: {out_path}")
    else:
        # Display
        scale = 0.5
        display = cv2.resize(combined, None, fx=scale, fy=scale)
        cv2.imshow("Segmentation Check", display)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

