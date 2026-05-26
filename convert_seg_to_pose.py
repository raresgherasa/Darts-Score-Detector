import os
import sys
import shutil
import math
from pathlib import Path
import cv2
import numpy as np

# Regulation board geometry and sectors
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

def dart_tip_from_endpoints(poly, end_a, end_b):
    n = len(poly)
    k = max(2, n // 12)
    def _angle(ep):
        idx = int(np.argmin((poly[:, 0] - ep[0]) ** 2 + (poly[:, 1] - ep[1]) ** 2))
        v1 = poly[(idx - k) % n] - poly[idx]
        v2 = poly[(idx + k) % n] - poly[idx]
        d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if d1 < 1e-6 or d2 < 1e-6:
            return math.pi
        return math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / (d1 * d2))))
    return end_a if _angle(end_a) <= _angle(end_b) else end_b

def main():
    src_dir = Path(__file__).parent / 'yolov11 dataset'
    dst_dir = Path(__file__).parent / 'yolov11_pose_dataset'
    
    if not src_dir.is_dir():
        print(f"Source dataset directory not found: {src_dir}")
        sys.exit(1)
        
    print(f"Converting segmentation dataset {src_dir} to pose dataset {dst_dir}...")
    
    # Create directory structure
    for split in ['train', 'valid', 'test']:
        (dst_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
    # Copy images and process labels
    for split in ['train', 'valid', 'test']:
        src_lbl_dir = src_dir / split / 'labels'
        src_img_dir = src_dir / split / 'images'
        
        if not src_lbl_dir.is_dir() or not src_img_dir.is_dir():
            print(f"Warning: split {split} has missing image or label directory. Skipping.")
            continue
            
        label_files = list(src_lbl_dir.glob('*.txt'))
        print(f"Processing split '{split}': {len(label_files)} label files...")
        
        for lbl_path in label_files:
            # Find corresponding image
            img_name = lbl_path.stem
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                temp_path = src_img_dir / (img_name + ext)
                if temp_path.exists():
                    img_path = temp_path
                    break
                    
            if img_path is None:
                print(f"Warning: Image file not found for label {lbl_path.name}")
                continue
                
            # Read image shape
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error: Could not read image {img_path}")
                continue
            h, w = img.shape[:2]
            
            # Copy image to destination
            shutil.copy(str(img_path), str(dst_dir / split / 'images' / img_path.name))
            
            # Read and parse labels
            pose_lines = []
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            # First pass: find bull center (class 2) as reference point
            bull_cx = None
            bull_cy = None
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id == 2:  # bull
                    coords = [float(x) for x in parts[1:]]
                    poly_norm = np.array(coords).reshape(-1, 2)
                    poly_px = poly_norm * [w, h]
                    M = cv2.moments(poly_px.astype(np.float32))
                    if M['m00'] > 1e-3:
                        bull_cx = M['m10'] / M['m00']
                        bull_cy = M['m01'] / M['m00']
                    else:
                        bull_cx = float(np.mean(poly_px[:, 0]))
                        bull_cy = float(np.mean(poly_px[:, 1]))
                    break
            
            # If bull not found, try board (class 1)
            if bull_cx is None:
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    if class_id == 1:  # board
                        coords = [float(x) for x in parts[1:]]
                        poly_norm = np.array(coords).reshape(-1, 2)
                        poly_px = poly_norm * [w, h]
                        M = cv2.moments(poly_px.astype(np.float32))
                        if M['m00'] > 1e-3:
                            bull_cx = M['m10'] / M['m00']
                            bull_cy = M['m01'] / M['m00']
                        else:
                            bull_cx = float(np.mean(poly_px[:, 0]))
                            bull_cy = float(np.mean(poly_px[:, 1]))
                        break
            
            # Fallback to image center
            if bull_cx is None:
                bull_cx = w / 2.0
                bull_cy = h / 2.0

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                
                # We only create keypoints for darts (class 0)
                if class_id != 0:
                    continue
                    
                # Parse normalized polygon points
                coords = [float(x) for x in parts[1:]]
                poly_norm = np.array(coords).reshape(-1, 2)
                
                # Scale to pixels
                poly_px = poly_norm * [w, h]
                
                if len(poly_px) < 3:
                    continue
                    
                # Find extreme points
                cx_p = float(np.mean(poly_px[:, 0]))
                cy_p = float(np.mean(poly_px[:, 1]))
                d_c = (poly_px[:, 0] - cx_p) ** 2 + (poly_px[:, 1] - cy_p) ** 2
                end_a = poly_px[int(np.argmax(d_c))]
                d_a = (poly_px[:, 0] - end_a[0]) ** 2 + (poly_px[:, 1] - end_a[1]) ** 2
                end_b = poly_px[int(np.argmax(d_a))]
                
                # Identify tip and tail (tip is always closer to the bull center)
                dist_a = np.hypot(end_a[0] - bull_cx, end_a[1] - bull_cy)
                dist_b = np.hypot(end_b[0] - bull_cx, end_b[1] - bull_cy)
                if dist_a < dist_b:
                    tip = end_a
                    tail = end_b
                else:
                    tip = end_b
                    tail = end_a
                
                # Normalize keypoints
                tip_norm = tip / [w, h]
                tail_norm = tail / [w, h]
                
                # Calculate bounding box
                xmin = min(poly_norm[:, 0])
                xmax = max(poly_norm[:, 0])
                ymin = min(poly_norm[:, 1])
                ymax = max(poly_norm[:, 1])
                box_w = xmax - xmin
                box_h = ymax - ymin
                box_cx = (xmin + xmax) / 2.0
                box_cy = (ymin + ymax) / 2.0
                
                # Write YOLO Pose line
                # Format: class_id x_center y_center width height kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis
                # Class ID is 0 (darts)
                pose_line = f"0 {box_cx:.6f} {box_cy:.6f} {box_w:.6f} {box_h:.6f} {tip_norm[0]:.6f} {tip_norm[1]:.6f} 2 {tail_norm[0]:.6f} {tail_norm[1]:.6f} 2\n"
                pose_lines.append(pose_line)
                
            # Write new label file
            dst_lbl_path = dst_dir / split / 'labels' / lbl_path.name
            with open(dst_lbl_path, 'w') as f:
                f.writelines(pose_lines)
                
    # Write data.yaml for Pose training
    yaml_cfg = {
        'path': str(dst_dir.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'kpt_shape': [2, 3],  # 2 keypoints (tip, tail), each with (x, y, visibility)
        'nc': 1,
        'names': ['dart'],
    }
    
    import yaml
    with open(dst_dir / 'data.yaml', 'w') as f:
        yaml.safe_dump(yaml_cfg, f, sort_keys=False)
    print(f"Wrote data.yaml config to {dst_dir / 'data.yaml'}")
    print("Conversion completed successfully!")

if __name__ == '__main__':
    main()
