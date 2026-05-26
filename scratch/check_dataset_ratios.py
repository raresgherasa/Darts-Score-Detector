import os
from pathlib import Path
import numpy as np

def main():
    dataset_dir = Path(str(Path(__file__).resolve().parents[1] / 'yolov11 dataset'))
    label_files = list((dataset_dir / 'train' / 'labels').glob('*.txt'))
    
    ratios = []
    board_widths = []
    bull_widths = []
    
    for lbl_path in label_files:
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        
        board_box = None
        bull_box = None
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) < 4:
                continue
                
            # If it's a polygon, find bounding box
            xs = coords[0::2]
            ys = coords[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w = xmax - xmin
            h = ymax - ymin
            diag = np.hypot(w, h)
            
            if class_id == 1:  # board
                board_box = diag
            elif class_id == 2:  # bulls
                bull_box = diag
                
        if board_box is not None and bull_box is not None:
            ratios.append(board_box / bull_box)
            board_widths.append(board_box)
            bull_widths.append(bull_box)
            
    if ratios:
        print(f"Number of images with both board and bull: {len(ratios)}")
        print(f"Average ratio of Board/Bull: {np.mean(ratios):.4f}")
        print(f"Median ratio of Board/Bull: {np.median(ratios):.4f}")
        print(f"Min ratio: {np.min(ratios):.4f}")
        print(f"Max ratio: {np.max(ratios):.4f}")
    else:
        print("No images found with both board and bull annotations.")

if __name__ == '__main__':
    main()
