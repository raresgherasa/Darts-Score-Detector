import sys
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/best.pt')
    yolo = YOLO(model_path)
    
    images_dir = Path(str(Path(__file__).resolve().parents[1] / 'data/images'))
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    
    name_to_id = {v: k for k, v in yolo.names.items()}
    bull_cls = name_to_id.get('bull', name_to_id.get('bulls'))
    board_cls = name_to_id.get('board', name_to_id.get('boards'))
    
    print(f"Image Name | Board Major Radius | Bull Major Radius | Board/Bull Ratio")
    print("-" * 75)
    
    count = 0
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
            
        results = yolo(frame, conf=0.35, verbose=False)[0]
        seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
        
        board_maj = None
        bull_maj = None
        
        if board_cls is not None and seg_polys is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == board_cls and len(seg_polys[i]) >= 5:
                    try:
                        _, (ax1, ax2), _ = cv2.fitEllipse(seg_polys[i].astype(np.float32))
                        board_maj = max(ax1, ax2) / 2.0
                    except Exception:
                        pass
                        
        if bull_cls is not None and seg_polys is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == bull_cls and len(seg_polys[i]) >= 5:
                    try:
                        _, (ax1, ax2), _ = cv2.fitEllipse(seg_polys[i].astype(np.float32))
                        bull_maj = max(ax1, ax2) / 2.0
                    except Exception:
                        pass
                        
        if board_maj is not None and bull_maj is not None:
            ratio = board_maj / bull_maj
            print(f"{path.name:30s} | {board_maj:18.2f} | {bull_maj:17.2f} | {ratio:16.4f}")
            count += 1
            if count >= 15:
                break

if __name__ == '__main__':
    main()
