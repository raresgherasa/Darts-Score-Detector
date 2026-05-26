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
    
    for path in image_paths[:15]:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
            
        results = yolo(frame, conf=0.35, verbose=False)[0]
        seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
        
        board_center = None
        bull_center = None
        bull_conf = 0.0
        
        if board_cls is not None and seg_polys is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == board_cls and len(seg_polys[i]) >= 5:
                    try:
                        (ex, ey), _, _ = cv2.fitEllipse(seg_polys[i].astype(np.float32))
                        board_center = (ex, ey)
                    except Exception:
                        pass
                        
        if bull_cls is not None and seg_polys is not None:
            best_idx = -1
            best_conf = 0.0
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == bull_cls:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_idx = i
            if best_idx >= 0:
                poly = seg_polys[best_idx].astype(np.float32)
                M = cv2.moments(poly)
                if M['m00'] > 1e-3:
                    bull_cx = M['m10'] / M['m00']
                    bull_cy = M['m01'] / M['m00']
                    bull_center = (bull_cx, bull_cy)
                    bull_conf = best_conf
                    
        if board_center is not None:
            if bull_center is not None:
                dist = np.hypot(board_center[0] - bull_center[0], board_center[1] - bull_center[1])
                print(f"{path.name:30s} | Board Center: ({board_center[0]:.1f}, {board_center[1]:.1f}) | Bull Center: ({bull_center[0]:.1f}, {bull_center[1]:.1f}) | Conf: {bull_conf:.2f} | Dist: {dist:.1f}")
            else:
                print(f"{path.name:30s} | Board Center: ({board_center[0]:.1f}, {board_center[1]:.1f}) | Bull Center: None")

if __name__ == '__main__':
    main()
