import sys
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darts_score_detection_offline import (
    score_geometric, perspective_correct_vec, refine_dart_tip,
    BULL_TO_DOUBLE_OUTER, DOUBLE_OUTER, TRIPLE_OUTER, TRIPLE_INNER, DOUBLE_INNER
)

def main():
    model_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/best.pt')
    yolo = YOLO(model_path)
    
    images_dir = Path(str(Path(__file__).resolve().parents[1] / 'data/images'))
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    
    name_to_id = {v: k for k, v in yolo.names.items()}
    board_cls = name_to_id.get('board', name_to_id.get('boards'))
    bull_cls = name_to_id.get('bull', name_to_id.get('bulls'))
    dart_cls = name_to_id.get('dart', name_to_id.get('darts', name_to_id.get('arrow', name_to_id.get('arrows'))))
    
    doubles_found = 0
    triples_found = 0
    
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
            
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = yolo(frame, conf=0.35, verbose=False)[0]
        seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
        
        # Board
        ema_ellipse = None
        if board_cls is not None and seg_polys is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == board_cls and len(seg_polys[i]) >= 5:
                    try:
                        (ex, ey), (ax1, ax2), ea = cv2.fitEllipse(seg_polys[i].astype(np.float32))
                        major, minor = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
                        if ax1 > ax2:
                            ea = (ea + 90.0) % 180.0
                        ema_ellipse = (ex, ey, major, minor, ea)
                    except Exception:
                        pass
                        
        if ema_ellipse is None:
            continue
            
        cx, cy, major, minor, ea = ema_ellipse
        bull_radius = major * BULL_TO_DOUBLE_OUTER
        
        # Bull Center override
        if bull_cls is not None and seg_polys is not None:
            best_bull_idx, best_bull_conf = -1, 0.0
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == bull_cls:
                    conf = float(box.conf[0])
                    if conf > best_bull_conf:
                        best_bull_conf, best_bull_idx = conf, i
            if best_bull_idx >= 0:
                poly = seg_polys[best_bull_idx]
                bull_cx = bull_cy = None
                if len(poly) >= 3:
                    M = cv2.moments(poly.astype(np.float32))
                    if M['m00'] > 1e-3:
                        bull_cx = M['m10'] / M['m00']
                        bull_cy = M['m01'] / M['m00']
                if bull_cx is not None:
                    cx, cy = bull_cx, bull_cy
                    
        # Darts
        if dart_cls is not None and seg_polys is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == dart_cls and len(seg_polys[i]) >= 3:
                    poly = seg_polys[i].astype(np.float32)
                    cx_p = float(np.mean(poly[:, 0]))
                    cy_p = float(np.mean(poly[:, 1]))
                    d_c = (poly[:, 0] - cx_p) ** 2 + (poly[:, 1] - cy_p) ** 2
                    end_a = poly[int(np.argmax(d_c))]
                    d_a = (poly[:, 0] - end_a[0]) ** 2 + (poly[:, 1] - end_a[1]) ** 2
                    end_b = poly[int(np.argmax(d_a))]
                    
                    dist_a = (end_a[0] - cx) ** 2 + (end_a[1] - cy) ** 2
                    dist_b = (end_b[0] - cx) ** 2 + (end_b[1] - cy) ** 2
                    tip = end_a if dist_a < dist_b else end_b
                    tail = end_b if dist_a < dist_b else end_a
                    
                    tip_x, tip_y = refine_dart_tip(gray, (tip[0], tip[1]), (tail[0], tail[1]))
                    
                    dx = tip_x - cx
                    dy = tip_y - cy
                    dx_c, dy_c = perspective_correct_vec(dx, dy, ema_ellipse)
                    corr_r = np.hypot(dx_c, dy_c)
                    ratio = corr_r / bull_radius
                    
                    if DOUBLE_INNER <= ratio <= DOUBLE_OUTER:
                        print(f"Double: {path.name} | Dart {i} at Ratio {ratio:.3f} (Corr R: {corr_r:.1f} px)")
                        doubles_found += 1
                    elif TRIPLE_INNER <= ratio <= TRIPLE_OUTER:
                        print(f"Triple: {path.name} | Dart {i} at Ratio {ratio:.3f} (Corr R: {corr_r:.1f} px)")
                        triples_found += 1
                        
    print(f"\nScan completed. Found {doubles_found} doubles and {triples_found} triples with the calibrated setup.")

if __name__ == '__main__':
    main()
