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
    
    pose_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/pose_best.pt')
    yolo_pose = YOLO(pose_path) if os.path.exists(pose_path) else None
    
    images_dir = Path(str(Path(__file__).resolve().parents[1] / 'data/images'))
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    
    name_to_id = {v: k for k, v in yolo.names.items()}
    bull_cls = name_to_id.get('bull', name_to_id.get('bulls'))
    board_cls = name_to_id.get('board', name_to_id.get('boards'))
    dart_cls = name_to_id.get('dart', name_to_id.get('darts', name_to_id.get('arrow', name_to_id.get('arrows'))))
    
    print(f"TRIPLE_INNER: {TRIPLE_INNER:.4f}, TRIPLE_OUTER: {TRIPLE_OUTER:.4f}")
    print(f"DOUBLE_INNER: {DOUBLE_INNER:.4f}, DOUBLE_OUTER: {DOUBLE_OUTER:.4f}")
    print("-" * 80)
    
    for path in image_paths[:30]:  # Let's inspect first 30 images
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        results = yolo(frame, conf=0.35, verbose=False)[0]
        seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
        
        results_pose = None
        if yolo_pose is not None:
            results_pose = yolo_pose(frame, conf=0.35, verbose=False)[0]
            
        # Board
        ema_ellipse = None
        board_poly_active = None
        if board_cls is not None:
            best_board_idx, best_board_conf = -1, 0.0
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == board_cls:
                    conf = float(box.conf[0])
                    if conf > best_board_conf:
                        best_board_conf, best_board_idx = conf, i
            if best_board_idx >= 0 and best_board_conf > 0.5 and seg_polys is not None:
                board_poly = seg_polys[best_board_idx].astype(np.float32)
                board_poly_active = board_poly
                try:
                    (ex, ey), (ax1, ax2), ea = cv2.fitEllipse(board_poly)
                    major, minor = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
                    if ax1 > ax2:
                        ea = (ea + 90.0) % 180.0
                    if minor > 10.0:
                        ema_ellipse = (ex, ey, major, minor, ea)
                except Exception:
                    pass
                    
        # Bull
        center_x, center_y, bull_radius = w / 2.0, h / 2.0, 19.0
        if ema_ellipse is not None:
            bex, bey, bmaj, _, _ = ema_ellipse
            center_x, center_y = bex, bey
            bull_radius = bmaj * BULL_TO_DOUBLE_OUTER
            
        best_bull_idx, best_bull_conf = -1, 0.0
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) == bull_cls:
                conf = float(box.conf[0])
                if conf > best_bull_conf:
                    best_bull_conf, best_bull_idx = conf, i
        if best_bull_idx >= 0:
            bx1, by1, bx2, by2 = results.boxes[best_bull_idx].xyxy[0].tolist()
            poly = seg_polys[best_bull_idx] if (seg_polys is not None and len(seg_polys[best_bull_idx]) >= 3) else None
            bull_cx = bull_cy = None
            if poly is not None:
                poly32 = poly.astype(np.float32)
                M = cv2.moments(poly32)
                if M['m00'] > 1e-3:
                    bull_cx = M['m10'] / M['m00']
                    bull_cy = M['m01'] / M['m00']
            if bull_cx is None:
                bull_cx = (bx1 + bx2) / 2.0
                bull_cy = (by1 + by2) / 2.0
            center_x, center_y = bull_cx, bull_cy
            
        # Angle offset
        angle_offset = 0.0
        if board_poly_active is not None and ema_ellipse is not None:
            n_top = max(3, len(board_poly_active) // 10)
            top_indices = np.argsort(board_poly_active[:, 1])[:n_top]
            raw_dx = float(np.mean(board_poly_active[top_indices, 0])) - center_x
            raw_dy = float(np.mean(board_poly_active[top_indices, 1])) - center_y
            c_dx, c_dy = perspective_correct_vec(raw_dx, raw_dy, ema_ellipse)
            angle_offset = np.degrees(np.arctan2(c_dy, c_dx)) + 90.0
            
        # Darts
        use_pose = (results_pose is not None)
        dart_source = results_pose if use_pose else results
        active_dart_cls = 0 if use_pose else dart_cls
        
        dart_boxes, dart_confs, dart_orig = [], [], []
        for i, box in enumerate(dart_source.boxes):
            if int(box.cls[0]) == active_dart_cls:
                dart_boxes.append(box.xyxy[0].tolist())
                dart_confs.append(float(box.conf[0]))
                dart_orig.append(i)
                
        kept = []
        # Basic NMS
        from darts_score_detection_offline import nms_indices
        indices = nms_indices(dart_boxes, dart_confs)
        
        for k in indices:
            x1, y1, x2, y2 = dart_boxes[k]
            orig_i = dart_orig[k]
            tip_cx, tip_cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if use_pose:
                if dart_source.keypoints is not None and dart_source.keypoints.xy is not None:
                    kpts = dart_source.keypoints.xy[orig_i].cpu().numpy()
                    if len(kpts) >= 2 and not np.allclose(kpts, 0.0):
                        pk_tip, pk_tail = kpts[0], kpts[1]
                        tip_cx, tip_cy = float(pk_tip[0]), float(pk_tip[1])
                        tail_x, tail_y = float(pk_tail[0]), float(pk_tail[1])
                        dist_tip = (tip_cx - center_x) ** 2 + (tip_cy - center_y) ** 2
                        dist_tail = (tail_x - center_x) ** 2 + (tail_y - center_y) ** 2
                        if dist_tail < dist_tip:
                            tip_cx, tip_cy, tail_x, tail_y = tail_x, tail_y, tip_cx, tip_cy
                        tip_cx, tip_cy = refine_dart_tip(gray, (tip_cx, tip_cy), (tail_x, tail_y), search_px=18)
            else:
                if seg_polys is not None and len(seg_polys[orig_i]) >= 3:
                    poly = seg_polys[orig_i].astype(np.float32)
                    cx_p = float(np.mean(poly[:, 0]))
                    cy_p = float(np.mean(poly[:, 1]))
                    d_c = (poly[:, 0] - cx_p) ** 2 + (poly[:, 1] - cy_p) ** 2
                    end_a = poly[int(np.argmax(d_c))]
                    d_a = (poly[:, 0] - end_a[0]) ** 2 + (poly[:, 1] - end_a[1]) ** 2
                    end_b = poly[int(np.argmax(d_a))]
                    dist_a = (end_a[0] - center_x) ** 2 + (end_a[1] - center_y) ** 2
                    dist_b = (end_b[0] - center_x) ** 2 + (end_b[1] - center_y) ** 2
                    if dist_a < dist_b:
                        tip_cx, tip_cy = float(end_a[0]), float(end_a[1])
                        tail_x, tail_y = float(end_b[0]), float(end_b[1])
                    else:
                        tip_cx, tip_cy = float(end_b[0]), float(end_b[1])
                        tail_x, tail_y = float(end_a[0]), float(end_a[1])
                    tip_cx, tip_cy = refine_dart_tip(gray, (tip_cx, tip_cy), (tail_x, tail_y))
            
            dx, dy = tip_cx - center_x, tip_cy - center_y
            dx_c, dy_c = perspective_correct_vec(dx, dy, ema_ellipse)
            raw_r = np.hypot(dx, dy) / bull_radius
            corr_r = np.hypot(dx_c, dy_c) / bull_radius
            label, _ = score_geometric(center_x, center_y, center_x + dx_c, center_y + dy_c, bull_radius, angle_offset)
            print(f"{path.name:30s} | Label: {label:5s} | Raw Ratio: {raw_r:6.3f} | Corr Ratio: {corr_r:6.3f}")

if __name__ == '__main__':
    main()
