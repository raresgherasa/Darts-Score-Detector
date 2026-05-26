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
    
    img_path = str(Path(__file__).resolve().parents[1] / 'data/images/darts_1779198352576.jpg')
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    results = yolo(frame, conf=0.35, verbose=False)[0]
    seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
    
    # Board
    ema_ellipse = None
    board_poly_active = None
    best_board_idx, best_board_conf = -1, 0.0
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == 1: # board
            conf = float(box.conf[0])
            if conf > best_board_conf:
                best_board_conf, best_board_idx = conf, i
    if best_board_idx >= 0 and seg_polys is not None:
        board_poly = seg_polys[best_board_idx].astype(np.float32)
        (ex, ey), (ax1, ax2), ea = cv2.fitEllipse(board_poly)
        major, minor = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
        if ax1 > ax2:
            ea = (ea + 90.0) % 180.0
        ema_ellipse = (ex, ey, major, minor, ea)
        print(f"Board ellipse: center=({ex:.2f}, {ey:.2f}), major={major:.2f}, minor={minor:.2f}, angle={ea:.2f}")

    # Bull
    center_x, center_y, bull_radius = w / 2.0, h / 2.0, 19.0
    if ema_ellipse is not None:
        bex, bey, bmaj, _, _ = ema_ellipse
        center_x, center_y = bex, bey
        bull_radius = bmaj * BULL_TO_DOUBLE_OUTER
        print(f"Board-derived bull radius: {bull_radius:.2f}")
        
    best_bull_idx, best_bull_conf = -1, 0.0
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == 2: # bull
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
        print(f"Bull detected: center=({bull_cx:.2f}, {bull_cy:.2f})")

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
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == 0: # dart
            poly = seg_polys[i].astype(np.float32)
            cx_p = float(np.mean(poly[:, 0]))
            cy_p = float(np.mean(poly[:, 1]))
            d_c = (poly[:, 0] - cx_p) ** 2 + (poly[:, 1] - cy_p) ** 2
            end_a = poly[int(np.argmax(d_c))]
            d_a = (poly[:, 0] - end_a[0]) ** 2 + (poly[:, 1] - end_a[1]) ** 2
            end_b = poly[int(np.argmax(d_a))]
            
            dist_a = (end_a[0] - center_x) ** 2 + (end_a[1] - center_y) ** 2
            dist_b = (end_b[0] - center_x) ** 2 + (end_b[1] - center_y) ** 2
            if dist_a < dist_b:
                tip = end_a
                tail = end_b
            else:
                tip = end_b
                tail = end_a
                
            ref_x, ref_y = refine_dart_tip(gray, (tip[0], tip[1]), (tail[0], tail[1]))
            
            dx = ref_x - center_x
            dy = ref_y - center_y
            dx_c, dy_c = perspective_correct_vec(dx, dy, ema_ellipse)
            corr_r = np.hypot(dx_c, dy_c)
            ratio = corr_r / bull_radius
            
            label, score = score_geometric(center_x, center_y, center_x + dx_c, center_y + dy_c, bull_radius, angle_offset)
            print(f"\nDart {i}:")
            print(f"  Refined Tip: ({ref_x:.2f}, {ref_y:.2f})")
            print(f"  Corr Dist: {corr_r:.2f} px")
            print(f"  Ratio to Bull: {ratio:.3f}")
            print(f"  Label: {label} | Score: {score}")

if __name__ == '__main__':
    main()
