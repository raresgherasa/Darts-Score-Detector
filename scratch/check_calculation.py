from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darts_score_detection_offline import (
    dart_tip_from_endpoints, refine_dart_tip, perspective_correct_vec, score_geometric,
    BULL_TO_DOUBLE_OUTER, SECTORS
)

def main():
    model_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/best.pt')
    yolo = YOLO(model_path)
    
    img_path = str(Path(__file__).resolve().parents[1] / 'data/images/darts_1779195243354.jpg')
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = yolo(frame, conf=0.35, verbose=False)[0]
    
    name_to_id = {v: k for k, v in yolo.names.items()}
    bull_cls = name_to_id.get('bull', name_to_id.get('bulls'))
    board_cls = name_to_id.get('board', name_to_id.get('boards'))
    dart_cls = name_to_id.get('dart', name_to_id.get('darts', name_to_id.get('arrow', name_to_id.get('arrows'))))
    
    seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
    
    # ── Board ──
    ema_ellipse = None
    board_poly_active = None
    if board_cls is not None:
        best_board_idx = -1
        best_board_conf = 0.0
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
                major = max(ax1, ax2) / 2.0
                minor = min(ax1, ax2) / 2.0
                if ax1 > ax2:
                    ea = (ea + 90.0) % 180.0
                if minor > 10.0:
                    ema_ellipse = (ex, ey, major, minor, ea)
                    print(f"Board ellipse: center=({ex:.2f}, {ey:.2f}), major={major:.2f}, minor={minor:.2f}, angle={ea:.2f}")
            except Exception as e:
                print("Board ellipse fit failed:", e)

    # ── Bull center & radius ──
    center_x, center_y, bull_radius = w / 2.0, h / 2.0, 19.0
    if ema_ellipse is not None:
        bex, bey, bmaj, _, _ = ema_ellipse
        center_x, center_y = bex, bey
        bull_radius = bmaj * BULL_TO_DOUBLE_OUTER
        print(f"Board-derived center: ({center_x:.2f}, {center_y:.2f}), bull_radius={bull_radius:.2f}")

    best_bull_idx, best_bull_conf = -1, 0.0
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == bull_cls:
            conf = float(box.conf[0])
            if conf > best_bull_conf:
                best_bull_conf, best_bull_idx = conf, i

    if best_bull_idx >= 0:
        bx1, by1, bx2, by2 = results.boxes[best_bull_idx].xyxy[0].tolist()
        poly = seg_polys[best_bull_idx] if (seg_polys is not None and len(seg_polys[best_bull_idx]) >= 3) else None
        
        # Centroid
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
            
        print(f"Bull detected: center=({bull_cx:.2f}, {bull_cy:.2f}), conf={best_bull_conf:.2f}")
        center_x, center_y = bull_cx, bull_cy

    # ── Angle offset ──
    angle_offset = 0.0
    if board_poly_active is not None and ema_ellipse is not None:
        n_top = max(3, len(board_poly_active) // 10)
        top_indices = np.argsort(board_poly_active[:, 1])[:n_top]
        raw_dx = float(np.mean(board_poly_active[top_indices, 0])) - center_x
        raw_dy = float(np.mean(board_poly_active[top_indices, 1])) - center_y
        c_dx, c_dy = perspective_correct_vec(raw_dx, raw_dy, ema_ellipse)
        angle_offset = np.degrees(np.arctan2(c_dy, c_dx)) + 90.0
        print(f"Calculated angle_offset: {angle_offset:.2f} degrees")

    # ── Darts ──
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == dart_cls:
            poly = seg_polys[i].astype(np.float32)
            cx_p = float(np.mean(poly[:, 0]))
            cy_p = float(np.mean(poly[:, 1]))
            d_c = (poly[:, 0] - cx_p) ** 2 + (poly[:, 1] - cy_p) ** 2
            end_a = poly[int(np.argmax(d_c))]
            d_a = (poly[:, 0] - end_a[0]) ** 2 + (poly[:, 1] - end_a[1]) ** 2
            end_b = poly[int(np.argmax(d_a))]
            
            tip = dart_tip_from_endpoints(poly, end_a, end_b)
            tail = end_b if np.allclose(tip, end_a, atol=1.0) else end_a
            
            # Refine
            ref_x, ref_y = refine_dart_tip(gray, (tip[0], tip[1]), (tail[0], tail[1]))
            
            print(f"\nDart {i}:")
            print(f"  Approx Tip: ({tip[0]:.2f}, {tip[1]:.2f})")
            print(f"  Refined Tip: ({ref_x:.2f}, {ref_y:.2f})")
            print(f"  Tail: ({tail[0]:.2f}, {tail[1]:.2f})")
            
            # Distance and angle
            dx = ref_x - center_x
            dy = ref_y - center_y
            raw_r = np.hypot(dx, dy)
            r_ratio = raw_r / bull_radius
            print(f"  Vector from center: dx={dx:.2f}, dy={dy:.2f}, distance={raw_r:.2f} pixels, ratio={r_ratio:.3f} bull_radii")
            
            dx_c, dy_c = perspective_correct_vec(dx, dy, ema_ellipse)
            corr_r = np.hypot(dx_c, dy_c)
            corr_ratio = corr_r / bull_radius
            print(f"  Corrected vector: dx_c={dx_c:.2f}, dy_c={dy_c:.2f}, distance={corr_r:.2f} pixels, ratio={corr_ratio:.3f} bull_radii")
            
            label, score = score_geometric(center_x, center_y, ref_x, ref_y, bull_radius, angle_offset)
            print(f"  Score: {label} (value={score})")

if __name__ == '__main__':
    main()
