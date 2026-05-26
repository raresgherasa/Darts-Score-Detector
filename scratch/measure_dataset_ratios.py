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
    board_cls = name_to_id.get('board', name_to_id.get('boards'))
    
    ratios_list = []
    
    for path in image_paths[:10]:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
            
        h, w = frame.shape[:2]
        results = yolo(frame, conf=0.35, verbose=False)[0]
        seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
        
        # Get board center using ellipse fit
        ema_ellipse = None
        if seg_polys is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == 1 and len(seg_polys[i]) >= 5: # board
                    (ex, ey), (ax1, ax2), ea = cv2.fitEllipse(seg_polys[i].astype(np.float32))
                    major, minor = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
                    if ax1 > ax2:
                        ea = (ea + 90.0) % 180.0
                    ema_ellipse = (ex, ey, major, minor, ea)
                    
        if ema_ellipse is None:
            continue
            
        cx, cy, major, minor, ea = ema_ellipse
        scale = major / minor
        ea_rad = np.radians(ea)
        cos_a, sin_a = np.cos(ea_rad), np.sin(ea_rad)
        
        # Convert frame to HSV to detect red/green
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        target_mask = cv2.bitwise_or(green_mask, red_mask)
        
        distances = []
        for angle_deg in range(0, 360, 4):
            angle_rad = np.radians(angle_deg)
            for r in range(1, int(major * 1.5)):
                dx_c = r * np.cos(angle_rad)
                dy_c = r * np.sin(angle_rad)
                
                u = cos_a * dx_c + sin_a * dy_c
                v = -sin_a * dx_c + cos_a * dy_c
                v /= scale
                dx = cos_a * u - sin_a * v
                dy = sin_a * u + cos_a * v
                
                px = int(round(cx + dx))
                py = int(round(cy + dy))
                
                if 0 <= px < w and 0 <= py < h:
                    if target_mask[py, px] > 0:
                        distances.append(r)
                        
        if not distances:
            continue
            
        # Group into clusters using simple thresholds relative to board major radius
        bull_r = [r for r in distances if r < major * 0.15]
        triple_r = [r for r in distances if major * 0.4 < r < major * 0.65]
        double_r = [r for r in distances if major * 0.7 < r < major * 0.95]
        
        if bull_r and triple_r and double_r:
            bull_max = np.percentile(bull_r, 95) # use 95th percentile to avoid outlier noise
            trip_min = np.percentile(triple_r, 5)
            trip_max = np.percentile(triple_r, 95)
            double_min = np.percentile(double_r, 5)
            double_max = np.percentile(double_r, 95)
            
            # Print ratios relative to bull_max
            r_trip_in = trip_min / bull_max
            r_trip_out = trip_max / bull_max
            r_double_in = double_min / bull_max
            r_double_out = double_max / bull_max
            
            print(f"Image: {path.name:30s} | Bull: {bull_max:.1f} | Trip: {r_trip_in:.3f}-{r_trip_out:.3f} | Double: {r_double_in:.3f}-{r_double_out:.3f}")
            ratios_list.append((r_trip_in, r_trip_out, r_double_in, r_double_out))
            
    if ratios_list:
        avg_ratios = np.mean(ratios_list, axis=0)
        print("-" * 80)
        print(f"Average Calibrated Ratios:")
        print(f"  TRIPLE_INNER: {avg_ratios[0]:.3f}")
        print(f"  TRIPLE_OUTER: {avg_ratios[1]:.3f}")
        print(f"  DOUBLE_INNER: {avg_ratios[2]:.3f}")
        print(f"  DOUBLE_OUTER: {avg_ratios[3]:.3f}")

if __name__ == '__main__':
    main()
