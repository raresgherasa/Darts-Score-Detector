from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/best.pt')
    yolo = YOLO(model_path)
    
    img_path = str(Path(__file__).resolve().parents[1] / 'data/images/darts_1779198352576.jpg')
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    
    # We define the three refined tip locations we got:
    # Dart 2: (790.50, 351.52)
    # Dart 3: (515.11, 383.52)
    # Dart 4: (739.17, 185.82)
    tips = {
        2: (790.50, 351.52),
        3: (515.11, 383.52),
        4: (739.17, 185.82)
    }
    
    # Convert to HSV to detect red/green
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    target_mask = cv2.bitwise_or(green_mask, red_mask)
    
    # Let's find the coordinates of all target pixels
    target_pts = np.argwhere(target_mask > 0)[:, ::-1] # (x, y) format
    
    # Find board center and ellipse to filter targets by radius
    results = yolo(frame, conf=0.35, verbose=False)[0]
    seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
    
    ema_ellipse = None
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == 1 and len(seg_polys[i]) >= 5: # board
            (ex, ey), (ax1, ax2), ea = cv2.fitEllipse(seg_polys[i].astype(np.float32))
            major, minor = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
            if ax1 > ax2:
                ea = (ea + 90.0) % 180.0
            ema_ellipse = (ex, ey, major, minor, ea)
            
    cx, cy, major, minor, ea = ema_ellipse
    scale = major / minor
    ea_rad = np.radians(ea)
    cos_a, sin_a = np.cos(ea_rad), np.sin(ea_rad)
    
    for dart_idx, (tx, ty) in tips.items():
        # Let's find the nearest red/green pixel
        dists = np.hypot(target_pts[:, 0] - tx, target_pts[:, 1] - ty)
        min_idx = np.argmin(dists)
        nearest_pt = target_pts[min_idx]
        min_dist = dists[min_idx]
        
        # Calculate corrected distance of nearest pt from board center
        dx = nearest_pt[0] - cx
        dy = nearest_pt[1] - cy
        
        # Apply perspective correction to vector
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        v *= scale
        dx_c = cos_a * u - sin_a * v
        dy_c = sin_a * u + cos_a * v
        corr_r = np.hypot(dx_c, dy_c)
        
        # Determine if it's in triple or double ring based on corrected radius
        ring_type = "Unknown"
        # Since double outer is around 222 px, triple is around 135 px
        if 115 < corr_r < 155:
            ring_type = "Triple Ring"
        elif 195 < corr_r < 235:
            ring_type = "Double Ring"
        elif corr_r < 30:
            ring_type = "Bullseye"
            
        print(f"Dart {dart_idx} (Tip: {tx:.1f}, {ty:.1f}):")
        print(f"  Nearest red/green pixel: ({nearest_pt[0]}, {nearest_pt[1]}) at distance {min_dist:.2f} px")
        print(f"  Corrected distance of that pixel: {corr_r:.1f} px ({ring_type})")

if __name__ == '__main__':
    main()
