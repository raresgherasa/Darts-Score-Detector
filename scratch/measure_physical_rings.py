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
    bull_cls = name_to_id.get('bull', name_to_id.get('bulls'))
    
    # Let's analyze a few images
    for path in image_paths[:5]:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
            
        h, w = frame.shape[:2]
        results = yolo(frame, conf=0.35, verbose=False)[0]
        seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
        
        # Get board center using ellipse fit
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
        scale = major / minor
        ea_rad = np.radians(ea)
        cos_a, sin_a = np.cos(ea_rad), np.sin(ea_rad)
        
        # Convert frame to HSV to detect red/green
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Color masks
        # Green: Hue 35-85
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        # Red: Hue 0-10 or 170-180
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine red and green
        target_mask = cv2.bitwise_or(green_mask, red_mask)
        
        # We sample along 360 radial directions
        distances = []
        for angle_deg in range(0, 360, 5):
            angle_rad = np.radians(angle_deg)
            # Generate a line of points from center outward in the corrected circle space,
            # then project to the foreshortened image space
            for r in range(1, int(major * 1.2)):
                # Corrected circular coords
                dx_c = r * np.cos(angle_rad)
                dy_c = r * np.sin(angle_rad)
                
                # Inverse perspective mapping
                u = cos_a * dx_c + sin_a * dy_c
                v = -sin_a * dx_c + cos_a * dy_c
                v /= scale
                dx = cos_a * u - sin_a * v
                dy = sin_a * u + cos_a * v
                
                px = int(round(cx + dx))
                py = int(round(cy + dy))
                
                if 0 <= px < w and 0 <= py < h:
                    if target_mask[py, px] > 0:
                        distances.append((angle_deg, r))
                        
        # Group distances by radius to identify the rings
        # We expect three distinct clusters along the radius:
        # Cluster 1: Bullseye (0 to ~35 pixels)
        # Cluster 2: Triple Ring (~160 to ~200 pixels)
        # Cluster 3: Double Ring (~280 to ~310 pixels)
        
        bull_pts = [r for _, r in distances if r < major * 0.2]
        triple_pts = [r for _, r in distances if major * 0.4 < r < major * 0.8]
        double_pts = [r for _, r in distances if major * 0.8 < r < major * 1.1]
        
        print(f"\nImage: {path.name}")
        print(f"Board Major Radius: {major:.2f} px")
        if bull_pts:
            print(f"  Bull: min={np.min(bull_pts):.1f}, max={np.max(bull_pts):.1f}")
        if triple_pts:
            print(f"  Triple Ring: min={np.min(triple_pts):.1f}, max={np.max(triple_pts):.1f} | Ratios: {np.min(triple_pts)/major:.3f} - {np.max(triple_pts)/major:.3f}")
        if double_pts:
            print(f"  Double Ring: min={np.min(double_pts):.1f}, max={np.max(double_pts):.1f} | Ratios: {np.min(double_pts)/major:.3f} - {np.max(double_pts)/major:.3f}")

if __name__ == '__main__':
    main()
