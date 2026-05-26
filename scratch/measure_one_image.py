import sys
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/best.pt')
    yolo = YOLO(model_path)
    
    img_path = str(Path(__file__).resolve().parents[1] / 'data/images/darts_1779198352576.jpg')
    frame = cv2.imread(img_path)
    if frame is None:
        print("Could not read image")
        return
        
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
        print("Board not detected")
        return
        
    cx, cy, major, minor, ea = ema_ellipse
    scale = major / minor
    ea_rad = np.radians(ea)
    cos_a, sin_a = np.cos(ea_rad), np.sin(ea_rad)
    
    # Convert frame to HSV to detect red/green
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Color masks
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    target_mask = cv2.bitwise_or(green_mask, red_mask)
    
    # Let's sample along 360 radial directions
    distances = []
    for angle_deg in range(0, 360, 2):
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
                    distances.append((angle_deg, r))
                    
    # Print sorted distances to see the density clusters
    all_r = [r for _, r in distances]
    hist, bin_edges = np.histogram(all_r, bins=50, range=(0, int(major * 1.2)))
    
    print(f"Board Major Radius: {major:.2f} px")
    print("\nDistance histogram of red/green pixels from center:")
    for i in range(len(hist)):
        if hist[i] > 10:
            print(f"  Range {bin_edges[i]:6.1f} - {bin_edges[i+1]:6.1f} px : {'*' * min(50, hist[i] // 10)} ({hist[i]})")

if __name__ == '__main__':
    main()
