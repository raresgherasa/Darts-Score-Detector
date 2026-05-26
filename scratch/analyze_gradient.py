from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darts_score_detection_offline import perspective_correct_vec

def main():
    model_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/best.pt')
    yolo = YOLO(model_path)
    
    img_path = str(Path(__file__).resolve().parents[1] / 'data/images/darts_1779195243354.jpg')
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = yolo(frame, conf=0.35, verbose=False)[0]
    
    seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
    dart_cls = 0 # dart
    
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == dart_cls:
            poly = seg_polys[i].astype(np.float32)
            cx_p = float(np.mean(poly[:, 0]))
            cy_p = float(np.mean(poly[:, 1]))
            d_c = (poly[:, 0] - cx_p) ** 2 + (poly[:, 1] - cy_p) ** 2
            end_a = poly[int(np.argmax(d_c))]
            d_a = (poly[:, 0] - end_a[0]) ** 2 + (poly[:, 1] - end_a[1]) ** 2
            end_b = poly[int(np.argmax(d_a))]
            
            # Let's assume end_a is tip and end_b is tail, or vice versa
            # We can use dart_tip_from_endpoints logic
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from darts_score_detection_offline import dart_tip_from_endpoints
            tip = dart_tip_from_endpoints(poly, end_a, end_b)
            tail = end_b if np.allclose(tip, end_a, atol=1.0) else end_a
            
            print(f"\n--- Dart {i} ---")
            print(f"Approx tip: {tip}, tail: {tail}")
            
            # Perform gradient analysis
            tip_pt = np.array(tip, dtype=np.float64)
            tail_pt = np.array(tail, dtype=np.float64)
            shaft = tip_pt - tail_pt
            shaft_len = float(np.linalg.norm(shaft))
            shaft_dir = shaft / shaft_len
            perp_dir = np.array([-shaft_dir[1], shaft_dir[0]])
            
            margin = 30
            x0 = max(0, int(tip_pt[0]) - margin)
            y0 = max(0, int(tip_pt[1]) - margin)
            x1 = min(w, int(tip_pt[0]) + margin + 1)
            y1 = min(h, int(tip_pt[1]) + margin + 1)
            roi = gray[y0:y1, x0:x1]
            gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx * gx + gy * gy)
            
            strip_half = 4
            print(f"{'t':>5} | {'x':>7} | {'y':>7} | {'Peak Grad':>10}")
            print("-" * 40)
            for t in range(25, -26, -1):
                center_pt = tip_pt + t * shaft_dir
                best = 0.0
                for s in range(-strip_half, strip_half + 1):
                    pt = center_pt + s * perp_dir
                    xi = int(round(pt[0])) - x0
                    yi = int(round(pt[1])) - y0
                    if 0 <= xi < grad.shape[1] and 0 <= yi < grad.shape[0]:
                        v = float(grad[yi, xi])
                        if v > best:
                            best = v
                print(f"{t:>5} | {center_pt[0]:>7.2f} | {center_pt[1]:>7.2f} | {best:>10.2f}")

if __name__ == '__main__':
    main()
