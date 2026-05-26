from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darts_score_detection_offline import perspective_correct_vec, dart_tip_from_endpoints, BULL_TO_DOUBLE_OUTER

def main():
    model_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/best.pt')
    yolo = YOLO(model_path)
    
    img_path = str(Path(__file__).resolve().parents[1] / 'data/images/darts_1779195247599.jpg')
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = yolo(frame, conf=0.35, verbose=False)[0]
    
    seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
    
    # Board center
    board_cls = 1
    bull_cls = 2
    
    center_x, center_y = w / 2.0, h / 2.0
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == bull_cls:
            poly = seg_polys[i]
            M = cv2.moments(poly.astype(np.float32))
            if M['m00'] > 1e-3:
                center_x = M['m10'] / M['m00']
                center_y = M['m01'] / M['m00']
                
    print(f"Bull center: ({center_x:.2f}, {center_y:.2f})")
    
    dart_cls = 0
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
            
            dist_a = np.hypot(end_a[0] - center_x, end_a[1] - center_y)
            dist_b = np.hypot(end_b[0] - center_x, end_b[1] - center_y)
            
            # Which is closer to bull
            closer_to_bull = end_a if dist_a < dist_b else end_b
            further_from_bull = end_b if dist_a < dist_b else end_a
            
            print(f"\nDart {i}:")
            print(f"  end_a: {end_a} | dist to bull: {dist_a:.2f}")
            print(f"  end_b: {end_b} | dist to bull: {dist_b:.2f}")
            print(f"  Heuristic tip chosen: {tip}")
            print(f"  Closer to bull: {closer_to_bull}")
            print(f"  Is heuristic same as closer to bull? {np.allclose(tip, closer_to_bull, atol=1.0)}")

if __name__ == '__main__':
    main()
