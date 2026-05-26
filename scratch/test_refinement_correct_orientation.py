import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darts_score_detection_offline import perspective_correct_vec

def proposed_refine_dart_tip(gray, approx_tip, tail_pt, search_px=18, strip_half=4):
    tip = np.array(approx_tip, dtype=np.float64)
    tail = np.array(tail_pt, dtype=np.float64)
    shaft = tip - tail
    shaft_len = float(np.linalg.norm(shaft))
    if shaft_len < 5.0:
        return approx_tip
    shaft_dir = shaft / shaft_len
    perp_dir = np.array([-shaft_dir[1], shaft_dir[0]])

    h, w = gray.shape[:2]
    margin = search_px + strip_half + 6
    x0 = max(0, int(tip[0]) - margin)
    y0 = max(0, int(tip[1]) - margin)
    x1 = min(w, int(tip[0]) + margin + 1)
    y1 = min(h, int(tip[1]) + margin + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return approx_tip
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    def _peak(center_pt):
        best = 0.0
        for s in range(-strip_half, strip_half + 1):
            pt = center_pt + s * perp_dir
            xi = int(round(pt[0])) - x0
            yi = int(round(pt[1])) - y0
            if 0 <= xi < grad.shape[1] and 0 <= yi < grad.shape[0]:
                v = float(grad[yi, xi])
                if v > best:
                    best = v
        return best

    t_vals = list(range(-10, search_px + 1))
    profile = []
    for t in t_vals:
        profile.append(_peak(tip + t * shaft_dir))
        
    profile = np.array(profile)
    
    inside_indices = [i for i, t in enumerate(t_vals) if t <= 2]
    max_inside_grad = np.max(profile[inside_indices]) if len(inside_indices) > 0 else 100.0
    thresh = max(40.0, 0.35 * max_inside_grad)
    
    best_t = 0
    for i, t in enumerate(t_vals):
        if t < 0:
            continue
        if profile[i] >= thresh:
            best_t = t
        else:
            if i + 1 < len(t_vals) and profile[i+1] < thresh:
                break
                
    refined = tip + best_t * shaft_dir
    return (float(refined[0]), float(refined[1]))

def main():
    model_path = str(Path(__file__).resolve().parents[1] / 'models/yolo/best.pt')
    yolo = YOLO(model_path)
    
    images_dir = Path(str(Path(__file__).resolve().parents[1] / 'data/images'))
    img_paths = sorted(images_dir.glob('*.jpg'))[:10]  # Let's test first 10 images
    
    for path in img_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        results = yolo(frame, conf=0.35, verbose=False)[0]
        seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None
        
        bull_cls = 2
        center_x, center_y = w / 2.0, h / 2.0
        if seg_polys is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == bull_cls:
                    poly = seg_polys[i]
                    M = cv2.moments(poly.astype(np.float32))
                    if M['m00'] > 1e-3:
                        center_x = M['m10'] / M['m00']
                        center_y = M['m01'] / M['m00']
        
        dart_cls = 0
        print(f"\nImage: {path.name}")
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) == dart_cls:
                poly = seg_polys[i].astype(np.float32)
                cx_p = float(np.mean(poly[:, 0]))
                cy_p = float(np.mean(poly[:, 1]))
                d_c = (poly[:, 0] - cx_p) ** 2 + (poly[:, 1] - cy_p) ** 2
                end_a = poly[int(np.argmax(d_c))]
                d_a = (poly[:, 0] - end_a[0]) ** 2 + (poly[:, 1] - end_a[1]) ** 2
                end_b = poly[int(np.argmax(d_a))]
                
                # Correct orientation: tip is closer to bull center
                dist_a = np.hypot(end_a[0] - center_x, end_a[1] - center_y)
                dist_b = np.hypot(end_b[0] - center_x, end_b[1] - center_y)
                if dist_a < dist_b:
                    tip, tail = end_a, end_b
                else:
                    tip, tail = end_b, end_a
                
                new_tip = proposed_refine_dart_tip(gray, tip, tail)
                new_diff = np.linalg.norm(np.array(new_tip) - tip)
                
                print(f"  Dart {i}:")
                print(f"    Approx Tip: ({tip[0]:.1f}, {tip[1]:.1f})")
                print(f"    Proposed Refined: ({new_tip[0]:.1f}, {new_tip[1]:.1f}) [shifted {new_diff:.1f} px]")

if __name__ == '__main__':
    main()
