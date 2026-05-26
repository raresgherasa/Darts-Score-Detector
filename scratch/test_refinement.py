import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darts_score_detection_offline import perspective_correct_vec, dart_tip_from_endpoints

def original_refine_dart_tip(gray, approx_tip, tail_pt, search_px=18, strip_half=4, grad_thresh=20):
    tip = np.array(approx_tip, dtype=np.float64)
    tail = np.array(tail_pt, dtype=np.float64)
    shaft = tip - tail
    shaft_len = float(np.linalg.norm(shaft))
    if shaft_len < 5.0:
        return approx_tip
    shaft_dir = shaft / shaft_len
    perp_dir = np.array([-shaft_dir[1], shaft_dir[0]])

    h, w = gray.shape[:2]
    margin = search_px + strip_half + 4
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

    for t in range(search_px, -(search_px + 1), -1):
        if _peak(tip + t * shaft_dir) > grad_thresh:
            refined = tip + t * shaft_dir
            return (float(refined[0]), float(refined[1]))

    return approx_tip

def proposed_refine_dart_tip(gray, approx_tip, tail_pt, search_px=18, strip_half=4):
    """Proposed refinement using a profile analysis of the gradient along the shaft.
    Scans outwards from the tail/approx_tip and finds where the shaft gradient drops.
    """
    tip = np.array(approx_tip, dtype=np.float64)
    tail = np.array(tail_pt, dtype=np.float64)
    shaft = tip - tail
    shaft_len = float(np.linalg.norm(shaft))
    if shaft_len < 5.0:
        return approx_tip
    shaft_dir = shaft / shaft_len
    perp_dir = np.array([-shaft_dir[1], shaft_dir[0]])

    h, w = gray.shape[:2]
    # We want to scan a bit deeper inside the shaft and further outside
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

    # 1. Sample gradient profile from t = -10 (inside shaft) to t = search_px (outside)
    t_vals = list(range(-10, search_px + 1))
    profile = []
    for t in t_vals:
        profile.append(_peak(tip + t * shaft_dir))
        
    profile = np.array(profile)
    
    # 2. Define the baseline/maximum shaft gradient inside the dart body (t <= 2)
    inside_indices = [i for i, t in enumerate(t_vals) if t <= 2]
    max_inside_grad = np.max(profile[inside_indices]) if len(inside_indices) > 0 else 100.0
    # Ensure a reasonable minimum threshold to avoid noise in flat areas
    thresh = max(40.0, 0.35 * max_inside_grad)
    
    # 3. Scan outwards from t = 0 (approx tip) to find the first point where the gradient
    # drops below the threshold and stays below it (or drops significantly).
    # Since we want to find the exact tip of the needle, we look for the last point
    # along the search direction (going outwards) that is still part of the high-contrast shaft/needle.
    best_t = 0
    for i, t in enumerate(t_vals):
        if t < 0:
            continue
        # If the gradient at this point is still high, it's likely still the needle
        if profile[i] >= thresh:
            best_t = t
        else:
            # If we drop below threshold, make sure it's not a temporary dip by checking the next point
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
                
                tip = dart_tip_from_endpoints(poly, end_a, end_b)
                tail = end_b if np.allclose(tip, end_a, atol=1.0) else end_a
                
                orig_tip = original_refine_dart_tip(gray, tip, tail)
                new_tip = proposed_refine_dart_tip(gray, tip, tail)
                
                orig_diff = np.linalg.norm(np.array(orig_tip) - tip)
                new_diff = np.linalg.norm(np.array(new_tip) - tip)
                
                print(f"  Dart {i}: Approx: ({tip[0]:.1f}, {tip[1]:.1f})")
                print(f"    Original Refined: ({orig_tip[0]:.1f}, {orig_tip[1]:.1f}) [shifted {orig_diff:.1f} px]")
                print(f"    Proposed Refined: ({new_tip[0]:.1f}, {new_tip[1]:.1f}) [shifted {new_diff:.1f} px]")

if __name__ == '__main__':
    main()
