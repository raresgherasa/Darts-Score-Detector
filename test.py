"""Quick visual test for the darts score detector on static images.

Usage:
  python test.py                        # random images — press any key for next, 'q' to quit
  python test.py --image path/to/img   # specific image
  python test.py --all                  # loop through every image in order, press any key to advance
  python test.py --no-rings             # hide the scoring-ring overlay (rings are shown by default)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

def smooth_polygon(poly, w, h, kernel_size=7):
    """Smooth a pixelated polygon by drawing it, blurring, and extracting the contour."""
    if len(poly) < 3:
        return poly
    # Bounding box of the polygon to keep mask size small
    xs = poly[:, 0]
    ys = poly[:, 1]
    x1, y1 = max(0, int(np.min(xs)) - 5), max(0, int(np.min(ys)) - 5)
    x2, y2 = min(w, int(np.max(xs)) + 5), min(h, int(np.max(ys)) + 5)
    mw, mh = x2 - x1, y2 - y1
    if mw <= 0 or mh <= 0:
        return poly
    mask = np.zeros((mh, mw), dtype=np.uint8)
    shifted_poly = (poly - np.array([x1, y1])).astype(np.int32)
    cv2.fillPoly(mask, [shifted_poly], 255)
    
    # Smooth using Gaussian blur
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best_contour = max(contours, key=cv2.contourArea)
        smoothed = best_contour.reshape(-1, 2) + np.array([x1, y1])
        return smoothed.astype(np.float32)
    return poly


DRAW_RINGS = True   # toggled by --no-rings flag

# Import geometry helpers from the main detection script
sys.path.insert(0, str(Path(__file__).parent))
from darts_score_detection_offline import (
    score_geometric,
    label_to_score,
    perspective_correct_vec,
    nms_indices,
    dart_tip_tail_poly,
    refine_dart_tip,
    resolve_class,
    BULL_TO_DOUBLE_OUTER,
    BULL_POLYGON_SCALE,
    INNER_BULL_ANNOT_RADIUS,
    INNER_BULL_EDGE,
    OUTER_BULL_EDGE,
    TRIPLE_INNER,
    TRIPLE_OUTER,
    DOUBLE_INNER,
    DOUBLE_OUTER,
    iou,
)


def draw_scoring_rings(frame, cx, cy, bull_radius, ema_ellipse, angle_offset):
    """Overlay the regulation scoring rings on the image for visual debugging.
    Rings are drawn in perspective by mapping each radius into the ellipse coords.
    """
    if bull_radius is None or bull_radius <= 0:
        return
    # We draw concentric "rings" by sampling angles around the bull and mapping
    # each point back through the inverse of perspective_correct_vec.
    if ema_ellipse is not None:
        _, _, major, minor, ea = ema_ellipse
        scale = min(major / max(minor, 1.0), 4.0)
        ar = math.radians(ea)
        cos_a, sin_a = math.cos(ar), math.sin(ar)
    else:
        scale = 1.0
        cos_a, sin_a = 1.0, 0.0

    ring_radii = [INNER_BULL_EDGE, OUTER_BULL_EDGE, TRIPLE_INNER, TRIPLE_OUTER, DOUBLE_INNER, DOUBLE_OUTER]
    ring_colors = [(0,255,255), (0,255,255), (255,180,0), (255,180,0), (0,180,255), (0,180,255)]
    for radius_units, color in zip(ring_radii, ring_colors):
        r_pixels = radius_units * bull_radius
        pts = []
        for ang in np.linspace(0, 2 * math.pi, 96):
            # Build the corrected vector at this radius and angle
            dx_c = r_pixels * math.cos(ang)
            dy_c = r_pixels * math.sin(ang)
            # Inverse perspective: undo the minor-axis stretch
            u =  cos_a * dx_c + sin_a * dy_c
            v = -sin_a * dx_c + cos_a * dy_c
            v /= scale
            dx = cos_a * u - sin_a * v
            dy = sin_a * u + cos_a * v
            pts.append((int(cx + dx), int(cy + dy)))
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, color, 1)

    # Draw sector spokes
    for k in range(20):
        # Sector boundary k is at angle k*18 - 9 (centered at k*18) — boundaries
        # at -9, 9, 27, ... in board-rotated frame
        boundary_deg = k * 18.0 - 9.0
        # Convert to image angle: board frame 0 = sector-20 direction = up = -90 in image
        # The score code does: a = atan2(dy,dx) - angle_offset + 99 ;  boundary at a=k*18.
        # → boundary in image space: atan2 = k*18 - 99 + angle_offset
        img_ang = math.radians(k * 18.0 - 99.0 + angle_offset)
        # corrected vector at edge:
        r_pixels = DOUBLE_OUTER * bull_radius
        dx_c = r_pixels * math.cos(img_ang)
        dy_c = r_pixels * math.sin(img_ang)
        u =  cos_a * dx_c + sin_a * dy_c
        v = -sin_a * dx_c + cos_a * dy_c
        v /= scale
        dx = cos_a * u - sin_a * v
        dy = sin_a * u + cos_a * v
        # Start at outer bull
        r_start = OUTER_BULL_EDGE * bull_radius
        dx_s_c = r_start * math.cos(img_ang)
        dy_s_c = r_start * math.sin(img_ang)
        us = cos_a * dx_s_c + sin_a * dy_s_c
        vs = -sin_a * dx_s_c + cos_a * dy_s_c
        vs /= scale
        sdx = cos_a * us - sin_a * vs
        sdy = sin_a * us + cos_a * vs
        cv2.line(frame, (int(cx + sdx), int(cy + sdy)),
                 (int(cx + dx), int(cy + dy)), (80, 80, 80), 1)

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_MODEL = 'models/yolo/best.pt'
DEFAULT_CONF  = 0.35


def detect_image(frame, yolo, yolo_pose, dart_cls, bull_cls, board_cls, sector_20_cls=None, angle_offset_fallback=0.0, disable_auto_rotation=False, bull50_cls=None):
    """Run detection on a single BGR frame. Returns annotated frame and list of score labels."""
    h, w = frame.shape[:2]
    # Clean grayscale conversion at the very beginning to prevent overlay contamination
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = yolo(frame, conf=DEFAULT_CONF, verbose=False, retina_masks=True)[0]
    seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None

    results_pose = None
    if yolo_pose is not None:
        # Match the pose training resolution (960) for precise tip localisation.
        results_pose = yolo_pose(frame, conf=DEFAULT_CONF, verbose=False, imgsz=960)[0]

    # ── Board FIRST — authoritative source for center, radius, perspective ──
    # Derive board geometry
    ema_ellipse, angle_offset = None, 0.0
    board_poly_active = None
    if board_cls is not None:
        best_board_idx, best_board_conf = -1, 0.0
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) == board_cls:
                conf = float(box.conf[0])
                if conf > best_board_conf:
                    best_board_conf, best_board_idx = conf, i
        if best_board_idx >= 0 and best_board_conf > 0.5 and \
                seg_polys is not None and len(seg_polys[best_board_idx]) >= 5:
            board_poly = seg_polys[best_board_idx].astype(np.float32)
            board_poly = smooth_polygon(board_poly, w, h, kernel_size=15)
            board_poly_active = board_poly
            pts = board_poly.astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 200, 0), 2)
            try:
                (ex, ey), (ax1, ax2), ea = cv2.fitEllipse(board_poly)
                # `ea` from fitEllipse is the WIDTH-axis angle; make it the MAJOR-axis
                # angle (rotate 90° when the height is the major axis).
                major, minor = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
                if ax2 > ax1:
                    ea = (ea + 90.0) % 180.0
                if minor > 10.0:
                    ema_ellipse = (ex, ey, major, minor, ea)
            except cv2.error:
                pass

    # ── Bull center & radius — board-derived when available ─────────────────
    best_bull_idx = -1
    best_bull_conf = 0.0
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == bull_cls:
            conf = float(box.conf[0])
            if conf > best_bull_conf:
                best_bull_conf, best_bull_idx = conf, i

    best_bull50_idx = -1
    best_bull50_conf = 0.0
    if bull50_cls is not None:
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) == bull50_cls:
                conf = float(box.conf[0])
                if conf > best_bull50_conf:
                    best_bull50_conf, best_bull50_idx = conf, i

    best_twenty_idx = -1
    best_twenty_conf = 0.0
    if sector_20_cls is not None:
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) == sector_20_cls:
                conf = float(box.conf[0])
                if conf > best_twenty_conf:
                    best_twenty_conf, best_twenty_idx = conf, i

    # Estimates of center and radius
    new_cx = new_cy = None
    bull_direct_r = None
    r_50_est = None
    r_twenty_est = None

    bull_cx = bull_cy = None
    bull50_cx = bull50_cy = None
    twenty_cx = twenty_cy = None
    twenty_poly = None

    # 1. 25-point circle (outer bull) processing
    if best_bull_idx >= 0:
        bx1, by1, bx2, by2 = results.boxes[best_bull_idx].xyxy[0].tolist()
        cv2.putText(frame, f'Bull25 {best_bull_conf:.2f}', (int(bx1), int(by1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        bull_poly = None
        if seg_polys is not None and len(seg_polys[best_bull_idx]) >= 3:
            bull_poly = smooth_polygon(seg_polys[best_bull_idx].astype(np.float32), w, h, kernel_size=9)

        if bull_poly is not None:
            pts = bull_poly.astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            ibx1, iby1, ibx2, iby2 = int(bx1), int(by1), int(bx2), int(by2)
            mw, mh = ibx2 - ibx1 + 1, iby2 - iby1 + 1
            if mw > 0 and mh > 0:
                mask = np.zeros((mh, mw), dtype=np.uint8)
                shifted_poly = bull_poly - np.array([ibx1, iby1])
                cv2.fillPoly(mask, [shifted_poly.astype(np.int32)], 255)
                M = cv2.moments(mask)
                if M['m00'] > 1e-3:
                    bull_cx = ibx1 + M['m10'] / M['m00']
                    bull_cy = iby1 + M['m01'] / M['m00']
        if bull_cx is None:
            bull_cx = (bx1 + bx2) / 2.0
            bull_cy = (by1 + by2) / 2.0

        if seg_polys is not None and len(seg_polys[best_bull_idx]) >= 5:
            try:
                poly = seg_polys[best_bull_idx].astype(np.float32)
                (ex, ey), (aw, ah), _ = cv2.fitEllipse(poly)
                major_b = max(aw, ah) / 2.0
                minor_b = min(aw, ah) / 2.0
                if minor_b > 1.0 and major_b / minor_b < 2.0:
                    bull_direct_r = major_b * BULL_POLYGON_SCALE
            except cv2.error:
                pass
        if bull_direct_r is None:
            bull_direct_r = (min(bx2 - bx1, by2 - by1) / 2.0) * BULL_POLYGON_SCALE

    # 2. 50-point circle (inner bull) processing
    if best_bull50_idx >= 0:
        b50_x1, b50_y1, b50_x2, b50_y2 = results.boxes[best_bull50_idx].xyxy[0].tolist()
        cv2.putText(frame, f'Bull50 {best_bull50_conf:.2f}', (int(b50_x1), int(b50_y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        bull50_poly = None
        if seg_polys is not None and len(seg_polys[best_bull50_idx]) >= 3:
            bull50_poly = smooth_polygon(seg_polys[best_bull50_idx].astype(np.float32), w, h, kernel_size=9)

        if bull50_poly is not None:
            pts = bull50_poly.astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 255))
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 200, 255), 2)

            ib50_x1, ib50_y1, ib50_x2, ib50_y2 = int(b50_x1), int(b50_y1), int(b50_x2), int(b50_y2)
            mw, mh = ib50_x2 - ib50_x1 + 1, ib50_y2 - ib50_y1 + 1
            if mw > 0 and mh > 0:
                mask = np.zeros((mh, mw), dtype=np.uint8)
                shifted_poly = bull50_poly - np.array([ib50_x1, ib50_y1])
                cv2.fillPoly(mask, [shifted_poly.astype(np.int32)], 255)
                M = cv2.moments(mask)
                if M['m00'] > 1e-3:
                    bull50_cx = ib50_x1 + M['m10'] / M['m00']
                    bull50_cy = ib50_y1 + M['m01'] / M['m00']
        if bull50_cx is None:
            bull50_cx = (b50_x1 + b50_x2) / 2.0
            bull50_cy = (b50_y1 + b50_y2) / 2.0

        if bull50_poly is not None and len(bull50_poly) >= 5:
            try:
                (ex, ey), (aw, ah), _ = cv2.fitEllipse(bull50_poly)
                major_b = max(aw, ah) / 2.0
                minor_b = min(aw, ah) / 2.0
                if minor_b > 1.0 and major_b / minor_b < 2.0:
                    r_50_est = major_b / INNER_BULL_ANNOT_RADIUS
            except cv2.error:
                pass
        if r_50_est is None:
            r_50_est = (min(b50_x2 - b50_x1, b50_y2 - b50_y1) / 2.0) / INNER_BULL_ANNOT_RADIUS

    # 3. Twenty segment processing (Double 20)
    if best_twenty_idx >= 0:
        tw_x1, tw_y1, tw_x2, tw_y2 = results.boxes[best_twenty_idx].xyxy[0].tolist()
        cv2.putText(frame, f'Twenty {best_twenty_conf:.2f}', (int(tw_x1), int(tw_y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        twenty_poly = None
        if seg_polys is not None and len(seg_polys[best_twenty_idx]) >= 3:
            twenty_poly = smooth_polygon(seg_polys[best_twenty_idx].astype(np.float32), w, h, kernel_size=9)

        if twenty_poly is not None:
            pts = twenty_poly.astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 180, 0))
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.polylines(frame, [pts], True, (255, 180, 0), 2)

            itw_x1, itw_y1, itw_x2, itw_y2 = int(tw_x1), int(tw_y1), int(tw_x2), int(tw_y2)
            mw, mh = itw_x2 - itw_x1 + 1, itw_y2 - itw_y1 + 1
            if mw > 0 and mh > 0:
                mask = np.zeros((mh, mw), dtype=np.uint8)
                shifted_poly = twenty_poly - np.array([itw_x1, itw_y1])
                cv2.fillPoly(mask, [shifted_poly.astype(np.int32)], 255)
                M = cv2.moments(mask)
                if M['m00'] > 1e-3:
                    twenty_cx = itw_x1 + M['m10'] / M['m00']
                    twenty_cy = itw_y1 + M['m01'] / M['m00']
        if twenty_cx is None:
            twenty_cx = (tw_x1 + tw_x2) / 2.0
            twenty_cy = (tw_y1 + tw_y2) / 2.0

    # Combine center coordinates using bullseye detections
    local_bull_cx = local_bull_cy = None
    if bull50_cx is not None and bull_cx is not None:
        local_bull_cx = 0.6 * bull50_cx + 0.4 * bull_cx
        local_bull_cy = 0.6 * bull50_cy + 0.4 * bull_cy
    elif bull50_cx is not None:
        local_bull_cx = bull50_cx
        local_bull_cy = bull50_cy
    elif bull_cx is not None:
        local_bull_cx = bull_cx
        local_bull_cy = bull_cy

    # 4. Final center selection (with board symmetry fallback)
    if ema_ellipse is not None and board_poly_active is not None and local_bull_cx is not None:
        bex, bey, _, _, _ = ema_ellipse
        min_x = np.min(board_poly_active[:, 0])
        max_x = np.max(board_poly_active[:, 0])
        board_sym_diff = abs((bex - min_x) - (max_x - bex))
        bull_sym_diff = abs((local_bull_cx - min_x) - (max_x - local_bull_cx))

        if board_sym_diff < 15.0 and bull_sym_diff > 20.0:
            center_x, center_y = bex, bey
        else:
            center_x, center_y = local_bull_cx, local_bull_cy
    elif local_bull_cx is not None:
        center_x, center_y = local_bull_cx, local_bull_cy
    elif ema_ellipse is not None:
        bex, bey, _, _, _ = ema_ellipse
        center_x, center_y = bex, bey
    else:
        center_x, center_y = w / 2.0, h / 2.0

    # 5. Estimate scale from twenty segment polygon boundaries (if center is known).
    # Each vertex's perspective-corrected distance from center is computed; the
    # 95th-percentile anchors to DOUBLE_OUTER and the 5th-percentile to DOUBLE_INNER,
    # giving a direct geometry-anchored estimate instead of a centroid approximation.
    if twenty_poly is not None and center_x is not None and len(twenty_poly) >= 5:
        dists = []
        for pt in twenty_poly:
            raw_dx = float(pt[0]) - center_x
            raw_dy = float(pt[1]) - center_y
            c_dx, c_dy = perspective_correct_vec(raw_dx, raw_dy, ema_ellipse)
            dists.append(math.hypot(c_dx, c_dy))
        dists_arr = np.array(dists)
        r_from_outer = float(np.percentile(dists_arr, 95)) / DOUBLE_OUTER
        r_from_inner = float(np.percentile(dists_arr, 5)) / DOUBLE_INNER
        r_twenty_est = (r_from_outer + r_from_inner) / 2.0
    elif twenty_cx is not None and center_x is not None:
        raw_dx = twenty_cx - center_x
        raw_dy = twenty_cy - center_y
        c_dx, c_dy = perspective_correct_vec(raw_dx, raw_dy, ema_ellipse)
        d_twenty = math.hypot(c_dx, c_dy)
        DOUBLE_MID = (DOUBLE_INNER + DOUBLE_OUTER) / 2.0
        r_twenty_est = d_twenty / DOUBLE_MID

    # 6. Select scale based on preference hierarchy (independent of board annotation style when possible)
    # Sanity-check r_twenty_est against the polygon estimate to reject outliers
    # caused by a bad center or false twenty detection.
    if r_twenty_est is not None:
        ref = bull_direct_r if bull_direct_r is not None else (r_50_est if r_50_est is not None else None)
        if ref is not None:
            if not (0.6 * ref <= r_twenty_est <= 1.7 * ref):
                r_twenty_est = None
        elif not (8.0 <= r_twenty_est <= 60.0):
            r_twenty_est = None

    if r_twenty_est is not None:
        bull_radius = r_twenty_est
    elif bull_direct_r is not None:
        bull_radius = bull_direct_r
    elif r_50_est is not None:
        bull_radius = r_50_est
    elif ema_ellipse is not None:
        bull_radius = ema_ellipse[2] * BULL_TO_DOUBLE_OUTER
    else:
        bull_radius = 19.0

    # ── Angle offset calculation ─────────────────────────────────────────────
    angle_offset = angle_offset_fallback
    if not disable_auto_rotation:
        sector_20_pos = None
        if twenty_cx is not None:
            sector_20_pos = (twenty_cx, twenty_cy)

        if sector_20_pos is not None:
            # 1. Class-derived auto-rotation
            raw_dx = sector_20_pos[0] - center_x
            raw_dy = sector_20_pos[1] - center_y
            c_dx, c_dy = perspective_correct_vec(raw_dx, raw_dy, ema_ellipse)
            angle_offset = math.degrees(math.atan2(c_dy, c_dx)) + 90.0
        elif board_poly_active is not None and ema_ellipse is not None:
            # 2. Fallback: topmost board vertex cluster
            n_top = max(3, len(board_poly_active) // 10)
            top_indices = np.argsort(board_poly_active[:, 1])[:n_top]
            raw_dx = float(np.mean(board_poly_active[top_indices, 0])) - center_x
            raw_dy = float(np.mean(board_poly_active[top_indices, 1])) - center_y
            c_dx, c_dy = perspective_correct_vec(raw_dx, raw_dy, ema_ellipse)
            angle_offset = math.degrees(math.atan2(c_dy, c_dx)) + 90.0

    # ── Geometric B50 boundary circle (always visible regardless of model detection) ──
    if center_x is not None and bull_radius is not None and bull_radius > 0:
        b50_px = max(2, int(round(INNER_BULL_EDGE * bull_radius)))
        cv2.circle(frame, (int(center_x), int(center_y)), b50_px, (0, 200, 255), 1)

    # ── Darts ─────────────────────────────────────────────────────────────────
    # Determine which model is authoritative for darts
    use_pose = (results_pose is not None)
    dart_source = results_pose if use_pose else results
    
    # Class ID for dart in the chosen model
    active_dart_cls = 0 if use_pose else dart_cls

    dart_boxes, dart_confs, dart_orig = [], [], []
    for i, box in enumerate(dart_source.boxes):
        if int(box.cls[0]) == active_dart_cls:
            dart_boxes.append(box.xyxy[0].tolist())
            dart_confs.append(float(box.conf[0]))
            dart_orig.append(i)

    labels = []
    total = 0
    for k in nms_indices(dart_boxes, dart_confs):
        x1, y1, x2, y2 = dart_boxes[k]
        orig_i = dart_orig[k]

        tip_cx, tip_cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if use_pose:
            # Draw matching segmentation polygon if available
            best_seg_idx = -1
            best_iou = 0.3
            for i_seg, box_seg in enumerate(results.boxes):
                if int(box_seg.cls[0]) == dart_cls:
                    seg_bbox = box_seg.xyxy[0].tolist()
                    curr_iou = iou(dart_boxes[k], seg_bbox)
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_seg_idx = i_seg

            if best_seg_idx >= 0 and seg_polys is not None and len(seg_polys[best_seg_idx]) >= 3:
                dart_poly = smooth_polygon(seg_polys[best_seg_idx].astype(np.float32), w, h, kernel_size=9)
                pts = dart_poly.astype(np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

            # Pose keypoints
            tail_x = tail_y = None
            if dart_source.keypoints is not None and dart_source.keypoints.xy is not None:
                kpts = dart_source.keypoints.xy[orig_i].cpu().numpy()
                if len(kpts) >= 2 and not np.allclose(kpts, 0.0):
                    pk_tip, pk_tail = kpts[0], kpts[1]
                    tip_cx, tip_cy = float(pk_tip[0]), float(pk_tip[1])
                    tail_x, tail_y = float(pk_tail[0]), float(pk_tail[1])
                    
                    # Snap the pose keypoint to the shaft edge (small window).
                    tip_cx, tip_cy = refine_dart_tip(gray, (tip_cx, tip_cy), (tail_x, tail_y), search_px=10)
            
            # Fallback if pose keypoints are missing/degenerate
            if tail_x is None:
                best_seg_idx = -1
                best_iou = 0.3
                for i_seg, box_seg in enumerate(results.boxes):
                    if int(box_seg.cls[0]) == dart_cls:
                        seg_bbox = box_seg.xyxy[0].tolist()
                        curr_iou = iou(dart_boxes[k], seg_bbox)
                        if curr_iou > best_iou:
                            best_iou = curr_iou
                            best_seg_idx = i_seg

                if best_seg_idx >= 0 and seg_polys is not None and len(seg_polys[best_seg_idx]) >= 3:
                    poly = seg_polys[best_seg_idx].astype(np.float32)
                    (tip_cx, tip_cy), (tail_x, tail_y) = dart_tip_tail_poly(poly)
                    tip_cx, tip_cy = refine_dart_tip(gray, (tip_cx, tip_cy), (tail_x, tail_y))

            # Draw lines and circles for tip and tail (if tail exists)
            if tail_x is not None:
                cv2.line(frame, (int(tail_x), int(tail_y)), (int(tip_cx), int(tip_cy)), (0, 255, 255), 2)
                cv2.circle(frame, (int(tail_x), int(tail_y)), 5, (255, 0, 0), -1) # Tail blue
        else:
            # Fallback to segmentation polygons
            if seg_polys is not None and len(seg_polys[orig_i]) >= 3:
                poly = seg_polys[orig_i].astype(np.float32)
                (tip_cx, tip_cy), (tail_x, tail_y) = dart_tip_tail_poly(poly)
                tip_cx, tip_cy = refine_dart_tip(
                    gray, (tip_cx, tip_cy), (tail_x, tail_y))
                
                # Draw filled segmentation polygon
                dart_poly = smooth_polygon(seg_polys[orig_i].astype(np.float32), w, h, kernel_size=9)
                pts = dart_poly.astype(np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

        dx, dy = tip_cx - center_x, tip_cy - center_y
        dx_c, dy_c = perspective_correct_vec(dx, dy, ema_ellipse)
        label, _ = score_geometric(center_x, center_y,
                                   center_x + dx_c, center_y + dy_c,
                                   bull_radius, angle_offset)
        labels.append(label)
        total += label_to_score(label)

        # Highlight tip with red dot
        cv2.circle(frame, (int(tip_cx), int(tip_cy)), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"score: {label}", (int(x1) + 5, int(y1) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if labels:
        cv2.putText(frame, f"total: {total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if DRAW_RINGS and center_x is not None:
        draw_scoring_rings(frame, center_x, center_y, bull_radius, ema_ellipse, angle_offset)

    # Draw sector-20 direction indicator — extend to just past the double ring
    if center_x is not None and center_x > 0:
        line_len = int(bull_radius * (DOUBLE_OUTER + 1.5))
        if twenty_cx is not None:
            raw_dx = twenty_cx - center_x
            raw_dy = twenty_cy - center_y
            dist = math.hypot(raw_dx, raw_dy)
            if dist > 0:
                vx = raw_dx / dist
                vy = raw_dy / dist
            else:
                vx, vy = 0.0, -1.0
        else:
            # Fallback: project the angle_offset through inverse perspective
            dir_rad = math.radians(angle_offset - 90.0)
            dx_c = math.cos(dir_rad)
            dy_c = math.sin(dir_rad)
            if ema_ellipse is not None:
                _, _, major, minor, ea = ema_ellipse
                scale = min(major / max(minor, 1.0), 4.0)
                ar = math.radians(ea)
                cos_a, sin_a = math.cos(ar), math.sin(ar)
                u =  cos_a * dx_c + sin_a * dy_c
                v = -sin_a * dx_c + cos_a * dy_c
                v /= scale
                vx = cos_a * u - sin_a * v
                vy = sin_a * u + cos_a * v
                dist = math.hypot(vx, vy)
                if dist > 0:
                    vx /= dist
                    vy /= dist
            else:
                vx, vy = dx_c, dy_c

        end_x = int(center_x + line_len * vx)
        end_y = int(center_y + line_len * vy)
        cv2.line(frame, (int(center_x), int(center_y)), (end_x, end_y), (0, 255, 255), 2)
        cv2.circle(frame, (end_x, end_y), 6, (0, 255, 255), 2)
        cv2.putText(frame, '20', (end_x + 8, end_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame, labels, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='', help='Path to a specific test image')
    ap.add_argument('--all', action='store_true', help='Test every image in data/images/')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--no-rings', dest='rings', action='store_false',
                    help='Disable overlaying the regulation scoring rings on the image')
    ap.set_defaults(rings=True)
    ap.add_argument('--save', default='', help='Optional folder to save annotated images instead of showing them')
    ap.add_argument('--angle-offset', type=float, default=0.0,
                    help="Rotation offset (degrees) to apply to sector lines. Align '20' sector at top.")
    ap.add_argument('--disable-auto-rotation', action='store_true',
                    help="Disable automatic calculation of angle offset from the board polygon/class.")
    args = ap.parse_args()

    global DRAW_RINGS
    DRAW_RINGS = args.rings

    if not Path(args.model).is_file():
        sys.exit(f"ERROR: model not found: {args.model}")

    yolo = YOLO(args.model)
    # Resolve by name so the old 3-class export and the new Darts.yolo26 5-class
    # export ('The 25/50-point circle','arrow','dartboard','twenty') both work.
    names = yolo.names
    bull_cls = resolve_class(names, 'bull', 'bulls', contains=['25'])
    bull50_cls = resolve_class(names, 'bull50', 'inner_bull', contains=['50'])
    dart_cls = resolve_class(names, 'dart', 'darts', 'arrow', 'arrows')
    board_cls = resolve_class(names, 'board', 'boards', 'dartboard', contains=['board'])
    if bull_cls is None or dart_cls is None:
        sys.exit(f"ERROR: model must have bull and dart classes. Found: {list(names.values())}")
    if bull50_cls is not None:
        print(f"INFO: 50-point inner bull class found (id={bull50_cls}, name='{names[bull50_cls]}')")

    sector_20_cls = resolve_class(names, '20', 'twenty', 'sector20', 'sector_20',
                                  contains=['twenty', 'sector20', 'sector_20'])
    if sector_20_cls is not None:
        print(f"INFO: sector-20 class found (id={sector_20_cls}, name='{names[sector_20_cls]}') — auto-rotation will align to it.")

    # Load Pose model if available for hybrid tracking
    yolo_pose = None
    pose_path = Path(args.model).parent / 'pose_best.pt'
    if not pose_path.is_file():
        pose_path = Path('models/yolo/pose_best.pt')
    if pose_path.is_file():
        yolo_pose = YOLO(str(pose_path))
        print(f"Hybrid mode active: using YOLOv11 Pose model {pose_path} for precise dart tip detection.")
    else:
        print("Segmentation mode active (no pose model found). Using segmentation polygons.")

    images_dir = Path('data') / 'images'
    all_imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS)

    if args.image:
        image_paths = [Path(args.image)]
    else:
        if not all_imgs:
            sys.exit(f"No images found in {images_dir}")
        image_paths = all_imgs

    print(f"Model: {args.model}  |  classes: bull={bull_cls} dart={dart_cls} board={board_cls}")
    if args.save:
        print(f"Saving annotations for {len(image_paths)} image(s) to folder: {args.save}\n")
    else:
        print(f"Testing {len(image_paths)} image(s) — press Left/Right Arrow to navigate, 'W' to toggle annotations, 'Q' to quit\n")

    save_dir = Path(args.save) if args.save else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if save_dir is not None:
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  SKIP (unreadable): {path}")
                continue
            out, labels, total = detect_image(frame, yolo, yolo_pose, dart_cls, bull_cls, board_cls,
                                              sector_20_cls=sector_20_cls,
                                              angle_offset_fallback=args.angle_offset,
                                              disable_auto_rotation=args.disable_auto_rotation,
                                              bull50_cls=bull50_cls)
            print(f"  {path.name:40s}  darts={labels}  total={total}")
            cv2.imwrite(str(save_dir / path.name), out)
    else:
        idx = 0
        show_annotations = True
        last_idx = -1
        last_show_annotations = None

        while True:
            if not image_paths:
                print("No images to test.")
                break

            path = image_paths[idx]
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  SKIP (unreadable): {path}")
                image_paths.pop(idx)
                if idx >= len(image_paths):
                    idx = 0
                continue

            if show_annotations:
                annotated_frame = frame.copy()
                out, labels, total = detect_image(annotated_frame, yolo, yolo_pose, dart_cls, bull_cls, board_cls,
                                                  sector_20_cls=sector_20_cls,
                                                  angle_offset_fallback=args.angle_offset,
                                                  disable_auto_rotation=args.disable_auto_rotation,
                                                  bull50_cls=bull50_cls)
                if idx != last_idx or show_annotations != last_show_annotations:
                    print(f"  {path.name:40s}  darts={labels}  total={total}")
                cv2.imshow('Darts Test', out)
            else:
                if idx != last_idx or show_annotations != last_show_annotations:
                    print(f"  {path.name:40s}  [Annotations Hidden]")
                cv2.imshow('Darts Test', frame)

            last_idx = idx
            last_show_annotations = show_annotations

            k = cv2.waitKey(0)
            k_code = k & 0xFF

            if k_code == ord('q') or k == 27:
                break
            elif k_code == ord('w') or k_code == ord('W'):
                show_annotations = not show_annotations
            elif k == 81 or k == 65361 or k == 2424832 or k_code == 81:
                idx = (idx - 1) % len(image_paths)
            elif k == 83 or k == 65363 or k == 2424834 or k_code == 83:
                idx = (idx + 1) % len(image_paths)

    if save_dir is None:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
