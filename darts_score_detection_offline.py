"""Offline darts score detector — YOLO11 for detection, deterministic geometric scoring.

The DNN classifier was removed (ACC-2): a regulation dartboard's score zones are
fully determined by (distance, angle) relative to the bull, so a lookup table is
more reliable and interpretable than a 60-class MLP, and removes the
features.tsv → training → ONNX dependency.

Requires:  pip install ultralytics

Example:
  python darts_score_detection_offline.py --input 0
  python darts_score_detection_offline.py --input path/to/video.mp4
  python darts_score_detection_offline.py --input path/to/image.jpg
  python darts_score_detection_offline.py --input rtsp://user:pass@ip/stream
  python darts_score_detection_offline.py --input 1 --capture
"""

import argparse
import math
import os
import pathlib
import sys
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO


# ── Dartboard geometry (regulation board, distances in OuterBull-radius units) ─
# OuterBull diameter = 31.8 mm → radius 15.9 mm is our unit.
OUTER_BULL_MM   = 15.9
DOUBLE_OUTER_MM = 170.0
# Calibrated geometric constants matching the Roboflow dataset annotation bias:
# 1. Board-derived scale factor (board polygon traces the entire board boundary)
BULL_TO_DOUBLE_OUTER = 0.07642

# 2. Outer Bull (class 0 'The 25-point circle') annotation scale correction factor
# Calibrated from 496 images: mean(r_twenty_est / ellipse_major) = 0.714
BULL_POLYGON_SCALE = 0.714

# 3. Inner Bull (class 1 'The 50-point circle') annotation scale correction divisor
INNER_BULL_ANNOT_RADIUS = 0.49448

INNER_BULL_EDGE = 6.35 / 15.9          # 0.399  — B50 inside this
OUTER_BULL_EDGE = 1.0                  #         — B25 up to here
TRIPLE_INNER    = 99.0 / 15.9          # 6.226
TRIPLE_OUTER    = 107.0 / 15.9         # 6.730
DOUBLE_INNER    = 162.0 / 15.9         # 10.189
DOUBLE_OUTER    = 170.0 / 15.9         # 10.692

# Sector numbers walking clockwise from "20" at the top of the board.
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def score_geometric(cx, cy, tip_x, tip_y, bull_radius, angle_offset_deg=0.0):
    """Return (label_str, score_int) from board geometry.

    cx, cy        — bull center in image pixels
    tip_x, tip_y  — dart tip in image pixels
    bull_radius   — OuterBull radius in image pixels
    angle_offset_deg — rotate so the "20" sector is at the top of the image
    """
    if bull_radius <= 0:
        return "?", 0
    dx = tip_x - cx
    dy = tip_y - cy
    r = math.hypot(dx, dy) / bull_radius

    if r < INNER_BULL_EDGE:
        return "B50", 50
    if r <= OUTER_BULL_EDGE:
        return "B25", 25
    if r > DOUBLE_OUTER:
        return "Miss", 0

    # Angle relative to image-right axis. Image y grows downward; "20" is at
    # top-center which is dy<0 → atan2 returns ~-90°. We shift so that the "20"
    # sector starts at 0° in the rotated frame.
    angle_deg = math.degrees(math.atan2(dy, dx)) - angle_offset_deg
    a = (angle_deg + 99.0) % 360.0   # +99° = +90° (move 20-center to 0) + 9° (half sector)
    sector_idx = int(a // 18.0) % 20
    sector_num = SECTORS[sector_idx]

    if TRIPLE_INNER <= r <= TRIPLE_OUTER:
        return f"T{sector_num}", 3 * sector_num
    if DOUBLE_INNER <= r <= DOUBLE_OUTER:
        return f"D{sector_num}", 2 * sector_num
    return f"S{sector_num}", sector_num


# ── Module-level helpers ──────────────────────────────────────────────────────
def load_env_file(filepath=".env"):
    """Load key-value pairs from a .env file into os.environ if it exists."""
    p = pathlib.Path(filepath)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                            v = v[1:-1]
                        os.environ[k] = v
        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)


def redact_rtsp_url(url: str) -> str:
    """Mask the credentials and IP/host of an RTSP URL for safe logging."""
    if not isinstance(url, str) or not url.lower().startswith(('rtsp://', 'rtsps://')):
        return url
    try:
        prefix = "rtsps://" if url.lower().startswith('rtsps://') else "rtsp://"
        rem = url[len(prefix):]
        if '@' in rem:
            creds, rest = rem.split('@', 1)
            if ':' in creds:
                user, _ = creds.split(':', 1)
                masked_creds = f"{user}:******"
            else:
                masked_creds = "******"
        else:
            masked_creds = ""
            rest = rem

        # Find first slash or question mark in rest
        sep_idx = len(rest)
        for char in ('/', '?'):
            idx = rest.find(char)
            if idx != -1:
                sep_idx = min(sep_idx, idx)

        host_port = rest[:sep_idx]
        path_query = rest[sep_idx:]

        if ':' in host_port:
            _, port = host_port.rsplit(':', 1)
            masked_host = f"<redacted_host>:{port}"
        else:
            masked_host = "<redacted_host>"

        creds_part = f"{masked_creds}@" if masked_creds else ""
        return f"{prefix}{creds_part}{masked_host}{path_query}"
    except Exception:
        return f"{url[:7]}******"


def resolve_class(names: dict, *aliases, contains=None):
    """Resolve a YOLO class id from {id: name} by exact alias or substring match.

    `aliases` are matched case-insensitively against the full class name first;
    `contains` (iterable of lowercase substrings) is tried second. Returns the
    class id or None. Lets the same code drive datasets whose class names differ
    (e.g. 'bulls' vs 'The 25-point circle', 'board' vs 'dartboard').
    """
    lower = {v.lower(): k for k, v in names.items()}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    if contains:
        for cid, name in names.items():
            nl = name.lower()
            if any(sub in nl for sub in contains):
                return cid
    return None


def label_to_score(label: str) -> int:
    """Convert a score label ('T20', 'D10', 'S5', 'B50', 'B25', 'Miss') to its integer value."""
    if label == 'B50':
        return 50
    if label == 'B25':
        return 25
    if len(label) >= 2 and label[0] in ('T', 'D', 'S'):
        try:
            n = int(label[1:])
            return {'T': 3, 'D': 2, 'S': 1}[label[0]] * n
        except ValueError:
            pass
    return 0


def iou(boxA, boxB):
    """IoU between two boxes given as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_indices(boxes, confs, iou_thresh=0.5):
    """Return indices of boxes kept after greedy NMS (sorted by conf desc)."""
    order = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    kept = []
    for idx in order:
        if any(iou(boxes[idx], boxes[k]) > iou_thresh for k in kept):
            continue
        kept.append(idx)
    return kept


def dart_tip_from_endpoints(poly, end_a, end_b):
    """Return the dart tip as the more acute of the two polygon endpoints.

    Uses interior angle at each endpoint — dart tips come to a sharp point,
    flights/tails are wider. Robust to board position and camera angle.
    """
    n = len(poly)
    k = max(2, n // 12)

    def _angle(ep):
        idx = int(np.argmin((poly[:, 0] - ep[0]) ** 2 + (poly[:, 1] - ep[1]) ** 2))
        v1 = poly[(idx - k) % n] - poly[idx]
        v2 = poly[(idx + k) % n] - poly[idx]
        d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if d1 < 1e-6 or d2 < 1e-6:
            return math.pi
        return math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / (d1 * d2))))

    return end_a if _angle(end_a) <= _angle(end_b) else end_b


def dart_tip_tail_poly(poly):
    """Return ((tip_x, tip_y), (tail_x, tail_y)) for a dart polygon.

    Endpoints from the PCA major axis; the tip is the *narrower* end (the barrel
    tapers to a point while the flight/tail is wide), with vertex acuteness as a
    tiebreak. This mirrors convert_seg_to_pose.py so inference and the trained
    labels agree on which end is the tip.
    """
    c = poly.mean(axis=0)
    pts_c = poly - c
    try:
        eigvecs = np.linalg.eigh(np.cov(pts_c.T))[1]
    except np.linalg.LinAlgError:
        return (float(poly[0][0]), float(poly[0][1])), (float(poly[-1][0]), float(poly[-1][1]))
    major, minor = eigvecs[:, -1], eigvecs[:, 0]
    proj_maj = pts_c @ major
    proj_min = pts_c @ minor
    end_a = poly[int(np.argmin(proj_maj))]
    end_b = poly[int(np.argmax(proj_maj))]

    L = float(proj_maj.max() - proj_maj.min())
    if L >= 1e-6:
        band = 0.25 * L
        a_band = proj_min[proj_maj <= proj_maj.min() + band]
        b_band = proj_min[proj_maj >= proj_maj.max() - band]
        width_a = float(a_band.max() - a_band.min()) if a_band.size >= 2 else 0.0
        width_b = float(b_band.max() - b_band.min()) if b_band.size >= 2 else 0.0
        if abs(width_a - width_b) > 0.15 * max(width_a, width_b, 1e-6):
            tip, tail = (end_a, end_b) if width_a < width_b else (end_b, end_a)
            return (float(tip[0]), float(tip[1])), (float(tail[0]), float(tail[1]))

    tip = dart_tip_from_endpoints(poly, end_a, end_b)
    tail = end_b if np.allclose(tip, end_a) else end_a
    return (float(tip[0]), float(tip[1])), (float(tail[0]), float(tail[1]))


def refine_dart_tip(gray, approx_tip, tail_pt, search_px=14, strip_half=4, grad_thresh=None):
    """Refine approximate dart tip along the shaft using gradient magnitude profile.

    Scans outwards from the tail/approx_tip and finds where the shaft gradient drops.
    Only commits a refinement when a *confirmed* gradient drop is found (the shaft
    edge ends); on a cluttered board where the gradient never drops the original
    tip is kept rather than drifting outward into a neighbouring ring.
    Falls back to approx_tip when geometry is degenerate.
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
    # Scan a bit deeper inside the shaft and further outside
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

    # Sample gradient profile along the shaft
    t_vals = list(range(-10, search_px + 1))
    profile = []
    for t in t_vals:
        profile.append(_peak(tip + t * shaft_dir))
        
    profile = np.array(profile)

    # Baseline shaft gradient from samples inside the dart body (t <= 0).
    inside_indices = [i for i, t in enumerate(t_vals) if t <= 0]
    if not inside_indices:
        return approx_tip
    max_inside_grad = float(np.max(profile[inside_indices]))
    if max_inside_grad < 1e-3:
        return approx_tip
    thresh = max(40.0, 0.45 * max_inside_grad)

    # The tip is where the strong shaft gradient ends. Walk outward and accept
    # the first sample that is still strong but is followed by a *confirmed*
    # drop (two consecutive samples below threshold). If no such drop exists
    # within the window the scene is too cluttered to trust — keep the original
    # tip instead of pushing it to the end of the search range.
    best_t = None
    for i, t in enumerate(t_vals):
        if t < 0:
            continue
        ahead = profile[i + 1:i + 3]
        if profile[i] >= thresh and ahead.size >= 2 and np.all(ahead < thresh):
            best_t = t
            break
    if best_t is None:
        return approx_tip

    refined = tip + best_t * shaft_dir
    return (float(refined[0]), float(refined[1]))


def perspective_correct_vec(dx, dy, ema_ellipse):
    """Apply fitEllipse-derived affine de-foreshortening to a (dx, dy) vector.

    Stretches the minor-axis component by major/minor to restore the circular
    board geometry distorted by a non-perpendicular camera.
    Returns corrected (dx, dy). If ema_ellipse is None or degenerate, returns input unchanged.
    """
    if ema_ellipse is None:
        return dx, dy
    _ex, _ey, major, minor, ea = ema_ellipse
    if minor < 1.0 or major < minor:
        return dx, dy
    scale = min(major / minor, 4.0)
    angle_rad = math.radians(ea)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    u =  cos_a * dx + sin_a * dy   # component along major axis (unchanged)
    v = -sin_a * dx + cos_a * dy   # component along minor axis (compressed)
    v *= scale
    return cos_a * u - sin_a * v, sin_a * u + cos_a * v


# ── Projective (homography) board rectification ───────────────────────────────
# The affine de-foreshortening above only removes the ellipse axis-compression;
# a tilted plane also has a *projective* foreshortening gradient (the near edge
# images larger than the far edge) that the affine model cannot capture. That
# residual is largest at the rim — the thin double/triple bands. The functions
# below rectify the board plane exactly from a single ellipse + the bull centre.
def _ellipse_to_conic(cx, cy, a, b, theta_deg):
    """Return the 3x3 conic matrix C (p^T C p = 0 on the ellipse), p=[x,y,1]."""
    th = math.radians(theta_deg)
    ct, st = math.cos(th), math.sin(th)
    R = np.array([[ct, st], [-st, ct]])           # world -> ellipse frame (u = R(x-c))
    D = np.array([[1.0 / (a * a), 0.0], [0.0, 1.0 / (b * b)]])
    Q = R.T @ D @ R
    c = np.array([cx, cy], dtype=np.float64)
    Qc = Q @ c
    return np.array([
        [Q[0, 0], Q[0, 1], -Qc[0]],
        [Q[1, 0], Q[1, 1], -Qc[1]],
        [-Qc[0], -Qc[1], float(c @ Q @ c - 1.0)],
    ])


def _conic_to_ellipse(C):
    """Return (cx, cy, a, b, theta_deg) of the ellipse described by conic C, or None."""
    A = C[0, 0]; B = 2.0 * C[0, 1]; Cc = C[1, 1]
    Dd = 2.0 * C[0, 2]; E = 2.0 * C[1, 2]; F = C[2, 2]
    M = np.array([[A, B / 2.0], [B / 2.0, Cc]])
    det_m = A * Cc - (B / 2.0) ** 2
    if det_m <= 1e-12:                              # not an ellipse (parabola/hyperbola)
        return None
    try:
        cen = np.linalg.solve(np.array([[2 * A, B], [B, 2 * Cc]]), np.array([-Dd, -E]))
    except np.linalg.LinAlgError:
        return None
    Fc = (A * cen[0] ** 2 + B * cen[0] * cen[1] + Cc * cen[1] ** 2
          + Dd * cen[0] + E * cen[1] + F)
    evals, evecs = np.linalg.eigh(M)
    if np.any(evals <= 0) or Fc >= 0:              # degenerate / empty conic
        return None
    axes = np.sqrt(-Fc / evals)
    i_maj = int(np.argmax(axes)); i_min = 1 - i_maj
    v = evecs[:, i_maj]
    theta = math.degrees(math.atan2(v[1], v[0]))
    return float(cen[0]), float(cen[1]), float(axes[i_maj]), float(axes[i_min]), theta


def board_rectifier(ema_ellipse, cx, cy):
    """Build a homography mapping image pixels → a metric board plane.

    Returns (H, Hinv) such that ``H @ [px,py,1]`` (de-homogenised) is the offset
    from the bull centre in a rectified frame where the scoring rings are true
    concentric circles and sectors are exactly 18°. The rectified scale is fixed
    so the board radius equals the ellipse major axis (keeping bull_radius in the
    same pixel range as the affine path). Returns None when the geometry is
    degenerate or implies an implausible tilt — the caller then falls back to the
    affine `perspective_correct_vec`.
    """
    if ema_ellipse is None:
        return None
    ex, ey, major, minor, ea = ema_ellipse
    if minor < 5.0 or major < minor:
        return None

    C = _ellipse_to_conic(ex, ey, major, minor, ea)
    b = np.array([cx, cy, 1.0])
    l = C @ b                                       # vanishing line = polar of bull centre
    if abs(l[2]) < 1e-12:
        return None
    l = l / l[2]
    # Reject implausible perspective: the vanishing line should be far outside the
    # board (a small l-gradient). |l0|,|l1| ~ 1/distance-to-horizon in px^-1.
    if math.hypot(l[0], l[1]) * major > 2.0:
        return None

    Hp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [l[0], l[1], 1.0]])
    try:
        Hp_inv = np.linalg.inv(Hp)
    except np.linalg.LinAlgError:
        return None

    C_q = Hp_inv.T @ C @ Hp_inv                     # board conic after perspective removal
    ell_q = _conic_to_ellipse(C_q)
    if ell_q is None:
        return None
    _oqx, _oqy, aq, bq, phiq = ell_q
    if bq < 1e-6 or aq < 1e-6:
        return None

    th = math.radians(phiq); ct, st = math.cos(th), math.sin(th)
    Rphi = np.array([[ct, -st], [st, ct]])          # ellipse frame -> q
    Sdiag = np.array([[1.0, 0.0], [0.0, aq / bq]])  # stretch minor up to major
    S = (Rphi @ Sdiag @ Rphi.T) * (major / aq)      # isotropise + scale to `major`

    Oq = Hp @ b
    Oq = Oq[:2] / Oq[2]                              # bull centre in q-space = new origin
    Aq = np.eye(3)
    Aq[:2, :2] = S
    Aq[:2, 2] = -S @ Oq
    H = Aq @ Hp
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    return H, Hinv


def rect_offset(H, px, py):
    """Map an image pixel to its rectified offset from the bull centre."""
    v = H @ np.array([px, py, 1.0])
    return v[0] / v[2], v[1] / v[2]


def rect_to_image(Hinv, ox, oy):
    """Map a rectified offset (from the bull centre) back to an image pixel."""
    v = Hinv @ np.array([ox, oy, 1.0])
    return v[0] / v[2], v[1] / v[2]


def rectifier_circularity(H, ellipse_params):
    """std/mean of the rectified radii of an ellipse — 0 means it became a circle.

    A correct board homography must map any concentric board circle (e.g. the
    25-point bull) to a circle in the rectified plane. A large value means the
    recovered geometry is wrong (bad board ellipse or bull centre) → fall back.
    """
    cx, cy, a, b, ang = ellipse_params
    th = math.radians(ang); ct, st = math.cos(th), math.sin(th)
    rs = []
    for t in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        u, v = a * math.cos(t), b * math.sin(t)
        ox, oy = rect_offset(H, cx + u * ct - v * st, cy + u * st + v * ct)
        rs.append(math.hypot(ox, oy))
    rs = np.array(rs)
    return float(rs.std() / (rs.mean() + 1e-9))


def draw_scoring_rings(frame, cx, cy, bull_radius, ema_ellipse):
    """Overlay the regulation scoring rings as perspective ellipses.

    Each ring is a circle of radius (r_units * bull_radius) in the de-foreshortened
    board frame; we map every sampled point back through the inverse of
    perspective_correct_vec so the drawn ring matches the tilted board exactly
    (a tilted circle images as an ellipse, not a circle).
    """
    if bull_radius is None or bull_radius <= 0:
        return
    if ema_ellipse is not None:
        _, _, major, minor, ea = ema_ellipse
        scale = min(major / max(minor, 1.0), 4.0)
        ar = math.radians(ea)
        cos_a, sin_a = math.cos(ar), math.sin(ar)
    else:
        scale = 1.0
        cos_a, sin_a = 1.0, 0.0

    rings = [
        (INNER_BULL_EDGE, (0, 200, 255)),
        (OUTER_BULL_EDGE, (0, 255, 0)),
        (TRIPLE_INNER,    (255, 100,   0)),
        (TRIPLE_OUTER,    (255,  60,   0)),
        (DOUBLE_INNER,    (  0, 140, 255)),
        (DOUBLE_OUTER,    (  0, 100, 255)),
    ]
    angs = np.linspace(0, 2 * math.pi, 96)
    for r_units, color in rings:
        r_px = r_units * bull_radius
        pts = []
        for ang in angs:
            dx_c = r_px * math.cos(ang)
            dy_c = r_px * math.sin(ang)
            # Inverse perspective: undo the minor-axis stretch.
            u =  cos_a * dx_c + sin_a * dy_c
            v = -sin_a * dx_c + cos_a * dy_c
            v /= scale
            pts.append((int(cx + cos_a * u - sin_a * v),
                        int(cy + sin_a * u + cos_a * v)))
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, color, 1)


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


class Track:
    """Per-dart temporal track that votes on a stable label across frames."""
    __slots__ = ('bbox', 'votes', 'last_seen_frame')

    def __init__(self, bbox, label, last_seen_frame):
        self.bbox = bbox
        self.votes = Counter([label])
        self.last_seen_frame = last_seen_frame

    def push(self, label):
        self.votes[label] += 1

    def majority(self):
        top, top_count = self.votes.most_common(1)[0]
        total = sum(self.votes.values())
        return top, top_count, total


def main():
    load_env_file()
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=os.environ.get('RTSP_URL', '0'),
                   help="Webcam index (e.g. '0'), RTSP URL, or path to video/image file")
    p.add_argument('--yolo-model', default='models/yolo/best.pt',
                   help="Path to YOLO11/YOLOv8 weights (.pt)")
    p.add_argument('--yolo-pose', default='models/yolo/pose_best.pt',
                   help="Path to YOLO11 Pose weights (.pt). If present, enables hybrid tracking.")
    p.add_argument('--conf', type=float, default=0.35,
                   help="YOLO confidence threshold")
    p.add_argument('--output', default='',
                   help="Optional output video path (e.g. out.mp4)")
    p.add_argument('--capture', action='store_true',
                   help="Enable SPACE-to-save screenshots of the raw (un-annotated) frame")
    p.add_argument('--capture-dir', default='data/images',
                   help="Folder where screenshots are saved when --capture is set")
    p.add_argument('--raw-view', action='store_true',
                   help="Show the raw camera fxeed in the display window without overlays")
    p.add_argument('--angle-offset', type=float, default=0.0,
                   help="Rotation offset (degrees) to apply to sector lines. Align '20' sector at top.")
    p.add_argument('--disable-auto-rotation', action='store_true',
                   help="Disable automatic calculation of angle offset from the board polygon/class.")
    args = p.parse_args()

    # ── INF-3: verify YOLO weights exist before doing anything ──────────────
    if not pathlib.Path(args.yolo_model).is_file():
        print(f"ERROR: YOLO weights missing: {args.yolo_model} — run yolo_training.py to produce them.",
              file=sys.stderr)
        sys.exit(1)

    # ── YOLO detector ───────────────────────────────────────────────────────
    yolo = YOLO(args.yolo_model)
    print(f"YOLO model loaded: {args.yolo_model}")
    print(f"Class names: {yolo.names}")

    # Load Pose model if available for hybrid tracking
    yolo_pose = None
    if pathlib.Path(args.yolo_pose).is_file():
        yolo_pose = YOLO(args.yolo_pose)
        print(f"YOLO Pose model loaded: {args.yolo_pose} (Hybrid mode active)")
    else:
        pose_fallback = pathlib.Path(args.yolo_model).parent / 'pose_best.pt'
        if pose_fallback.is_file():
            yolo_pose = YOLO(str(pose_fallback))
            print(f"YOLO Pose model loaded: {pose_fallback} (Hybrid mode active)")
        else:
            print("INFO: No YOLO Pose weights found. Falling back to segmentation polygons.")

    # Resolve class indices by name so class order in data.yaml doesn't matter.
    # Works across the old 3-class export ('arrow','board','bulls') and the new
    # Darts.yolo26 5-class export ('The 25/50-point circle','arrow','dartboard','twenty').
    names = yolo.names
    # BULL = outer-bull / 25-point circle: its fitted radius is the scoring unit.
    BULL_CLS = resolve_class(names, 'bull', 'bulls', contains=['25'])
    BULL50_CLS = resolve_class(names, 'bull50', 'inner_bull', contains=['50'])
    DART_CLS = resolve_class(names, 'dart', 'darts', 'arrow', 'arrows')
    if BULL_CLS is None or DART_CLS is None:
        print(f"ERROR: model must have a bull and a dart class. Found: {list(names.values())}", file=sys.stderr)
        sys.exit(1)
    if BULL50_CLS is not None:
        print(f"INFO: 50-point inner bull class found (id={BULL50_CLS}, name='{names[BULL50_CLS]}')")
    BOARD_CLS = resolve_class(names, 'board', 'boards', 'dartboard', contains=['board'])
    if BOARD_CLS is None:
        print("INFO: model has no 'board' class — perspective correction and auto-rotation disabled.")
    else:
        print(f"INFO: 'board' class found (id={BOARD_CLS}) — perspective correction and auto-rotation enabled.")

    # Sector-20 landmark: matches '20'/'sector20'/'sector_20' or the word 'twenty'.
    SECTOR_20_CLS = resolve_class(names, '20', 'twenty', 'sector20', 'sector_20',
                                  contains=['twenty', 'sector20', 'sector_20'])
    if SECTOR_20_CLS is not None:
        print(f"INFO: sector-20 class found (id={SECTOR_20_CLS}, name='{names[SECTOR_20_CLS]}') — auto-rotation will align to it.")

    # ── Input source ────────────────────────────────────────────────────────
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    is_image = os.path.isfile(args.input) and args.input.lower().endswith(image_exts)
    is_rtsp = args.input.lower().startswith(('rtsp://', 'rtsps://'))

    if is_image:
        frames = [cv2.imread(args.input)]
        if frames[0] is None:
            print(f"Failed to read image: {redact_rtsp_url(args.input)}", file=sys.stderr)
            sys.exit(1)
        cap = None
    else:
        if args.input.isdigit():
            src = int(args.input)
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
        elif is_rtsp:
            cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Failed to open input: {redact_rtsp_url(args.input)}", file=sys.stderr)
            if not is_rtsp:
                # BUG-3: explicit loop to avoid leaking VideoCapture handles
                available = []
                for i in range(4):
                    probe = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    if probe.isOpened():
                        available.append(i)
                    probe.release()
                print("Available devices: " + " ".join(str(i) for i in available),
                      file=sys.stderr)
            sys.exit(1)
        if is_rtsp:
            print(f"RTSP stream opened: {redact_rtsp_url(args.input)}")
        frames = None

    writer = None
    writer_w = None
    writer_h = None
    # ACC-9: EMA-smoothed bull center/radius across frames
    ema_cx = None
    ema_cy = None
    ema_r = None
    bull_radius_fallback = 19.0  # fallback until YOLO detects the bull

    # ACC-8: temporal majority vote per dart track (only used for video)
    tracks = []
    frame_idx = [0]
    TRACK_IOU_THRESH = 0.4
    TRACK_MAX_AGE = 30
    ema_ellipse = None        # (ex, ey, major, minor, ea) — board ellipse EMA
    ema_angle_offset = None   # auto-detected board rotation (deg); None → use --angle-offset

    def process(frame, use_tracker=True):
        nonlocal writer, writer_w, writer_h
        nonlocal ema_cx, ema_cy, ema_r
        nonlocal ema_ellipse, ema_angle_offset
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = yolo(frame, conf=args.conf, verbose=False, retina_masks=True)[0]
        results_pose = None
        if yolo_pose is not None:
            # Match the pose training resolution (960) so tiny dart tips are
            # localised at the precision the model was trained for.
            results_pose = yolo_pose(frame, conf=args.conf, verbose=False, imgsz=960)[0]

        seg_polys = None
        if results.masks is not None and results.masks.xy is not None:
            seg_polys = results.masks.xy  # list of (N,2) arrays, one per box

        # ── Board detection FIRST — authoritative source for center, radius, perspective ─
        # The board polygon is large and immune to dart occlusion. The bull polygon
        # often gets contaminated by overlapping dart shafts, so we derive bull
        # center + radius from the board ellipse via the regulation ratio
        # (OuterBull 15.9 mm / Double-outer 170 mm = 0.0935).
        board_poly_active = None
        if BOARD_CLS is not None:
            best_board_idx = -1
            best_board_conf = 0.0
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) != BOARD_CLS:
                    continue
                conf = float(box.conf[0])
                if conf > best_board_conf:
                    best_board_conf = conf
                    best_board_idx = i

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
                    # cv2.fitEllipse returns (width, height) and `ea` is the angle of
                    # the WIDTH axis. We want `ea` to be the MAJOR-axis angle: when the
                    # height is the major axis (ax2 > ax1) rotate by 90°. (Previously
                    # this branch was inverted, leaving ea on the minor axis so the
                    # affine de-foreshortening stretched the wrong direction.)
                    major = max(ax1, ax2) / 2.0
                    minor = min(ax1, ax2) / 2.0
                    if ax2 > ax1:   # height > width: major axis is along ea+90°
                        ea = (ea + 90.0) % 180.0
                    if minor > 10.0:
                        if ema_ellipse is None:
                            ema_ellipse = (ex, ey, major, minor, ea)
                        else:
                            oex, oey, omaj, omin, oea = ema_ellipse
                            da = ((ea - oea + 90.0) % 180.0) - 90.0
                            ema_ellipse = (
                                0.85 * oex  + 0.15 * ex,
                                0.85 * oey  + 0.15 * ey,
                                0.85 * omaj + 0.15 * major,
                                0.85 * omin + 0.15 * minor,
                                oea + 0.15 * da,
                            )
                except cv2.error:
                    pass

        # ── Bull center & radius ────────────────────────────────────────────
        # Preferred path: derive from board ellipse (robust to dart occlusion).
        # Fallback path: trust the bull polygon when (a) no board class in the
        # model, or (b) board confidence too low. Even then, prefer the MAJOR
        # axis of the bull ellipse — the major axis equals the true diameter
        # (foreshortening only shrinks the minor axis).
        best_bull_idx = -1
        best_bull_conf = 0.0
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) == BULL_CLS:
                conf = float(box.conf[0])
                if conf > best_bull_conf:
                    best_bull_conf = conf
                    best_bull_idx = i

        best_bull50_idx = -1
        best_bull50_conf = 0.0
        if BULL50_CLS is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == BULL50_CLS:
                    conf = float(box.conf[0])
                    if conf > best_bull50_conf:
                        best_bull50_conf = conf
                        best_bull50_idx = i

        best_twenty_idx = -1
        best_twenty_conf = 0.0
        if SECTOR_20_CLS is not None:
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == SECTOR_20_CLS:
                    conf = float(box.conf[0])
                    if conf > best_twenty_conf:
                        best_twenty_conf = conf
                        best_twenty_idx = i

        # Estimates of center and radius
        new_cx = new_cy = None
        bull_direct_r = None
        r_50_est = None
        r_twenty_est = None

        bull_cx = bull_cy = None
        bull50_cx = bull50_cy = None
        twenty_cx = twenty_cy = None
        twenty_poly = None
        bull_ellipse_params = None   # (cx,cy,major,minor,angle) — validates the homography

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
                    (ex, ey), (aw, ah), bang = cv2.fitEllipse(poly)
                    major_b = max(aw, ah) / 2.0
                    minor_b = min(aw, ah) / 2.0
                    if minor_b > 1.0 and major_b / minor_b < 2.0:
                        bull_direct_r = major_b * BULL_POLYGON_SCALE
                        if ah > aw:
                            bang = (bang + 90.0) % 180.0
                        bull_ellipse_params = (ex, ey, major_b, minor_b, bang)
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
                new_cx, new_cy = bex, bey
            else:
                new_cx, new_cy = local_bull_cx, local_bull_cy
        elif local_bull_cx is not None:
            new_cx, new_cy = local_bull_cx, local_bull_cy
        elif ema_ellipse is not None:
            bex, bey, _, _, _ = ema_ellipse
            new_cx, new_cy = bex, bey

        # ── Board rectifier (projective de-foreshortening) ────────────────────
        # Build one homography per frame mapping image px → metric board plane
        # (rings become true concentric circles). Anchored on the stable EMA bull
        # centre when available. Falls back to the affine model if the geometry is
        # degenerate or the homography fails to round-trip the bull to a circle.
        rect_cx = ema_cx if ema_cx is not None else new_cx
        rect_cy = ema_cy if ema_cy is not None else new_cy
        rectifier = None
        if ema_ellipse is not None and rect_cx is not None:
            cand = board_rectifier(ema_ellipse, rect_cx, rect_cy)
            if cand is not None and (bull_ellipse_params is None
                                     or rectifier_circularity(cand[0], bull_ellipse_params) < 0.15):
                rectifier = cand

        def corrected_offset(px, py):
            """Offset of an image point from the bull centre in de-foreshortened units."""
            if rectifier is not None:
                return rect_offset(rectifier[0], px, py)
            cx0 = rect_cx if rect_cx is not None else 0.0
            cy0 = rect_cy if rect_cy is not None else 0.0
            return perspective_correct_vec(px - cx0, py - cy0, ema_ellipse)

        # 5. Estimate scale from twenty segment polygon boundaries (if center is known).
        # Each vertex's de-foreshortened distance from center is computed; the
        # 95th-percentile anchors to DOUBLE_OUTER and the 5th-percentile to DOUBLE_INNER,
        # giving a direct geometry-anchored estimate instead of a centroid approximation.
        if twenty_poly is not None and new_cx is not None and len(twenty_poly) >= 5:
            dists = []
            for pt in twenty_poly:
                c_dx, c_dy = corrected_offset(float(pt[0]), float(pt[1]))
                dists.append(math.hypot(c_dx, c_dy))
            dists_arr = np.array(dists)
            r_from_outer = float(np.percentile(dists_arr, 95)) / DOUBLE_OUTER
            r_from_inner = float(np.percentile(dists_arr, 5)) / DOUBLE_INNER
            r_twenty_est = (r_from_outer + r_from_inner) / 2.0
        elif twenty_cx is not None and new_cx is not None:
            c_dx, c_dy = corrected_offset(twenty_cx, twenty_cy)
            d_twenty = math.hypot(c_dx, c_dy)
            DOUBLE_MID = (DOUBLE_INNER + DOUBLE_OUTER) / 2.0
            r_twenty_est = d_twenty / DOUBLE_MID

        # 6. Select scale based on preference hierarchy.
        # Sanity-check r_twenty_est: reject if it's implausible relative to the
        # polygon estimate (outlier caused by a bad center or false detection).
        if r_twenty_est is not None:
            ref = bull_direct_r if bull_direct_r is not None else (r_50_est if r_50_est is not None else None)
            if ref is not None:
                # Sanity: r_twenty_est must be within 60%–170% of the polygon estimate.
                # With the corrected BULL_POLYGON_SCALE both estimates should agree well.
                if not (0.6 * ref <= r_twenty_est <= 1.7 * ref):
                    r_twenty_est = None  # discard bad estimate
            elif not (8.0 <= r_twenty_est <= 60.0):
                r_twenty_est = None

        if r_twenty_est is not None:
            new_r = r_twenty_est
        elif bull_direct_r is not None:
            new_r = bull_direct_r
        elif r_50_est is not None:
            new_r = r_50_est
        elif ema_ellipse is not None:
            new_r = ema_ellipse[2] * BULL_TO_DOUBLE_OUTER
        else:
            new_r = None

        # EMA smoothing
        if new_cx is not None:
            # Always update if board/multiple markers drove the estimation
            if ema_ellipse is not None or best_bull_conf > 0.6 or best_bull50_conf > 0.6:
                if ema_cx is None:
                    ema_cx, ema_cy, ema_r = new_cx, new_cy, new_r
                elif math.hypot(new_cx - ema_cx, new_cy - ema_cy) < 50.0:
                    ema_cx = 0.85 * ema_cx + 0.15 * new_cx
                    ema_cy = 0.85 * ema_cy + 0.15 * new_cy
                    if new_r is not None:
                        ema_r  = 0.85 * ema_r  + 0.15 * new_r

        if ema_cx is not None:
            center_x, center_y, bull_radius = ema_cx, ema_cy, ema_r
        else:
            center_x, center_y, bull_radius = 0.0, 0.0, bull_radius_fallback

        # ── Auto-rotation: sector-20 direction ───────────────────────────────
        if not args.disable_auto_rotation:
            sector_20_pos = None
            if twenty_cx is not None:
                sector_20_pos = (twenty_cx, twenty_cy)

            new_ao = None
            if sector_20_pos is not None:
                # 1. Class-derived auto-rotation
                c_dx, c_dy = corrected_offset(sector_20_pos[0], sector_20_pos[1])
                new_ao = math.degrees(math.atan2(c_dy, c_dx)) + 90.0
            elif board_poly_active is not None and ema_ellipse is not None:
                # 2. Fallback: topmost board vertex
                n_top = max(3, len(board_poly_active) // 10)
                top_indices = np.argsort(board_poly_active[:, 1])[:n_top]
                c_dx, c_dy = corrected_offset(float(np.mean(board_poly_active[top_indices, 0])),
                                              float(np.mean(board_poly_active[top_indices, 1])))
                new_ao = math.degrees(math.atan2(c_dy, c_dx)) + 90.0

            if new_ao is not None:
                if ema_angle_offset is None:
                    ema_angle_offset = new_ao
                else:
                    da2 = ((new_ao - ema_angle_offset + 180.0) % 360.0) - 180.0
                    ema_angle_offset = ema_angle_offset + 0.15 * da2

        # active_angle: manual when auto-rotation is disabled, auto when available, CLI fallback otherwise
        if args.disable_auto_rotation:
            active_angle = args.angle_offset
        else:
            active_angle = ema_angle_offset if ema_angle_offset is not None else args.angle_offset

        # ── Geometric ring boundaries (always visible for validation) ──────────
        # Drawn as perspective ellipses so they overlay the tilted board rings.
        if ema_cx is not None and bull_radius is not None and bull_radius > 0:
            draw_scoring_rings(frame, center_x, center_y, bull_radius, ema_ellipse)

        # ── Dart detections ────────────────────────────────────────────────
        # Determine which model is authoritative for darts
        use_pose = (results_pose is not None)
        dart_source = results_pose if use_pose else results
        
        # Class ID for dart in the chosen model
        active_dart_cls = 0 if use_pose else DART_CLS

        dart_boxes = []
        dart_confs = []
        dart_orig_idx = []
        for i, box in enumerate(dart_source.boxes):
            if int(box.cls[0]) != active_dart_cls:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            dart_boxes.append((x1, y1, x2, y2))
            dart_confs.append(conf)
            dart_orig_idx.append(i)

        # ACC-12: NMS to dedupe overlapping dart boxes
        keep = nms_indices(dart_boxes, dart_confs, iou_thresh=0.5)

        detections = []
        for k in keep:
            x1, y1, x2, y2 = dart_boxes[k]
            orig_i = dart_orig_idx[k]

            tip_cx = (x1 + x2) / 2
            tip_cy = (y1 + y2) / 2
            
            if use_pose:
                # Pose keypoints
                tail_cx, tail_cy = None, None
                if dart_source.keypoints is not None and dart_source.keypoints.xy is not None:
                    kpts = dart_source.keypoints.xy[orig_i].cpu().numpy()
                    if len(kpts) >= 2 and not np.allclose(kpts, 0.0):
                        pk_tip, pk_tail = kpts[0], kpts[1]
                        tip_cx, tip_cy = float(pk_tip[0]), float(pk_tip[1])
                        tail_cx, tail_cy = float(pk_tail[0]), float(pk_tail[1])
                        
                        # Snap the pose keypoint to the shaft edge (small window —
                        # it is already near-correct, only a few px of refinement).
                        tip_cx, tip_cy = refine_dart_tip(gray, (tip_cx, tip_cy), (tail_cx, tail_cy), search_px=10)
                
                # Fallback if pose keypoints are missing/degenerate
                if tail_cx is None:
                    best_seg_idx = -1
                    best_iou = 0.3
                    for i_seg, box_seg in enumerate(results.boxes):
                        if int(box_seg.cls[0]) == DART_CLS:
                            seg_bbox = box_seg.xyxy[0].tolist()
                            curr_iou = iou(dart_boxes[k], seg_bbox)
                            if curr_iou > best_iou:
                                best_iou = curr_iou
                                best_seg_idx = i_seg

                    if best_seg_idx >= 0 and seg_polys is not None and len(seg_polys[best_seg_idx]) >= 3:
                        poly = seg_polys[best_seg_idx].astype(np.float32)
                        (tip_cx, tip_cy), (tail_cx, tail_cy) = dart_tip_tail_poly(poly)
                        tip_cx, tip_cy = refine_dart_tip(gray, (tip_cx, tip_cy), (tail_cx, tail_cy))
            else:
                # Fallback to segmentation polygons
                tail_cx, tail_cy = None, None
                if seg_polys is not None and len(seg_polys[orig_i]) >= 3:
                    poly = seg_polys[orig_i].astype(np.float32)
                    (tip_cx, tip_cy), (tail_cx, tail_cy) = dart_tip_tail_poly(poly)
                    tip_cx, tip_cy = refine_dart_tip(
                        gray, (tip_cx, tip_cy), (tail_cx, tail_cy))

            tip_dx, tip_dy = tip_cx - center_x, tip_cy - center_y
            tip_dx_c, tip_dy_c = perspective_correct_vec(tip_dx, tip_dy, ema_ellipse)
            tip_cx_s = center_x + tip_dx_c
            tip_cy_s = center_y + tip_dy_c
            label, score = score_geometric(center_x, center_y, tip_cx_s, tip_cy_s,
                                           bull_radius, active_angle)
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'score': score,
                'tip': (tip_cx, tip_cy),
                'tail': (tail_cx, tail_cy) if (tail_cx is not None) else None,
                'orig_idx': orig_i,
            })

        # ACC-8: temporal smoothing via IoU tracking (majority vote per track)
        display_items = []  # (bbox, label, low_conf, tip, tail, orig_idx)
        if use_tracker:
            current_frame = frame_idx[0]
            used_tracks = set()
            for det in detections:
                best_t = -1
                best_iou = TRACK_IOU_THRESH
                for ti, tr in enumerate(tracks):
                    if ti in used_tracks:
                        continue
                    iv = iou(det['bbox'], tr.bbox)
                    if iv > best_iou:
                        best_iou = iv
                        best_t = ti
                if best_t >= 0:
                    tr = tracks[best_t]
                    tr.bbox = det['bbox']
                    tr.push(det['label'])
                    tr.last_seen_frame = current_frame
                    used_tracks.add(best_t)
                    top, top_count, total = tr.majority()
                    low_conf = (total >= 3) and (top_count / total < 0.6)
                    display_items.append((det['bbox'], top, low_conf, det['tip'], det['tail'], det['orig_idx']))
                else:
                    new_tr = Track(det['bbox'], det['label'], current_frame)
                    tracks.append(new_tr)
                    display_items.append((det['bbox'], det['label'], False, det['tip'], det['tail'], det['orig_idx']))
            tracks[:] = [tr for tr in tracks
                         if current_frame - tr.last_seen_frame <= TRACK_MAX_AGE]
            frame_idx[0] = current_frame + 1
        else:
            for det in detections:
                display_items.append((det['bbox'], det['label'], False, det['tip'], det['tail'], det['orig_idx']))

        # ── Draw dart annotations ──────────────────────────────────────────
        total_round_score = 0
        for bbox, label, low_conf, tip, tail, orig_i in display_items:
            x1, y1, x2, y2 = bbox
            display_label = label + ("?" if low_conf else "")
            text_color = (0, 165, 255) if low_conf else (255, 255, 255)

            if tail is not None:
                # Draw skeleton line and tail dot
                cv2.line(frame, (int(tail[0]), int(tail[1])), (int(tip[0]), int(tip[1])), (0, 255, 255), 2)
                cv2.circle(frame, (int(tail[0]), int(tail[1])), 5, (255, 0, 0), -1)
            elif seg_polys is not None and len(seg_polys[orig_i]) >= 3:
                dart_poly = smooth_polygon(seg_polys[orig_i].astype(np.float32), w, h, kernel_size=9)
                pts = dart_poly.astype(np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

            # ACC-1: draw dart tip
            cv2.circle(frame, (int(tip[0]), int(tip[1])), 4, (0, 0, 255), -1)

            cv2.putText(frame, f"score: {display_label}", (int(x1) + 5, int(y1) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

            total_round_score += label_to_score(label)

        if display_items:
            cv2.putText(frame, f"total: {total_round_score}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Sector-20 direction indicator (cyan) — extends to just past the double ring
        if center_x is not None and center_x > 0:
            # Reach the double-ring midpoint so the "20" label lands at the correct ring
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
                # Fallback: project the active_angle through inverse perspective
                dir_rad = math.radians(active_angle - 90.0)
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ── VideoWriter handling with size-change recovery (QA-4) ─────────
        if args.output:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))
                writer_w, writer_h = w, h
            elif (w, h) != (writer_w, writer_h):
                print(f"WARNING: frame size changed from ({writer_w},{writer_h}) to "
                      f"({w},{h}); recreating VideoWriter.", file=sys.stderr)
                writer.release()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))
                writer_w, writer_h = w, h
            writer.write(frame)
        return frame

    # ── Run ─────────────────────────────────────────────────────────────────
    if is_image:
        raw = frames[0].copy()
        out = process(frames[0], use_tracker=False)
        display_frame = raw if args.raw_view else out
        cv2.imshow('Darts Score Detection', display_frame)
        print("Press any key in the window to exit.")
        cv2.waitKey(0)
    else:
        if args.capture:
            os.makedirs(args.capture_dir, exist_ok=True)
            print(f"Capture mode on — press SPACE to save a raw frame to {args.capture_dir}")
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_rtsp:
                    print("RTSP stream lost — reconnecting…", file=sys.stderr)
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue
                break
            raw = frame.copy()
            out = process(frame, use_tracker=True)
            display_frame = raw if args.raw_view else out
            if args.capture:
                # Copy to avoid drawing status text on the clean 'raw' image
                display_frame = display_frame.copy()
                cv2.putText(display_frame, f"SPACE=save  q=quit  saved={saved}",
                            (10, display_frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Darts Score Detection', display_frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if args.capture and k == 32:  # SPACE
                name = f"darts_{int(time.time() * 1000)}.jpg"
                path = os.path.join(args.capture_dir, name)
                cv2.imwrite(path, raw)
                saved += 1
                print(f"[capture #{saved}] saved {path}")
        cap.release()

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
