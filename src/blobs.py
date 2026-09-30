"""White retro-reflective marker candidates from a single BGR frame."""
import cv2
import numpy as np

# Markers are ~20-25 px spheres at 1080p; motion blur smears them into streaks up to ~70 px.
TOPHAT_K = 45
MIN_AREA, MAX_AREA = 25, 2500


def whiteness(img):
    """Per-pixel min(B,G,R): high only for white/grey. Skin (low B), the blue star and
    stylus (low R), and the red holder (low G,B) all score low."""
    return img.min(axis=2)


def detect_candidates(img, roi=None, th_tophat=45, th_white=150):
    """Return an (N, 8) array: x, y, area, mean_whiteness, peak_tophat, elongation,
    angle_deg, score. Coordinates are intensity-weighted centroids in full-frame pixels.

    roi: optional uint8 mask; components whose centroid falls outside it are dropped.
    """
    w = whiteness(img)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_K, TOPHAT_K))
    th = cv2.morphologyEx(w, cv2.MORPH_TOPHAT, k)
    mask = ((th > th_tophat) & (w > th_white)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        if area < MIN_AREA:
            continue
        sl = (slice(y0, y0 + bh), slice(x0, x0 + bw))
        m = lab[sl] == i
        feat = _describe(m, th[sl], w[sl], x0, y0)
        if area <= MAX_AREA and feat[5] <= 6:
            out.append(feat)
        else:
            # A marker touching another bright structure (e.g. the iPad stand edge) merges
            # into one large/elongated component; recover marker-sized peaks inside it.
            out.extend(_peaks_inside(m, th[sl], w[sl], x0, y0))
    out = [f for f in out if roi is None or roi[int(round(f[1])), int(round(f[0]))]]
    return np.asarray(out, np.float32).reshape(-1, 8)


def _describe(m, th, w, x0, y0):
    ys, xs = np.nonzero(m)
    ww = th[ys, xs].astype(np.float32)
    cx = x0 + (xs * ww).sum() / ww.sum()
    cy = y0 + (ys * ww).sum() / ww.sum()
    # Second moments -> elongation (1 = round, >1 = streak) and streak direction.
    cov = np.cov(np.stack([xs, ys])) if len(xs) > 2 else np.eye(2)
    ev, evec = np.linalg.eigh(cov)
    elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
    ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
    area = float(m.sum())
    mw = float(w[m].mean())
    pk = float(th[m].max())
    # Round, bright, marker-sized blobs score highest.
    size_term = np.exp(-((np.sqrt(area) - 18) / 14) ** 2)
    score = (pk / 255.0) * (mw / 255.0) * size_term / elong ** 0.5
    return [cx, cy, area, mw, pk, elong, ang, score]


def _peaks_inside(m, th, w, x0, y0, r=11):
    sm = cv2.GaussianBlur(th.astype(np.float32) * m, (0, 0), 4)
    loc = (sm == cv2.dilate(sm, np.ones((2 * r + 1, 2 * r + 1), np.uint8))) & (sm > 60) & m
    feats = []
    for py, px in zip(*np.nonzero(loc)):
        ya, yb = max(0, py - r), min(m.shape[0], py + r + 1)
        xa, xb = max(0, px - r), min(m.shape[1], px + r + 1)
        sub = m[ya:yb, xa:xb] & (th[ya:yb, xa:xb] > 0.5 * th[py, px])
        if sub.sum() < MIN_AREA or w[ya:yb, xa:xb][sub].mean() < 190:
            continue
        f = _describe(sub, th[ya:yb, xa:xb], w[ya:yb, xa:xb], x0 + xa, y0 + ya)
        if f[5] <= 3:
            feats.append(f)
    return feats
