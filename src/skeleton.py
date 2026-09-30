"""Fuse the pose models into one left-arm skeleton and score how well they agree.

Chain used everywhere downstream: shoulder -> elbow -> wrist -> hand (MCP centre).
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from common import out_dir

CHAIN = ["shoulder", "elbow", "wrist", "hand"]
# Per-model confidence floors (scores are on different scales: heatmap max for ViTPose,
# SimCC max for RTMPose, landmark visibility for MediaPipe).
THR = {"vitpose": 0.3, "rtmpose": 0.3, "mediapipe": 0.5}


def model_chain(tag, name):
    """(T, 4, 2) chain joints and (T, 4) scores for one model; hand = mean(index, pinky MCP)."""
    z = np.load(out_dir(tag, "models") / f"{name}.npz")
    arm, sc = z["arm"], z["arm_score"]
    pts = np.full(arm.shape[:1] + (4, 2), np.nan, np.float32)
    s = np.zeros(arm.shape[:1] + (4,), np.float32)
    pts[:, :3], s[:, :3] = arm[:, :3], sc[:, :3]
    if not np.isnan(arm[:, 3]).all():
        pts[:, 3] = (arm[:, 3] + arm[:, 4]) / 2
        s[:, 3] = np.minimum(sc[:, 3], sc[:, 4])
    ok = s >= THR[name]
    pts[~ok] = np.nan
    return pts, s


def _robust_fuse(stack, outlier_px):
    """Cross-model median with outlier rejection, then gap fill + light smoothing.
    stack: (M, T, J, 2) with NaN where a model is missing or below its floor."""
    med = np.nanmedian(stack, axis=0)
    # Drop any model more than outlier_px from the cross-model median, then re-median.
    dev = np.linalg.norm(stack - med[None], axis=-1)
    stack = np.where((dev > outlier_px)[..., None], np.nan, stack)
    fused = np.nanmedian(stack, axis=0)
    n_models = np.sum(~np.isnan(stack[..., 0]), axis=0)  # (T, J)
    T = fused.shape[0]
    t = np.arange(T)
    for j in range(fused.shape[1]):
        for c in range(2):
            v = fused[:, j, c]
            g = ~np.isnan(v)
            if g.sum() < 2:
                continue
            v = np.interp(t, t[g], v[g])
            v = median_filter(v, size=5, mode="nearest")
            fused[:, j, c] = savgol_filter(v, 7, 2)
    return fused, n_models


def fuse(tag, models=("vitpose", "rtmpose", "mediapipe"), outlier_px=90):
    per = {m: model_chain(tag, m)[0] for m in models}
    fused, n_models = _robust_fuse(np.stack(list(per.values())), outlier_px)
    return fused, n_models, per


def fuse_right_shoulder(tag, models=("vitpose", "rtmpose", "mediapipe"), outlier_px=90):
    """(T, 2) fused right shoulder joint; with the left one it defines the trunk frame."""
    stack = []
    for m in models:
        z = np.load(out_dir(tag, "models") / f"{m}.npz")
        p = z["arm"][:, 5].copy()
        p[z["arm_score"][:, 5] < THR[m]] = np.nan
        stack.append(p[:, None])
    return _robust_fuse(np.stack(stack), outlier_px)[0][:, 0]


def agreement(per, ref):
    """Median pixel distance of each model's joints to a reference track (T, 4, 2)."""
    rows = {}
    for m, p in per.items():
        d = np.linalg.norm(p - ref, axis=-1)
        rows[m] = {CHAIN[j]: float(np.nanmedian(d[:, j])) if np.isfinite(d[:, j]).any()
                   else np.nan for j in range(4)}
        rows[m]["valid_frac"] = float(np.mean(~np.isnan(p[..., 0])))
    return rows


def point_to_chain(p, joints):
    """Project image points p (N,2) onto the 3-segment chain.

    Returns seg (N,) in {0,1,2}, u (N,) along that segment (0 = proximal joint, 1 = distal), v (N,) signed
    perpendicular offset in units of that segment's length, and dist (N,) in px.
    """
    best = None
    # The acromion marker sits proximal to the shoulder joint centre and the hand marker
    # distal to the MCP estimate, so the end segments may extend past their joints.
    lo = {0: -0.5, 1: 0.0, 2: 0.0}
    hi = {0: 1.0, 1: 1.0, 2: 2.0}
    for s in range(3):
        a, b = joints[s], joints[s + 1]
        ab = b - a
        L = np.linalg.norm(ab) + 1e-6
        u = np.clip(((p - a) @ ab) / L ** 2, lo[s], hi[s])
        foot = a + u[:, None] * ab
        d = np.linalg.norm(p - foot, axis=1)
        v = ((p - a) @ np.array([-ab[1], ab[0]])) / L ** 2
        cur = np.stack([np.full(len(p), s), u, v, d], 1)
        best = cur if best is None else np.where((d < best[:, 3])[:, None], cur, best)
    return best[:, 0].astype(int), best[:, 1], best[:, 2], best[:, 3]


def chain_to_point(seg, u, v, joints):
    a, b = joints[seg], joints[seg + 1]
    ab = b - a
    return a + u * ab + v * np.array([-ab[1], ab[0]])
