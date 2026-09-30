"""Stage 2: find, label and track the white markers of the lab's Marker Placement protocol
(Vicon Plug-in Gait Upper Body): left arm + the trunk markers visible from above.

    python src/markers.py

Per video writes outputs/<tag>/:
    candidates.npz       every white-blob candidate inside the left-limb ROI, per frame
    skeleton_fused.npz   fused shoulder/elbow/wrist/hand track from the pose models
    markers_long.csv     frame, time_s, marker, x_px, y_px, quality, blob_score
    markers_wide.csv     one row per frame, <marker>_x / _y / _q columns
    marker_template.csv  learned marker layout (arm: limb coords, trunk: shoulder frame),
                         support = fraction of frames found, analysed = support >= 0.3

quality: 1 detected blob, 2 interpolated (short gap), 3 DINOv3 re-detection, 0 missing.
"""
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from blobs import detect_candidates
from common import list_videos, out_dir, read_video, video_tag
from skeleton import CHAIN, agreement, chain_to_point, fuse, fuse_right_shoulder

LEFT_ARM = (6, 7, 11)      # Sapiens2: Left_Hand, Left_Lower_Arm, Left_Upper_Arm
# On this overhead back view Sapiens2 labels most of the left arm "Left_Hand", so the
# three classes are used together as one silhouette, never to tell segments apart.
APPAREL = 1
RIGHT_ARM = (15, 16, 20)
MAX_GAP_INTERP = 6         # frames (200 ms @ 30 fps) bridged by PCHIP


# ----------------------------------------------------------------------------- ROI
def limb_roi(labels, joints, shape, capsule_r=45):
    """Left upper-limb ROI = dilated Sapiens2 left-arm silhouette (+ the marker blobs it
    labels Apparel)  OR  a capsule around the fused skeleton, minus the Sapiens2 right
    arm. Markers protrude from the silhouette edge, hence close + dilate."""
    H, W = shape
    roi = np.zeros((H, W), np.uint8)
    if labels is not None:
        arm = np.isin(labels, LEFT_ARM).astype(np.uint8)
        arm = cv2.morphologyEx(arm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
        near = cv2.dilate(arm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
        # Sapiens2 segments the markers themselves as small "Apparel" blobs; keep the
        # ones touching the left arm (edge markers otherwise fall outside the mask).
        roi |= near | ((labels == APPAREL) & cv2.dilate(near, np.ones((25, 25), np.uint8)).astype(bool))
    if not np.isnan(joints).any():
        p = joints.astype(int)
        for s in range(3):
            cv2.line(roi, tuple(p[s]), tuple(p[s + 1]), 1, 2 * capsule_r)
        cv2.circle(roi, tuple(p[0]), int(capsule_r * 1.4), 1, -1)  # acromion sits above the joint centre
    if labels is not None:
        right = np.isin(labels, RIGHT_ARM) & ~np.isin(labels, LEFT_ARM)
        roi[right] = 0
    return roi


def nms(c, r=24):
    """Merge candidates closer than r px - a sphere and its white base plate, or a blur
    streak broken in two - keeping the highest-scoring (roundest, brightest) one."""
    if len(c) < 2:
        return c
    keep = []
    for i in np.argsort(-c[:, 7]):
        if all(np.hypot(*(c[i, :2] - c[j, :2])) > r for j in keep):
            keep.append(i)
    return c[sorted(keep)]


# ----------------------------------------------------------------------------- templates
# Marker Placement protocol (Vicon Plug-in Gait Upper Body), left arm, as a prior in limb
# coordinates: c = segment index + fraction along it (0 shoulder, 1 elbow, 2 wrist,
# 3 hand-MCP centre, from the fused pose skeleton), v = perpendicular offset / segment
# length, +v = thumb (radial) side (checked against RTMPose thumb/pinky keypoints).
ARM_PRIOR = pd.DataFrame([
    ("LSHO", 0.05, 0.00),   # acromion tip
    ("LUPA", 0.55, 0.00),   # 1-2 in below deltoid tip, dorsal
    ("LELB", 1.00, 0.00),   # lateral epicondyle
    ("LFRM", 1.55, -0.10),  # mid forearm, ulnar side
    ("LWRA", 1.95, 0.12),   # radial styloid
    ("LWRB", 1.95, -0.12),  # ulnar styloid
    ("LHND", 2.95, 0.10),   # under the middle-finger knuckle
    ("LFIN", 3.85, 0.20),   # index fingertip
], columns=["label", "c", "v"])

# Trunk markers in a shoulder frame: origin = midpoint of the fused shoulder joints,
# c along left->right shoulder, v perpendicular pointing down the back (away from the
# head in this overhead view); both in units of shoulder width. CLAV/STRN are on the
# front and never visible from this camera.
# The overhead camera sees the back steeply foreshortened, so T10 appears only ~0.6
# shoulder widths below the shoulder line (priors read off a blob histogram in this frame).
TRUNK_PRIOR = pd.DataFrame([
    ("RSHO", 0.45, -0.10),
    ("C7", 0.00, 0.05),
    ("RBAK", 0.40, 0.25),
    ("T10", -0.05, 0.60),
], columns=["label", "c", "v"])
MIN_SUPPORT = 0.3  # markers found in fewer frames are reported but not analysed


def seg_of(c):
    seg = np.clip(np.floor(np.asarray(c)), 0, 2).astype(int)
    return seg, np.asarray(c) - seg


def on_segment(p, J, s):
    """(c, v) of point p relative to chain segment s, unclipped."""
    a, b = J[s], J[s + 1]
    ab = b - a
    L2 = ab @ ab + 1e-6
    return s + ((p - a) @ ab) / L2, ((p - a) @ np.array([-ab[1], ab[0]])) / L2


class ArmFrame:
    """Marker anchors along the fused shoulder-elbow-wrist-hand chain."""

    def __init__(self, joints):
        self.J = joints

    def xy(self, tpl, t):
        seg, u = seg_of(tpl.c.values)
        return np.array([chain_to_point(s, uu, vv, self.J[t])
                         for s, uu, vv in zip(seg, u, tpl.v.values)])

    def coords(self, p, t, c_prior):
        return on_segment(p, self.J[t], seg_of(c_prior)[0])

    def gates(self, tpl, t, frac, floor):
        seglen = np.linalg.norm(self.J[t][1:] - self.J[t][:-1], axis=1)
        return np.maximum(floor, frac * seglen[seg_of(tpl.c.values)[0]])


class TrunkFrame:
    """Marker anchors in the shoulder frame (see TRUNK_PRIOR)."""

    def __init__(self, lsh, rsh):
        self.M = (lsh + rsh) / 2
        d = rsh - lsh
        self.W = np.linalg.norm(d, axis=1)[:, None] + 1e-6
        self.e1 = d / self.W
        e2 = np.stack([-self.e1[:, 1], self.e1[:, 0]], 1)
        self.e2 = e2 * np.sign(e2[:, 1:2] + 1e-9)  # point towards image-down (down the back)

    def xy(self, tpl, t):
        return self.M[t] + self.W[t] * (tpl.c.values[:, None] * self.e1[t] + tpl.v.values[:, None] * self.e2[t])

    def coords(self, p, t, c_prior=None):
        r = (p - self.M[t]) / self.W[t]
        return r @ self.e1[t], r @ self.e2[t]

    def gates(self, tpl, t, frac, floor):
        return np.full(len(tpl), max(floor, frac * float(self.W[t, 0])))


def assign(c, pred_tpl, g_tpl, pred_tmp=None, g_tmp=None, w_tmp=0.5):
    """Hungarian match of K markers to N candidates. The template term keeps identities
    anatomically consistent every frame (so a swap cannot persist); the temporal term
    breaks ties between close markers. Returns {k: j}."""
    if len(c) == 0:
        return {}
    D1 = np.linalg.norm(pred_tpl[:, None] - c[None, :, :2], axis=2) / g_tpl[:, None]
    cost = D1.copy()
    bad = D1 > 1.0
    if pred_tmp is not None:
        have = ~np.isnan(pred_tmp[:, 0])
        D2 = np.linalg.norm(pred_tmp[:, None] - c[None, :, :2], axis=2) / g_tmp[:, None]
        cost[have] = (1 - w_tmp) * D1[have] + w_tmp * D2[have]
        # a marker seen last frame may drift from the template, but not jump
        bad[have] = (D1[have] > 1.3) | (D2[have] > 1.0)
    cost = cost - 0.2 * c[None, :, 7]
    cost[bad] = 1e6
    r, q = linear_sum_assignment(cost)
    return {k: j for k, j in zip(r, q) if cost[k, j] < 1e5}


def refine_template(cands, frame, tpl, iters=4, frac=0.35, floor=30):
    """EM-style: assign every frame against the current template, then move each marker's
    (c, v) to the median of what it was assigned. Adapts the protocol prior to this
    participant's actual marker placement. `support` = fraction of frames it was found."""
    tpl = tpl.copy()
    T = len(cands)
    for _ in range(iters):
        obs = [[] for _ in range(len(tpl))]
        for t in range(T):
            m = assign(cands[t], frame.xy(tpl, t), frame.gates(tpl, t, frac, floor))
            for k, j in m.items():
                obs[k].append(frame.coords(cands[t][j, :2], t, tpl.c.values[k]))
        for k, o in enumerate(obs):
            if len(o) >= 0.1 * T:
                tpl.loc[k, ["c", "v"]] = np.median(np.array(o), axis=0)
        tpl["support"] = [len(o) / T for o in obs]
    return tpl


# ----------------------------------------------------------------------------- tracking
def track(cands, frame, tpl, frac=0.35, floor=30):
    """Per-frame assignment against the refined template, with the previous frame's
    position (carried along by the frame's motion) as a secondary cue."""
    T, K = len(cands), len(tpl)
    pos = np.full((T, K, 2), np.nan, np.float32)
    score = np.zeros((T, K), np.float32)
    for t in range(T):
        ptpl = frame.xy(tpl, t)
        ptmp = pos[t - 1] + (ptpl - frame.xy(tpl, t - 1)) if t > 0 else None
        m = assign(cands[t], ptpl, frame.gates(tpl, t, frac, floor), ptmp,
                   frame.gates(tpl, t, 0.2, 25))
        for k, j in m.items():
            pos[t, k] = cands[t][j, :2]
            score[t, k] = cands[t][j, 7]
    return pos, score


def drop_outliers(pos, thr=30):
    """Remove samples that jump off an otherwise smooth path (single-frame mislabels)."""
    from scipy.ndimage import median_filter
    out = pos.copy()
    for k in range(pos.shape[1]):
        for c in range(2):
            v = pos[:, k, c]
            g = ~np.isnan(v)
            if g.sum() < 5:
                continue
            filled = np.interp(np.arange(len(v)), np.flatnonzero(g), v[g])
            med = median_filter(filled, size=7, mode="nearest")
            out[g & (np.abs(filled - med) > thr), k] = np.nan
    return out


def fill_gaps(pos, max_gap=MAX_GAP_INTERP):
    """PCHIP across gaps of <= max_gap frames. Returns filled positions + quality codes."""
    T, K, _ = pos.shape
    q = np.where(np.isnan(pos[..., 0]), 0, 1).astype(np.int8)
    out = pos.copy()
    t = np.arange(T)
    for k in range(K):
        g = ~np.isnan(pos[:, k, 0])
        if g.sum() < 4:
            continue
        f = PchipInterpolator(t[g], pos[g, k], axis=0, extrapolate=False)
        # label each run of missing frames
        miss = ~g
        start = None
        for i in range(T + 1):
            if i < T and miss[i]:
                start = i if start is None else start
            elif start is not None:
                if start > 0 and i < T and i - start <= max_gap:
                    out[start:i, k] = f(t[start:i])
                    q[start:i, k] = 2
                start = None
    return out, q


# ----------------------------------------------------------------------------- main
def run_group(cands, frame, prior, frac, floor):
    tpl = refine_template(cands, frame, prior, frac=frac, floor=floor)
    pos, bscore = track(cands, frame, tpl, frac=frac, floor=floor)
    pos = drop_outliers(pos)
    bscore[np.isnan(pos[..., 0])] = 0
    filled, q = fill_gaps(pos)
    return tpl, filled, q, bscore


def process(vp, args):
    tag = video_tag(vp)
    d = out_dir(tag)
    frames, ts, fps = read_video(vp)
    H, W = frames[0].shape[:2]
    T = len(frames)

    joints, n_models, per = fuse(tag)
    rsh = fuse_right_shoulder(tag)
    agree = pd.DataFrame(agreement(per, joints)).T
    agree.to_csv(d / "pose_model_agreement_vs_fused.csv")
    np.savez(d / "skeleton_fused.npz", joints=joints, n_models=n_models, r_shoulder=rsh,
             t=ts, chain=np.array(CHAIN))
    arm_f = ArmFrame(joints)
    trunk_f = TrunkFrame(joints[:, 0], rsh)

    sp = out_dir(tag, "models") / "sapiens2.npz"
    labels = np.load(sp)["labels"] if sp.exists() and not args.no_sapiens else None

    arm_c, trunk_c = [], []
    for t in tqdm(range(T), desc=f"blobs {tag}"):
        c = detect_candidates(frames[t])
        xi = np.clip(c[:, 0].round().astype(int), 0, W - 1)
        yi = np.clip(c[:, 1].round().astype(int), 0, H - 1)
        roi = limb_roi(labels[t] if labels is not None else None, joints[t], (H, W))
        in_arm = roi[yi, xi].astype(bool)
        cc, vv = trunk_f.coords(c[:, :2], t)
        in_trunk = ~in_arm & (cc > -0.4) & (cc < 0.8) & (vv > -0.35) & (vv < 1.2)
        if labels is not None:  # right forearm / hand markers are not trunk markers
            in_trunk &= ~np.isin(labels[t][yi, xi], (15, 16))
        arm_c.append(nms(c[in_arm]))
        trunk_c.append(nms(c[in_trunk]))
    np.savez(d / "candidates.npz", **{f"f{t}": c for t, c in enumerate(arm_c)},
             **{f"trunk{t}": c for t, c in enumerate(trunk_c)}, t=ts)

    groups = [("arm", run_group(arm_c, arm_f, ARM_PRIOR, 0.35, 30)),
              ("trunk", run_group(trunk_c, trunk_f, TRUNK_PRIOR, 0.2, 25))]
    tpls, long = [], []
    for gname, (tpl, filled, q, bscore) in groups:
        tpl = tpl.assign(group=gname, analysed=tpl.support >= MIN_SUPPORT)
        tpls.append(tpl)
        for k, n in enumerate(tpl.label):
            long.append(pd.DataFrame(dict(frame=np.arange(T), time_s=ts, marker=n,
                                          x_px=filled[:, k, 0], y_px=filled[:, k, 1],
                                          quality=q[:, k], blob_score=bscore[:, k])))
    tpl = pd.concat(tpls, ignore_index=True)
    print(tpl.round(3).to_string())
    tpl.to_csv(d / "marker_template.csv", index=False)
    names = tpl.label.tolist()
    long = pd.concat(long, ignore_index=True)
    long.to_csv(d / "markers_long.csv", index=False, float_format="%.4f")
    wide = long.pivot(index=["frame", "time_s"], columns="marker",
                      values=["x_px", "y_px", "quality"])
    suffix = {"x_px": "x", "y_px": "y", "quality": "q"}
    wide.columns = [f"{m}_{suffix[v]}" for v, m in wide.columns]
    wide = wide[[f"{n}_{s}" for n in names for s in ("x", "y", "q")]].reset_index()
    wide.to_csv(d / "markers_wide.csv", index=False, float_format="%.4f")

    q = long.pivot(index="frame", columns="marker", values="quality")
    cov = pd.Series({n: f"det {np.mean(q[n] == 1):.0%} / +interp {np.mean(q[n] >= 1):.0%}" for n in names})
    print(f"[{tag}] coverage:\n{cov.to_string()}")
    return tag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_sapiens", action="store_true", help="ROI from skeleton capsule only")
    args = ap.parse_args()
    for vp in list_videos():
        process(vp, args)
