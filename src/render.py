"""QA renders: annotated overlay video and a contact sheet of sampled frames.

    python src/render.py [--sheet_only] [--frames 0 50 100]
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from common import list_videos, out_dir, read_video, video_tag

PALETTE = [(255, 99, 71), (255, 215, 0), (50, 205, 50), (0, 191, 255), (147, 112, 219),
           (255, 105, 180), (0, 255, 255), (255, 140, 0), (173, 255, 47), (220, 220, 220)]
QCOL = {1: None, 2: (0, 165, 255), 3: (255, 0, 255)}  # detected / interpolated / DINO


def draw(img, t, markers, names, joints, cands):
    out = img.copy()
    if cands is not None:
        for x, y, *_ in cands:
            cv2.circle(out, (int(x), int(y)), 16, (160, 160, 160), 1)
    J = joints[t].astype(int)
    for s in range(3):
        cv2.line(out, tuple(J[s]), tuple(J[s + 1]), (255, 255, 255), 2)
    for j in J:
        cv2.circle(out, tuple(j), 5, (255, 255, 255), -1)
    for k, n in enumerate(names):
        x, y, q = markers[n][t]
        if q == 0 or np.isnan(x):
            continue
        col = PALETTE[k % len(PALETTE)][::-1]
        cv2.circle(out, (int(x), int(y)), 13, col, 3)
        if QCOL.get(int(q)):
            cv2.circle(out, (int(x), int(y)), 18, QCOL[int(q)][::-1], 2)
        cv2.putText(out, n, (int(x) + 16, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return out


def load(tag):
    d = out_dir(tag)
    long = pd.read_csv(d / "markers_long.csv")
    names = pd.read_csv(d / "marker_template.csv").label.tolist()
    mk = {n: long[long.marker == n].sort_values("frame")[["x_px", "y_px", "quality"]].values for n in names}
    joints = np.load(d / "skeleton_fused.npz")["joints"]
    cz = np.load(d / "candidates.npz")
    cands = [cz[f"f{t}"] for t in range(len(joints))]
    return names, mk, joints, cands


def arm_window(joints, W, H, pad=140, extra=None):
    pts = joints.reshape(-1, 2) if extra is None else np.vstack([joints.reshape(-1, 2), extra])
    x0, y0 = np.nanmin(pts, 0) - pad
    x1, y1 = np.nanmax(pts, 0) + pad
    return int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet_only", action="store_true")
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--n_sheet", type=int, default=12)
    args = ap.parse_args()
    for vp in list_videos():
        tag = video_tag(vp)
        frames, ts, fps = read_video(vp)
        H, W = frames[0].shape[:2]
        names, mk, joints, cands = load(tag)
        seen = np.vstack([v[v[:, 2] == 1, :2] for v in mk.values()])
        x0, y0, x1, y1 = arm_window(joints, W, H, extra=seen)
        idx = args.frames or np.linspace(0, len(frames) - 1, args.n_sheet).astype(int).tolist()
        tiles = []
        for t in idx:
            im = draw(frames[t], t, mk, names, joints, cands[t])[y0:y1, x0:x1]
            im = cv2.resize(im, (480, int(480 * (y1 - y0) / (x1 - x0))))
            cv2.putText(im, f"f{t} {ts[t]:.2f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(im)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(out_dir(tag) / "contact_sheet.jpg"), sheet)
        if args.sheet_only:
            continue
        vw = cv2.VideoWriter(str(out_dir(tag) / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for t in range(len(frames)):
            im = draw(frames[t], t, mk, names, joints, None)
            cv2.putText(im, f"frame {t}  t={ts[t]:.3f}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            vw.write(im)
        vw.release()
        print(f"[{tag}] wrote contact_sheet.jpg, overlay.mp4")
