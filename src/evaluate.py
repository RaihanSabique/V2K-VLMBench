"""Stage 5: score each pose/segmentation model against the white-marker tracks.

    python src/evaluate.py      -> outputs/model_comparison.csv (+ per-video copies)

Markers sit on the skin surface, not at joint centres, so each joint is compared with
its nearest marker proxy and two errors are reported:
    median_err_px   raw distance (includes the constant marker-to-joint-centre offset)
    jitter_px       RMS distance after removing each model's median offset vector -
                    how consistently the model follows the limb frame to frame
    pck25 / pck50   fraction of frames within 25 / 50 px (raw)
    speed_r         Pearson r between the model's filtered joint speed and the marker
                    proxy's speed (does the model reproduce the kinematic profile?)
    peak_speed_ratio  model / marker peak speed (motion-blur damping shows up as < 1)
Only frames where the proxy marker(s) were directly detected (quality 1) are scored.
"""
import numpy as np
import pandas as pd

from common import OUT, list_videos, out_dir, video_tag
from kinematics import lowpass_runs
from skeleton import CHAIN, model_chain

PROXY = {"shoulder": ["LSHO"], "elbow": ["LELB"], "wrist": ["LWRA", "LWRB"], "hand": ["LHND"]}
MODELS = ["vitpose", "rtmpose", "mediapipe", "fused"]


def marker_proxies(tag):
    long = pd.read_csv(out_dir(tag) / "markers_long.csv")
    out = {}
    for j, ms in PROXY.items():
        xy, ok = [], []
        for m in ms:
            g = long[long.marker == m].sort_values("frame")
            xy.append(g[["x_px", "y_px"]].values)
            ok.append(g.quality.values == 1)
        out[j] = (np.mean(xy, axis=0), np.all(ok, axis=0))
    t = long[long.marker == "LSHO"].sort_values("frame").time_s.values
    return out, t


def speed(xy, t, fs):
    f = np.stack([lowpass_runs(xy[:, i].astype(float), fs, 6.0) for i in range(2)], 1)
    return np.linalg.norm(np.gradient(f, t, axis=0), axis=1)


def evaluate(tag):
    prox, t = marker_proxies(tag)
    fs = (len(t) - 1) / (t[-1] - t[0])
    rows = []
    for name in MODELS:
        if name == "fused":
            pts = np.load(out_dir(tag) / "skeleton_fused.npz")["joints"]
        else:
            pts, _ = model_chain(tag, name)
        for j, jn in enumerate(CHAIN):
            ref, ok = prox[jn]
            p = pts[:, j]
            valid = ok & ~np.isnan(p[:, 0])
            if valid.sum() < 20:
                rows.append(dict(video=tag, model=name, joint=jn, n_frames=int(valid.sum())))
                continue
            e = p[valid] - ref[valid]
            d = np.linalg.norm(e, axis=1)
            jit = np.sqrt(np.mean(np.sum((e - np.median(e, 0)) ** 2, 1)))
            # speed agreement on the frames where both exist (gaps left as NaN)
            pm = p.copy()
            pm[~valid] = np.nan
            rm = ref.copy()
            rm[~valid] = np.nan
            sp_m, sp_r = speed(pm, t, fs), speed(rm, t, fs)
            both = ~np.isnan(sp_m) & ~np.isnan(sp_r)
            r = np.corrcoef(sp_m[both], sp_r[both])[0, 1] if both.sum() > 20 else np.nan
            rows.append(dict(video=tag, model=name, joint=jn, n_frames=int(valid.sum()),
                             coverage=float(np.mean(~np.isnan(p[:, 0]))),
                             median_err_px=float(np.median(d)), jitter_px=float(jit),
                             pck25=float(np.mean(d < 25)), pck50=float(np.mean(d < 50)),
                             speed_r=float(r),
                             peak_speed_ratio=float(np.nanpercentile(sp_m[both], 99) /
                                                    np.nanpercentile(sp_r[both], 99))))
    # Sapiens2: fraction of detected markers inside the left-arm mask, raw (closed) and as
    # the dilated ROI the tracker uses (markers protrude from the silhouette edge).
    lab = np.load(out_dir(tag, "models") / "sapiens2.npz")["labels"]
    long = pd.read_csv(out_dir(tag) / "markers_long.csv")
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    for m in ["LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LHND", "LFIN"]:
        g = long[(long.marker == m) & (long.quality == 1)]
        hit, hit_roi = [], []
        for f, x, y in g[["frame", "x_px", "y_px"]].values[::3]:
            arm = cv2.morphologyEx(np.isin(lab[int(f)], (6, 7, 11)).astype(np.uint8), cv2.MORPH_CLOSE, k)
            hit.append(bool(arm[int(y), int(x)]))
            hit_roi.append(bool(cv2.dilate(arm, kd)[int(y), int(x)]))
        rows.append(dict(video=tag, model="sapiens2", joint=m, n_frames=len(hit),
                         seg_contains_marker=float(np.mean(hit)) if hit else np.nan,
                         seg_roi_contains_marker=float(np.mean(hit_roi)) if hit else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    allr = []
    for vp in list_videos():
        tag = video_tag(vp)
        df = evaluate(tag)
        df.to_csv(out_dir(tag) / "model_comparison.csv", index=False, float_format="%.3f")
        allr.append(df)
    allr = pd.concat(allr)
    allr.to_csv(OUT / "model_comparison.csv", index=False, float_format="%.3f")
    pose = allr[allr.model != "sapiens2"]
    print(pose.pivot_table(index=["model", "joint"], columns="video",
                           values=["median_err_px", "jitter_px", "speed_r", "peak_speed_ratio"]).round(2).to_string())
    print(allr[allr.model == "sapiens2"].pivot_table(index="joint", columns="video",
                                                     values=["seg_contains_marker", "seg_roi_contains_marker"]).round(2))
