"""Stage 4: kinematic time series and per-movement reach features from the marker tracks.

    python src/kinematics.py [--cutoff_hz 6] [--px_per_cm X]

Per video writes outputs/<tag>/:
    kinematics_timeseries.csv  per frame: filtered marker positions, wrist-centre and LHND
                               velocity/speed/acceleration/jerk, protocol segment lengths,
                               elbow / wrist / shoulder angles, trunk marker displacement,
                               shoulder-line rotation, movement id / type
    movements.csv              one row per segmented movement with reach features
    session_summary.csv        per-video aggregates
All positions are image-plane pixels (2-D projection of 3-D motion) unless --px_per_cm
is given, in which case lengths are also reported in cm.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from common import list_videos, out_dir, video_tag

# Marker Placement protocol (Plug-in Gait Upper Body), markers visible from above.
MARKERS = ["LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LHND", "LFIN",
           "RSHO", "C7", "RBAK", "T10"]


# ----------------------------------------------------------------------------- signal
def lowpass_runs(x, fs, fc, order=4, min_len=16):
    """Zero-lag Butterworth applied to each contiguous non-NaN run (gaps stay NaN)."""
    b, a = butter(order, fc / (fs / 2))
    y = np.full_like(x, np.nan, dtype=float)
    ok = ~np.isnan(x)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= min_len:
            y[s:e] = filtfilt(b, a, x[s:e], padlen=min(3 * max(len(a), len(b)), e - s - 1))
        else:
            y[s:e] = x[s:e]
    return y


def deriv(x, t):
    return np.gradient(x, t, axis=0)


def sparc(speed, fs, pad=4, fc=10.0, amp_th=0.05):
    """Spectral arc length (Balasubramanian et al. 2015). More negative = less smooth."""
    n = int(2 ** (np.ceil(np.log2(len(speed))) + pad))
    f = np.arange(n) * fs / n
    M = np.abs(np.fft.fft(speed, n))
    M = M / M.max()
    sel = f <= fc
    f, M = f[sel], M[sel]
    idx = np.flatnonzero(M >= amp_th)
    f, M = f[idx[0]:idx[-1] + 1], M[idx[0]:idx[-1] + 1]
    if len(f) < 2:
        return np.nan
    return -np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(M) ** 2))


def ldlj(speed, t):
    """Log dimensionless jerk from the speed profile (Balasubramanian et al. 2015)."""
    mt = t[-1] - t[0]
    vpk = speed.max()
    if mt <= 0 or vpk <= 0:
        return np.nan
    j = np.gradient(np.gradient(speed, t), t)
    return -np.log((mt ** 3 / vpk ** 2) * np.trapezoid(j ** 2, t))


def angle_at(b, a, c):
    """Angle ABC in degrees (a, c are the neighbours of vertex b). 180 = straight."""
    u, v = a - b, c - b
    cos = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# ----------------------------------------------------------------------------- endpoint
def wrist_centre(P):
    """Midpoint of LWRA/LWRB. When one is missing, use the other plus the median
    A->centre offset seen while both were visible."""
    a, b = P["LWRA"], P["LWRB"]
    both = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    wc = (a + b) / 2
    if both.sum() > 10:
        oa = np.median(wc[both] - a[both], axis=0)
        ob = np.median(wc[both] - b[both], axis=0)
        only_a = ~np.isnan(a[:, 0]) & np.isnan(b[:, 0])
        only_b = np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
        wc[only_a] = a[only_a] + oa
        wc[only_b] = b[only_b] + ob
    return wc


# ----------------------------------------------------------------------------- segmentation
def segment(speed, t, rel_thr=0.10, merge_s=0.10, min_dur_s=0.15):
    """Movements = runs where speed exceeds rel_thr x the clip's robust peak speed
    (95th percentile). Runs separated by < merge_s are merged; each run is then widened
    to where speed falls below rel_thr x its own peak (standard onset/offset rule)."""
    s = np.nan_to_num(speed)
    vref = np.nanpercentile(speed, 95)
    on = s > rel_thr * vref
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    runs = [[a, b] for a, b in zip(edges[::2], edges[1::2])]
    merged = []
    for r in runs:
        if merged and t[r[0]] - t[merged[-1][1] - 1] < merge_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for a, b in merged:
        pk = s[a:b].max()
        while a > 0 and s[a - 1] > rel_thr * pk and s[a - 1] <= s[a]:
            a -= 1
        while b < len(s) and s[b] > rel_thr * pk and s[b] <= s[b - 1]:
            b += 1
        if t[min(b, len(t) - 1)] - t[a] >= min_dur_s:
            out.append((a, b))
    return out


def split_reversals(moves, y, speed, min_dy):
    """Split a movement at the speed minimum between two opposite-direction excursions
    along the reach axis (e.g. a partial return that turns back to the iPad without
    the speed dropping below the onset threshold)."""
    out = []
    stack = list(moves)
    while stack:
        a, b = stack.pop(0)
        yy = y[a:b]
        best = None
        for k in range(2, len(yy) - 2):
            if speed[a + k] <= speed[a + k - 1] and speed[a + k] <= speed[a + k + 1]:
                d1, d2 = yy[k] - yy[0], yy[-1] - yy[k]
                if abs(d1) >= min_dy and abs(d2) >= min_dy and np.sign(d1) != np.sign(d2):
                    if best is None or speed[a + k] < speed[a + best]:
                        best = k
        if best is None:
            out.append((a, b))
        else:
            stack[:0] = [(a, a + best + 1), (a + best, b)]
    return out


def classify(p0, p1, target_y, home_y):
    """Reach axis = image vertical (the iPad is at the top of the frame, the home star
    near the bottom). A movement covering >= 1/4 of the observed endpoint range along that
    axis is a reach (towards the iPad) or return (away); shorter ones are named by zone."""
    span = home_y - target_y
    dy = p1[1] - p0[1]
    if abs(dy) >= span / 4:
        return "reach" if dy < 0 else "return"
    z = lambda y: "target" if y < target_y + span / 3 else ("home" if y > home_y - span / 3 else "mid")
    z0, z1 = z(p0[1]), z(p1[1])
    if z0 == z1 == "target":
        return "on_target"
    if z0 == z1 == "home":
        return "at_home"
    return "other"


# ----------------------------------------------------------------------------- main
def process(tag, args):
    d = out_dir(tag)
    long = pd.read_csv(d / "markers_long.csv")
    tpl = pd.read_csv(d / "marker_template.csv")
    # markers found in < 30 % of frames are not analysed (see markers.MIN_SUPPORT)
    names = [m for m in MARKERS if m in set(tpl.label[tpl.analysed])]
    t = long[long.marker == names[0]].sort_values("frame").time_s.values
    fs = (len(t) - 1) / (t[-1] - t[0])
    P, Q = {}, {}
    for m in names:
        g = long[long.marker == m].sort_values("frame")
        xy = g[["x_px", "y_px"]].values.astype(float)
        xy[g.quality.values == 0] = np.nan
        P[m] = np.stack([lowpass_runs(xy[:, i], fs, args.cutoff_hz) for i in range(2)], 1)
        Q[m] = g.quality.values

    wc = wrist_centre(P)
    ts = pd.DataFrame({"frame": np.arange(len(t)), "time_s": t})
    for m in names:
        ts[f"{m}_x"], ts[f"{m}_y"] = P[m][:, 0], P[m][:, 1]
    ts["WRC_x"], ts["WRC_y"] = wc[:, 0], wc[:, 1]

    def kin(prefix, xy):
        v = deriv(xy, t)
        sp = np.linalg.norm(v, axis=1)
        acc = deriv(sp, t)  # tangential acceleration
        jerk = deriv(acc, t)
        ts[f"{prefix}_vx"], ts[f"{prefix}_vy"] = v[:, 0], v[:, 1]
        ts[f"{prefix}_speed"], ts[f"{prefix}_acc"], ts[f"{prefix}_jerk"] = sp, acc, jerk
        return sp

    sp = kin("WRC", wc)
    sp_hnd = kin("LHND", P["LHND"]) if "LHND" in P else None
    nan = np.full(len(t), np.nan)
    has = lambda *ms: all(m in P for m in ms)

    # Protocol segments (marker-based): upper arm LSHO-LELB, forearm LELB-LWRB,
    # hand wrist centre-LHND. Their 2-D lengths vary with out-of-plane rotation.
    ts["upper_arm_len_px"] = np.linalg.norm(P["LSHO"] - P["LELB"], axis=1)
    ts["forearm_len_px"] = np.linalg.norm(P["LELB"] - P["LWRB"], axis=1)
    ts["hand_len_px"] = np.linalg.norm(wc - P["LHND"], axis=1) if has("LHND") else nan

    # Joint angles, 2-D projected (180 = segments in line)
    ts["elbow_angle_deg"] = angle_at(P["LELB"], P["LSHO"], wc)
    ts["elbow_angvel_dps"] = deriv(ts["elbow_angle_deg"].values, t)
    ts["wrist_angle_deg"] = angle_at(wc, P["LELB"], P["LHND"]) if has("LHND") else nan
    # Shoulder: angle at LSHO between the shoulder line (towards RSHO, or the fused right
    # shoulder joint when RSHO is not visible) and the upper arm. ~90 = arm pointing
    # forward, larger = arm swung out laterally.
    sk = np.load(d / "skeleton_fused.npz")
    rref = P["RSHO"] if has("RSHO") else sk["r_shoulder"]
    ts["shoulder_ref"] = "RSHO" if has("RSHO") else "pose_R_shoulder"
    ts["shoulder_angle_deg"] = angle_at(P["LSHO"], rref, P["LELB"])

    # Trunk: displacement of each trunk marker from its median position, and rotation of
    # the shoulder line (RSHO->LSHO) about its median direction.
    for m in ("C7", "T10", "RBAK", "LSHO"):
        ts[f"{m}_disp_px"] = np.linalg.norm(P[m] - np.nanmedian(P[m], 0), axis=1) if has(m) else nan
    if has("RSHO"):
        z = (P["LSHO"] - P["RSHO"]) @ np.array([1, 1j])
        z = z / np.abs(z)
        ref = np.nansum(z) / np.abs(np.nansum(z))  # mean direction; avoids +/-180 wrap
        ts["shoulder_line_rot_deg"] = np.degrees(np.angle(z * np.conj(ref)))
    else:
        ts["shoulder_line_rot_deg"] = nan

    moves = segment(sp, t, args.rel_thr)
    target_y, home_y = np.nanpercentile(wc[:, 1], [5, 95])
    moves = split_reversals(moves, wc[:, 1], np.nan_to_num(sp), (home_y - target_y) / 4)
    ts["movement_id"] = -1
    ts["movement_type"] = ""
    rows = []
    good_wr = (Q["LWRA"] >= 1) | (Q["LWRB"] >= 1)
    rng = lambda x: np.nanmax(x) - np.nanmin(x) if np.isfinite(x).any() else np.nan
    for i, (a, b) in enumerate(moves):
        b = min(b, len(t))
        seg_t = t[a:b]
        xy = wc[a:b]
        s = sp[a:b]
        if np.isnan(xy).any() or len(seg_t) < 5:
            continue
        path = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
        if path < args.min_path_px:  # tracking jitter while the finger rests on the screen
            continue
        kind = classify(xy[0], xy[-1], target_y, home_y)
        ts.loc[a:b - 1, "movement_id"] = i
        ts.loc[a:b - 1, "movement_type"] = kind
        disp = np.linalg.norm(xy[-1] - xy[0])
        ipk = int(np.nanargmax(s))
        acc = ts["WRC_acc"].values[a:b]
        el = ts["elbow_angle_deg"].values[a:b]
        w = lambda c: ts[c].values[a:b]
        npk = len(find_peaks(s, prominence=0.1 * s.max())[0])
        hand_path = np.nansum(np.linalg.norm(np.diff(P["LHND"][a:b], axis=0), axis=1)) if has("LHND") else np.nan
        row = dict(
            movement_id=i, type=kind, onset_s=seg_t[0], offset_s=seg_t[-1],
            truncated=bool(a == 0 or b >= len(t)),  # cut by the start/end of the clip
            duration_s=seg_t[-1] - seg_t[0],
            start_x=xy[0, 0], start_y=xy[0, 1], end_x=xy[-1, 0], end_y=xy[-1, 1],
            path_length_px=path, displacement_px=disp,
            straightness=disp / path if path > 0 else np.nan,
            peak_speed_pxps=s.max(), mean_speed_pxps=path / (seg_t[-1] - seg_t[0]),
            time_to_peak_speed_s=seg_t[ipk] - seg_t[0],
            time_to_peak_speed_pct=100 * (seg_t[ipk] - seg_t[0]) / (seg_t[-1] - seg_t[0]),
            peak_acc_pxps2=np.nanmax(acc), peak_dec_pxps2=-np.nanmin(acc),
            deceleration_time_pct=100 * (seg_t[-1] - seg_t[ipk]) / (seg_t[-1] - seg_t[0]),
            n_speed_peaks=npk, ldlj=ldlj(s, seg_t), sparc=sparc(s, fs),
            hand_path_length_px=hand_path,
            hand_peak_speed_pxps=np.nanmax(sp_hnd[a:b]) if sp_hnd is not None else np.nan,
            elbow_angle_start_deg=el[0], elbow_angle_end_deg=el[-1],
            elbow_rom_deg=rng(el),
            peak_elbow_angvel_dps=np.nanmax(np.abs(w("elbow_angvel_dps"))),
            wrist_rom_deg=rng(w("wrist_angle_deg")),
            shoulder_angle_start_deg=w("shoulder_angle_deg")[0],
            shoulder_angle_end_deg=w("shoulder_angle_deg")[-1],
            shoulder_rom_deg=rng(w("shoulder_angle_deg")),
            upper_arm_len_cv_pct=100 * np.nanstd(w("upper_arm_len_px")) / np.nanmean(w("upper_arm_len_px")),
            forearm_len_cv_pct=100 * np.nanstd(w("forearm_len_px")) / np.nanmean(w("forearm_len_px")),
            trunk_C7_disp_range_px=rng(w("C7_disp_px")),
            trunk_T10_disp_range_px=rng(w("T10_disp_px")),
            shoulder_line_rot_range_deg=rng(w("shoulder_line_rot_deg")),
            LSHO_disp_range_px=rng(w("LSHO_disp_px")),
            wrist_detected_frac=float(np.mean(good_wr[a:b])),
        )
        rows.append(row)
    mv = pd.DataFrame(rows)
    if args.px_per_cm and len(mv):
        for c in [c for c in mv.columns if c.endswith("_px") or "_pxps" in c]:
            mv[c.replace("_px", "_cm")] = mv[c] / args.px_per_cm
    ts.to_csv(d / "kinematics_timeseries.csv", index=False, float_format="%.4f")
    mv.to_csv(d / "movements.csv", index=False, float_format="%.4f")

    summ = dict(video=tag, duration_s=t[-1], fs_hz=fs, n_movements=len(mv),
                **{f"n_{k}": int((mv.type == k).sum()) if len(mv) else 0
                   for k in ["reach", "return", "on_target", "other"]},
                pct_time_moving=100 * np.mean(ts.movement_id >= 0),
                **{f"coverage_{m}_detected_pct": 100 * np.mean(Q[m] == 1) for m in names},
                **{f"coverage_{m}_usable_pct": 100 * np.mean(Q[m] >= 1) for m in names})
    if len(mv):
        for c in ["duration_s", "peak_speed_pxps", "time_to_peak_speed_pct", "straightness",
                  "n_speed_peaks", "ldlj", "sparc", "elbow_rom_deg", "wrist_rom_deg",
                  "shoulder_rom_deg", "trunk_C7_disp_range_px", "trunk_T10_disp_range_px",
                  "shoulder_line_rot_range_deg"]:
            r = mv[(mv.type == "reach") & ~mv.truncated][c]
            summ[f"reach_{c}_mean"] = r.mean()
            summ[f"reach_{c}_sd"] = r.std()
    pd.DataFrame([summ]).to_csv(d / "session_summary.csv", index=False, float_format="%.4f")
    print(f"[{tag}] fs={fs:.2f} Hz, {len(mv)} movements:",
          mv.type.value_counts().to_dict() if len(mv) else {})
    return ts, mv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff_hz", type=float, default=6.0)
    ap.add_argument("--rel_thr", type=float, default=0.10)
    ap.add_argument("--min_path_px", type=float, default=30.0)
    ap.add_argument("--px_per_cm", type=float, default=None)
    args = ap.parse_args()
    for vp in list_videos():
        process(video_tag(vp), args)
