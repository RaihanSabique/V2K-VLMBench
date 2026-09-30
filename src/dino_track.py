"""Stage 3 (optional): DINOv3 dense-feature re-detection of markers the blob tracker lost.

    python src/dino_track.py [--backbone facebook/dinov3-vitl16-pretrain-lvd1689m]

For each marker, a reference descriptor is averaged from DINOv3 patch features at frames
where it was detected cleanly (quality 1). Where a marker is still missing after
markers.py (quality 0), its skeleton-predicted neighbourhood is searched for the peak
cosine similarity; accepted hits are snapped to the nearest weak white blob if one exists.
Updates markers_long.csv / markers_wide.csv in place, marking those samples quality 3,
and writes dino_similarity_<backbone>.csv: per-marker similarity floor, a hold-out test
(re-detect the marker on frames where the blob tracker found it and measure the error),
and how many missing frames were filled. Only analysed arm markers are handled, and a
marker's gaps are filled only if its hold-out test passes (reliable_holdout).
--dry_run computes everything but leaves the marker CSVs untouched.

DINOv3 is gated on Hugging Face: accept the licence on the model page and run
`hf auth login` once before this stage.
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from blobs import detect_candidates
from common import list_videos, out_dir, read_video, video_tag
from skeleton import chain_to_point

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class DenseFeatures:
    def __init__(self, repo):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(repo)
        # fp32: in fp16 DINOv3 ViT-L overflows to NaN features on V100 (no bf16 there)
        self.model = AutoModel.from_pretrained(repo, dtype=torch.float32).to(DEV).eval()
        cfg = self.model.config
        self.patch = cfg.patch_size
        self.n_prefix = 1 + getattr(cfg, "num_register_tokens", 0)
        self.mean = torch.tensor(self.proc.image_mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(self.proc.image_std, device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, crop_bgr):
        """L2-normalised patch features (C, h, w) for a crop whose sides are multiples of patch."""
        x = torch.from_numpy(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).to(DEV)
        x = x.permute(2, 0, 1)[None].float() / 255.0
        x = (x - self.mean) / self.std
        h, w = x.shape[2] // self.patch, x.shape[3] // self.patch
        tok = self.model(pixel_values=x).last_hidden_state[0, self.n_prefix:]
        f = tok.float().T.reshape(-1, h, w)
        return F.normalize(f, dim=0)


def arm_crop(joints, H, W, margin=110, scale=2.0, patch=16):
    """Box around the fused arm, upsampled x`scale` so one patch ~ 8 px of the original."""
    x0, y0 = np.floor(joints.min(0) - margin).astype(int)
    x1, y1 = np.ceil(joints.max(0) + margin).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    ow = int(round((x1 - x0) * scale / patch)) * patch
    oh = int(round((y1 - y0) * scale / patch)) * patch
    return (x0, y0, x1, y1), (ow, oh)


def sample(feat, pts, box, size):
    """Bilinearly sample normalised features at full-frame points pts (N,2)."""
    x0, y0, x1, y1 = box
    ow, oh = size
    gx = (pts[:, 0] - x0) / (x1 - x0) * 2 - 1
    gy = (pts[:, 1] - y0) / (y1 - y0) * 2 - 1
    g = torch.tensor(np.stack([gx, gy], 1), device=DEV, dtype=torch.float32).view(1, -1, 1, 2)
    s = F.grid_sample(feat[None], g, align_corners=False)[0, :, :, 0].T
    return F.normalize(s, dim=1)


def reliable_holdout(errs, n_rej, max_median=10.0, min_within=0.8):
    """Fill a marker's gaps only if DINOv3 re-found it accurately on held-out frames:
    median error <= 10 px and >= 80 % of accepted re-detections within 10 px."""
    e = np.asarray(errs)
    return bool(len(e) >= 10 and np.median(e) <= max_median and np.mean(e <= 10) >= min_within)


def process(vp, fx, args):
    tag = video_tag(vp)
    d = out_dir(tag)
    frames, ts, _ = read_video(vp)
    H, W = frames[0].shape[:2]
    joints = np.load(d / "skeleton_fused.npz")["joints"]
    long = pd.read_csv(d / "markers_long.csv")
    tpl = pd.read_csv(d / "marker_template.csv")
    names = tpl.label.tolist()
    T, K = len(frames), len(names)
    P = np.stack([long[long.marker == n].sort_values("frame")[["x_px", "y_px"]].values for n in names], 1)
    Q = np.stack([long[long.marker == n].sort_values("frame")["quality"].values for n in names], 1)

    # 1) reference descriptors from clean detections, spread over the clip
    feats_cache = {}

    def get(t):
        if t not in feats_cache:
            box, size = arm_crop(joints[t], H, W, scale=args.scale, patch=fx.patch)
            x0, y0, x1, y1 = box
            crop = cv2.resize(frames[t][y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC)
            if len(feats_cache) > 64:
                feats_cache.clear()
            feats_cache[t] = (fx(crop), box, size)
        return feats_cache[t]

    # Only analysed arm markers: the crop and the chain-based search prediction are
    # defined on the arm (trunk templates live in a different coordinate frame).
    use = [k for k in range(K) if tpl.group[k] == "arm" and bool(tpl.analysed[k])]
    seg = np.clip(np.floor(tpl.c.values), 0, 2).astype(int)
    u = tpl.c.values - seg
    v = tpl.v.values

    ref, ref_frames = {}, {}
    for k in use:
        good = np.flatnonzero(Q[:, k] == 1)
        if not len(good):
            continue
        pick = good[np.linspace(0, len(good) - 1, min(args.n_ref, len(good))).astype(int)]
        vecs = []
        for t in pick:
            f, b, sz = get(t)
            vecs.append(sample(f, P[t, k][None], b, sz))
        ref[k] = F.normalize(torch.cat(vecs).mean(0), dim=0)
        ref_frames[k] = set(pick.tolist())

    # similarity of the clean detections to their own reference -> per-marker acceptance floor
    sims_ok = {k: [] for k in ref}
    for t in range(0, T, 5):
        f, b, sz = get(t)
        for k in ref:
            if Q[t, k] == 1:
                sims_ok[k].append(float(sample(f, P[t, k][None], b, sz)[0] @ ref[k]))
    # accept a re-detection only if it looks like this marker's clean detections do;
    # never below 0.5 cosine similarity, however noisy the marker's own history
    floor = {k: max(0.5, np.percentile(x, 5)) if len(x) > 5 else 0.6 for k, x in sims_ok.items()}

    def locate(t, k):
        """DINOv3 re-detection of marker k in frame t near its skeleton prediction.
        Returns the position or None if the best match is below the acceptance floor."""
        f, (x0, y0, x1, y1), (ow, oh) = get(t)
        pred = chain_to_point(seg[k], u[k], v[k], joints[t])
        sim = torch.einsum("chw,c->hw", f, ref[k])
        sm = F.interpolate(sim[None, None], size=(oh, ow), mode="bilinear")[0, 0].cpu().numpy()
        sx, sy = (ow / (x1 - x0)), (oh / (y1 - y0))
        px, py = (pred[0] - x0) * sx, (pred[1] - y0) * sy
        yy, xx = np.mgrid[0:oh, 0:ow]
        sm[(xx - px) ** 2 + (yy - py) ** 2 > (args.search_px * sx) ** 2] = -1
        j = int(np.argmax(sm))
        best, (by, bx) = sm.flat[j], divmod(j, ow)
        if not np.isfinite(best) or best < floor[k]:
            return None
        hit = np.array([x0 + bx / sx, y0 + by / sy])
        # snap to a weak blob (relaxed thresholds) within 15 px, if any
        r0 = 30
        lx, ly = int(max(0, hit[0] - r0)), int(max(0, hit[1] - r0))
        patch = frames[t][ly:int(hit[1] + r0), lx:int(hit[0] + r0)]
        c = detect_candidates(patch, th_tophat=25, th_white=120) if patch.size else np.zeros((0, 8))
        if len(c):
            c[:, :2] += [lx, ly]
            dd = np.linalg.norm(c[:, :2] - hit, axis=1)
            if dd.min() < 15:
                hit = c[np.argmin(dd), :2]
        return hit

    # 2) hold-out check: hide frames where the blob tracker did find the marker (not used
    #    for the reference descriptor) and measure how far DINOv3's re-detection lands.
    hold = {}
    rng = np.random.default_rng(0)
    for k in ref:
        cand = [t for t in np.flatnonzero(Q[:, k] == 1) if t not in ref_frames[k]]
        errs, n_rej = [], 0
        for t in rng.choice(cand, size=min(args.n_holdout, len(cand)), replace=False):
            hit = locate(int(t), k)
            if hit is None:
                n_rej += 1
            else:
                errs.append(float(np.linalg.norm(hit - P[t, k])))
        hold[k] = (errs, n_rej)

    # 3) fill the gaps - only for markers whose hold-out re-detection is accurate
    reliable = {k: reliable_holdout(*hold[k]) for k in ref}
    n_filled = {k: 0 for k in ref}
    for t in tqdm(range(T), desc=f"dino {tag}"):
        for k in ref:
            if Q[t, k] == 0 and reliable[k]:
                hit = locate(t, k)
                if hit is not None:
                    P[t, k] = hit
                    Q[t, k] = 3
                    n_filled[k] += 1

    rows = []
    for k in ref:
        errs, n_rej = hold[k]
        e = np.array(errs)
        rows.append(dict(marker=names[k], sim_floor=floor[k],
                         sim_clean_median=np.median(sims_ok[k]) if sims_ok[k] else np.nan,
                         holdout_n=len(e) + n_rej, holdout_rejected=n_rej,
                         holdout_median_err_px=np.median(e) if len(e) else np.nan,
                         holdout_p90_err_px=np.percentile(e, 90) if len(e) else np.nan,
                         holdout_within_10px=float(np.mean(e <= 10)) if len(e) else np.nan,
                         reliable=reliable[k],
                         missing_before=int(np.sum(Q[:, k] == 3) + np.sum(Q[:, k] == 0)),
                         frames_filled=n_filled[k]))
    stats = pd.DataFrame(rows)
    tagb = args.backbone.split("/")[-1]
    stats.to_csv(d / f"dino_similarity_{tagb}.csv", index=False)
    print(stats.to_string())
    if args.dry_run:
        return
    for k, n in enumerate(names):
        m = long.marker == n
        order = long[m].sort_values("frame").index
        long.loc[order, ["x_px", "y_px"]] = P[:, k]
        long.loc[order, "quality"] = Q[:, k]
    long.to_csv(d / "markers_long.csv", index=False, float_format="%.4f")
    wide = long.pivot(index=["frame", "time_s"], columns="marker", values=["x_px", "y_px", "quality"])
    suffix = {"x_px": "x", "y_px": "y", "quality": "q"}
    wide.columns = [f"{m}_{suffix[v]}" for v, m in wide.columns]
    wide = wide[[f"{n}_{s}" for n in names for s in ("x", "y", "q")]].reset_index()
    wide.to_csv(d / "markers_wide.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--search_px", type=float, default=60)
    ap.add_argument("--n_ref", type=int, default=40)
    ap.add_argument("--n_holdout", type=int, default=40)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    fx = DenseFeatures(args.backbone)
    for vp in list_videos():
        process(vp, fx, args)
