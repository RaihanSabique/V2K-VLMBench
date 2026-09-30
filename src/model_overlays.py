"""One overlay video per model, plus a 2x2 comparison, in outputs/<video>/overlays/.

    python src/model_overlays.py                       # vitpose rtmpose mediapipe sapiens2 markers
    python src/model_overlays.py --models dinov3       # DINOv3 feature PCA (slow on CPU)
    python src/model_overlays.py --compose             # 3x2 grid of all six overlays

Pose models are drawn from the cached outputs of run_models.py (keypoints with score
below the model's floor are hidden). Participant's left side = orange, right = blue.
Videos are H.264 (bundled ffmpeg) so they play in browsers and VS Code.
"""
import argparse
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm

from common import ROOT, list_videos, out_dir, read_video, video_tag
from skeleton import THR

LEFT, RIGHT, MID = (52, 104, 235), (214, 120, 42), (122, 175, 27)  # BGR: orange, blue, green

COCO_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
              (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
WB_FEET = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
MP_EDGES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16),
            (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
            (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]


def side_coco(i):  # COCO / whole-body body part: odd = left, even = right, 0 = nose
    return MID if i == 0 else (LEFT if i % 2 else RIGHT)


def side_mp(i):  # BlazePose: 1,2,3,7,9 and odd >= 11 are left
    if i == 0:
        return MID
    return LEFT if (i in (1, 2, 3, 7, 9) or (i >= 11 and i % 2)) else RIGHT


def draw_graph(img, pts, ok, edges, color_of, r=6, th=3):
    for a, b in edges:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     color_of(a) if color_of(a) == color_of(b) else MID, th, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(img, tuple(pts[i].astype(int)), r, color_of(i), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[i].astype(int)), r, (255, 255, 255), 1, cv2.LINE_AA)


def label(img, text, t):
    cv2.rectangle(img, (0, 0), (img.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(img, text, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, f"t = {t:6.3f} s", (img.shape[1] - 330, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------- renderers
def r_vitpose(tag):
    z = np.load(out_dir(tag, "models") / "vitpose.npz")
    k, s = z["kpts"], z["scores"] >= THR["vitpose"]

    def f(img, t, sec):
        draw_graph(img, k[t], s[t], COCO_EDGES, side_coco)
        return label(img, "ViTPose+ base (COCO-17)", sec)
    return f


def r_rtmpose(tag):
    z = np.load(out_dir(tag, "models") / "rtmpose.npz")
    k, s = z["kpts"], z["scores"] >= THR["rtmpose"]
    lh = [(a + 91, b + 91) for a, b in HAND_EDGES]
    rh = [(a + 112, b + 112) for a, b in HAND_EDGES]

    def f(img, t, sec):
        for i in np.flatnonzero(s[t, 23:91]) + 23:  # face points, small
            cv2.circle(img, tuple(k[t, i].astype(int)), 2, MID, -1)
        draw_graph(img, k[t], s[t], COCO_EDGES + WB_FEET, side_coco)
        draw_graph(img, k[t], s[t], lh, lambda i: LEFT, r=3, th=2)
        draw_graph(img, k[t], s[t], rh, lambda i: RIGHT, r=3, th=2)
        return label(img, "RTMPose-Body2d (whole-body 133)", sec)
    return f


def r_mediapipe(tag):
    z = np.load(out_dir(tag, "models") / "mediapipe.npz")
    k = z["kpts"]
    s = (z["scores"] >= THR["mediapipe"]) & ~np.isnan(k[..., 0])
    hand = z["hand"]
    # same 150 px gate as run_models: drop a "left hand" that is really the right hand
    far = ~(np.linalg.norm(hand[:, 0] - k[:, 15], axis=1) <= 150)
    hs = ~np.isnan(hand[..., 0]) & ~far[:, None]

    def f(img, t, sec):
        draw_graph(img, k[t], s[t], MP_EDGES, side_mp)
        draw_graph(img, hand[t], hs[t], HAND_EDGES, lambda i: LEFT, r=3, th=2)
        return label(img, "MediaPipe BlazePose heavy + hand", sec)
    return f


def r_sapiens2(tag):
    sys.path.insert(0, str(ROOT / "third_party/sapiens2"))
    from sapiens.dense.src.datasets.seg.seg_utils import DOME_CLASSES_29
    pal = np.array([v["color"][::-1] for v in DOME_CLASSES_29.values()], np.uint8)  # RGB -> BGR
    names = [v["name"] for v in DOME_CLASSES_29.values()]
    lab = np.load(out_dir(tag, "models") / "sapiens2.npz")["labels"]

    def f(img, t, sec):
        L = lab[t]
        col = pal[L]
        fg = L > 0
        img[fg] = (0.45 * img[fg] + 0.55 * col[fg]).astype(np.uint8)
        # legend: classes present in this frame, largest first
        ids, cnt = np.unique(L[fg], return_counts=True)
        y = 90
        for i in ids[np.argsort(-cnt)][:8]:
            cv2.rectangle(img, (20, y - 18), (44, y + 4), tuple(int(c) for c in pal[i]), -1)
            cv2.putText(img, names[i], (52, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 32
        return label(img, "Sapiens2-0.4B body-part segmentation", sec)
    return f


def r_markers(tag):
    from render import draw, load
    names, mk, joints, _ = load(tag)

    def f(img, t, sec):
        return label(draw(img, t, mk, names, joints, None), "White markers (protocol labels)", sec)
    return f


def r_dinov3(tag, frames, backbone="facebook/dinov3-vitl16-pretrain-lvd1689m", size=(896, 504)):
    """Foreground PCA of DINOv3 patch features -> RGB, the visualisation style of the DINOv3
    paper. Over the whole frame the top components on this scene follow image position (the
    same floor is cos 0.34 left vs right), so the PCA is fitted only on patches Sapiens2 marks
    as the person, on 12 frames spread over the clip; colours are then comparable over time.
    Background is shown dimmed and grey."""
    import torch
    from dino_track import DenseFeatures
    fx = DenseFeatures(backbone)
    H, W = frames[0].shape[:2]
    lab = np.load(out_dir(tag, "models") / "sapiens2.npz")["labels"]

    def feats(img):
        return fx(cv2.resize(img, size, interpolation=cv2.INTER_AREA))  # (C, h, w), L2-normalised

    def fg_grid(t, h, w):
        return cv2.resize((lab[t] > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5

    rows = []
    for i in np.linspace(0, len(frames) - 1, 12).astype(int):
        fe = feats(frames[i])
        rows.append(fe.flatten(1).T[torch.from_numpy(fg_grid(i, *fe.shape[1:]).ravel())])
    sample = torch.cat(rows)
    mean = sample.mean(0)
    _, _, V = torch.pca_lowrank(sample - mean, q=3, center=False)
    proj = (sample - mean) @ V
    lo, hi = torch.quantile(proj, 0.02, dim=0), torch.quantile(proj, 0.98, dim=0)

    def f(img, t, sec):
        fe = feats(frames[t])
        c, h, w = fe.shape
        p = ((fe.flatten(1).T - mean) @ V - lo) / (hi - lo)
        rgb = (p.clamp(0, 1).reshape(h, w, 3).cpu().numpy() * 255).astype(np.uint8)
        rgb = cv2.resize(rgb[..., ::-1], (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        m = cv2.resize(fg_grid(t, h, w).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        grey = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.45
        person = 0.35 * img + 0.65 * rgb
        img = (m * person + (1 - m) * grey).clip(0, 255).astype(np.uint8)
        return label(img, "DINOv3 ViT-L/16 feature PCA on the person (RGB = top 3 components)", sec)
    return f


# ----------------------------------------------------------------------------- writer
def writer(path, fps, size):
    w = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", quality=None,
                                    output_params=["-crf", "23", "-preset", "medium"],
                                    pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=8)
    w.send(None)
    return w


def render_all(vp, models, grid):
    tag = video_tag(vp)
    frames, ts, fps = read_video(vp)
    H, W = frames[0].shape[:2]
    d = out_dir(tag, "overlays")
    makers = {m: (r_dinov3(tag, frames) if m == "dinov3" else globals()[f"r_{m}"](tag)) for m in models}
    outs = {m: writer(d / f"{m}.mp4", fps, (W, H)) for m in models}
    g = writer(d / "comparison_grid.mp4", fps, (W, H)) if grid else None
    for t in tqdm(range(len(frames)), desc=tag):
        tiles = {}
        for m, fn in makers.items():
            im = fn(frames[t].copy(), t, ts[t])
            outs[m].send(np.ascontiguousarray(im[..., ::-1]))
            tiles[m] = im
        if g is not None:
            order = [m for m in ("vitpose", "rtmpose", "mediapipe", "sapiens2") if m in tiles]
            small = [cv2.resize(tiles[m], (W // 2, H // 2), interpolation=cv2.INTER_AREA) for m in order]
            top, bot = np.hstack(small[:2]), np.hstack(small[2:4])
            g.send(np.ascontiguousarray(np.vstack([top, bot])[..., ::-1]))
    for w in list(outs.values()) + ([g] if g is not None else []):
        w.close()
    print(f"[{tag}] wrote", ", ".join(f"{m}.mp4" for m in models) + (", comparison_grid.mp4" if grid else ""))


ALL_ORDER = ["vitpose", "rtmpose", "mediapipe", "sapiens2", "dinov3", "markers"]


def compose_all(tag):
    """3x2 grid of the six finished per-model overlays -> overlays/all_models_grid.mp4."""
    d = out_dir(tag, "overlays")
    caps = [cv2.VideoCapture(str(d / f"{m}.mp4")) for m in ALL_ORDER]
    n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    tw, th = 640, 360
    w = writer(d / "all_models_grid.mp4", fps, (3 * tw, 2 * th))
    for _ in tqdm(range(n), desc=f"grid {tag}"):
        tiles = []
        for c in caps:
            ok, im = c.read()
            tiles.append(cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA))
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        w.send(np.ascontiguousarray(grid[..., ::-1]))
    w.close()
    print(f"[{tag}] wrote all_models_grid.mp4 ({n} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["vitpose", "rtmpose", "mediapipe", "sapiens2", "markers"])
    ap.add_argument("--video", type=int, default=None, help="index into list_videos() (default: all)")
    ap.add_argument("--compose", action="store_true", help="only tile the six finished overlays")
    args = ap.parse_args()
    if args.compose:
        for vp in list_videos():
            compose_all(video_tag(vp))
        sys.exit()
    grid = all(m in args.models for m in ("vitpose", "rtmpose", "mediapipe", "sapiens2"))
    vids = list_videos() if args.video is None else [list_videos()[args.video]]
    for vp in vids:
        render_all(vp, args.models, grid)
