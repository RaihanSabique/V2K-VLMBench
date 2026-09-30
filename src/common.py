"""Shared helpers: video I/O with per-frame timestamps, keypoint index maps, paths."""
import os
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))
# Weights are cached in the project, but the login token stays where `hf auth login` puts it.
os.environ.setdefault("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))

# Unified left-upper-limb joint set every pose model is mapped onto.
ARM_JOINTS = ["L_shoulder", "L_elbow", "L_wrist", "L_index", "L_pinky", "R_shoulder"]

# Index of each ARM_JOINTS entry in each model's native keypoint layout.
# COCO-17 (ViTPose) has no hand points -> -1.
JOINT_MAP = {
    "vitpose": [5, 7, 9, -1, -1, 6],
    # COCO-WholeBody-133: left hand block 91..111 (91 wrist, 96 index MCP, 108 pinky MCP)
    "rtmpose": [5, 7, 9, 96, 108, 6],
    # MediaPipe BlazePose-33: 17 left pinky, 19 left index (knuckle-level landmarks)
    "mediapipe": [11, 13, 15, 19, 17, 12],
}


def video_tag(path):
    """Short, filesystem-safe id for a video file."""
    stem = Path(path).stem
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def list_videos():
    return sorted(DATA.glob("*.mp4"))


def read_video(path, max_frames=None):
    """Decode all frames (BGR) and the container timestamp of each frame in seconds.

    Timestamps come from the demuxer (CAP_PROP_POS_MSEC), not frame_idx / fps, so a
    variable-frame-rate webcam recording keeps its true timing.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, ts = [], []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    ts = np.asarray(ts)
    # Some builds report the timestamp of the *next* frame, or 0 throughout.
    if len(ts) > 1 and (np.all(ts == 0) or np.any(np.diff(ts) <= 0)):
        ts = np.arange(len(frames)) / fps
    ts = ts - ts[0]
    return frames, ts, fps


def out_dir(tag, *sub):
    d = OUT / tag
    for s in sub:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def box_to_affine(box, out_w, out_h, pad=1.25):
    """Affine that maps an (x0,y0,x1,y1) box, padded and aspect-fixed, onto an out_w x out_h crop."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * pad, (y1 - y0) * pad
    aspect = out_w / out_h
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    s = out_w / w
    M = np.array([[s, 0, out_w / 2 - s * cx], [0, s, out_h / 2 - s * cy]], np.float32)
    return M


def invert_affine_points(M, pts):
    Mi = cv2.invertAffineTransform(M)
    pts = np.asarray(pts, np.float32)
    return pts @ Mi[:, :2].T + Mi[:, 2]
