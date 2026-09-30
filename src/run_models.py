"""Stage 1: run each pretrained model over every frame and cache raw outputs.

    python src/run_models.py --stages detect vitpose rtmpose sapiens2 mediapipe

Writes outputs/<video_tag>/models/<stage>.npz. Each pose stage stores its native
keypoints plus `arm` / `arm_score`, the left-limb joints in ARM_JOINTS order.
"""
import argparse
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import (ARM_JOINTS, JOINT_MAP, MODELS, ROOT, box_to_affine,
                    invert_affine_points, list_videos, out_dir, read_video,
                    video_tag)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def arm_from(kpts, scores, name):
    idx = JOINT_MAP[name]
    T = len(kpts)
    arm = np.full((T, len(ARM_JOINTS), 2), np.nan, np.float32)
    sc = np.zeros((T, len(ARM_JOINTS)), np.float32)
    for j, k in enumerate(idx):
        if k >= 0:
            arm[:, j] = kpts[:, k]
            sc[:, j] = scores[:, k]
    return arm, sc


# ----------------------------------------------------------------------------- detect
def stage_detect(frames, ts, tag):
    from transformers import AutoProcessor, RTDetrForObjectDetection

    rid = "PekingU/rtdetr_r50vd_coco_o365"
    proc = AutoProcessor.from_pretrained(rid)
    model = RTDetrForObjectDetection.from_pretrained(rid).to(DEV).eval()
    H, W = frames[0].shape[:2]
    boxes = np.full((len(frames), 4), np.nan, np.float32)
    bscore = np.zeros(len(frames), np.float32)
    bs = 16
    for i in tqdm(range(0, len(frames), bs), desc="rtdetr"):
        batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[i:i + bs]]
        inp = proc(images=batch, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_object_detection(
            out, target_sizes=[(H, W)] * len(batch), threshold=0.3)
        for k, r in enumerate(res):
            person = r["labels"] == 0
            if person.any():
                # Largest confident person = the participant (nobody else is in view).
                b = r["boxes"][person].cpu().numpy()
                s = r["scores"][person].cpu().numpy()
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                j = np.argmax(area * s)
                boxes[i + k], bscore[i + k] = b[j], s[j]
    # Fill misses from neighbours, then a 5-frame running median so crops don't jitter.
    good = ~np.isnan(boxes[:, 0])
    for c in range(4):
        boxes[:, c] = np.interp(np.arange(len(boxes)), np.flatnonzero(good), boxes[good, c])
    from scipy.ndimage import median_filter
    boxes = median_filter(boxes, size=(5, 1), mode="nearest")
    np.savez(out_dir(tag, "models") / "detect.npz", boxes=boxes, score=bscore, t=ts)
    print(f"[detect] person found in {good.mean():.1%} of frames")


def load_boxes(tag):
    return np.load(out_dir(tag, "models") / "detect.npz")["boxes"]


# ----------------------------------------------------------------------------- vitpose
def stage_vitpose(frames, ts, tag):
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    rid = "usyd-community/vitpose-plus-base"
    proc = AutoProcessor.from_pretrained(rid)
    model = VitPoseForPoseEstimation.from_pretrained(rid).to(DEV).eval()
    boxes = load_boxes(tag)
    xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]
    T = len(frames)
    kpts = np.zeros((T, 17, 2), np.float32)
    scores = np.zeros((T, 17), np.float32)
    bs = 16
    for i in tqdm(range(0, T, bs), desc="vitpose"):
        imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[i:i + bs]]
        bx = [[xywh[i + k].tolist()] for k in range(len(imgs))]
        inp = proc(imgs, boxes=bx, return_tensors="pt").to(DEV)
        # ViTPose+ is a mixture-of-experts over datasets; expert 0 = COCO.
        inp["dataset_index"] = torch.zeros(len(imgs), dtype=torch.long, device=DEV)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_pose_estimation(out, boxes=bx)
        for k, r in enumerate(res):
            kpts[i + k] = r[0]["keypoints"].cpu().numpy()
            scores[i + k] = r[0]["scores"].cpu().numpy()
    arm, arm_score = arm_from(kpts, scores, "vitpose")
    np.savez(out_dir(tag, "models") / "vitpose.npz", kpts=kpts, scores=scores,
             arm=arm, arm_score=arm_score, t=ts)


# ----------------------------------------------------------------------------- rtmpose
def stage_rtmpose(frames, ts, tag):
    import onnxruntime as ort

    onnx = MODELS / "rtmpose_body2d/rtmpose_body2d-onnx-float/rtmpose_body2d.onnx"
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(onnx), providers=prov)
    boxes = load_boxes(tag)
    T = len(frames)
    kpts = np.zeros((T, 133, 2), np.float32)
    scores = np.zeros((T, 133), np.float32)
    for i in tqdm(range(T), desc="rtmpose"):
        M = box_to_affine(boxes[i], 192, 256, pad=1.25)
        crop = cv2.warpAffine(frames[i], M, (192, 256), flags=cv2.INTER_LINEAR)
        # The exported graph subtracts mean=[123.7,116.3,103.5] directly, so it wants
        # 0-255 despite metadata.json saying [0, 1]. It also flips channels before the
        # RGB mean, so feeding BGR reproduces mmpose's training normalisation (and gives
        # clearly higher keypoint confidence than feeding RGB).
        x = crop.astype(np.float32)[None]
        px, py = sess.run(None, {"image": x.transpose(0, 3, 1, 2)})
        # SimCC decode (split ratio 2): argmax bin / 2 = pixel in the 192x256 crop.
        xi, yi = px[0].argmax(-1), py[0].argmax(-1)
        pts = np.stack([xi, yi], -1).astype(np.float32) / 2.0
        kpts[i] = invert_affine_points(M, pts)
        scores[i] = np.minimum(px[0].max(-1), py[0].max(-1))
    arm, arm_score = arm_from(kpts, scores, "rtmpose")
    np.savez(out_dir(tag, "models") / "rtmpose.npz", kpts=kpts, scores=scores,
             arm=arm, arm_score=arm_score, t=ts)


# ----------------------------------------------------------------------------- sapiens2
SAPIENS_CLASSES = ["Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot",
                   "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock",
                   "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing", "Right_Foot",
                   "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe",
                   "Right_Sock", "Right_Upper_Arm", "Right_Upper_Leg", "Torso",
                   "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth",
                   "Tongue"]


def stage_sapiens2(frames, ts, tag):
    import sys
    sys.path.insert(0, str(ROOT / "third_party/sapiens2"))
    import torch.nn.functional as F
    from sapiens.dense.models import init_model

    cfg = ROOT / ("third_party/sapiens2/sapiens/dense/configs/seg/shutterstock_goliath/"
                  "sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py")
    ckpt = MODELS / "sapiens2/seg/sapiens2_0.4b_seg.safetensors"
    model = init_model(str(cfg), str(ckpt), device=DEV)
    boxes = load_boxes(tag)
    H, W = frames[0].shape[:2]
    labels = np.zeros((len(frames), H, W), np.uint8)
    for i in tqdm(range(len(frames)), desc="sapiens2"):
        # Portrait 768x1024 crop around the person: the model's native aspect, so the
        # arm is not squashed the way a resize of the whole 16:9 frame would squash it.
        M = box_to_affine(boxes[i], 768, 1024, pad=1.15)
        crop = cv2.warpAffine(frames[i], M, (768, 1024), flags=cv2.INTER_LINEAR)
        data = model.data_preprocessor(model.pipeline(dict(img=crop)))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            logits = model(data["inputs"])
        logits = F.interpolate(logits.float(), size=(1024, 768), mode="bilinear")
        lab = logits.argmax(1)[0].byte().cpu().numpy()
        labels[i] = cv2.warpAffine(lab, cv2.invertAffineTransform(M), (W, H),
                                   flags=cv2.INTER_NEAREST, borderValue=0)
    np.savez_compressed(out_dir(tag, "models") / "sapiens2.npz", labels=labels, t=ts,
                        classes=np.array(SAPIENS_CLASSES))


# ----------------------------------------------------------------------------- mediapipe
def stage_mediapipe(frames, ts, tag):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "mediapipe/pose_landmarker_heavy.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.3, min_tracking_confidence=0.3))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "mediapipe/hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3))
    H, W = frames[0].shape[:2]
    T = len(frames)
    kpts = np.full((T, 33, 2), np.nan, np.float32)
    scores = np.zeros((T, 33), np.float32)
    hand = np.full((T, 21, 2), np.nan, np.float32)
    hand_score = np.zeros(T, np.float32)
    for i in tqdm(range(T), desc="mediapipe"):
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        ms = int(round(ts[i] * 1000)) + i  # strictly increasing
        r = pose.detect_for_video(img, ms)
        if r.pose_landmarks:
            lm = r.pose_landmarks[0]
            kpts[i] = [[p.x * W, p.y * H] for p in lm]
            scores[i] = [p.visibility for p in lm]
        rh = hands.detect_for_video(img, ms)
        if rh.hand_landmarks:
            # MediaPipe's handedness label assumes a mirrored selfie view, which this
            # over-the-shoulder camera is not; pick the hand nearest the pose's left wrist.
            cand = [np.array([[p.x * W, p.y * H] for p in h]) for h in rh.hand_landmarks]
            ref = kpts[i, 15] if not np.isnan(kpts[i, 15, 0]) else None
            if ref is None:
                j = int(np.argmin([c[0, 0] for c in cand]))  # image-left = participant's left
            else:
                j = int(np.argmin([np.linalg.norm(c[0] - ref) for c in cand]))
            # only if it really sits at the left wrist: the model often finds just the
            # resting right hand, which would otherwise be stored as the left one
            if ref is None or np.linalg.norm(cand[j][0] - ref) <= 150:
                hand[i] = cand[j]
                hand_score[i] = rh.handedness[j][0].score
    pose.close()
    hands.close()
    arm, arm_score = arm_from(kpts, scores, "mediapipe")
    np.savez(out_dir(tag, "models") / "mediapipe.npz", kpts=kpts, scores=scores, arm=arm,
             arm_score=arm_score, hand=hand, hand_score=hand_score, t=ts)


STAGES = dict(detect=stage_detect, vitpose=stage_vitpose, rtmpose=stage_rtmpose,
              sapiens2=stage_sapiens2, mediapipe=stage_mediapipe)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=list(STAGES))
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()
    for vp in list_videos():
        tag = video_tag(vp)
        frames, ts, fps = read_video(vp, args.max_frames)
        print(f"== {tag}: {len(frames)} frames, {fps:.2f} fps, {ts[-1]:.2f} s")
        np.save(out_dir(tag) / "timestamps.npy", ts)
        for s in args.stages:
            t0 = time.time()
            STAGES[s](frames, ts, tag)
            print(f"[{s}] {time.time() - t0:.1f}s")
