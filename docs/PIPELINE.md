# VICONMapper — pipeline reference

Pulls left upper-limb white-marker trajectories (with timestamps) and reach kinematics
out of the Logitech overhead videos of MHEALTH125 playing the iPad reaching game.

## Run

```bash
sbatch slurm/pipeline.sbatch          # everything, ~6 min on one V100 (partition ai)
```

Or stage by stage (always `module load python/3.10.13; source .venv/bin/activate` first;
`run_models.py` and `dino_track.py` need a GPU):

| Stage | Script | What it does |
|---|---|---|
| 1 | `src/run_models.py` | RT-DETR person box → ViTPose+-B, RTMPose (Qualcomm ONNX), Sapiens2-0.4B seg, MediaPipe pose+hand. Caches `outputs/<video>/models/*.npz` |
| 2 | `src/markers.py` | Fuses the pose models into a shoulder–elbow–wrist–hand skeleton, detects white blobs in the left-limb ROI, labels them as Plug-in Gait markers and tracks them |
| 3 | `src/dino_track.py` | *Optional.* DINOv3 dense features re-detect markers the blob tracker lost (quality 3) |
| 4 | `src/kinematics.py` | 6 Hz Butterworth, derivatives, movement segmentation, per-movement features |
| 5 | `src/plots.py`, `src/evaluate.py`, `src/render.py` | QA figure, model-vs-marker scores, overlay video + contact sheet |
| 6 | `src/model_overlays.py` | One H.264 overlay per model (vitpose, rtmpose, mediapipe, sapiens2, dinov3, markers) + a 2×2 and a 3×2 all-models grid in `outputs/<video>/overlays/`. DINOv3 (`--models dinov3`): `sbatch slurm/dino_overlay.sbatch` on GPU, or `VIDEO=0/1 sbatch slurm/dino_overlay_cpu.sbatch` (~15 min per video on CPU). Then `--compose` builds `all_models_grid.mp4` |
| 7 | `src/report_docx.py` | Findings report → `documents/MHEALTH125_Marker_Tracking_Findings.docx` (tables built from the CSVs) |

DINOv3 is gated: accept the licence at
https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m, then log in once:
`LD_LIBRARY_PATH=/cm/shared/apps/python/3.10.13/lib .venv/bin/hf auth login`
(the library path is needed when `module load python/3.10.13` hasn't run in that shell).
`sbatch slurm/dino.sbatch` runs DINOv3 and everything downstream. Compute nodes have no
internet, so jobs run with `HF_HUB_OFFLINE=1` on pre-downloaded weights. DINOv3 runs in
float32 (float16 overflows to NaN on the V100). A marker's gaps are filled only if its
hold-out test passes: median re-detection error ≤ 10 px and ≥ 80 % within 10 px, on frames
where the blob tracker found it. In this data only V2's LHND passed.

## Outputs (`outputs/<video>/`)

| File | Contents |
|---|---|
| `markers_long.csv` | `frame, time_s, marker, x_px, y_px, quality, blob_score` — the marker time series |
| `markers_wide.csv` | same, one row per frame |
| `kinematics_timeseries.csv` | filtered markers, WRC and LHND velocity/speed/acc/jerk, segment lengths, elbow/wrist/shoulder angles, trunk-marker displacement, shoulder-line rotation, movement id/type |
| `movements.csv` | one row per movement: onset/offset, duration, path length, straightness, peak/mean speed, time-to-peak (%), peak acc/dec, deceleration time, # speed peaks, LDLJ, SPARC, elbow/wrist/shoulder ROM, C7/T10 shift, shoulder-line rotation, segment-length CV, `truncated` |
| `session_summary.csv` | counts, % time moving, marker coverage, reach-feature means/SDs |
| `model_comparison.csv` | each model's joints vs marker proxies (also pooled in `outputs/model_comparison.csv`) |
| `kinematics_qa.png`, `contact_sheet.jpg`, `overlay.mp4` | visual QA |

`quality`: 1 = blob detected, 2 = PCHIP-interpolated gap ≤ 6 frames (200 ms), 3 = DINOv3
re-detection, 0 = missing (NaN). Time comes from container timestamps (both clips are
constant 30 fps, 1920×1080).

**Markers** follow `documents/Marker Placement.docx` (Vicon Plug-in Gait Upper Body).
Left arm: LSHO acromion, LUPA upper arm, LELB lateral epicondyle, LFRM ulnar mid-forearm,
LWRA radial styloid (thumb side), LWRB ulnar styloid, LHND under the middle-finger knuckle,
LFIN index fingertip. Trunk (tracked in a shoulder frame): RSHO, C7, RBAK, T10. CLAV, STRN
and the front head markers face away from the camera. Markers found in < 30 % of frames are
kept in the CSVs but flagged `analysed = False` in `marker_template.csv` and skipped by the
kinematics. The thumb side was checked against RTMPose thumb/pinky keypoints.

**Segments and angles** use the protocol's segments: upper arm LSHO→LELB, forearm
LELB→LWRB, hand wrist centre (WRC = LWRA/LWRB midpoint)→LHND. Joint angles are elbow
(LSHO-LELB-WRC), wrist (LELB-WRC-LHND) and shoulder (RSHO-LSHO-LELB, or the pose right
shoulder when RSHO is not visible). Trunk measures are C7/T10/RBAK displacement and
shoulder-line rotation. The kinematic endpoint is WRC.

## Caveats

- **Units are image pixels and angles are 2-D projections.** One overhead camera gives no
  depth, so reach toward the iPad is foreshortened and the elbow angle is not the 3-D
  joint angle. Pass `--px_per_cm` to `kinematics.py` once a scale is known.
- Reach/return labels use the image vertical as the reach axis (iPad at the top).
- Neither clip starts or ends at rest, so movements touching the clip boundary are flagged
  `truncated` and left out of the session reach means.
- Sapiens2-0.4B labels nearly the whole left arm "Left_Hand" on this view, so its part
  labels are only used as a silhouette ROI, never to identify segments. It does segment
  each marker as a small "Apparel" blob, which the ROI uses.
- The Qualcomm MediaPipe-Pose repo ships only Snapdragon-specific QNN binaries, so the
  official Google MediaPipe BlazePose (heavy) + HandLandmarker models are used instead.
- The Qualcomm RTMPose ONNX wants **0–255 BGR** input, despite its metadata saying 0–1 RGB.
