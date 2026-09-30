"""Build documents/MHEALTH125_Marker_Tracking_Findings.docx from the pipeline outputs.

    python src/report_docx.py

Tables are generated from the CSVs in outputs/, so the report always matches the data.
Narrative text lives in NARRATIVE below.
"""
from datetime import date

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from common import OUT, ROOT

VIDEOS = [("MHEALTH125_V2_Logitech_AB", "V2"), ("MHEALTH125_AsteroidBelt_logitech", "AsteroidBelt")]
DINO_CSV = "dino_similarity_dinov3-vitl16-pretrain-lvd1689m.csv"
OUT_DOCX = ROOT / "documents" / "MHEALTH125_Marker_Tracking_Findings.docx"
HEADER_FILL = "D9E2F3"


# ----------------------------------------------------------------------------- helpers
def shade(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexfill)
    tcPr.append(shd)


def table(doc, header, rows, widths=None, font=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(font)
        shade(c, HEADER_FILL)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
    # repeat header row on each page
    trPr = t.rows[0]._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader")
    h.set(qn("w:val"), "true")
    trPr.append(h)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(" " + it[1])
        else:
            p.add_run(it)


def numbered(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Number")


def para(doc, text, italic=False, size=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    return p


def figure(doc, path, caption, width=6.5):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = para(doc, caption, italic=True, size=9)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def pct(x):
    return f"{100 * x:.0f} %"


# ----------------------------------------------------------------------------- data
def coverage_rows():
    per = {}
    for tag, lab in VIDEOS:
        L = pd.read_csv(OUT / tag / "markers_long.csv")
        tp = pd.read_csv(OUT / tag / "marker_template.csv").set_index("label")
        for m, g in L.groupby("marker", sort=False):
            q = g.quality
            note = "" if tp.loc[m, "analysed"] else " (not analysed)"
            per.setdefault(m, {})[lab] = (pct((q == 1).mean()) + note, pct((q == 3).mean()), pct((q >= 1).mean()))
    rows = []
    for m, d in per.items():
        v2, ab = d.get("V2", ("-",) * 3), d.get("AsteroidBelt", ("-",) * 3)
        rows.append([m, *v2, *ab])
    return rows


def model_rows():
    m = pd.read_csv(OUT / "model_comparison.csv")
    m = m[m.model != "sapiens2"]
    names = {"vitpose": "ViTPose+ base", "rtmpose": "RTMPose-Body2d",
             "mediapipe": "MediaPipe BlazePose", "fused": "Fused (median of three)"}
    rows = []
    for key, nm in names.items():
        g = m[m.model == key].set_index(["video", "joint"])

        def val(vid, joint, col, fmt):
            try:
                x = g.loc[(vid, joint), col]
                return fmt.format(x) if pd.notna(x) else "-"
            except KeyError:
                return "-"
        v2, ab = VIDEOS[0][0], VIDEOS[1][0]
        rows.append([nm,
                     f"{val(v2, 'wrist', 'median_err_px', '{:.0f}')} / {val(ab, 'wrist', 'median_err_px', '{:.0f}')}",
                     f"{val(v2, 'wrist', 'speed_r', '{:.2f}')} / {val(ab, 'wrist', 'speed_r', '{:.2f}')}",
                     f"{val(v2, 'elbow', 'speed_r', '{:.2f}')} / {val(ab, 'elbow', 'speed_r', '{:.2f}')}",
                     "no hand keypoints" if key == "vitpose" else
                     f"{val(v2, 'hand', 'median_err_px', '{:.0f}')} / {val(ab, 'hand', 'median_err_px', '{:.0f}')} px; "
                     f"r {val(v2, 'hand', 'speed_r', '{:.2f}')} / {val(ab, 'hand', 'speed_r', '{:.2f}')}"])
    return rows


def dino_rows():
    rows = []
    for tag, lab in VIDEOS:
        f = OUT / tag / DINO_CSV
        if not f.exists():
            continue
        for _, r in pd.read_csv(f).iterrows():
            rows.append([lab, r.marker, f"{r.sim_clean_median:.2f}", f"{r.sim_floor:.2f}",
                         f"{int(r.holdout_n - r.holdout_rejected)}/{int(r.holdout_n)}",
                         f"{r.holdout_median_err_px:.1f}" if pd.notna(r.holdout_median_err_px) else "-",
                         pct(r.holdout_within_10px) if pd.notna(r.holdout_within_10px) else "-",
                         "pass" if r.reliable else "fail",
                         f"{int(r.frames_kept)} of {int(r.missing_before)}"])
    return rows


def movement_rows():
    rows = []
    for tag, lab in VIDEOS:
        mv = pd.read_csv(OUT / tag / "movements.csv")
        for _, r in mv[mv.type.isin(["reach", "return"])].iterrows():
            if r.truncated and r.duration_s < 0.5:
                continue
            kind = r.type.capitalize() + (" (clip edge)" if r.truncated else "")
            trunk = f"{r.trunk_C7_disp_range_px:.0f} / " + (f"{r.trunk_T10_disp_range_px:.0f}"
                                                            if pd.notna(r.trunk_T10_disp_range_px) else "-")
            rows.append([lab, kind, f"{r.onset_s:.2f}", f"{r.duration_s:.2f}", f"{r.path_length_px:.0f}",
                         f"{r.straightness:.2f}", f"{r.peak_speed_pxps:.0f}", f"{r.time_to_peak_speed_pct:.0f}",
                         f"{int(r.n_speed_peaks)}", f"{r.sparc:.2f}",
                         f"{r.elbow_angle_start_deg:.0f} → {r.elbow_angle_end_deg:.0f}",
                         f"{r.shoulder_rom_deg:.0f}", trunk])
    return rows


# ----------------------------------------------------------------------------- document
def build(narr):
    doc = Document()
    # python-docx's template writes <w:zoom> without the schema-required percent attribute
    for z in doc.settings.element.findall(qn("w:zoom")):
        z.set(qn("w:percent"), "100")
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)

    doc.add_heading("MHEALTH125 Video Marker Tracking and Reach Kinematics", level=0)
    p = para(doc, f"Findings report · {date.today():%d %B %Y}", italic=True, size=10)
    p.runs[0].font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    doc.add_heading("Summary", level=1)
    para(doc, narr["summary_lead"])
    bullets(doc, narr["summary_bullets"])

    doc.add_heading("Data and marker set", level=1)
    para(doc, narr["data_lead"])
    table(doc, ["Video", "Frames", "Frame rate", "Duration", "Resolution"],
          [["MHEALTH125 V2 - Logitech AB", "468", "30 fps (constant)", "15.6 s", "1920 × 1080"],
           ["MHEALTH125 AsteroidBelt - logitech", "463", "30 fps (constant)", "15.4 s", "1920 × 1080"]])
    para(doc, narr["protocol"])
    table(doc, ["Marker", "Segment", "Placement (protocol)"], narr["marker_table"], widths=[1.4, 1.4, 3.7])
    para(doc, "Segments follow the protocol's measurement definitions, applied to the markers "
              "visible from above:")
    table(doc, ["Segment", "Protocol definition", "Marker-based definition used here"],
          narr["segment_table"], widths=[1.2, 2.5, 2.8])

    doc.add_heading("Method", level=1)
    para(doc, narr["method_lead"])
    numbered(doc, narr["method_steps"])

    doc.add_heading("Marker tracking results", level=1)
    para(doc, narr["tracking_lead"])
    table(doc, ["Marker", "V2 detected", "V2 DINOv3-filled", "V2 usable",
                "AB detected", "AB DINOv3-filled", "AB usable"], coverage_rows(), font=8)
    para(doc, narr["tracking_defs"], italic=True, size=9)
    bullets(doc, narr["tracking_notes"])
    figure(doc, OUT / VIDEOS[1][0] / "contact_sheet.jpg",
           "Figure 1. Sampled AsteroidBelt frames with protocol labels (left-arm chain LSHO → LFIN; "
           "trunk RSHO, C7, RBAK, T10).")

    doc.add_heading("DINOv3 gap filling", level=1)
    para(doc, narr["dino_lead"])
    table(doc, ["Video", "Marker", "Clean sim.", "Accept floor", "Hold-out accepted",
                "Hold-out median err (px)", "Hold-out ≤ 10 px", "Accuracy rule", "Gaps filled"], dino_rows(), font=8)
    para(doc, narr["dino_defs"], italic=True, size=9)
    bullets(doc, narr["dino_notes"])

    doc.add_heading("Model comparison", level=1)
    para(doc, narr["model_lead"])
    table(doc, ["Model", "Wrist error px (V2 / AB)", "Wrist speed r (V2 / AB)",
                "Elbow speed r (V2 / AB)", "Hand vs LHND (V2 / AB)"], model_rows(), font=8,
          widths=[1.5, 1.2, 1.2, 1.2, 1.4])
    para(doc, narr["model_defs"], italic=True, size=9)
    bullets(doc, narr["model_notes"])
    still = OUT / VIDEOS[1][0] / "overlays" / "all_models_frame.jpg"
    if still.exists():
        figure(doc, still, "Figure 2. The same AsteroidBelt frame (t = 14.0 s) in all six overlays. Top: "
                           "ViTPose+, RTMPose, MediaPipe. Bottom: Sapiens2, DINOv3 feature PCA, tracked markers.")

    doc.add_heading("Reach kinematics", level=1)
    para(doc, narr["kin_lead"])
    sec_l = doc.add_section()
    sec_l.orientation = 1  # landscape for the wide table
    sec_l.page_width, sec_l.page_height = Inches(11), Inches(8.5)
    table(doc, ["Video", "Movement", "Onset (s)", "Dur. (s)", "Path (px)", "Straight.", "Peak speed (px/s)",
                "Time to peak (% MT)", "Speed peaks", "SPARC", "Elbow start → end (°)",
                "Shoulder ROM (°)", "C7 / T10 shift (px)"], movement_rows(), font=8)
    para(doc, narr["kin_defs"], italic=True, size=9)
    bullets(doc, narr["kin_notes"])
    sec_p = doc.add_section()
    sec_p.orientation = 0
    sec_p.page_width, sec_p.page_height = Inches(8.5), Inches(11)
    figure(doc, OUT / VIDEOS[1][0] / "kinematics_qa.png",
           "Figure 3. AsteroidBelt: wrist-centre speed, reach-axis position, elbow angle and C7 "
           "displacement; movements shaded by type.")
    figure(doc, OUT / VIDEOS[0][0] / "kinematics_qa.png",
           "Figure 4. V2: same panels.")

    doc.add_heading("Limitations and next steps", level=1)
    para(doc, narr["lim_lead"])
    bullets(doc, narr["lim_notes"])
    para(doc, "Next steps:")
    bullets(doc, narr["next_steps"])
    para(doc, narr["repro"])

    OUT_DOCX.parent.mkdir(exist_ok=True)
    doc.save(OUT_DOCX)
    return OUT_DOCX


# ----------------------------------------------------------------------------- narrative
NARRATIVE = dict(
    summary_lead=(
        "All eight left-arm markers from the lab's Marker Placement protocol (Vicon Plug-in Gait "
        "Upper Body) can be tracked from the two overhead videos. Seven are directly detected in "
        "85-100 % of frames per video; the exception is V2's forearm marker at 74 %. Every position "
        "is time-stamped at 30 fps."),
    summary_bullets=[
        ("Trunk markers.", "RSHO, C7, RBAK and T10 are tracked in AsteroidBelt. V2's framing leaves only C7."),
        ("Pose models.", "ViTPose+ base follows the wrist within 7-11 px and reproduces its speed profile "
                         "(r = 0.89-0.97). RTMPose is comparable and also gives hand points. MediaPipe is too "
                         "noisy for kinematics on this view. Sapiens2 is useful only as an arm outline."),
        ("DINOv3.", "Its re-detections were checked on frames where the true marker position is known. "
                    "They are accurate for some markers only: 0.6 px median error for V2's hand marker, "
                    "but 10-16 px with frequent large misses for the forearm and fingertip. Only the 6 "
                    "LHND fills that passed this check were kept."),
        ("Reaches.", "V2's reaches are fast, near-straight and single-peaked. AsteroidBelt's reach and "
                     "return each show three speed peaks, straightness 0.67-0.82, and about 40 deg of "
                     "shoulder excursion. Its reach moves the upper trunk (C7 60 px) far more than the "
                     "lower trunk (T10 7 px)."),
        ("Main caveat.", "All measures are 2-D, in image pixels, from one camera. They still need checking "
                         "against the Vicon 3-D data."),
    ],
    data_lead=(
        "Two overhead Logitech videos show participant MHEALTH125 reaching to an iPad with the left "
        "hand, starting from a star-shaped home mark on the table. The camera looks down from behind "
        "the participant, so it sees the back and the dorsal side of the arms."),
    protocol=(
        "Markers follow the lab's Marker Placement protocol (documents/Marker Placement.docx), which is "
        "the Vicon Plug-in Gait Upper Body model. Markers were applied with the participant seated and "
        "the elbows at 90 deg. On the left side, the upper-arm and forearm markers sit 1-2 in more proximal "
        "than on the right, and the arm markers form a triangle rather than a line."),
    marker_table=[
        ["LSHO", "Shoulder", "Tip (\"cliff\") of the acromion"],
        ["LUPA", "Upper arm", "1-2 in below the deltoid tip, dorsal"],
        ["LELB", "Upper arm / elbow", "Anterior surface of the lateral epicondyle"],
        ["LFRM", "Forearm", "Mid forearm, ulnar (pinky) side, toward the top"],
        ["LWRA", "Forearm / wrist", "Radial styloid (thumb side)"],
        ["LWRB", "Forearm / wrist", "Ulnar styloid (pinky side)"],
        ["LHND", "Hand", "Directly under the middle-finger knuckle"],
        ["LFIN (*S*)", "Hand", "Tip of the index finger"],
        ["C7", "Thorax", "Most prominent cervical vertebra"],
        ["T10", "Thorax", "Mid/low back at the level of T9 on the front"],
        ["RBAK", "Thorax", "Right scapula, directly below the scapular spine"],
        ["RSHO", "Shoulder", "Tip of the right acromion"],
        ["CLAV, STRN", "Thorax", "Jugular notch, sternum (front; not visible)"],
        ["LFHD, RFHD, LBHD, RBHD (*S*)", "Head", "On the headband (not tracked)"],
    ],
    segment_table=[
        ["Upper arm", "Posterior acromion to olecranon", "LSHO to LELB"],
        ["Forearm", "Olecranon to ulnar styloid", "LELB to LWRB"],
        ["Hand", "Mid-wrist to middle-finger tip", "Wrist centre (LWRA+LWRB)/2 to LHND"],
        ["Shoulder width", "Right to left posterior acromion", "RSHO to LSHO"],
        ["Trunk", "C7, T10, CLAV, STRN, RBAK", "C7 to T10 (CLAV and STRN face away from the camera)"],
    ],
    method_lead=(
        "The white markers are found with classical image processing. The pretrained models serve to "
        "locate the limb, so the markers can be named and followed through fast, blurred reaches. Each "
        "frame's labels come from the protocol layout, adapted to this participant; the previous frame "
        "only breaks ties, so a label swap cannot persist."),
    method_steps=[
        "Pose skeleton. An RT-DETR person box feeds ViTPose+ base, RTMPose-Body2d and MediaPipe BlazePose. "
        "Their left shoulder, elbow, wrist and hand points are fused by a confidence-gated median; any model "
        "more than 90 px from the others is dropped.",
        "Marker candidates. A white top-hat filter on min(B, G, R) finds bright, grey-white blobs of marker "
        "size, ignoring skin, the blue star and stylus, and the red holder. Blobs merged with the iPad stand "
        "are split at their intensity peaks; motion-blurred streaks are kept.",
        "Search regions. The left-limb region joins the Sapiens2 left-arm silhouette, a 45 px band around "
        "the fused skeleton, and blobs Sapiens2 labels as markers. The trunk region is a box in a shoulder "
        "frame, with origin between the shoulders and axes along and across the shoulder line.",
        "Labelling. Each protocol marker starts at its expected place along the shoulder-elbow-wrist-hand "
        "chain (or in the shoulder frame for trunk markers). Four assignment-and-update passes (EM) move "
        "each marker to where it sits on this participant; a Hungarian assignment then labels every frame.",
        "DINOv3 gap filling. Where a marker is still missing, DINOv3 dense features search near its "
        "predicted position. Fills are kept only for markers that pass a hold-out accuracy test (below).",
        "Cleaning. Single-frame jumps over 30 px are removed. Gaps of up to 6 frames (200 ms) are filled "
        "with shape-preserving (PCHIP) interpolation and flagged. Positions are low-pass filtered with a "
        "4th-order zero-lag Butterworth filter at 6 Hz.",
        "Kinematics. The endpoint is the wrist centre. A movement starts and ends where speed crosses 10 % "
        "of its peak; movements that reverse direction along the reach axis are split.",
    ],
    tracking_lead=(
        "All eight left-arm markers are tracked in both videos, and seven are directly detected in at least "
        "85 % of frames. Trunk tracking depends on framing: AsteroidBelt sees RSHO, C7, RBAK and T10, while "
        "V2's frame cuts off at the shoulders and shows only C7."),
    tracking_defs=(
        "Detected = a white blob matched in that frame. DINOv3-filled = a gap filled by a validated DINOv3 "
        "re-detection. Usable = detected, DINOv3-filled, or a gap of up to 200 ms bridged by interpolation. "
        "Markers found in fewer than 30 % of frames are reported but not analysed. AB = AsteroidBelt."),
    tracking_notes=[
        ("LFRM in V2.", "The protocol places it on the ulnar side of the forearm. In V2 the forearm is "
                        "pronated, so the marker faces away from the camera for about a quarter of the clip."),
        ("C7 in AsteroidBelt.", "C7 sits at the neck base under the participant's braids and is hidden "
                                "in 45 % of frames."),
        ("LFIN.", "The index-fingertip marker sits beside the blue stylus, so its label is the least "
                  "certain of the arm set."),
        ("Checks.", "Labels were checked visually on sampled frames, including motion-blurred ones. Two "
                    "errors found this way were fixed: the ulnar wrist marker merging with the iPad stand, "
                    "and the hand label jumping to a knuckle highlight."),
    ],
    dino_lead=(
        "DINOv3 ViT-L/16 was used to re-find markers the blob detector lost. To test it, each marker's "
        "reference descriptor was built from frames where the blob detector found it. DINOv3 then re-detected "
        "the marker on up to 40 other such frames, where the true position is known. A marker's gaps were "
        "filled only if the median hold-out error was at most 10 px and at least 80 % of re-detections "
        "landed within 10 px. Only V2's LHND passed among the markers that had gaps: 6 of its 7 missing "
        "frames were filled."),
    dino_defs=(
        "Clean sim. = median cosine similarity of clean detections to the marker's reference descriptor. "
        "Accept floor = minimum similarity for a re-detection to count. Hold-out accepted = re-detections "
        "above the floor out of those tried. Accuracy rule = median hold-out error <= 10 px and >= 80 % within 10 px. "
        "Gaps filled = DINOv3 fills kept, out of frames missing; markers that fail the rule keep their gaps."),
    dino_notes=[
        ("Where it works.", "Markers with a distinctive neighbourhood: V2's LHND (0.6 px median error), and "
                            "LSHO, LUPA and LELB in V2 (7-9 px)."),
        ("Where it fails.", "V2 LFRM: 9.9 px median, but only 50 % within 10 px and a 90th percentile of "
                            "62 px. LFIN: 11-16 px median. V2 LWRA: 94 px, because its neighbourhood looks "
                            "like LWRB's. The two wrist markers sit only 63-68 px apart on similar skin."),
        ("Why.", "At 16 px per patch (8 px after 2x upsampling), a marker spans only 2-3 patches, and DINOv3 "
                 "describes the surrounding limb more than the white sphere itself. Its features also carry a "
                 "strong position component on this scene: the same wooden floor at the left and right of the "
                 "frame has cosine similarity 0.34, barely above left floor vs left table (0.22). Two nearby "
                 "markers on similar skin therefore look alike. DINOv3 suits coarse re-localisation better "
                 "than sub-marker precision."),
        ("Setup notes.", "The model must run in float32 on the V100 (float16 overflowed to NaN features), and "
                         "offline on the compute nodes, which have no internet."),
    ],
    model_lead=(
        "ViTPose+ base tracks the left arm best, and RTMPose is the only one that also gives usable hand "
        "points. Fusing the pose models matches the best single model at the wrist. MediaPipe is too noisy "
        "for kinematics on this view. Each model was scored against the marker tracks, on frames where the "
        "reference marker was directly detected."),
    model_defs=(
        "Wrist error = median distance to the midpoint of LWRA and LWRB. Speed r = correlation between the "
        "model's speed profile and the marker's, after the same 6 Hz filter; it shows whether markerless "
        "tracking alone would reproduce the kinematics. Elbow and hand errors include the fixed offset "
        "between a skin marker and the joint centre, so speed r is the fairer comparison there."),
    model_notes=[
        ("Shoulder.", "Every model is poor at the shoulder: median error to LSHO is 36-102 px and speed "
                      "r at most 0.57. From above and behind, the joint is hidden under the head and braids."),
        ("MediaPipe.", "Wrist jitter (RMS scatter after removing the constant offset) is 119 px in V2 and "
                       "31 px in AsteroidBelt, against 9 and 18 px for ViTPose+. Its hand model often finds "
                       "only the resting right hand: in 14 % of V2 frames the 'left hand' was more than "
                       "150 px from the left wrist. Such hands are now rejected. Google's official BlazePose "
                       "models were used, because the Qualcomm repository ships only Snapdragon binaries."),
        ("RTMPose.", "The Qualcomm ONNX export needs 0-255 BGR input, despite metadata stating 0-1 RGB."),
        ("Sapiens2-0.4B.", "On this overhead back view it labels nearly the whole left arm 'Left_Hand', so "
                           "its part labels cannot separate segments. As a silhouette it contains 90-100 % of "
                           "wrist and hand markers but only 0-72 % of LELB. It segments each marker as a small "
                           "'Apparel' region, which helps find markers at the arm's edge."),
        ("Overlay videos.", "outputs/<video>/overlays/ holds one H.264 overlay per model (vitpose, rtmpose, "
                            "mediapipe, sapiens2, dinov3, markers), a 2x2 grid of the four pose/segmentation "
                            "models, and a 3x2 grid of all six (all_models_grid.mp4, Figure 2). The DINOv3 "
                            "overlay shows a PCA of its features fitted on the person only; a whole-frame PCA "
                            "was dominated by image position."),
    ],
    kin_lead=(
        "V2's reaches and returns are fast, straight and single-peaked. AsteroidBelt's are slower to organise, "
        "with three speed peaks each, straightness 0.67-0.82, and about twice the shoulder excursion. Most of "
        "each clip is spent working on the screen: the hand is moving 82 % of the time in V2 and 70 % in "
        "AsteroidBelt, including 5 and 9 short on-screen movements."),
    kin_defs=(
        "Endpoint = wrist centre (midpoint of LWRA and LWRB). Straightness = straight-line distance / path "
        "length. SPARC = spectral-arc-length smoothness; more negative is less smooth. Shoulder ROM = change "
        "in the angle at LSHO between the shoulder line and the upper arm; the shoulder line runs to RSHO in "
        "AsteroidBelt and to the pose model's right shoulder in V2. A 0.33 s reach at the very start of V2 is "
        "cut off by the clip and omitted."),
    kin_notes=[
        ("Elbow.", "Each reach extends the elbow by 35-49 deg, from about 79-97 deg at home to 128-138 deg at "
                   "the iPad. While working on the screen in AsteroidBelt the elbow reaches up to 173 deg, "
                   "nearly straight."),
        ("Trunk.", "In AsteroidBelt's reach, C7 at the neck base shifts 60 px while T10 on the lower back "
                   "shifts 7 px, and the shoulder line rotates 10.6 deg. The movement is concentrated in the "
                   "upper trunk and shoulder girdle, not a whole-trunk lean. LSHO itself moves 41 px."),
        ("Timing.", "Peak speed comes early in V2's reaches (20-59 % of movement time) and late in "
                    "AsteroidBelt's return (82 %), meaning a long acceleration phase and a short deceleration."),
        ("Segment lengths.", "In 2-D, upper-arm length (LSHO-LELB) varies by 13-17 % over each clip and "
                             "forearm length by 7-14 %. The limb moves out of the image plane, so projected "
                             "angles and path lengths understate the true 3-D values."),
    ],
    lim_lead=(
        "These results come from one overhead 2-D camera, so they describe image-plane motion in pixels. "
        "Comparing them with the Vicon 3-D data from the same session is the next step."),
    lim_notes=[
        ("Units and projection.", "Lengths and speeds are in pixels and angles are projected; motion toward "
                                  "the iPad is foreshortened. A scale reference would give cm; the pipeline "
                                  "accepts --px_per_cm."),
        ("Short, open-ended clips.", "Each clip is about 15 s with one or two full reach-return cycles, and "
                                     "neither starts or ends at rest."),
        ("Reach axis.", "Reach and return are defined along the image vertical, with the iPad at the top; this "
                        "suits this camera placement only."),
        ("Trunk.", "Only C7 is visible in V2, and C7 is hidden under the braids in 45 % of AsteroidBelt "
                   "frames. CLAV and STRN are never visible, so Plug-in Gait trunk angles cannot be computed."),
        ("Marker identity.", "Labels come from the protocol layout plus visual checks on sampled frames, not "
                             "manual labelling of every frame."),
    ],
    next_steps=[
        "Align these trajectories with the Vicon C3D for the same trials and report agreement.",
        "Add a scale reference (for example, the iPad's known screen width) to report cm and cm/s.",
        "Record longer trials that start and end at the home star.",
        "If gap filling matters for LFRM, test a keypoint tracker (e.g. CoTracker) against the same "
        "hold-out rule used for DINOv3.",
    ],
    repro=(
        "Code, outputs and a README are in ~/VICONMapper on the cluster. sbatch slurm/pipeline.sbatch reruns "
        "everything on one V100. Per-frame marker positions are in markers_long.csv, per-movement features in "
        "movements.csv, model scores in model_comparison.csv, and DINOv3 hold-out results in "
        "dino_similarity_dinov3-vitl16-pretrain-lvd1689m.csv."),
)

if __name__ == "__main__":
    print(build(NARRATIVE))
