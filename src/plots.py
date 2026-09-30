"""QA figure per video: wrist-centre speed, reach-axis position and elbow angle over time,
with segmented movements shaded by type.   python src/plots.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import list_videos, out_dir, video_tag

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LINE = "#2a78d6"  # categorical slot 1
# Fixed categorical order (reference palette slots 1-5); colour follows movement type.
TYPE_COL = {"reach": "#2a78d6", "return": "#eb6834", "on_target": "#1baf7a",
            "at_home": "#eda100", "other": "#e87ba4"}


def figure(tag):
    d = out_dir(tag)
    ts = pd.read_csv(d / "kinematics_timeseries.csv")
    mv = pd.read_csv(d / "movements.csv")
    t = ts.time_s
    panels = [("WRC_speed", "Wrist-centre speed (px/s)"),
              ("WRC_y", "Wrist-centre y (px)\n↑ towards iPad"),
              ("elbow_angle_deg", "Elbow angle (°, 2-D)\n180 = straight"),
              ("C7_disp_px", "Trunk: C7 displacement\nfrom median (px)")]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True, facecolor=SURF)
    for ax, (col, lab) in zip(axes, panels):
        ax.set_facecolor(SURF)
        for _, m in mv.iterrows():
            ax.axvspan(m.onset_s, m.offset_s, color=TYPE_COL.get(m.type, "#888"), alpha=0.18, lw=0)
        ax.plot(t, ts[col], color=LINE, lw=2)
        ax.set_ylabel(lab, color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[1].invert_yaxis()  # image y grows downward; flip so "up" = towards the iPad
    axes[-1].set_xlabel("Time (s)", color=INK2)
    present = [k for k in TYPE_COL if k in set(mv.type)]
    axes[0].legend(handles=[Patch(color=TYPE_COL[k], alpha=0.35, label=k.replace("_", " ")) for k in present],
                   loc="upper right", ncol=len(present), frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"{tag}: left upper-limb kinematics from video markers", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(d / "kinematics_qa.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    for vp in list_videos():
        figure(video_tag(vp))
