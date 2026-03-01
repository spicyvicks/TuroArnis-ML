"""
visualize_reference_poses.py
─────────────────────────────
Renders reference pose stick figures from feature_templates.json.
Each pose is reconstructed from the mean joint heights and angles stored
in the template (forward-kinematics style).

Usage:
    python visualize_reference_poses.py            # front view (default)
    python visualize_reference_poses.py left       # left view
    python visualize_reference_poses.py right      # right view
    python visualize_reference_poses.py all        # all 3 viewpoints

Output:
    reference_poses_<viewpoint>.png  saved in this directory
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Path to the template file (relative from ML repo to app repo) ────────────
TEMPLATE_PATH = (
    Path(__file__).parent.parent
    / "TuroArnis" / "app" / "models" / "gcn" / "feature_templates.json"
)

# ── Drawing constants ────────────────────────────────────────────────────────
TORSO_H  = 0.26   # hip → shoulder height (normalised, hip = origin)
HEAD_R   = 0.052  # head circle radius
BG_COLOR = "#0f0f1a"


# ── Helpers ──────────────────────────────────────────────────────────────────
def mean(template, key, fallback=0.0):
    """Safely extract the mean value from a feature entry."""
    return template.get(key, {}).get("mean", fallback)


def reconstruct(t):
    """
    Build 2D landmark positions from a feature-template dict.
    All coordinates are normalised with hip centre = (0, 0), Y-up.
    """
    # ── Fixed torso anchors ──────────────────────────────────────────────────
    hip   = np.array([0.0,  0.0])
    r_hip = np.array([ 0.07, 0.0])
    l_hip = np.array([-0.07, 0.0])
    shc   = np.array([0.0,  TORSO_H])
    r_sh  = np.array([ 0.10, TORSO_H])
    l_sh  = np.array([-0.10, TORSO_H])
    head  = np.array([0.0,  TORSO_H + 0.07])

    # ── Legs (knee angle → ankle position) ───────────────────────────────────
    thigh = 0.10

    def leg(hip_pt, side_sign, knee_angle_deg):
        ka = np.radians(knee_angle_deg)
        knee = hip_pt + np.array([0.0, -thigh])
        # deviation from straight caused by bend
        ankle = knee + thigh * np.array(
            [side_sign * np.sin(np.pi - ka), -np.cos(np.pi - ka)]
        )
        return knee, ankle

    r_knee, r_ankle = leg(r_hip,  1, mean(t, "right_knee_angle", 175))
    l_knee, l_ankle = leg(l_hip, -1, mean(t, "left_knee_angle",  175))

    stagger = mean(t, "foot_stagger", 0.0)
    r_ankle[0] += stagger * 0.5
    l_ankle[0] -= stagger * 0.5

    # ── Arms via wrist/elbow height means ────────────────────────────────────
    r_wrist = np.array([mean(t, "right_wrist_x", 0.0), mean(t, "right_wrist_height", 0.10)])
    l_wrist = np.array([mean(t, "left_wrist_x",  0.0), mean(t, "left_wrist_height",  0.10)])

    def elbow(shoulder, wrist, elbow_h):
        """Linearly interpolate elbow along shoulder→wrist at the given height."""
        dh = wrist[1] - shoulder[1]
        tv = np.clip((elbow_h - shoulder[1]) / dh, 0.1, 0.9) if abs(dh) > 1e-4 else 0.5
        return shoulder + tv * (wrist - shoulder)

    r_elbow = elbow(r_sh, r_wrist, mean(t, "right_elbow_height", 0.12))
    l_elbow = elbow(l_sh, l_wrist, mean(t, "left_elbow_height",  0.12))

    # ── Stick ────────────────────────────────────────────────────────────────
    grip = np.array([mean(t, "stick_grip_x",      0.0), mean(t, "stick_grip_height", 0.10)])
    tip  = np.array([mean(t, "stick_tip_x",       0.0), mean(t, "stick_tip_height",  0.15)])

    return dict(
        hip=hip, r_hip=r_hip, l_hip=l_hip,
        shc=shc, r_sh=r_sh, l_sh=l_sh, head=head,
        r_knee=r_knee, r_ankle=r_ankle,
        l_knee=l_knee, l_ankle=l_ankle,
        r_wrist=r_wrist, r_elbow=r_elbow,
        l_wrist=l_wrist, l_elbow=l_elbow,
        grip=grip, tip=tip,
    )


# ── Drawing ──────────────────────────────────────────────────────────────────
def draw(ax, pts, title=""):
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.set_xlim(-0.50, 0.50)
    ax.set_ylim(-0.42, 0.48)
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=7.5, pad=3, fontweight="bold")

    def seg(a, b, color="#dfe6e9", lw=2.5):
        ax.plot(
            [pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
            color=color, lw=lw, solid_capstyle="round", zorder=3,
        )

    def dot(name, r=0.011, color="#dfe6e9"):
        ax.add_patch(plt.Circle(pts[name], r, color=color, zorder=5))

    # Torso
    seg("l_sh",  "r_sh",  lw=2.8)
    seg("l_hip", "r_hip", lw=2.2)
    seg("l_sh",  "l_hip", lw=2.0)
    seg("r_sh",  "r_hip", lw=2.0)
    seg("hip",   "shc",   lw=2.8)  # spine

    # Head
    ax.add_patch(plt.Circle(pts["head"], HEAD_R, color="#dfe6e9", fill=False, lw=2.5, zorder=5))

    # Right arm — yellow (striking / weapon arm)
    seg("r_sh",    "r_elbow", color="#f9ca24", lw=3.2)
    seg("r_elbow", "r_wrist", color="#f9ca24", lw=3.2)
    dot("r_elbow", color="#f9ca24")
    dot("r_wrist", color="#f9ca24", r=0.013)

    # Left arm — cyan (guard arm)
    seg("l_sh",    "l_elbow", color="#74b9ff", lw=2.5)
    seg("l_elbow", "l_wrist", color="#74b9ff", lw=2.5)
    dot("l_elbow", color="#74b9ff")
    dot("l_wrist", color="#74b9ff")

    # Legs
    for a, b in [("r_hip", "r_knee"), ("r_knee", "r_ankle"),
                 ("l_hip", "l_knee"), ("l_knee", "l_ankle")]:
        seg(a, b, color="#b2bec3", lw=2.2)
    for j in ("r_knee", "l_knee", "r_ankle", "l_ankle"):
        dot(j, r=0.009, color="#b2bec3")

    # Stick — orange
    ax.plot(
        [pts["grip"][0], pts["tip"][0]],
        [pts["grip"][1], pts["tip"][1]],
        color="#fd9644", lw=4.5, solid_capstyle="round", zorder=6,
    )
    dot("grip", r=0.013, color="#e55039")  # grip end (red)
    dot("tip",  r=0.010, color="#ffdd57")  # tip end  (yellow)


# ── Main ─────────────────────────────────────────────────────────────────────
def render(viewpoint: str, templates: dict):
    vp = viewpoint.lower()
    items = {k: v for k, v in templates.items() if k.startswith(vp + "_")}

    if not items:
        print(f"[!] No templates found for viewpoint '{vp}'. Available prefixes:")
        prefixes = sorted({k.split("_")[0] for k in templates})
        print("   ", prefixes)
        return

    keys_sorted = sorted(items)
    n    = len(keys_sorted)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.8))
    fig.patch.set_facecolor(BG_COLOR)
    axes = np.array(axes).flatten()

    for idx, key in enumerate(keys_sorted):
        label = (
            key.replace(f"{vp}_", "")
               .replace("_correct", "")
               .replace("_", " ")
               .title()
        )
        pts = reconstruct(items[key])
        draw(axes[idx], pts, title=label)

    for ax in axes[n:]:
        ax.set_visible(False)

    # Legend
    legend = [
        mpatches.Patch(color="#f9ca24", label="Right (striking) arm"),
        mpatches.Patch(color="#74b9ff", label="Left (guard) arm"),
        mpatches.Patch(color="#fd9644", label="Stick"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Reference Poses — {vp.title()} View  ({n} techniques)",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out = Path(__file__).parent / f"reference_poses_{vp}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[✓] Saved → {out}")
    plt.show()


def main():
    if not TEMPLATE_PATH.exists():
        print(f"[!] Template file not found:\n    {TEMPLATE_PATH}")
        print("    Adjust TEMPLATE_PATH at the top of the script.")
        sys.exit(1)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        templates = json.load(f)

    viewpoints = sys.argv[1:] if len(sys.argv) > 1 else ["front"]
    if viewpoints == ["all"]:
        viewpoints = ["front", "left", "right"]

    for vp in viewpoints:
        render(vp, templates)


if __name__ == "__main__":
    main()
