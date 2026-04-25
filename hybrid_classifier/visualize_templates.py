"""
Visualize Template Poses (Mean per Class)
=========================================
Reconstructs "average" stick figures by computing mean node features
for each class. Also draws standard deviation ellipses around joints
to show pose variability.

Usage:
    python hybrid_classifier/visualize_templates.py \
        --input hybrid_classifier/hybrid_features_v3/train_features_front.pt \
        --output hybrid_classifier/visualizations/template_poses_front.png

Output: Grid of 13 mean poses with variability ellipses
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch


def load_aspect_config(config_path, group1_default, group2_default):
    """
    Load per-class aspect ratio overrides from JSON.
    Returns a dict: class_idx -> aspect_ratio.
    Unlisted classes fall back to group defaults (0-5 -> group1, 6-12 -> group2).
    """
    overrides = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
        raw = data.get('aspect_ratios', {})
        for k, v in raw.items():
            overrides[int(k)] = float(v)
        print(f"[INFO] Loaded {len(overrides)} per-class overrides from {config_path}")
    else:
        if config_path:
            print(f"[WARN] Config not found: {config_path}, using group defaults only")

    result = {}
    for class_idx in range(13):
        if class_idx in overrides:
            result[class_idx] = overrides[class_idx]
        elif class_idx <= 5:
            result[class_idx] = group1_default
        else:
            result[class_idx] = group2_default
    return result


# Class names
CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

JOINT_COLORS = {
    'body': '#dfe6e9',
    'left_arm': '#74b9ff',
    'right_arm': '#f9ca24',
    'stick': '#fd9644',
    'grip': '#e55039',
    'tip': '#ffdd57',
}


def draw_template(ax, mean_nodes, std_nodes, title="", aspect_ratio=1.0):
    """Draw mean pose with std ellipses for joint variability."""
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=10, pad=4, fontweight="bold")

    pts = mean_nodes[:, :2].copy()  # [35, 2]
    # Option C: Auto-estimated aspect ratio correction
    pts[:, 0] = pts[:, 0] * aspect_ratio
    # Flip Y: MediaPipe y=0 is top, matplotlib y=0 is bottom
    pts[:, 1] = 1.0 - pts[:, 1]
    # Center horizontally on hip center
    hip_center_x = (pts[23, 0] + pts[24, 0]) / 2
    pts[:, 0] = pts[:, 0] - hip_center_x
    stds = std_nodes[:, :2].copy()
    stds[:, 0] = stds[:, 0] * aspect_ratio  # scale x std by aspect ratio
    stds[:, 1] = stds[:, 1]  # std is magnitude, no flip needed
    visibility = mean_nodes[:, 3]

    has_stick = (abs(mean_nodes[33, 0]) > 1e-6 or abs(mean_nodes[33, 1]) > 1e-6) and mean_nodes[33, 3] > 0.01

    # Helper: draw segment
    def seg(i, j, color="#dfe6e9", lw=2.5, ls="-"):
        if visibility[i] < 0.01 or visibility[j] < 0.01:
            return
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (abs(xi) < 1e-6 and abs(yi) < 1e-6) or (abs(xj) < 1e-6 and abs(yj) < 1e-6):
            return
        ax.plot([xi, xj], [yi, yj], color=color, lw=lw, solid_capstyle="round", linestyle=ls, zorder=3)

    # Helper: draw ellipse for std
    def ellipse(i, color="#dfe6e9", alpha=0.25):
        if visibility[i] < 0.01:
            return
        xi, yi = pts[i]
        sx, sy = stds[i]
        if abs(xi) < 1e-6 and abs(yi) < 1e-6:
            return
        # Use 2*std for ~95% confidence
        width = max(sx * 4, 0.005)
        height = max(sy * 4, 0.005)
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (xi, yi), width, height,
            angle=0, color=color, alpha=alpha, zorder=2
        ))

    # Helper: dot
    def dot(i, r=0.012, color="#dfe6e9", z=5):
        if visibility[i] < 0.01:
            return
        xi, yi = pts[i]
        if abs(xi) < 1e-6 and abs(yi) < 1e-6:
            return
        ax.add_patch(plt.Circle((xi, yi), r, color=color, zorder=z))

    # Draw std ellipses for key joints (underneath)
    for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 33, 34]:
        color = JOINT_COLORS['body']
        if i in [13, 15]:
            color = JOINT_COLORS['left_arm']
        elif i in [14, 16]:
            color = JOINT_COLORS['right_arm']
        elif i in [33]:
            color = JOINT_COLORS['grip']
        elif i in [34]:
            color = JOINT_COLORS['tip']
        ellipse(i, color=color)

    # Torso
    seg(11, 12, lw=3.0)
    seg(11, 23, lw=2.5)
    seg(12, 24, lw=2.5)
    seg(23, 24, lw=2.5)

    # Arms
    seg(11, 13, color=JOINT_COLORS['left_arm'], lw=3.0)
    seg(13, 15, color=JOINT_COLORS['left_arm'], lw=3.0)
    seg(12, 14, color=JOINT_COLORS['right_arm'], lw=3.0)
    seg(14, 16, color=JOINT_COLORS['right_arm'], lw=3.0)

    # Legs
    seg(23, 25, color="#b2bec3", lw=2.2)
    seg(25, 27, color="#b2bec3", lw=2.2)
    seg(24, 26, color="#b2bec3", lw=2.2)
    seg(26, 28, color="#b2bec3", lw=2.2)

    # Head: data-driven radius, always clearly above shoulders
    if visibility[0] > 0.5 and visibility[11] > 0.5 and visibility[12] > 0.5:
        x0, y0 = pts[0]
        shoulder_y = (pts[11, 1] + pts[12, 1]) / 2
        shoulder_width = abs(pts[11, 0] - pts[12, 0])
        radius = max(shoulder_width / 4.0, 0.02)
        radius = min(radius, 0.12)
        center_y = y0 + radius * 0.3
        min_center_y = shoulder_y + radius + 0.03
        center_y = max(center_y, min_center_y)
        ax.add_patch(plt.Circle((x0, center_y), radius, color=JOINT_COLORS['body'], fill=False, lw=2.5, zorder=5))

    # Stick
    if has_stick:
        grip_x, grip_y = pts[33]
        tip_x, tip_y = pts[34]
        ax.plot([grip_x, tip_x], [grip_y, tip_y],
                color=JOINT_COLORS['stick'], lw=4.5, solid_capstyle="round", zorder=6)
        ax.add_patch(plt.Circle((grip_x, grip_y), 0.014, color=JOINT_COLORS['grip'], zorder=7))
        ax.add_patch(plt.Circle((tip_x, tip_y), 0.011, color=JOINT_COLORS['tip'], zorder=7))
        seg(15, 33, color="#dfe6e9", lw=1.5, ls="--")
        seg(16, 33, color="#dfe6e9", lw=1.5, ls="--")
    else:
        ax.text(0.85, 0.95, "No stick", color="red", fontsize=7, ha="right")

    # Dots
    for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        if i in [13, 15]:
            dot(i, color=JOINT_COLORS['left_arm'])
        elif i in [14, 16]:
            dot(i, color=JOINT_COLORS['right_arm'])
        else:
            dot(i, color="#b2bec3")


def main():
    parser = argparse.ArgumentParser(description="Visualize template (mean) poses per class")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to .pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    parser.add_argument("--aspect_ratio_group1", type=float, default=0.4,
                        help="X-axis scale for classes 0-5 (default: 0.4)")
    parser.add_argument("--aspect_ratio_group2", type=float, default=1.3,
                        help="X-axis scale for classes 6-12 (default: 1.3, wider)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config with per-class aspect_ratio overrides")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    data = torch.load(input_path, map_location='cpu')
    node_features = data['node_features'].numpy()
    labels = data['labels'].numpy()

    print(f"Loaded {len(labels)} samples")

    ar_map = load_aspect_config(args.config, args.aspect_ratio_group1, args.aspect_ratio_group2)
    print(f"[INFO] Group defaults: 0-5 -> {args.aspect_ratio_group1:.4f}, 6-12 -> {args.aspect_ratio_group2:.4f}")
    for idx in sorted(ar_map.keys()):
        print(f"  Class {idx} ({CLASS_NAMES[idx]}): AR = {ar_map[idx]:.4f}")

    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = Path("hybrid_classifier/visualizations") / f"{stem}_templates.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(CLASS_NAMES)
    cols = 4
    rows = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    axes = np.array(axes).flatten()

    for class_idx in range(n_classes):
        mask = labels == class_idx
        class_nodes = node_features[mask]  # [count, 35, 6]

        if len(class_nodes) == 0:
            axes[class_idx].set_visible(False)
            continue

        mean_nodes = class_nodes.mean(axis=0)  # [35, 6]
        std_nodes = class_nodes.std(axis=0)    # [35, 6]

        label = CLASS_NAMES[class_idx].replace("_correct", "").replace("_", " ").title()
        title = f"{label}\n(n={len(class_nodes)})"
        draw_template(axes[class_idx], mean_nodes, std_nodes, title=title, aspect_ratio=ar_map[class_idx])

    # Hide unused axes
    for idx in range(n_classes, len(axes)):
        axes[idx].set_visible(False)

    legend = [
        mpatches.Patch(color=JOINT_COLORS['right_arm'], label="Right (striking) arm"),
        mpatches.Patch(color=JOINT_COLORS['left_arm'], label="Left (guard) arm"),
        mpatches.Patch(color=JOINT_COLORS['stick'], label="Stick"),
        mpatches.Patch(color="#dfe6e9", label="Body / Legs"),
        mpatches.Patch(color="#dfe6e9", alpha=0.25, label="Std ellipse (95%)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.01))

    stem = input_path.stem
    fig.suptitle(f"Template Poses (Mean + Std) — {stem}",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] Saved -> {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
