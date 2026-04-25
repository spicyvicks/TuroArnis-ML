"""
Visualize Features from .pt Files
=================================
Reconstructs stick figures from pre-computed node_features tensors.
Useful for understanding what the model "sees" in the training data.

Usage:
    python hybrid_classifier/visualize_features.py \
        --input hybrid_classifier/hybrid_features_v3/train_features_front.pt \
        --output hybrid_classifier/visualizations/features_front.png \
        --samples_per_class 2

Output: Grid of stick figures (samples_per_class columns x 13 rows)
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

    # Build full map with fallbacks
    result = {}
    for class_idx in range(13):
        if class_idx in overrides:
            result[class_idx] = overrides[class_idx]
        elif class_idx <= 5:
            result[class_idx] = group1_default
        else:
            result[class_idx] = group2_default
    return result
# Class names matching feature extraction
CLASS_NAMES = [
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'
]

# Skeleton edges for drawing (undirected, unique pairs)
DRAW_EDGES = [
    # Torso
    (11, 12),   # shoulders
    (11, 23), (12, 24),  # shoulder to hip
    (23, 24),   # hips
    # Left arm
    (11, 13), (13, 15),  # L shoulder -> elbow -> wrist
    # Right arm
    (12, 14), (14, 16),  # R shoulder -> elbow -> wrist
    # Left leg
    (23, 25), (25, 27),  # L hip -> knee -> ankle
    # Right leg
    (24, 26), (26, 28),  # R hip -> knee -> ankle
    # Stick
    (15, 33), (16, 33),  # wrists to grip
    (33, 34),   # grip to tip
    # Head / spine
    (0, 11), (0, 12),   # nose to shoulders (simplified)
]

# Joint colors
JOINT_COLORS = {
    'body': '#dfe6e9',
    'left_arm': '#74b9ff',
    'right_arm': '#f9ca24',
    'stick': '#fd9644',
    'grip': '#e55039',
    'tip': '#ffdd57',
}


def is_stick_valid(node_features):
    """Check if stick nodes (33, 34) have valid coordinates."""
    grip = node_features[33]
    tip = node_features[34]
    # Valid if not all zeros and visibility > 0
    return (grip[0] != 0 or grip[1] != 0) and grip[3] > 0.01


def draw_pose(ax, node_features, title="", highlight_arms=True, aspect_ratio=1.0):
    """Draw a single pose from node_features [35, 6]."""
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=8, pad=3, fontweight="bold")

    # Extract x, y coordinates (nodes 0-32 are body, 33=grip, 34=tip)
    pts = node_features[:, :2].copy()  # [35, 2]
    # Option C: Auto-estimated aspect ratio correction
    # Images are portrait (w/h << 1); without correction, horizontal distances
    # appear exaggerated when plotted with equal aspect ratio.
    pts[:, 0] = pts[:, 0] * aspect_ratio
    # Flip Y: MediaPipe y=0 is top, matplotlib y=0 is bottom
    pts[:, 1] = 1.0 - pts[:, 1]
    # Center horizontally on hip center for consistent alignment
    hip_center_x = (pts[23, 0] + pts[24, 0]) / 2
    pts[:, 0] = pts[:, 0] - hip_center_x
    visibility = node_features[:, 3]  # [35]

    # Check stick validity
    has_stick = is_stick_valid(node_features)

    # Helper to draw a segment
    def seg(i, j, color="#dfe6e9", lw=2.5, ls="-"):
        if visibility[i] < 0.01 or visibility[j] < 0.01:
            return
        xi, yi = pts[i]
        xj, yj = pts[j]
        # Skip if both are effectively zero (uninitialized)
        if (abs(xi) < 1e-6 and abs(yi) < 1e-6) or (abs(xj) < 1e-6 and abs(yj) < 1e-6):
            return
        ax.plot([xi, xj], [yi, yj], color=color, lw=lw, solid_capstyle="round", linestyle=ls, zorder=3)

    # Helper to draw a joint dot
    def dot(i, r=0.012, color="#dfe6e9", z=5):
        if visibility[i] < 0.01:
            return
        xi, yi = pts[i]
        if abs(xi) < 1e-6 and abs(yi) < 1e-6:
            return
        ax.add_patch(plt.Circle((xi, yi), r, color=color, zorder=z))

    # Torso
    seg(11, 12, lw=3.0)
    seg(11, 23, lw=2.5)
    seg(12, 24, lw=2.5)
    seg(23, 24, lw=2.5)

    # Arms
    if highlight_arms:
        seg(11, 13, color=JOINT_COLORS['left_arm'], lw=3.0)
        seg(13, 15, color=JOINT_COLORS['left_arm'], lw=3.0)
        seg(12, 14, color=JOINT_COLORS['right_arm'], lw=3.0)
        seg(14, 16, color=JOINT_COLORS['right_arm'], lw=3.0)
    else:
        seg(11, 13, lw=2.5)
        seg(13, 15, lw=2.5)
        seg(12, 14, lw=2.5)
        seg(14, 16, lw=2.5)

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
        # Data-driven head radius: ~1/4 of shoulder width, clamped
        radius = max(shoulder_width / 4.0, 0.02)
        radius = min(radius, 0.12)
        # Default: center so nose is at lower part of circle
        center_y = y0 + radius * 0.3
        # ENFORCE: circle bottom must be clearly above shoulders
        min_center_y = shoulder_y + radius + 0.03
        center_y = max(center_y, min_center_y)
        ax.add_patch(plt.Circle((x0, center_y), radius, color=JOINT_COLORS['body'], fill=False, lw=2.5, zorder=5))

    # Stick
    if has_stick:
        # Draw stick line
        grip_x, grip_y = pts[33]
        tip_x, tip_y = pts[34]
        ax.plot([grip_x, tip_x], [grip_y, tip_y],
                color=JOINT_COLORS['stick'], lw=4.5, solid_capstyle="round", zorder=6)
        # Grip dot
        ax.add_patch(plt.Circle((grip_x, grip_y), 0.014, color=JOINT_COLORS['grip'], zorder=7))
        # Tip dot
        ax.add_patch(plt.Circle((tip_x, tip_y), 0.011, color=JOINT_COLORS['tip'], zorder=7))
        # Connections from wrists to grip
        seg(15, 33, color="#dfe6e9", lw=1.5, ls="--")
        seg(16, 33, color="#dfe6e9", lw=1.5, ls="--")
    else:
        # Draw a small X to indicate missing stick
        ax.text(0.85, 0.95, "No stick", color="red", fontsize=7, ha="right")

    # Joint dots
    for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        if i in [13, 15]:
            dot(i, color=JOINT_COLORS['left_arm'])
        elif i in [14, 16]:
            dot(i, color=JOINT_COLORS['right_arm'])
        else:
            dot(i, color="#b2bec3")


def main():
    parser = argparse.ArgumentParser(description="Visualize poses from .pt feature files")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to .pt file (e.g., train_features_front.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: auto-generated)")
    parser.add_argument("--samples_per_class", type=int, default=2,
                        help="Number of sample columns per class (default: 2)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for sample selection")
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

    # Load data
    data = torch.load(input_path, map_location='cpu')
    node_features = data['node_features'].numpy()  # [N, 35, 6]
    labels = data['labels'].numpy()  # [N]

    print(f"Loaded {len(labels)} samples from {input_path}")
    print(f"Node feature shape: {node_features.shape}")

    # Load per-class aspect ratio config (overrides group defaults)
    ar_map = load_aspect_config(args.config, args.aspect_ratio_group1, args.aspect_ratio_group2)
    print(f"[INFO] Group defaults: 0-5 -> {args.aspect_ratio_group1:.4f}, 6-12 -> {args.aspect_ratio_group2:.4f}")
    for idx in sorted(ar_map.keys()):
        print(f"  Class {idx} ({CLASS_NAMES[idx]}): AR = {ar_map[idx]:.4f}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = Path("hybrid_classifier/visualizations") / f"{stem}_poses.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select random samples per class
    np.random.seed(args.random_seed)
    n_classes = len(CLASS_NAMES)
    cols = args.samples_per_class
    rows = n_classes

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 5.0))
    fig.patch.set_facecolor("#0f0f1a")

    if cols == 1:
        axes = axes.reshape(-1, 1)
    axes = np.array(axes).reshape(rows, cols)

    for class_idx in range(n_classes):
        # Find all samples for this class
        mask = labels == class_idx
        class_samples = node_features[mask]
        class_count = len(class_samples)

        # Select up to `cols` random samples
        if class_count == 0:
            selected = []
        elif class_count <= cols:
            selected = list(range(class_count))
        else:
            selected = np.random.choice(class_count, size=cols, replace=False)

        for col_idx in range(cols):
            ax = axes[class_idx, col_idx]
            if col_idx < len(selected):
                sample_idx = selected[col_idx]
                sample = class_samples[sample_idx]
                label = CLASS_NAMES[class_idx].replace("_correct", "").replace("_", " ").title()
                title = f"{label}\n(sample {sample_idx})"
                draw_pose(ax, sample, title=title, aspect_ratio=ar_map[class_idx])
            else:
                ax.set_visible(False)

    # Legend
    legend = [
        mpatches.Patch(color=JOINT_COLORS['right_arm'], label="Right (striking) arm"),
        mpatches.Patch(color=JOINT_COLORS['left_arm'], label="Left (guard) arm"),
        mpatches.Patch(color=JOINT_COLORS['stick'], label="Stick"),
        mpatches.Patch(color="#dfe6e9", label="Body / Legs"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.01))

    stem = input_path.stem
    fig.suptitle(f"Feature Visualization — {stem}  ({len(labels)} samples, {cols} per class)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] Saved -> {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
