"""
Visualize Synthetic vs Real Poses
==================================
Side-by-side comparison of real parent poses and their synthetic children.
Useful for human validation of synthetic quality before training.

Usage:
    python hybrid_classifier/visualize_synthetic.py \
        --real hybrid_classifier/hybrid_features_v3/train_features_front.pt \
        --synthetic hybrid_classifier/hybrid_features_v3/synthetic_train_features_front_3x.pt \
        --output hybrid_classifier/visualizations/synthetic_comparison_train.png \
        --samples_per_class 3

Output: Grid showing [Real | Synthetic A | Synthetic B] for each class
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


def is_stick_valid(node_features):
    grip = node_features[33]
    return (grip[0] != 0 or grip[1] != 0) and grip[3] > 0.01


def draw_pose(ax, node_features, title="", show_displacement=False, parent_nodes=None, aspect_ratio=1.0):
    """Draw a single pose. If parent_nodes given and show_displacement, color joints by movement."""
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=7, pad=2, fontweight="bold")

    pts = node_features[:, :2].copy()
    # Option C: Auto-estimated aspect ratio correction
    pts[:, 0] = pts[:, 0] * aspect_ratio
    # Flip Y: MediaPipe y=0 is top, matplotlib y=0 is bottom
    pts[:, 1] = 1.0 - pts[:, 1]
    # Center horizontally on hip center
    hip_center_x = (pts[23, 0] + pts[24, 0]) / 2
    pts[:, 0] = pts[:, 0] - hip_center_x
    visibility = node_features[:, 3]
    has_stick = is_stick_valid(node_features)

    # Displacement colors if parent provided (compute before flip)
    if show_displacement and parent_nodes is not None:
        raw_disp = np.linalg.norm(node_features[:, :2] - parent_nodes[:, :2], axis=1)
        max_disp = raw_disp.max() + 1e-8
        disp_norm = raw_disp / max_disp
    else:
        disp_norm = None

    def seg(i, j, color="#dfe6e9", lw=2.5, ls="-"):
        if visibility[i] < 0.01 or visibility[j] < 0.01:
            return
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (abs(xi) < 1e-6 and abs(yi) < 1e-6) or (abs(xj) < 1e-6 and abs(yj) < 1e-6):
            return
        ax.plot([xi, xj], [yi, yj], color=color, lw=lw, solid_capstyle="round", linestyle=ls, zorder=3)

    def dot(i, r=0.012, color="#dfe6e9", z=5):
        if visibility[i] < 0.01:
            return
        xi, yi = pts[i]
        if abs(xi) < 1e-6 and abs(yi) < 1e-6:
            return
        if show_displacement and disp_norm is not None and i in [13, 14, 15, 16, 33, 34]:
            d = disp_norm[i]
            if d < 0.3:
                c = '#4caf50'
            elif d < 0.6:
                c = '#ff9800'
            else:
                c = '#f44336'
            ax.add_patch(plt.Circle((xi, yi), r * 1.3, color=c, zorder=z + 1))
        else:
            ax.add_patch(plt.Circle((xi, yi), r, color=color, zorder=z))

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
        ax.text(0.85, 0.95, "No stick", color="red", fontsize=6, ha="right")

    # Dots
    for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        if i in [13, 15]:
            dot(i, color=JOINT_COLORS['left_arm'])
        elif i in [14, 16]:
            dot(i, color=JOINT_COLORS['right_arm'])
        else:
            dot(i, color="#b2bec3")


def main():
    parser = argparse.ArgumentParser(description="Compare real vs synthetic poses")
    parser.add_argument("--real", type=str, required=True,
                        help="Path to real .pt file")
    parser.add_argument("--synthetic", type=str, required=True,
                        help="Path to synthetic .pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    parser.add_argument("--samples_per_class", type=int, default=3,
                        help="Number of real parents to show per class")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--aspect_ratio_group1", type=float, default=0.4,
                        help="X-axis scale for classes 0-5 (default: 0.4)")
    parser.add_argument("--aspect_ratio_group2", type=float, default=1.3,
                        help="X-axis scale for classes 6-12 (default: 1.3, wider)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config with per-class aspect_ratio overrides")
    args = parser.parse_args()

    real_path = Path(args.real)
    syn_path = Path(args.synthetic)

    if not real_path.exists():
        print(f"Error: Real file not found: {real_path}")
        return
    if not syn_path.exists():
        print(f"Error: Synthetic file not found: {syn_path}")
        return

    real_data = torch.load(real_path, map_location='cpu')
    syn_data = torch.load(syn_path, map_location='cpu')

    real_nodes = real_data['node_features'].numpy()
    real_labels = real_data['labels'].numpy()
    syn_nodes = syn_data['node_features'].numpy()
    syn_labels = syn_data['labels'].numpy()
    syn_meta = syn_data.get('meta', [{}] * len(syn_labels))

    print(f"Real samples: {len(real_labels)}")
    print(f"Synthetic samples: {len(syn_labels)}")

    ar_map = load_aspect_config(args.config, args.aspect_ratio_group1, args.aspect_ratio_group2)
    print(f"[INFO] Group defaults: 0-5 -> {args.aspect_ratio_group1:.4f}, 6-12 -> {args.aspect_ratio_group2:.4f}")
    for idx in sorted(ar_map.keys()):
        print(f"  Class {idx} ({CLASS_NAMES[idx]}): AR = {ar_map[idx]:.4f}")

    # Determine how many synthetics per real (from meta)
    synthetics_per_parent = 2  # Default assumption for train

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("hybrid_classifier/visualizations") / f"synthetic_comparison_{real_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(CLASS_NAMES)
    cols = 1 + synthetics_per_parent  # Real + synthetics
    rows = args.samples_per_class

    fig, axes = plt.subplots(rows * n_classes, cols, figsize=(cols * 3.8, rows * n_classes * 4.5))
    fig.patch.set_facecolor("#0f0f1a")

    if n_classes == 1:
        axes = axes.reshape(1, -1)
    axes = np.array(axes).reshape(rows * n_classes, cols)

    np.random.seed(args.random_seed)

    row_idx = 0
    for class_idx in range(n_classes):
        # Get real samples for this class
        real_mask = real_labels == class_idx
        real_class_indices = np.where(real_mask)[0]

        # Get synthetic samples for this class
        syn_mask = syn_labels == class_idx
        syn_class_indices = np.where(syn_mask)[0]

        # Group synthetics by parent
        syn_by_parent = {}
        for si in syn_class_indices:
            meta = syn_meta[si] if si < len(syn_meta) else {}
            parent = meta.get('parent_idx', -1)
            if parent not in syn_by_parent:
                syn_by_parent[parent] = []
            syn_by_parent[parent].append(si)

        # Select parents that have synthetics
        valid_parents = [p for p in real_class_indices if p in syn_by_parent]
        if len(valid_parents) == 0:
            continue

        selected_parents = np.random.choice(valid_parents,
                                            size=min(args.samples_per_class, len(valid_parents)),
                                            replace=False)

        for parent_idx in selected_parents:
            # Draw real parent
            real_node = real_nodes[parent_idx]
            label = CLASS_NAMES[class_idx].replace("_correct", "").replace("_", " ").title()
            draw_pose(axes[row_idx, 0], real_node,
                      title=f"{label}\nReal #{parent_idx}",
                      show_displacement=False, aspect_ratio=ar_map[class_idx])

            # Draw synthetics
            syn_indices = syn_by_parent.get(parent_idx, [])
            for col_idx, syn_i in enumerate(syn_indices[:synthetics_per_parent]):
                if col_idx + 1 < cols:
                    syn_node = syn_nodes[syn_i]
                    meta = syn_meta[syn_i] if syn_i < len(syn_meta) else {}
                    lam = meta.get('lambda', '?')
                    sim = meta.get('similarity', 0)
                    checks = meta.get('checks', {})
                    ok_str = "OK" if all(checks.values()) else "WARN"
                    draw_pose(axes[row_idx, col_idx + 1], syn_node,
                              title=f"Synthetic λ={lam:.2f}\n{ok_str} sim={sim:.3f}",
                              show_displacement=True, parent_nodes=real_node, aspect_ratio=ar_map[class_idx])

            # Hide unused columns
            for c in range(1 + len(syn_indices), cols):
                if c < cols:
                    axes[row_idx, c].set_visible(False)

            row_idx += 1

    # Hide unused rows
    for r in range(row_idx, rows * n_classes):
        for c in range(cols):
            axes[r, c].set_visible(False)

    # Legend
    legend = [
        mpatches.Patch(color=JOINT_COLORS['right_arm'], label="Right (striking) arm"),
        mpatches.Patch(color=JOINT_COLORS['left_arm'], label="Left (guard) arm"),
        mpatches.Patch(color=JOINT_COLORS['stick'], label="Stick"),
        mpatches.Patch(color='#4caf50', label="Low displacement"),
        mpatches.Patch(color='#ff9800', label="Medium displacement"),
        mpatches.Patch(color='#f44336', label="High displacement"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6,
               facecolor="#1a1a2e", edgecolor="none",
               labelcolor="white", fontsize=8, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(f"Synthetic Validation: {real_path.stem}",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] Saved -> {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
