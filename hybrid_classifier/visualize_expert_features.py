"""
Visualize Expert Features (Stick Detection + Body Skeleton)

Draws the corrected stick, body skeleton, and expert feature values
on a sample image so you can visually verify the detection logic.

Usage:
    # Random test image:
    python hybrid_classifier/visualize_expert_features.py

    # Specific image:
    python hybrid_classifier/visualize_expert_features.py --image path/to/image.jpg

    # Force viewpoint (skips auto-detect):
    python hybrid_classifier/visualize_expert_features.py --image path/to/image.jpg --viewpoint front
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
import random
import argparse
import sys

# ── Config ────────────────────────────────────────────────────────────────────
STICK_MODEL   = "runs/pose/arnis_stick_detector/weights/best.pt"
DATASET_ROOT  = Path("dataset_split/test")
OUTPUT_PATH   = "visualized_features.jpg"

STICK_LENGTH_M       = 0.71
FRONT_VIEW_THRESHOLD = 0.45

# MediaPipe skeleton connections (body only, indices 11-28)
BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),   # left arm
    (12, 14), (14, 16),             # right arm
    (11, 23), (12, 24),             # torso sides
    (23, 24),                       # hips
    (23, 25), (25, 27),             # left leg
    (24, 26), (26, 28),             # right leg
]

# ── Stick Detection (Updated Method 4) ────────────────────────────────────────
def apply_stick_method4_correction(raw_grip_px, raw_tip_px, kpts, img_width, img_height, world_landmarks, viewpoint=None):
    """
    Updated Method 4: shin-based length, right-pinky anchor (front), front-only swap.
    Identical to the version in 2b_generate_node_hybrid_features.py.
    """
    def to_pixels(lm):
        return np.array([lm[0] * img_width, lm[1] * img_height])

    left_shoulder  = to_pixels(kpts[11])
    right_shoulder = to_pixels(kpts[12])
    left_hip       = to_pixels(kpts[23])
    right_hip      = to_pixels(kpts[24])
    left_wrist     = to_pixels(kpts[15])
    right_wrist    = to_pixels(kpts[16])

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    avg_torso_px   = (np.linalg.norm(left_shoulder - left_hip) +
                      np.linalg.norm(right_shoulder - right_hip)) / 2.0

    if viewpoint is not None:
        is_front_view = (viewpoint == 'front')
    else:
        view_ratio    = shoulder_width / (avg_torso_px + 1e-6)
        is_front_view = view_ratio > FRONT_VIEW_THRESHOLD

    grip_px = np.array(raw_grip_px, dtype=float)
    tip_px  = np.array(raw_tip_px,  dtype=float)

    # Grip/tip swap — front view only
    if is_front_view:
        dist_grip_r = np.linalg.norm(grip_px - right_wrist)
        dist_grip_l = np.linalg.norm(grip_px - left_wrist)
        dist_tip_r  = np.linalg.norm(tip_px  - right_wrist)
        dist_tip_l  = np.linalg.norm(tip_px  - left_wrist)
        if min(dist_tip_r, dist_tip_l) < min(dist_grip_r, dist_grip_l):
            grip_px, tip_px = tip_px, grip_px

    # Grip anchor — right pinky for front, raw YOLO for side
    if is_front_view:
        grip_px = to_pixels(kpts[18])   # right pinky

    # Shin-based stick length
    lk = to_pixels(kpts[25]); la = to_pixels(kpts[27])
    rk = to_pixels(kpts[26]); ra = to_pixels(kpts[28])

    def world_pt(idx):
        lm = world_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    shin_m  = (np.linalg.norm(world_pt(25) - world_pt(27)) +
               np.linalg.norm(world_pt(26) - world_pt(28))) / 2.0
    shin_px = (np.linalg.norm(lk - la) + np.linalg.norm(rk - ra)) / 2.0

    stick_px = shin_px * (STICK_LENGTH_M / (shin_m + 1e-6))
    stick_px = min(stick_px, avg_torso_px * 2.5)

    direction      = tip_px - grip_px
    direction_unit = direction / (np.linalg.norm(direction) + 1e-6)
    corrected_tip  = grip_px + direction_unit * stick_px

    return tuple(grip_px), tuple(corrected_tip), is_front_view


# ── Feature Computation ────────────────────────────────────────────────────────
def compute_expert_features(kpts, stick_grip_norm, stick_tip_norm):
    """Compute the same expert features used in training."""
    def dist(a, b): return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    def angle(p1, p2, p3):
        v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
        v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
        c  = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    hip_center_x = (kpts[23][0] + kpts[24][0]) / 2
    hip_center_y = (kpts[23][1] + kpts[24][1]) / 2
    shoulder_y   = (kpts[11][1] + kpts[12][1]) / 2
    nose_y       = kpts[0][1]

    stick_vec = np.array([stick_tip_norm[0] - stick_grip_norm[0],
                          stick_tip_norm[1] - stick_grip_norm[1]])
    stick_len = np.linalg.norm(stick_vec) + 1e-6

    return {
        "left_elbow_angle":   angle(kpts[11], kpts[13], kpts[15]),
        "right_elbow_angle":  angle(kpts[12], kpts[14], kpts[16]),
        "left_knee_angle":    angle(kpts[23], kpts[25], kpts[27]),
        "right_knee_angle":   angle(kpts[24], kpts[26], kpts[28]),
        "stick_angle":        np.degrees(np.arctan2(stick_vec[1], stick_vec[0])),
        "stick_length":       dist(stick_grip_norm, stick_tip_norm),
        "tip_vs_nose":        stick_tip_norm[1] - nose_y,
        "tip_vs_shoulder":    stick_tip_norm[1] - shoulder_y,
        "tip_vs_hip":         stick_tip_norm[1] - hip_center_y,
        "grip_side":          stick_grip_norm[0] - hip_center_x,
        "tip_side":           stick_tip_norm[0]  - hip_center_x,
        "r_hand_vs_shoulder": kpts[16][1] - shoulder_y,
        "foot_stagger":       kpts[27][1] - kpts[28][1],
        "hands_distance":     dist(kpts[15], kpts[16]),
    }


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_text_block(img, lines, start_y=30, x=10):
    """Draw a block of text with black outline for readability."""
    y = start_y
    for text, color in lines:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 24
    return y


def visualize(image_path, viewpoint=None, output_path=OUTPUT_PATH):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    h, w = img.shape[:2]

    # ── MediaPipe Pose ─────────────────────────────────────────────────────────
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("No pose detected.")
        return

    kpts = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark])
    world_lms = results.pose_world_landmarks.landmark

    def to_px(norm):
        return (int(norm[0] * w), int(norm[1] * h))

    # ── YOLO Stick ─────────────────────────────────────────────────────────────
    stick_detector = YOLO(STICK_MODEL)
    stick_results  = stick_detector(str(image_path), verbose=False)[0]

    has_stick = (stick_results.keypoints is not None and
                 len(stick_results.keypoints.data) > 0)

    if has_stick:
        sk = stick_results.keypoints.data[0].cpu().numpy()
        raw_grip = (sk[0, 0], sk[0, 1])
        raw_tip  = (sk[1, 0], sk[1, 1])

        corrected_grip, corrected_tip, is_front = apply_stick_method4_correction(
            raw_grip, raw_tip, kpts, w, h, world_lms, viewpoint=viewpoint
        )

        # Normalized for feature computation
        grip_norm = (corrected_grip[0] / w, corrected_grip[1] / h)
        tip_norm  = (corrected_tip[0]  / w, corrected_tip[1]  / h)
        features  = compute_expert_features(kpts, grip_norm, tip_norm)
    else:
        print("No stick detected by YOLO.")
        corrected_grip = corrected_tip = None
        features = {}

    # ── Draw Skeleton ──────────────────────────────────────────────────────────
    for i, j in BODY_CONNECTIONS:
        if kpts[i][3] > 0.4 and kpts[j][3] > 0.4:
            cv2.line(img, to_px(kpts[i]), to_px(kpts[j]), (180, 180, 180), 2)

    for i in range(33):
        if kpts[i][3] > 0.4:
            color = (0, 255, 255)
            # Highlight key landmarks
            if i == 18:  color = (255, 0, 255)   # Right pinky (front grip anchor) — magenta
            if i in (15, 16): color = (0, 200, 255)  # Wrists — orange-yellow
            cv2.circle(img, to_px(kpts[i]), 5, color, -1)

    # ── Draw Stick ─────────────────────────────────────────────────────────────
    if corrected_grip and corrected_tip:
        g = (int(corrected_grip[0]), int(corrected_grip[1]))
        t = (int(corrected_tip[0]),  int(corrected_tip[1]))

        # Raw YOLO stick (thin, semi-transparent gray)
        rg = (int(raw_grip[0]), int(raw_grip[1]))
        rt = (int(raw_tip[0]),  int(raw_tip[1]))
        overlay = img.copy()
        cv2.line(overlay, rg, rt, (100, 100, 100), 3)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # Corrected stick (thick orange)
        cv2.line(img, g, t, (0, 140, 255), 6)

        # Endpoints
        cv2.circle(img, g, 9, (0, 0, 255), -1)    # Grip — Red
        cv2.circle(img, t, 9, (0, 255, 0), -1)    # Tip  — Green

        # Labels
        cv2.putText(img, "Grip", (g[0]+10, g[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(img, "Tip",  (t[0]+10, t[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # ── Legend ─────────────────────────────────────────────────────────────────
    view_label = ("FRONT" if (viewpoint == 'front' or
                  (has_stick and corrected_grip and
                   (viewpoint is None and features.get("grip_side", 0) != 0)))
                  else viewpoint.upper() if viewpoint else "AUTO")

    lines = [
        (f"Viewpoint: {view_label}", (255, 255, 100)),
        ("", (255,255,255)),
        ("--- Stick ---", (0, 200, 255)),
        (f"  Angle:  {features.get('stick_angle', 0):.1f} deg", (255, 255, 255)),
        (f"  Length: {features.get('stick_length', 0):.3f} (norm)", (255, 255, 255)),
        ("", (255,255,255)),
        ("--- Relative Heights ---", (180, 255, 180)),
        (f"  Tip vs Nose:     {features.get('tip_vs_nose', 0):+.3f}", (255, 255, 255)),
        (f"  Tip vs Shoulder: {features.get('tip_vs_shoulder', 0):+.3f}", (255, 255, 255)),
        (f"  Tip vs Hip:      {features.get('tip_vs_hip', 0):+.3f}", (255, 255, 255)),
        ("", (255,255,255)),
        ("--- Position ---", (255, 200, 180)),
        (f"  Grip Side: {features.get('grip_side', 0):+.3f} (>0=Right)", (255, 255, 255)),
        (f"  Tip Side:  {features.get('tip_side', 0):+.3f}", (255, 255, 255)),
        ("", (255,255,255)),
        ("--- Body Angles ---", (200, 180, 255)),
        (f"  R Elbow: {features.get('right_elbow_angle', 0):.1f} deg", (255, 255, 255)),
        (f"  L Elbow: {features.get('left_elbow_angle', 0):.1f} deg", (255, 255, 255)),
        (f"  R Knee:  {features.get('right_knee_angle', 0):.1f} deg", (255, 255, 255)),
        (f"  L Knee:  {features.get('left_knee_angle', 0):.1f} deg", (255, 255, 255)),
    ]

    # Semi-transparent background for text
    text_bg = img.copy()
    cv2.rectangle(text_bg, (5, 5), (280, len(lines) * 24 + 20), (0, 0, 0), -1)
    cv2.addWeighted(text_bg, 0.5, img, 0.5, 0, img)

    draw_text_block(img, lines)

    # ── Legend for colors ──────────────────────────────────────────────────────
    legend_y = h - 90
    cv2.circle(img, (15, legend_y),    7, (0, 0, 255), -1);   cv2.putText(img, "Grip (corrected)", (28, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.circle(img, (15, legend_y+22), 7, (0, 255, 0), -1);   cv2.putText(img, "Tip (corrected)",  (28, legend_y+27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.line(img, (8, legend_y+44), (22, legend_y+44), (0, 140, 255), 5); cv2.putText(img, "Corrected stick", (28, legend_y+49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.line(img, (8, legend_y+66), (22, legend_y+66), (100, 100, 100), 2); cv2.putText(img, "Raw YOLO stick", (28, legend_y+71), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.circle(img, (15, legend_y+88), 5, (255, 0, 255), -1); cv2.putText(img, "Right pinky (front anchor)", (28, legend_y+93), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # ── Save ───────────────────────────────────────────────────────────────────
    cv2.imwrite(output_path, img)
    print(f"\nSaved to: {output_path}")
    print("\nComputed Expert Features:")
    for k, v in features.items():
        print(f"  {k:25s}: {v:.4f}")


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize expert features on a single image.")
    parser.add_argument("--image",     type=str, help="Path to image (default: random from dataset_split/test)")
    parser.add_argument("--viewpoint", type=str, choices=["front", "left", "right"], help="Force viewpoint")
    parser.add_argument("--output",    type=str, default=OUTPUT_PATH, help="Output image path")
    args = parser.parse_args()

    img_path = args.image
    if not img_path:
        all_imgs = list(DATASET_ROOT.rglob("*.jpg")) + list(DATASET_ROOT.rglob("*.png"))
        if not all_imgs:
            print(f"No images found in {DATASET_ROOT}")
            sys.exit(1)
        img_path = random.choice(all_imgs)
        print(f"No image specified — picked random: {img_path}")

    visualize(img_path, viewpoint=args.viewpoint, output_path=args.output)
