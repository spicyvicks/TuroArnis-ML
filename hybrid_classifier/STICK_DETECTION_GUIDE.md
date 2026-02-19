# Stick Detection Guide (Method 4 — Updated)

This document describes the **current stick detection algorithm** used in `pose_analyzer.py` (the live app).
Use this to update `apply_stick_method4_correction()` in `2b_generate_node_hybrid_features.py` and `2c_extract_test_features.py` so training features match inference features.

---

## What Changed vs. Old Method 4

| | Old (training scripts) | New (app / this guide) |
|---|---|---|
| **Front length** | `avg_torso_px × 1.5` | Shin-based: `shin_px × (0.71 / shin_m)` |
| **Side length** | Forearm 3D ratio | Shin-based: `shin_px × (0.71 / shin_m)` |
| **Front anchor** | Raw YOLO grip | MediaPipe RIGHT pinky |
| **Side anchor** | Raw YOLO grip | Raw YOLO grip (unchanged) |
| **Grip/tip swap** | Always (all views) | Disabled (all views) |
| **Direction** | Pure YOLO (all views) | Pure YOLO (all views, unchanged) |

---

## Updated `apply_stick_method4_correction()` Function

Replace the existing function in both `2b_generate_node_hybrid_features.py` and `2c_extract_test_features.py`:

```python
def apply_stick_method4_correction(raw_grip_px, raw_tip_px, kpts, img_width, img_height, world_landmarks, viewpoint=None):
    """
    Apply Stick Detection Method 4 (Updated): Adaptive Stick Correction

    Changes from original:
    - Length: shin-based (knee→ankle 3D ratio) for ALL viewpoints
    - Front anchor: MediaPipe RIGHT pinky (index 18) instead of raw YOLO grip
    - Side anchor: raw YOLO grip (unchanged)
    - Grip/tip swap: Disabled for all viewpoints (trusts YOLO)

    Args:
        raw_grip_px: (x, y) in pixels - raw YOLO grip point
        raw_tip_px:  (x, y) in pixels - raw YOLO tip point
        kpts:        [33, 4] array - MediaPipe landmarks (x, y, z, visibility) normalized
        img_width:   image width in pixels
        img_height:  image height in pixels
        world_landmarks: MediaPipe world landmarks (3D in meters)
        viewpoint:   'front' | 'left' | 'right' | None (auto-detect if None)

    Returns:
        corrected_grip_px, corrected_tip_px: (x, y) tuples in pixels
    """
    STICK_LENGTH_M = 0.71   # Standard Arnis stick length in meters
    FRONT_VIEW_THRESHOLD = 0.45

    def to_pixels(lm):
        return np.array([lm[0] * img_width, lm[1] * img_height])

    # Body landmarks in pixels
    left_shoulder  = to_pixels(kpts[11])
    right_shoulder = to_pixels(kpts[12])
    left_hip       = to_pixels(kpts[23])
    right_hip      = to_pixels(kpts[24])
    left_wrist     = to_pixels(kpts[15])
    right_wrist    = to_pixels(kpts[16])

    # --- STEP 1: Determine viewpoint (front vs side) ---
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    avg_torso_px   = (np.linalg.norm(left_shoulder - left_hip) +
                      np.linalg.norm(right_shoulder - right_hip)) / 2.0

    if viewpoint is not None:
        is_front_view = (viewpoint == 'front')
    else:
        view_ratio    = shoulder_width / (avg_torso_px + 1e-6)
        is_front_view = view_ratio > FRONT_VIEW_THRESHOLD

    # --- STEP 2: Grip/tip swap (disabled) ---
    grip_px = np.array(raw_grip_px, dtype=float)
    tip_px  = np.array(raw_tip_px,  dtype=float)

    # Step 2 disabled: Trusts YOLO assignment for all viewpoints
    # if is_front_view:
    #     dist_grip_r = np.linalg.norm(grip_px - right_wrist)
    #     dist_grip_l = np.linalg.norm(grip_px - left_wrist)
    #     dist_tip_r  = np.linalg.norm(tip_px  - right_wrist)
    #     dist_tip_l  = np.linalg.norm(tip_px  - left_wrist)
    #     min_grip_dist = min(dist_grip_r, dist_grip_l)
    #     min_tip_dist  = min(dist_tip_r,  dist_tip_l)
    #     if min_tip_dist < min_grip_dist:
    #         grip_px, tip_px = tip_px, grip_px  # swap

    # --- STEP 3: Set grip anchor ---
    if is_front_view:
        # Front view: use MediaPipe RIGHT pinky (index 18) as grip anchor
        # More anatomically accurate than raw YOLO grip from front
        right_pinky = to_pixels(kpts[18])
        grip_px = right_pinky
    # Side view: keep raw YOLO grip (already set above)

    # --- UNIFIED LOGIC: Hand Proximity & Pinky Snap ---
    
    # 0. Foreshortening Check (New from App)
    # If raw YOLO stick is very short (pointing at camera), skip correction to avoid wild snaps
    grip_px = np.array(raw_grip_px, dtype=float)
    tip_px  = np.array(raw_tip_px,  dtype=float)

    raw_length = np.linalg.norm(tip_px - grip_px)
    FORESHORTEN_THRESHOLD_PX = 40
    
    if raw_length < FORESHORTEN_THRESHOLD_PX:
        return tuple(grip_px), tuple(tip_px)

    # 1. Identify Hand: Compare YOLO grip distance to Left vs Right Wristabove)

    # --- STEP 4: Shin-based stick length (both viewpoints) ---
    # Shin (knee→ankle) is more stable than torso or forearm across poses
    lk = to_pixels(kpts[25]);  la = to_pixels(kpts[27])   # left knee, left ankle
    rk = to_pixels(kpts[26]);  ra = to_pixels(kpts[28])   # right knee, right ankle

    # 3D shin length (meters)
    def world_pt(idx):
        lm = world_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    shin_m = (np.linalg.norm(world_pt(25) - world_pt(27)) +
              np.linalg.norm(world_pt(26) - world_pt(28))) / 2.0

    # 2D shin length (pixels)
    shin_px = (np.linalg.norm(lk - la) + np.linalg.norm(rk - ra)) / 2.0

    # Stick length in pixels via ratio
    stick_px = shin_px * (STICK_LENGTH_M / (shin_m + 1e-6))

    # Clamp to 2.5× torso (sanity check)
    stick_px = min(stick_px, avg_torso_px * 2.5)

    # --- STEP 5: Project corrected tip using YOLO direction ---
    direction = tip_px - grip_px
    direction_len = np.linalg.norm(direction) + 1e-6
    direction_unit = direction / direction_len

    corrected_tip_px = grip_px + direction_unit * stick_px

    return tuple(grip_px), tuple(corrected_tip_px)
```

---

## How to Update the Training Scripts

### `2b_generate_node_hybrid_features.py`

1. Replace `apply_stick_method4_correction()` with the function above.
2. Pass `viewpoint` to the function in `extract_raw_features()`:

```python
# In extract_raw_features(), add viewpoint parameter:
def extract_raw_features(image_path, stick_detector, viewpoint=None):
    ...
    corrected_grip_px, corrected_tip_px = apply_stick_method4_correction(
        raw_grip_px, raw_tip_px, kpts, w, h,
        results.pose_world_landmarks.landmark,
        viewpoint=viewpoint   # <-- pass this
    )
```

3. In `process_single_image()`, pass viewpoint through:

```python
raw_data = extract_raw_features(img_path, stick_detector, viewpoint=viewpoint)
```

### `2c_extract_test_features.py`

Apply the same changes as above.

---

## MediaPipe Landmark Indices Reference

| Index | Landmark |
|---|---|
| 11 | Left shoulder |
| 12 | Right shoulder |
| 15 | Left wrist |
| 16 | Right wrist |
| 18 | **Right pinky** ← new front anchor |
| 23 | Left hip |
| 24 | Right hip |
| 25 | Left knee |
| 26 | Right knee |
| 27 | Left ankle |
| 28 | Right ankle |

---

## Why These Changes

| Change | Reason |
|---|---|
| Shin-based length | Shin length is consistent across poses; torso compresses in bent poses, forearm varies with arm position |
| Front anchor | UNIFIED: MediaPipe Pinky (LEFT or RIGHT) based on YOLO proximity |
| Side anchor | UNIFIED: MediaPipe Pinky (LEFT or RIGHT) based on YOLO proximity |
| Grip/tip swap | Disabled (all views) |
| Direction | Pure YOLO (all views, unchanged) |
| **Foreshortening** | **Safety Check**: If raw YOLO stick length < 40px (pointing at camera), correction is SKIPPED to prevent wild snapping. |
