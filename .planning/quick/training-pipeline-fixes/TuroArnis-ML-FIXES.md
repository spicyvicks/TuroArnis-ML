# TuroArnis-ML Training Pipeline Fixes

## Summary

Apply these 3 fixes to the TuroArnis-ML repository to align training with TuroArnis app inference.

---

## Fix #2: Add Quality Validation Gates

**File:** `hybrid_classifier/1_extract_reference_features.py`

### Step 1: Add validation function (insert near top of file)

```python
def validate_features(features, pose_landmarks, stick_detected, stick_confidence=1.0):
    """Validate extracted features before including in template statistics."""
    # MediaPipe critical joint visibility check
    CRITICAL_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]
    MIN_VISIBILITY = 0.5
    
    for joint_idx in CRITICAL_JOINTS:
        if pose_landmarks.landmark[joint_idx].visibility < MIN_VISIBILITY:
            return False, f"Low visibility on joint {joint_idx}"
    
    # Stick detection check
    if not stick_detected:
        return False, "Stick not detected"
    
    # Stick confidence check
    if stick_confidence < 0.3:
        return False, f"Low stick confidence ({stick_confidence:.2f})"
    
    # Physical plausibility
    if features.get('stick_length', 0) == 0:
        return False, "Zero stick length"
    
    return True, "Valid"
```

### Step 2: Modify `analyze_reference_images()` loop (around line 300)

```python
all_features = []
rejected_count = 0
total_count = 0

for img_path in tqdm(images, ...):
    features = extract_geometric_features(img_path)
    total_count += 1
    
    if features is None:
        rejected_count += 1
        print(f"  [REJECTED] {img_path.name}: No pose detected")
        continue
    
    # NEW: Add validation gate
    is_valid, reason = validate_features(
        features, 
        results.pose_landmarks,  # Pass landmarks from extraction
        stick_was_detected,      # Track if YOLO found stick
        stick_confidence
    )
    
    if not is_valid:
        rejected_count += 1
        print(f"  [REJECTED] {img_path.name}: {reason}")
        continue
    # END validation gate
    
    all_features.append(features)

print(f"  Accepted: {len(all_features)}/{total_count}")
print(f"  Rejected: {rejected_count}/{total_count}")
```

**Expected Result:** Template STDs should decrease by >30%

---

## Fix #4: Use 3D Angles

**File:** `hybrid_classifier/1_extract_reference_features.py`

### Step 1: Replace `calculate_angle()` with 3D version

```python
def calculate_angle(p1, p2, p3):
    """Calculate 3D angle - matches TuroArnis app."""
    # Handle 2D input for backward compatibility
    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0.0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0.0]
    if len(p3) == 2:
        p3 = [p3[0], p3[1], 0.0]
    
    # 3D vectors
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)
```

### Step 2: Modify angle calculation to use world landmarks

In `extract_geometric_features()`, change:

```python
# OLD (2D):
features['left_elbow_angle'] = calculate_angle(kpts[11], kpts[13], kpts[15])

# NEW (3D):
world_landmarks = results.pose_world_landmarks.landmark
def get_world_point(idx):
    lm = world_landmarks[idx]
    return [lm.x, lm.y, lm.z]

features['left_elbow_angle'] = calculate_angle(
    get_world_point(11), get_world_point(13), get_world_point(15)
)
# ... repeat for all 6 angle features
```

**Expected Result:** Better discrimination between thrusts (forward) and blocks (sideways)

---

## Fix #5: NaN Sentinel for Missing Stick

**File:** `hybrid_classifier/1_extract_reference_features.py`

In `extract_geometric_features()`, change fallback values:

```python
# OLD (center fallback - corrupts templates):
else:
    stick_grip = [0.5, 0.5]
    stick_tip = [0.5, 0.5]

# NEW (NaN sentinel - validation will reject):
else:
    import math
    stick_grip = [float('nan'), float('nan')]
    stick_tip = [float('nan'), float('nan')]
```

**Expected Result:** Failed stick detections are properly rejected by validation gate

---

## Fix #3: Sync Deployment Package

**File:** `deployment_package/src/feature_extraction.py`

Copy the updated `feature_extraction.py` from TuroArnis app after all fixes are applied:

```bash
cp app/models/gcn/feature_extraction.py deployment_package/src/
```

---

## Verification Steps

After applying all fixes:

1. **Regenerate templates:**
   ```bash
   python hybrid_classifier/1_extract_reference_features.py
   ```

2. **Check rejection rate:** Should be 10-30% depending on image quality

3. **Verify tight STDs:**
   ```python
   import json
   with open("hybrid_classifier/feature_templates.json") as f:
       templates = json.load(f)
   
   # Check angle feature STDs - should be < 30°, not > 60°
   for feat, stats in templates["front_crown_thrust_correct"].items():
       if "angle" in feat and stats["std"] > 30:
           print(f"WARNING: {feat} std={stats['std']:.1f}° (too wide)")
   ```

4. **Retrain models:**
   ```bash
   python hybrid_classifier/4c_train_hybrid_gcn_v2.py --merged
   ```

---

## TuroArnis App Changes Already Applied

These fixes were already made in TuroArnis:

- ✅ Issue #1: Removed 'neutral' class (12 classes now match training)
- ✅ Issue #3: Fixed coordinate normalization (both to full frame)

Both committed and ready for testing.
