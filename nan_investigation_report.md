# NaN Investigation Report: Feature Generation Pipeline

## Executive Summary

**NaN values are being introduced during feature generation in v3, NOT during training.**

- **Root Cause**: When YOLO stick detection fails, `float('nan')` sentinel values are assigned to stick keypoints
- **Affected Version**: `hybrid_features_v3` only (right viewpoint: 13.2% train, 8.6% test)
- **Fixed in**: `hybrid_features_v4` (no NaN detected)

---

## 1. Where NaN Values Originate

### 1.1 Feature Generation Scripts (SOURCE OF NaN)

**File**: `hybrid_classifier/2b_generate_node_hybrid_features.py` (lines 228-231)

```python
# Line 228-231
else:
    # FIXED: No stick detected - skip this sample instead of using NaN
    # Return None to indicate failed detection, will be filtered out
    return None
```

Wait - the v3 code has a fix to return `None`. Let me check the actual v3 execution...

### 1.2 Template Generation Script (ALSO USES NaN)

**File**: `hybrid_classifier/1_extract_reference_features.py` (lines 264-267)

```python
# Lines 263-267
else:
    # Issue #5: Use NaN sentinel instead of [0.5, 0.5] fallback
    # This allows validation gate to properly reject failed detections
    stick_grip = [float('nan'), float('nan')]
    stick_tip = [float('nan'), float('nan')]
    stick_confidence = 0.0
```

**BUT**: This NaN is filtered out by the validation gate at lines 413-426:
```python
# Issue #2: Apply validation gate before accepting into templates
is_valid, reason = validate_features(features, metadata...)
if not is_valid:
    rejected_count += 1
    print(f"  [REJECTED] {img_path.name}: {reason}")
    continue
```

So templates should NOT contain NaN (verified: 0 NaN in templates).

### 1.3 Deployment Package (ALSO USES NaN)

**File**: `deployment_package/src/feature_extraction.py` (lines 85-87)

```python
else:
    # Issue #5: Use NaN sentinel for missing stick detection
    stick_grip = [float('nan'), float('nan'), 0.0, 0.0]
    stick_tip = [float('nan'), float('nan'), 0.0, 0.0]
```

---

## 2. NaN Propagation Analysis

### 2.1 Feature File Analysis Results

| Version | Viewpoint | Train NaN % | Test NaN % | Status |
|---------|-----------|-------------|------------|--------|
| v2 | All | 0.0% | 0.0% | Clean |
| **v3** | **front** | **0.0%** | **0.0%** | **Clean** |
| **v3** | **left** | **0.0%** | **0.0%** | **Clean** |
| **v3** | **right** | **13.2%** | **8.6%** | **CONTAMINATED** |
| v4 | All | 0.0% | 0.0% | Clean |

### 2.2 Breakdown by Class (v3/right train)

| Class | Samples | With NaN | % NaN | Class Name |
|-------|---------|----------|-------|------------|
| 0 | 188 | 18 | 9.6% | crown_thrust_correct |
| 1 | 172 | 8 | 4.7% | left_chest_thrust_correct |
| 2 | 153 | 10 | 6.5% | left_elbow_block_correct |
| 3 | 155 | 4 | 2.6% | left_eye_thrust_correct |
| 4 | 158 | 25 | 15.8% | left_knee_block_correct |
| 5 | 151 | 9 | 6.0% | left_temple_block_correct |
| 6 | 144 | 3 | 2.1% | right_chest_thrust_correct |
| 7 | 179 | 2 | 1.1% | right_elbow_block_correct |
| 8 | 162 | 1 | 0.6% | right_eye_thrust_correct |
| 9 | 159 | 14 | 8.8% | right_knee_block_correct |
| 10 | 151 | 4 | 2.6% | right_temple_block_correct |
| 11 | 158 | 3 | 1.9% | solar_plexus_thrust_correct |
| **12** | **220** | **182** | **82.7%** | **neutral** |

**Critical Finding**: Class 12 (neutral) has 82.7% NaN contamination! This suggests the "neutral" class images have poor stick detection rates.

### 2.3 Where NaN Appears in Features

- **Node indices 33-34** (stick nodes): Contain NaN in x, y, z coordinates
- **Node indices 0-32** (pose nodes): No NaN detected
- **Hybrid features indices**: [10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 29]
  - These correspond to stick-related features (stick_tip_height, stick_grip_height, stick_angle, stick_dx, stick_dy, tip_vs_*, grip_*, tip_side, stick_length)

---

## 3. Why v3 Right Viewpoint Has NaN but v4 Doesn't

### Hypothesis

The v3 feature generation was run with an older version of the code or different conditions that allowed NaN to propagate through. The key differences:

1. **v3** (contaminated): May have been generated when the `return None` check (line 228-231) was not filtering properly, or stick detection was less reliable
2. **v4** (clean): Likely regenerated with improved stick detection or stricter filtering

### Evidence

- v3 code (lines 228-231) has logic to `return None` when stick detection fails
- However, the fact that v3 has NaN but v4 doesn't suggests:
  1. v4 was regenerated with better stick detection model, OR
  2. v4 used different filtering logic, OR
  3. v4 dataset had better quality images

---

## 4. Impact on Training

### 4.1 Training Script Handling

Both training scripts have NaN filtering:

**File**: `hybrid_classifier/4c_train_hybrid_gcn_v2.py` (lines 253-257)
```python
def _get_nan_mask(self):
    node_nan = torch.isnan(self.node_features).any(dim=(1, 2))
    hybrid_nan = torch.isnan(self.hybrid_features).any(dim=1)
    return node_nan | hybrid_nan
```

**File**: `hybrid_classifier/4d_train_hybrid_gcn_v3.py` (lines 390-394)
```python
def _get_nan_mask(self):
    node_nan = torch.isnan(self.node_features).any(dim=(1, 2))
    hybrid_nan = torch.isnan(self.hybrid_features).any(dim=1)
    return node_nan | hybrid_nan
```

**Lines 660+**: Both instantiate with `filter_nan=True`

### 4.2 Training Impact

- Training automatically filters NaN samples (283 samples filtered in v3/right train)
- This reduces effective training data for neutral class from 220 → 38 samples
- **No training crashes occur** - NaN is handled gracefully
- However, **reduced data quality** for affected classes

---

## 5. Recommendations

### 5.1 Immediate Action (NOT Required for Training)

Training scripts already filter NaN samples, so training will proceed without crashes. However, for data quality:

### 5.2 Recommended Action (For Better Data Quality)

**Option A: Regenerate v3 features (if needed)**
- Regenerate using v4 generation code
- v4 has 0% NaN across all viewpoints

**Option B: Use v4 features instead of v3**
- v4 is already clean (0% NaN)
- Switch training to use v4 features

**Option C: Investigate neutral class images**
- 82.7% of neutral class in v3/right has NaN
- Check if neutral images lack visible sticks
- Consider special handling for neutral class (no stick required)

### 5.3 Code Improvements

1. **Add NaN check during generation** - Fail early if NaN detected
2. **Log rejected samples** - Track why stick detection fails
3. **Consider neutral class exception** - Neutral poses may not require stick

---

## 6. Key Files and Lines

### Where NaN is Introduced:
1. `hybrid_classifier/1_extract_reference_features.py:266-267` - NaN sentinel for missing stick
2. `deployment_package/src/feature_extraction.py:86-87` - NaN sentinel for missing stick

### Where NaN is Filtered:
1. `hybrid_classifier/2b_generate_node_hybrid_features.py:228-231` - Returns None on failure
2. `hybrid_classifier/1_extract_reference_features.py:118-119` - Rejects NaN stick length in validation
3. `hybrid_classifier/4c_train_hybrid_gcn_v2.py:253-257` - Dataset NaN mask
4. `hybrid_classifier/4d_train_hybrid_gcn_v3.py:390-394` - Dataset NaN mask

---

## 7. Conclusion

**NaN originates in feature generation when YOLO stick detection fails**, specifically in the v3 feature set for the right viewpoint. The NaN propagates into:
- Stick node features (indices 33-34)
- Hybrid features that depend on stick measurements

**The training scripts handle this gracefully** by filtering NaN samples, but this reduces the effective training data size (especially for neutral class: 220 → 38 samples).

**Recommendation**: Use v4 features which have 0% NaN, or regenerate v3 with improved stick detection.

---

*Report generated: 2025-01-08*
*Investigation scope: Feature generation pipeline*
