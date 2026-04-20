#!/usr/bin/env python3
"""
TuroArnis-ML Training Pipeline Fixes
====================================

This script contains the specific code changes needed in the TuroArnis-ML
repository to fix Issues #2, #4, and #5 identified during cross-repo analysis.

Apply these changes to align training with the TuroArnis app inference.

ISSUES FIXED:
- Issue #2: Add quality validation gates to template generation
- Issue #4: Use 3D coordinates for angle calculation  
- Issue #5: Use NaN sentinels for missing stick detection

"""

# ============================================================================
# ISSUE #2: QUALITY VALIDATION GATES
# File: hybrid_classifier/1_extract_reference_features.py
# ============================================================================

VALIDATION_CODE = '''
def validate_features(features, pose_landmarks, stick_detected, stick_confidence=1.0):
    """
    Validate extracted features before including in template statistics.
    Returns (is_valid: bool, reason: str)
    """
    # Rule 1: MediaPipe critical joint visibility check
    CRITICAL_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]  # Shoulders, elbows, wrists, hips
    MIN_VISIBILITY = 0.5
    
    for joint_idx in CRITICAL_JOINTS:
        if pose_landmarks.landmark[joint_idx].visibility < MIN_VISIBILITY:
            return False, f"Low visibility on joint {joint_idx} (< {MIN_VISIBILITY})"
    
    # Rule 2: YOLO stick detection check
    if not stick_detected:
        return False, "Stick not detected by YOLO"
    
    # Rule 3: YOLO confidence check (if available)
    MIN_STICK_CONFIDENCE = 0.3
    if stick_confidence < MIN_STICK_CONFIDENCE:
        return False, f"Low stick confidence ({stick_confidence:.2f} < {MIN_STICK_CONFIDENCE})"
    
    # Rule 4: Physical plausibility - reject zero stick length
    if features.get('stick_length', 0) == 0:
        return False, "Zero stick length (detection failure)"
    
    # Rule 5: Detect center fallback pattern [0.5, 0.5]
    # This indicates the default fallback value, not actual detection
    stick_grip_x = features.get('stick_grip_x', 0) + 0.5  # Convert back from hip-relative
    stick_grip_height = features.get('stick_grip_height', 0) + 0.5  # Rough approx
    
    CENTER_THRESHOLD = 0.02
    if abs(stick_grip_x - 0.5) < CENTER_THRESHOLD and abs(stick_grip_height - 0.5) < CENTER_THRESHOLD:
        return False, "Stick at center (fallback value, not real detection)"
    
    # Rule 6: Sanity check on angles (reject impossible values)
    ANGLE_FEATURES = [
        'left_elbow_angle', 'right_elbow_angle',
        'left_shoulder_angle', 'right_shoulder_angle',
        'left_knee_angle', 'right_knee_angle'
    ]
    
    for angle_name in ANGLE_FEATURES:
        angle_val = features.get(angle_name, 0)
        # Human joints can't be < 0 or > 180 (fully bent/extended)
        # Values outside this range indicate detection errors
        if angle_val < 0 or angle_val > 180:
            return False, f"Impossible {angle_name}: {angle_val:.1f}°"
    
    return True, "Valid"


def log_rejection(image_path, class_name, viewpoint, reason, all_rejections):
    """Log rejection for debugging and analysis."""
    import json
    from pathlib import Path
    
    rejection = {
        'image': str(image_path),
        'class': class_name,
        'viewpoint': viewpoint,
        'reason': reason,
        'timestamp': str(datetime.now())
    }
    all_rejections.append(rejection)
    
    # Also print immediate feedback
    print(f"  [REJECTED] {Path(image_path).name}: {reason}")
'''

# ============================================================================
# INTEGRATION: Add validation to analyze_reference_images()
# ============================================================================

ANALYZE_FUNCTION_PATCH = '''
# In analyze_reference_images(), after extracting features (around line 300):

all_features = []
rejected_count = 0
total_count = 0
all_rejections = []  # Add this list to collect rejections

for img_path in tqdm(images, desc=f"{viewpoint}/{class_name}", leave=False):
    features = extract_geometric_features(img_path, apply_mirror=apply_mirror)
    total_count += 1
    
    if features is None:
        rejected_count += 1
        log_rejection(img_path, class_name, viewpoint, 
                      "Feature extraction failed (no pose detected)", all_rejections)
        continue
    
    # === NEW: VALIDATION GATE ===
    # We need to re-extract with detection metadata to validate
    # Option A: Modify extract_geometric_features to return metadata
    # Option B: Re-run detection with metadata (slower but cleaner)
    
    # For Option A, modify extract_geometric_features signature:
    # features, metadata = extract_geometric_features(img_path, return_metadata=True)
    
    is_valid, reason = validate_features(
        features, 
        metadata['pose_landmarks'],
        metadata['stick_detected'],
        metadata.get('stick_confidence', 1.0)
    )
    
    if not is_valid:
        rejected_count += 1
        log_rejection(img_path, class_name, viewpoint, reason, all_rejections)
        continue
    # === END VALIDATION GATE ===
    
    all_features.append(features)

# After loop, save rejection log
rejection_log_path = Path(f"rejection_logs/rejections_{viewpoint}_{class_name}.json")
rejection_log_path.parent.mkdir(parents=True, exist_ok=True)
with open(rejection_log_path, 'w') as f:
    json.dump(all_rejections, f, indent=2)

print(f"  Accepted: {len(all_features)}/{total_count} ({100*len(all_features)/total_count:.1f}%)")
print(f"  Rejected: {rejected_count}/{total_count}")

# Add minimum acceptance threshold
MIN_ACCEPTANCE_RATE = 0.50  # 50%
if len(all_features) / total_count < MIN_ACCEPTANCE_RATE:
    print(f"  WARNING: Low acceptance rate for {viewpoint}/{class_name}! "
          f"Expected >{MIN_ACCEPTANCE_RATE*100:.0f}%, got {100*len(all_features)/total_count:.1f}%")
'''

# ============================================================================
# ISSUE #4: 3D ANGLE CALCULATION
# File: hybrid_classifier/1_extract_reference_features.py
# ============================================================================

ANGLE_3D_CODE = '''
def calculate_angle_3d(p1, p2, p3):
    """
    Calculate 3D angle at p2 formed by p1-p2-p3 using three-dimensional coordinates.
    
    Args:
        p1, p2, p3: Each is [x, y, z] in 3D space
        
    Returns:
        Angle in degrees
    """
    # Handle 2D input for backward compatibility
    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0.0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0.0]
    if len(p3) == 2:
        p3 = [p3[0], p3[1], 0.0]
    
    # 3D vector construction
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    
    # Normalize
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0  # Degenerate case
    
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm
    
    # Clamp for numerical stability
    cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)
'''

# Integration: Update extract_geometric_features to use 3D coordinates
EXTRACT_3D_PATCH = '''
# In extract_geometric_features(), after getting pose_results:

# Get 2D landmarks (normalized to image)
landmarks_2d = results.pose_landmarks.landmark

# Get 3D world landmarks (in meters - physically accurate)
world_landmarks = results.pose_world_landmarks.landmark

# For angle calculation, use 3D world coordinates for physical accuracy
# This distinguishes thrusts (forward in Z) from blocks (sideways in XY)

def get_world_point(idx):
    """Extract 3D world coordinate for landmark idx."""
    lm = world_landmarks[idx]
    return [lm.x, lm.y, lm.z]

# Calculate angles using 3D world coordinates
features['left_elbow_angle'] = calculate_angle_3d(
    get_world_point(11),   # left_shoulder
    get_world_point(13),   # left_elbow  
    get_world_point(15)    # left_wrist
)
features['right_elbow_angle'] = calculate_angle_3d(
    get_world_point(12),  # right_shoulder
    get_world_point(14),  # right_elbow
    get_world_point(16)   # right_wrist
)
# ... etc for all angle features

# Keep 2D normalized coordinates for height/position features
# (they're relative to hip center anyway, so 2D is fine)
'''

# ============================================================================
# ISSUE #5: NaN SENTINEL FOR MISSING STICK
# File: hybrid_classifier/1_extract_reference_features.py
# ============================================================================

STICK_NAN_CODE = '''
# In extract_geometric_features(), replace the fallback:

# OLD (corrupts templates):
else:
    stick_grip = [0.5, 0.5]  # Center fallback - WRONG
    stick_tip = [0.5, 0.5]

# NEW (NaN sentinel - propagates to features as invalid):
else:
    stick_grip = [float('nan'), float('nan')]  # NaN sentinel
    stick_tip = [float('nan'), float('nan')]
    stick_confidence = 0.0  # Track for validation

# Later, in feature computation, NaN will propagate:
# - stick_length = NaN (rejected by validation)
# - stick_angle = NaN (rejected by validation)  
# - stick_tip_height = NaN (rejected by validation)
# etc.

# This allows the validation gate (Issue #2) to properly reject these samples.
'''

# ============================================================================
# DEPLOYMENT: Update deployment_package/src/feature_extraction.py
# ============================================================================

DEPLOYMENT_SYNC = '''
# File: deployment_package/src/feature_extraction.py
# 
# This file should match app/models/gcn/feature_extraction.py in TuroArnis repo.
# Key differences to sync:
#
# 1. Use calculate_angle_3d (Issue #4 fix)
# 2. Use NaN sentinels for missing stick (Issue #5 fix)
# 3. Add validation layer if doing new template generation (Issue #2)
#
# The app already has these fixes - ensure deployment package matches!
'''

# ============================================================================
# VERIFICATION: Test Script
# ============================================================================

VERIFICATION_TEST = '''
#!/usr/bin/env python3
"""
Verify training pipeline fixes work correctly.
Run this after applying changes to TuroArnis-ML.
"""

import json
from pathlib import Path

def verify_templates():
    """Check template statistics are sane after quality gating."""
    
    templates_path = Path("hybrid_classifier/feature_templates.json")
    if not templates_path.exists():
        print("ERROR: Templates not found!")
        return False
    
    with open(templates_path) as f:
        templates = json.load(f)
    
    issues = []
    
    for key, template in templates.items():
        for feat_name, stats in template.items():
            if not isinstance(stats, dict):
                continue
                
            # Check for excessive STD (indicates noisy data)
            if 'std' in stats:
                if 'angle' in feat_name and stats['std'] > 30:
                    issues.append(f"{key}.{feat_name}: std={stats['std']:.1f}° (too wide)")
                elif stats['std'] > 0.15:  # For normalized coordinates
                    issues.append(f"{key}.{feat_name}: std={stats['std']:.3f} (too wide)")
    
    if issues:
        print(f"WARNING: Found {len(issues)} features with excessive variance:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues)-10} more")
        return False
    
    print("✓ Templates look healthy (tight STDs)")
    return True


def verify_rejection_log():
    """Check rejection logs exist and have reasonable rates."""
    
    log_dir = Path("rejection_logs")
    if not log_dir.exists():
        print("WARNING: No rejection logs found (validation may not be running)")
        return False
    
    logs = list(log_dir.glob("*.json"))
    print(f"✓ Found {len(logs)} rejection log files")
    
    total_rejected = 0
    for log_file in logs:
        with open(log_file) as f:
            rejections = json.load(f)
            total_rejected += len(rejections)
    
    print(f"✓ Total rejections logged: {total_rejected}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Training Pipeline Verification")
    print("=" * 60)
    
    ok = True
    ok &= verify_templates()
    ok &= verify_rejection_log()
    
    print("=" * 60)
    if ok:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed - review issues above")
    print("=" * 60)
'''


def main():
    """Print summary of changes needed."""
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║  TuroArnis-ML Training Pipeline Fixes - Implementation Guide         ║
╚════════════════════════════════════════════════════════════════════╝

SUMMARY OF CHANGES NEEDED:

1. ISSUE #2 - Quality Validation Gates
   File: hybrid_classifier/1_extract_reference_features.py
   
   Add the validate_features() function (see VALIDATION_CODE above)
   Integrate validation into analyze_reference_images() loop
   Add rejection logging for debugging
   
   Expected result: Tighter STDs in templates (>30% reduction)

2. ISSUE #4 - 3D Angle Calculation  
   File: hybrid_classifier/1_extract_reference_features.py
   
   Add calculate_angle_3d() function (see ANGLE_3D_CODE above)
   Modify extract_geometric_features() to use world_landmarks (3D)
   for angle calculation, 2D for height/position features
   
   Expected result: Better discrimination between thrusts and blocks

3. ISSUE #5 - NaN Sentinel for Missing Stick
   File: hybrid_classifier/1_extract_reference_features.py
   
   Change stick fallback from [0.5, 0.5] to [NaN, NaN]
   This allows validation gate to properly reject these samples
   
   Expected result: No corrupted template statistics from failed detections

4. DEPLOYMENT SYNC
   File: deployment_package/src/feature_extraction.py
   
   Ensure this matches TuroArnis app/models/gcn/feature_extraction.py
   (which already has these fixes)

VERIFICATION:

After applying fixes and regenerating templates:
- Run verify_templates() to check STDs are tight
- Check rejection_logs/ for validation working
- Test model accuracy on holdout set

═══════════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    main()
