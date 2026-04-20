# Quick Task: Fix Training Pipeline Incompatibilities

## Overview
Fix 5 critical incompatibilities between TuroArnis-ML training pipeline and TuroArnis app inference causing classification failures.

## Issues to Fix

### 1. Class Name Mismatch (P0)
**Problem:** Training uses 12 classes, app expects 13 (missing 'neutral')
- Training: 12 technique classes
- App: 12 techniques + 'neutral' class

**Fix:** Remove 'neutral' from app CLASS_NAMES to match training
**Files:** `app/models/gcn/model_architecture.py`

### 2. Missing Quality Validation Gates (P0)
**Problem:** Template generation accepts all features without validation
- Failed YOLO detections corrupt templates with center-fallback [0.5, 0.5]
- Wide STDs (60°+) make classifier non-discriminative

**Fix:** Add validation layer to reject failed detections before template stats
**Files:** 
- `hybrid_classifier/1_extract_reference_features.py` (in TuroArnis-ML repo)

### 3. Coordinate Space Inconsistency (P1)
**Problem:** Training normalizes to full frame, app normalizes to person crop

**Investigate first:** Debug coordinate mismatch between training and app

### 4. 2D-Only Angle Calculation (P1)
**Problem:** Training ignores z-depth from MediaPipe world landmarks

**Fix:** Update calculate_angle() to use 3D coordinates

### 5. Stick Fallback Handling (P2)
**Problem:** Training uses [0.5, 0.5] center fallback, app uses NaN sentinels

**Fix:** Align both to use NaN sentinel approach

## Execution Order
1. Issue #1 - Class mismatch (quick win)
2. Issue #3 - Coordinate investigation (decision point)
3. Issue #2 - Validation gates (in ML repo)
4. Issue #4 - 3D angles (in ML repo)
5. Issue #5 - Stick fallback (in ML repo)

## Success Criteria
- [ ] App CLASS_NAMES matches training (12 classes)
- [ ] Templates regenerate with tighter STDs (<30° for angles)
- [ ] Coordinate mismatch resolved
- [ ] Model retrained with clean data (optional, can use existing model for now)

## Risk: Low
These are alignment fixes, not algorithmic changes. Safe to implement incrementally.
