# Front Model Redo — Checkpoint Plan
# Created: 2026-04-27
# Status: In Progress — Deleting old augmented data
# Branch: ml-3

## User Decisions

1. **Synthetic ratios:** Generate BOTH 1x and 3x simultaneously, train both, compare.
2. **Mirroring:** YES — training images must match camera view. Camera sees mirrored person (my left = camera's right). This is NOT augmentation; it's preprocessing alignment.
3. **Horizontal flip augmentation:** REMOVE entirely. No random flips.
4. **Target:** Get front model to 70%+ real-only test accuracy.

## Mirroring Strategy

**Problem:** The app camera shows a mirrored view. Raw photos have the person's left on the image's left. The mirrored camera shows person's left on image's right. Train and test MUST be in the same coordinate space.

**Solution:**
- All training images in `dataset_split/train/front/` are PERMANENTLY flipped horizontally ONCE to match camera view. This is preprocessing, not augmentation.
- Labels DO NOT change. `left_chest_thrust` still means the technique uses the left side. The model learns that in camera space, left techniques appear on the right side of the image.
- This matches what `0c_flip_test_images.py` already does for test data.
- Reference poses should also be flipped (or use `feature_templates_mirrored.json` if already generated).

**Why this is safe:**
- Every image in the dataset gets the SAME transformation (consistent, not random).
- No label swapping needed because the transformation is uniform across the entire dataset.
- The model learns camera-space semantics: "stick on image-right = left technique" — this is exactly what the live app will see.

## Why Remove Horizontal Flip Augmentation

**Current bug:** `0c_augment_training_data.py` randomly flips 50% of augmented images but keeps labels the same. This means:
- Some `left_chest_thrust` images show stick on left side (not flipped)
- Some `left_chest_thrust` images show stick on right side (flipped)
- Model sees contradictory visual patterns for the same label

**Result:** Side-positioned classes (where body pose doesn't disambiguate left/right) fail at 0-7% accuracy.

**Fix:** Consistent mirroring for ALL images + no random flip augmentation.

## Execution Steps

### Step 1: Clean Slate
- [x] Delete all `_aug*.jpg` / `_aug*.png` files from `dataset_split/train/front/*/` (1,683 files)
- [ ] Permanently mirror ALL original images in `dataset_split/train/front/` to match camera view
- [ ] Verify ~512 original images remain after deletion, all now mirrored

### Step 2: No-Flip Augmentation
- [ ] Generate 2 augmented copies per original (AUGMENT_FACTOR=2)
- [ ] Augmentations: Rotate ±10°, Scale 0.85-1.15x, Perspective, Brightness/Contrast, GaussNoise, Blur
- [ ] NO horizontal flip in augmentation pipeline
- [ ] Expected: ~512 originals + ~1,024 augmented = ~1,536 total images

### Step 3: Templates
- [ ] Flip reference poses to match camera view (if not already done)
- [ ] Regenerate `feature_templates.json` from mirrored reference poses
- [ ] Verify 39 templates, 100% neutral acceptance

### Step 4: Feature Extraction
- [ ] Run `2b_generate_node_hybrid_features.py --viewpoint front` on mirrored + augmented dataset
- [ ] Expected output: `train_features_front.pt` (~1,500 samples), `test_features_front.pt` (~200 samples)

### Step 5: Synthetic Generation (Dual Track)
- [ ] Run `2c_generate_synthetic_features.py --train_factor 1` → `synthetic_train_features_front_1x.pt`
- [ ] Run `2c_generate_synthetic_features.py --train_factor 3` → `synthetic_train_features_front_3x.pt`
- [ ] Both use same real features, same perturbation sigma=0.02
- [ ] Combine real + 1x synthetic → `combined_synthetic_train_features_front_1x.pt`
- [ ] Combine real + 3x synthetic → `combined_synthetic_train_features_front_3x.pt`

### Step 6: Training Track A (1x Synthetic)
- [ ] Train with `combined_synthetic_train_features_front_1x.pt`
- [ ] Hyperparameters: DROPOUT=0.5, LR=0.005, HIDDEN_DIM=128, WEIGHT_DECAY=5e-5
- [ ] Evaluate real-only test
- [ ] Expected: Less overfitting than 3x, train-real gap ~15-20%

### Step 7: Training Track B (3x Synthetic)
- [ ] Train with `combined_synthetic_train_features_front_3x.pt`
- [ ] Same hyperparameters as Track A
- [ ] Evaluate real-only test
- [ ] Expected: More overfitting, train-real gap ~25-30%

### Step 8: Compare and Decide
- [ ] Compare real-only test accuracy: 1x vs 3x
- [ ] Compare train-real gap: smaller gap wins
- [ ] Compare per-class accuracy, especially side classes
- [ ] Pick winner → this becomes the front model

### Step 9: Deploy or Iterate
- [ ] If winner ≥70%: Save as `model_front.pth`, proceed to left/right viewpoints
- [ ] If winner 60-70%: Apply stronger regularization (dropout 0.7, AdamW, Focal Loss γ=1.5)
- [ ] If winner <60%: Investigate feature quality or architecture

## Why 10x Synthetic Would Fail

The synthetic generator only applies 2% joint noise + nearest-neighbor mixup. These are near-duplicates of the ~2,000 real samples. More synthetic = more memorization of perturbation artifacts, not more generalization. Evidence: 3x already shows 29-point train-real gap (86% train vs 56.6% real). 10x would make this worse, not better.

## Files to Modify

- `0c_augment_training_data.py` — Remove HorizontalFlip, adjust pipeline for front view only
- `hybrid_classifier/2b_generate_node_hybrid_features.py` — May need to remove internal flip logic if any
- `hybrid_classifier/1_extract_reference_features.py` — Use mirrored templates
- `hybrid_classifier/4e_train_hybrid_gcn_v2_with_synthetic.py` — Train twice with different combined files

## Blockers Cleared

- ✅ No random horizontal flip augmentation (replaced with consistent mirroring)
- ✅ Side classes should no longer see contradictory visual patterns
- ✅ Train/test coordinate space aligned (both mirrored)
- ✅ 1x vs 3x synthetic comparison planned

## Current Session Notes

- 1,683 augmented files found in `dataset_split/train/front/`
- 512 original images expected after cleanup
- All augmented files have `_aug` in filename
- Existing `0c_remove_augmented_data.py` can do global cleanup, but we're targeting front only
