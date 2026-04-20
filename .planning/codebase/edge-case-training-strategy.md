---
title: Edge-Case Training Strategy for Robustness
planted_date: 2026-04-18
trigger_condition: When basic quality-gated template generation works and lesson mode classification accuracy improves
source: /gsd-explore training pipeline optimization
---

## Concept

Once clean, quality-gated templates are working and lesson mode classification is accurate for "ideal" poses, consider collecting intentionally challenging examples as a separate "robustness" dataset.

## Why Separate?

The exploration revealed that mixing clean and noisy examples in one template set pollutes the ground truth. Wide STDs from corrupted features make the classifier non-discriminative.

Instead of reverting to inclusive training, maintain **two distinct data streams**:

1. **Clean Templates** (current priority)
   - Only verified-successful detections
   - Tight STDs, high discrimination
   - Used for primary classification

2. **Robustness Dataset** (future seed)
   - Intentionally challenging cases: occluded sticks, extreme angles, partial visibility
   - Tagged with detection quality metadata
   - Used for:
     - Confidence calibration ("when should the model say 'uncertain'?")
     - Fallback classification strategies (e.g., "stick unavailable → rely on body pose only")
     - Active learning triggers ("this failed consistently — add to training")

## What Counts as "Challenging"

- Stick partially occluded by body during overhead thrusts
- Extreme camera angles (low/high) that distort perspective
- Users with different body proportions, mobility aids
- Motion blur from fast techniques
- Low lighting conditions

## How to Use It Differently

Unlike clean templates that become Gaussian statistics, robustness data might:
- Train a separate "uncertainty estimator" head on the GCN
- Feed into a secondary classifier that handles degraded inputs
- Inform dynamic threshold adjustment (lower threshold when stick detection fails)
- Generate synthetic examples for data augmentation

## Success Criteria (Trigger Condition)

Implement when:
- [ ] Quality-gated template generation is deployed
- [ ] Lesson mode achieves >85% accuracy on ideal poses
- [ ] Classification failures correlate with specific degradation patterns (occlusion, angle, lighting)
- [ ] User feedback indicates "it works when I'm clear, fails when I'm occluded"

## Risk: Don't Do This Too Early

Adding robustness data before clean templates work will:
- Re-introduce the STD-widening problem we just solved
- Mask the root cause with "training on noise"
- Make debugging harder by conflating two failure modes

Wait for clean baseline first.
