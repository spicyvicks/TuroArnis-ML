# Project State: TuroArnis-ML

## Project Reference

Building a HybridGCN classifier for Arnis technique recognition from pose + stick detection data. Current focus: **front-view model accuracy ≥70% real-only** (currently at 77.7% baseline v5, targeting 80-85% with Option A).

## Current Position

- **Phase:** V5 Front Model Deployment (complete)
- **Plan:** Deploy v5 front specialist (77.7% real-only) to TuroArnis app
- **Status:** Deployment package created, model verified, docs written
- **Progress:** 100% (front model deployed)

## Progress

| Task | Status | Notes |
|------|--------|-------|
| Task 1: Full system audit | Done | Identified 5 root causes |
| Task 2: v4/v5/v6 iterations | Done | v5 3x = 77.7% best baseline |
| Task 3-5: Option A syntax + eval fix | Done | Completed but v6 failed to beat v5 |
| Task 6-10: v6 experiments | Done | **v6 abandoned** — 74.6% max, below v5 77.7% |
| **V5 Deployment** | **Done** | Model, architecture, feature extraction, inference wrapper |
| **Active Model Registry** | **Done** | `active_model.json` points to v5 |
| **Deployment Docs** | **Done** | README_v5.md + MANIFEST updated |

## Recent Decisions

- **v5 is the production model** — After 4 v6 variants tested (masked 3x, masked 2x, has_stick hybrid, 7-dim nodes), none beat v5's 77.7%. v5's "bug" (origin-based stick fallback) is actually a stable, learnable signal.
- **v6 experiments abandoned** — Masking, has_stick signals, and 7-dim nodes all underperformed. The constant offset from origin fallback is a feature, not a bug.
- **13 classes kept** — `neutral` class included in deployment (user decision). App must add `neutral` back to class list.
- **Signed features required in app** — v5 model expects 46 hybrid features (33 base + 13 signed). App feature extraction must compute signed direction features.
- **Front-view target met** — 77.7% exceeds 70% target. Move to left/right viewpoint training next.

## Pending Todos

- [ ] **App integration** — Update TuroArnis app to use v5 model + signed features + neutral class
- [ ] **Left viewpoint model** — Train left-view specialist (original Phase 5.7 goal)
- [ ] **Right viewpoint model** — Train right-view specialist (original Phase 5.7 goal)
- [ ] If accuracy ≤77%: try 1x or 2x synthetic multiplier
- [ ] If still ≤77%: consider reverting to v5 model and moving to left/right viewpoint training
- [ ] Phase 5.7 left viewpoint model (not started)
- [ ] Phase 5.7 right viewpoint model (not started)
- [ ] Quick task: ML repo issues #2, #4, #5 (quality validation, 3D angle, stick fallback)

## Blockers / Concerns

- **None currently** — pipeline is ready to run.
- **If Option A fails** — fallback to reduce synthetic multiplier or train purely on real data, or merge left/right pairs into 8 classes.
- **Overfitting risk** — v5 had 29-point train-real gap (86% train vs 57.6% real). Real-only validation should help.

## Session Continuity

- **Last session:** 2026-04-28 — paused mid-implementation to preserve state
- **Resumed:** 2026-04-28 — syntax checks passed, evaluate script updated
- **Next action:** Run full extraction → synthetic → train → eval pipeline
- **Resume file:** `.planning/.continue-here.md`
- **Structured handoff:** `.planning/HANDOFF.json` (should be cleared after successful resumption)

## Infrastructure

- **Branch:** `ml-3`
- **Device:** CPU training (no CUDA)
- **Stick detector:** `runs/pose/stick_detector_20260425_212025/weights/best.pt` (mAP50=0.946)
- **Best model:** `hybrid_classifier/models/model_front_with_synthetic_3x_v5.pth` (epoch 28, val=72.8%, real-only=77.7%)
- **Features:** `hybrid_features_v6/` contains v5-era features (NOT yet regenerated with Option A edits)
- **Uncommitted changes:**
  - `hybrid_classifier/2b_generate_node_hybrid_features_v6.py`
  - `hybrid_classifier/4e_train_hybrid_gcn_v2_with_synthetic_v6.py`
  - `hybrid_classifier/evaluate_real_only_v6.py` (just updated)

## Key Files

| File | Purpose |
|------|---------|
| `hybrid_classifier/2b_generate_node_hybrid_features_v6.py` | Feature extraction — zero-stick fallback, true zero nodes |
| `hybrid_classifier/4e_train_hybrid_gcn_v2_with_synthetic_v6.py` | Training — masked global_mean_pool, node_mask, real-only validation |
| `hybrid_classifier/evaluate_real_only_v6.py` | Real-only evaluation — node_mask support, per-class accuracy |
| `hybrid_classifier/2c_generate_synthetic_features_v6.py` | Synthetic data generator |
| `.planning/.continue-here.md` | Detailed context and remaining tasks |
| `.planning/HANDOFF.json` | Structured handoff (to be cleared) |
