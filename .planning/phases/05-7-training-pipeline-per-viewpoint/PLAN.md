# Phase 5.7-Per-Viewpoint: Per-Viewpoint Model Retraining

## Objective

Retrain HybridGCN models per viewpoint (front, left, right) to achieve 70%+ validation accuracy on each viewpoint independently. This is a pivot from the failed merged model approach.

## Background

**Previous Attempt (Merged Model):**
- Attempt 1 (dropout 0.7): 24.9% accuracy - FAILED
- Attempt 2 (dropout 0.5, Option A): 37.6% accuracy - PARTIAL
- Attempt 3 (dropout 0.3, 256 hidden): 9.0% accuracy - FAILED

**Key Insight:** The merged model cannot effectively learn from combined viewpoints. Per-viewpoint models (trained separately) have historically performed better.

**Existing Assets:**
- Quality-gated templates generated (39 templates, 50-70% acceptance)
- Training script with `--viewpoint` flag support
- Feature generation script with `--viewpoint` flag support
- Old per-viewpoint feature files (Feb 19) - need regeneration

---

## Plan Overview

| Phase | Task | Duration | Checkpoint |
|-------|------|----------|------------|
| 1 | Reset training script to Option A config | 5 min | Gate 1 |
| 2 | Regenerate per-viewpoint features | 30 min | Gate 2 |
| 3 | Train front viewpoint model | 25 min | Gate 3 |
| 4 | Train left viewpoint model | 25 min | Gate 4 |
| 5 | Train right viewpoint model | 25 min | Gate 5 |
| 6 | Verify all models 70%+ | 5 min | Gate 6 |
| 7 | Deploy to TuroArnis | 5 min | - |
| 8 | Integration test | 10 min | - |

**Total Duration:** ~130 minutes (2 hours 10 minutes)

---

## Checkpoint Gates

### Gate 1: Configuration Reset ✅
**Trigger:** Before feature generation  
**Verify:**
- [ ] Training script has Option A hyperparameters
- [ ] DROPOUT = 0.5 (not 0.3 or 0.7)
- [ ] LEARNING_RATE = 0.005
- [ ] HIDDEN_DIM = 128
- [ ] WEIGHT_DECAY = 5e-5

**Decision:** PASS → Continue to feature generation  
**Decision:** FAIL → Fix configuration, retry Gate 1

---

### Gate 2: Feature Generation Complete ✅
**Trigger:** After all 3 viewpoint feature files generated  
**Verify:**
- [ ] `train_features_front.pt` exists and >2MB
- [ ] `train_features_left.pt` exists and >2MB  
- [ ] `train_features_right.pt` exists and >2MB
- [ ] Each file has ~2000-2600 samples
- [ ] Files are newly generated (< 1 hour old)

**Decision:** PASS → Continue to front model training  
**Decision:** FAIL → Debug feature generation, check dataset_split structure

---

### Gate 3: Front Model Trained ✅
**Trigger:** After front viewpoint training completes  
**Verify:**
- [ ] `model_front.pth` exists
- [ ] Validation accuracy ≥ 70%
- [ ] No severe overfitting (train-test gap < 20%)
- [ ] Training converged (not stopped prematurely)

**Decision:** PASS (≥70%) → Continue to left model  
**Decision:** RETRY (60-70%) → Adjust hyperparameters (LR 0.005→0.01), retrain  
**Decision:** INVESTIGATE (<60%) → Check feature quality, debug training

---

### Gate 4: Left Model Trained ✅
**Trigger:** After left viewpoint training completes  
**Verify:**
- [ ] `model_left.pth` exists
- [ ] Validation accuracy ≥ 70%
- [ ] No severe overfitting (train-test gap < 20%)

**Decision:** PASS (≥70%) → Continue to right model  
**Decision:** RETRY (60-70%) → Adjust hyperparameters, retrain  
**Decision:** INVESTIGATE (<60%) → Check feature quality

---

### Gate 5: Right Model Trained ✅
**Trigger:** After right viewpoint training completes  
**Verify:**
- [ ] `model_right.pth` exists
- [ ] Validation accuracy ≥ 70%
- [ ] No severe overfitting (train-test gap < 20%)

**Decision:** PASS (≥70%) → Continue to deployment  
**Decision:** RETRY (60-70%) → Adjust hyperparameters, retrain  
**Decision:** INVESTIGATE (<60%) → Check feature quality

---

### Gate 6: All Models Validated ✅
**Trigger:** After all 3 models trained  
**Verify:**
- [ ] Front model: ≥70% validation accuracy
- [ ] Left model: ≥70% validation accuracy  
- [ ] Right model: ≥70% validation accuracy
- [ ] At least 2/3 models ≥70% (for ensemble fallback)

**Decision:** ALL PASS → Proceed to deployment  
**Decision:** 2/3 PASS → Deploy with fallback to per-viewpoint inference  
**Decision:** <2 PASS → Investigate failing viewpoints, consider architecture changes

---

## Detailed Tasks

### Task 1: Reset Training Script to Option A (5 min)

**Files:** `hybrid_classifier/4c_train_hybrid_gcn_v2.py`

**Changes:**
```python
# Model architecture - Option A (Working Configuration)
HIDDEN_DIM = 128              # Not 256
NUM_LAYERS = 3                # Not 4
DROPOUT = 0.5                 # Not 0.3 or 0.7
NODE_EMBED_DIM = 16           # Not 32

# Training - Option A
LEARNING_RATE = 0.005         # Not 0.01
WEIGHT_DECAY = 5e-5           # Not 1e-5
EPOCHS = 150                  # Not 200
PATIENCE = 20                 # Not 30
BATCH_SIZE = 64               # Keep
MAX_OVERFIT_GAP = 35.0        # Keep
```

**Verification:**
```bash
grep "HIDDEN_DIM = 128" hybrid_classifier/4c_train_hybrid_gcn_v2.py
grep "DROPOUT = 0.5" hybrid_classifier/4c_train_hybrid_gcn_v2.py
```

---

### Task 2: Regenerate Front Viewpoint Features (10 min)

**Command:**
```bash
cd C:\Users\HP\Documents\GitHub\TuroArnis-ML
python hybrid_classifier\2b_generate_node_hybrid_features.py --viewpoint front
```

**Outputs:**
- `hybrid_classifier/hybrid_features_v3/train_features_front.pt`
- `hybrid_classifier/hybrid_features_v3/test_features_front.pt`

**Expected:**
- ~2500-2600 training samples
- ~200-300 test samples
- 13 classes (12 techniques + neutral)
- Tensor shapes: [N, 35, 6] node, [N, 30] hybrid

**Checkpoint:** Gate 2

---

### Task 3: Regenerate Left Viewpoint Features (10 min)

**Command:**
```bash
python hybrid_classifier\2b_generate_node_hybrid_features.py --viewpoint left
```

**Outputs:**
- `train_features_left.pt`
- `test_features_left.pt`

**Checkpoint:** Gate 2

---

### Task 4: Regenerate Right Viewpoint Features (10 min)

**Command:**
```bash
python hybrid_classifier\2b_generate_node_hybrid_features.py --viewpoint right
```

**Outputs:**
- `train_features_right.pt`
- `test_features_right.pt`

**Checkpoint:** Gate 2

---

### Task 5: Train Front Viewpoint Model (20-30 min)

**Command:**
```bash
python hybrid_classifier\4c_train_hybrid_gcn_v2.py --viewpoint front
```

**Outputs:**
- `hybrid_classifier/models/model_front.pth`
- `hybrid_classifier/models/history_front.json`

**Expected:**
- Validation accuracy: ≥70%
- Train accuracy: 60-80%
- Epochs: 80-150
- Gap: <20%

**Retry if 60-70%:**
- Increase LR: 0.005 → 0.01
- Reduce dropout: 0.5 → 0.4
- Retrain

**Checkpoint:** Gate 3

---

### Task 6: Train Left Viewpoint Model (20-30 min)

**Command:**
```bash
python hybrid_classifier\4c_train_hybrid_gcn_v2.py --viewpoint left
```

**Outputs:**
- `model_left.pth`
- `history_left.json`

**Checkpoint:** Gate 4

---

### Task 7: Train Right Viewpoint Model (20-30 min)

**Command:**
```bash
python hybrid_classifier\4c_train_hybrid_gcn_v2.py --viewpoint right
```

**Outputs:**
- `model_right.pth`
- `history_right.json`

**Checkpoint:** Gate 5

---

### Task 8: Verify All Models 70%+ (5 min)

**Script:**
```python
import json
from pathlib import Path

for viewpoint in ['front', 'left', 'right']:
    history_file = Path(f'hybrid_classifier/models/history_{viewpoint}.json')
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        best_acc = max(history.get('val_acc', [0]))
        print(f"{viewpoint}: {best_acc:.1f}%")
    else:
        print(f"{viewpoint}: NO HISTORY FILE")
```

**Success Criteria:**
- All 3 models ≥70%: Proceed to deployment
- 2/3 models ≥70%: Deploy with per-viewpoint fallback
- <2 models ≥70%: Investigate and retrain failing viewpoints

**Checkpoint:** Gate 6

---

### Task 9: Deploy to TuroArnis App (5 min)

**Commands:**
```bash
copy "hybrid_classifier\models\model_front.pth" "..\TuroArnis\app\models\gcn\"
copy "hybrid_classifier\models\model_left.pth" "..\TuroArnis\app\models\gcn\"
copy "hybrid_classifier\models\model_right.pth" "..\TuroArnis\app\models\gcn\"
copy "hybrid_classifier\feature_templates.json" "..\TuroArnis\app\models\gcn\"
```

**Verification:**
- Check file sizes:
  - model_*.pth: ~1.4MB each
  - feature_templates.json: ~150KB

---

### Task 10: Integration Test (10 min)

**Test Scenarios:**
1. **Front View:** Perform thrusts/blocks facing camera
2. **Left View:** Perform techniques with left side to camera  
3. **Right View:** Perform techniques with right side to camera
4. **Neutral:** Stand still, verify neutral detection

**Commands:**
```bash
cd C:\Users\HP\Documents\GitHub\TuroArnis
python app\test_classification.py
# or
python app\app.py
```

**Success Criteria:**
- Each viewpoint model classifies its own viewpoint correctly >70%
- No crashes during 2-3 minute test
- Smooth viewpoint switching if applicable

---

## Hyperparameter Reference

### Option A (Primary Configuration)
```python
HIDDEN_DIM = 128
DROPOUT = 0.5
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-5
EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 64
```

### Option B (Retry Configuration if 60-70%)
```python
HIDDEN_DIM = 128
DROPOUT = 0.4          # Reduced from 0.5
LEARNING_RATE = 0.01   # Increased from 0.005
WEIGHT_DECAY = 5e-5
EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 64
```

### Option C (Investigation Configuration if <60%)
```python
HIDDEN_DIM = 256       # More capacity
DROPOUT = 0.4
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-5    # Less regularization
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 32        # Smaller batches
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Feature generation fails | Check dataset_split directory exists with front/left/right subdirectories |
| Training crashes (OOM) | Reduce BATCH_SIZE to 32 |
| Model stuck at ~37% (like merged) | Switch to Option C with higher capacity |
| One viewpoint underperforms | Deploy 2/3 models with viewpoint detection fallback |
| All viewpoints <70% | Investigate feature quality, consider Random Forest baseline |
| Training takes >30 min per model | Accept longer time, don't reduce epochs |

---

## Success Criteria

### Phase Complete When:
- [ ] All 3 per-viewpoint feature files regenerated (< 1 hour old)
- [ ] Front model: ≥70% validation accuracy
- [ ] Left model: ≥70% validation accuracy  
- [ ] Right model: ≥70% validation accuracy
- [ ] Models deployed to TuroArnis app/models/gcn/
- [ ] Templates copied with version metadata
- [ ] Integration test passes (classification works for all 3 viewpoints)

### Partial Success (Acceptable):
- [ ] 2/3 models ≥70%
- [ ] Deployed with per-viewpoint detection
- [ ] Failing viewpoint marked for future improvement

---

## Notes

- **Do NOT compromise on training time** - Each viewpoint needs 20-30 minutes minimum
- **Do NOT use old feature files** - Must regenerate with new templates
- **Do NOT skip checkpoint gates** - Each gate validates before proceeding
- **Do NOT merge models** - Keep per-viewpoint separate
- **DO commit after each successful gate** - Preserve progress

---

## Execution Status

| Task | Status | Commit | Accuracy |
|------|--------|--------|----------|
| Task 1: Reset Config | ⏳ PENDING | - | - |
| Task 2: Front Features | ⏳ PENDING | - | - |
| Task 3: Left Features | ⏳ PENDING | - | - |
| Task 4: Right Features | ⏳ PENDING | - | - |
| Task 5: Front Model | ⏳ PENDING | - | - |
| Task 6: Left Model | ⏳ PENDING | - | - |
| Task 7: Right Model | ⏳ PENDING | - | - |
| Task 8: Verify 70%+ | ⏳ PENDING | - | - |
| Task 9: Deploy | ⏳ PENDING | - | - |
| Task 10: Integration | ⏳ PENDING | - | - |

**Last Updated:** 2026-04-22  
**Started By:** Phase 5.7 Per-Viewpoint Pivot  
**Context:** Merged model failed (9-37% accuracy), pivoting to per-viewpoint training
