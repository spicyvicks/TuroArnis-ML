# Plan B: Add MediaPipe Hand Landmarks (77 Nodes Total)

## Goal
Give the model **hand geometry** (21 keypoints per hand) to distinguish left/right techniques and thrust/block dynamics. The current model only sees **wrist position** (nodes 15, 16), which varies significantly with arm length across different people.

## Expected Outcome
| Metric | Current | Target |
|--------|---------|--------|
| Val accuracy | 57.6% | **≥65%** |
| Left/right recall | 0–7% | **≥60%** |
| Block vs. thrust accuracy | ~50% | **≥75%** |

## Context: Why This Plan Exists

The test set consists of **different people, locations, and times** than training. Wrist position alone is **not person-invariant**:
- A 5'2" person and a 6'0" person have very different absolute wrist positions for the same technique
- **Hand geometry relative to the stick** is much more consistent across body sizes
- Finger curl, hand orientation, and hand-stick proximity are the defining visual features of Arnis techniques

MediaPipe Hands detects **21 landmarks per hand** (42 total), providing rich structural signal that the current 35-node graph completely discards.

## Implementation Steps

---

### Step 1: Integrate MediaPipe Hands into Feature Extraction
**File:** `hybrid_classifier/2b_generate_node_hybrid_features.py`

**New dependency:** Already available via `mediapipe` package (`mp.solutions.hands`).

**Function to add:**
```python
import mediapipe as mp
mp_hands = mp.solutions.hands

def extract_hand_landmarks(image, pose_keypoints):
    """
    Run MediaPipe Hands and align to pose wrists.
    Returns [42, 4] array: 21 right-hand + 21 left-hand landmarks.
    Missing hands filled with zeros, visibility=0.
    """
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3  # Lower threshold for front view occlusion
    )
    
    results = hands_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    hands_detector.close()
    
    right_hand = np.zeros((21, 4), dtype=np.float32)
    left_hand = np.zeros((21, 4), dtype=np.float32)
    
    if results.multi_hand_landmarks:
        for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            # MediaPipe Hands "handedness" is image-based (left/right of image)
            # We align to pose wrists by proximity
            wrist_lm = hand_landmarks.landmark[0]  # Hand wrist landmark
            hand_wrist = np.array([wrist_lm.x, wrist_lm.y, wrist_lm.z])
            
            dist_to_r = np.linalg.norm(hand_wrist - pose_keypoints[16, :3])
            dist_to_l = np.linalg.norm(hand_wrist - pose_keypoints[15, :3])
            
            target_array = right_hand if dist_to_r < dist_to_l else left_hand
            
            for i, lm in enumerate(hand_landmarks.landmark):
                target_array[i] = [lm.x, lm.y, lm.z, lm.visibility]
    
    return np.vstack([right_hand, left_hand])
```

**Critical: Hand-to-Pose Alignment**
MediaPipe Hands and MediaPipe Pose use the same normalized `[0,1]` image coordinate system, but the "wrist" in Hands (landmark 0) may not exactly match Pose wrist (landmark 15/16). We align by **minimum Euclidean distance** — whichever hand wrist is closer to the pose right wrist becomes the "right hand" for our graph.

**Fallback for missing hands:**
If a hand is not detected (common in front view when stick occludes hand), fill with zeros and set visibility=0. The model must learn to handle missing hand nodes via the visibility channel.

---

### Step 2: Update Graph Structure (77 Nodes Total)

**New node layout:**
| Node Indices | Content | Count | Notes |
|-------------|---------|-------|-------|
| 0–32 | MediaPipe Pose | 33 | Body skeleton |
| 33–34 | Stick grip, tip | 2 | From YOLO or fallback |
| 35–55 | Right hand landmarks | 21 | Thumb(0-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20) |
| 56–76 | Left hand landmarks | 21 | Same structure, offset by 21 |

**New edges to add in `GraphDataset.__getitem__`:**

```python
HAND_EDGES = []

# Right hand internal connections (simplified skeleton)
RIGHT_HAND_INTERNAL = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index finger
    (0,9),(9,10),(10,11),(11,12),  # middle finger
    (0,13),(13,14),(14,15),(15,16), # ring finger
    (0,17),(17,18),(18,19),(19,20), # pinky
]
for a, b in RIGHT_HAND_INTERNAL:
    HAND_EDGES.append((35 + a, 35 + b))
    HAND_EDGES.append((35 + b, 35 + a))

# Left hand internal (same structure, offset by 21)
for a, b in RIGHT_HAND_INTERNAL:
    HAND_EDGES.append((56 + a, 56 + b))
    HAND_EDGES.append((56 + b, 56 + a))

# Hand-to-pose wrist connections
HAND_EDGES.append((16, 35))  # right wrist to right hand root
HAND_EDGES.append((35, 16))
HAND_EDGES.append((15, 56))  # left wrist to left hand root
HAND_EDGES.append((56, 15))
```

**Dynamic hand-to-stick edge (per-sample):**
```python
# In GraphDataset.__getitem__
grip_pos = node_features[33, :3]  # stick grip
min_dist = float('inf')
nearest_node = -1

for hand_node in range(35, 77):  # all hand nodes
    if node_features[hand_node, 3] > 0.1:  # visibility check
        dist = np.linalg.norm(grip_pos - node_features[hand_node, :3])
        if dist < min_dist:
            min_dist = dist
            nearest_node = hand_node

if nearest_node >= 0 and min_dist < 0.15:  # threshold in normalized coords
    extra_edges = torch.tensor([[33, nearest_node], [nearest_node, 33]])
    edge_index = torch.cat([edge_index, extra_edges.t()], dim=1)
```

---

### Step 3: Update Node Features for Hands

**File:** `hybrid_classifier/2b_generate_node_hybrid_features.py`

Hand landmarks must use the **same person-normalized coordinates** as pose nodes (see Plan A, Step 2):

```python
# Hip center and shoulder width from pose (used as person scale)
hip_center = (pose_keypoints[23, :3] + pose_keypoints[24, :3]) / 2
shoulder_width = np.linalg.norm(pose_keypoints[11, :3] - pose_keypoints[12, :3])

# For each hand landmark
for i, kpt in enumerate(hand_keypoints):
    x, y, z, vis = kpt
    rel = (np.array([x, y, z]) - hip_center) / (shoulder_width + 1e-8)
    features.append([x, y, z, vis, rel[0], rel[1], rel[2], shoulder_width])
```

This ensures hand positions are **relative to body scale**, not absolute image coordinates. A right-hand index finger at `rel_x=+0.3` means "30% of shoulder-width to the right of the hip center" — invariant to camera distance and body height.

---

### Step 4: Update Synthetic Feature Generation

**File:** `hybrid_classifier/2c_generate_synthetic_features.py`

**Modify `apply_joint_perturbation`:**
```python
# Old: only arm joints
ARM_JOINTS = [11, 12, 13, 14, 15, 16, 33, 34]

# New: include hand joints
HAND_JOINTS = list(range(35, 77))  # 42 hand nodes
PERTURB_JOINTS = ARM_JOINTS + HAND_JOINTS

for joint in PERTURB_JOINTS:
    noise = np.random.normal(0, sigma, size=3)
    perturbed[joint, :3] += noise
```

**Stick-to-right-wrist constraint still applies:**
After perturbing right wrist (16), reattach stick grip (33). Hand landmarks naturally follow because their positions were originally detected relative to the wrist.

**Recompute hand features after perturbation:**
After adding noise to hand nodes, recompute their `dist_to_hip` and `angle_from_hip` using the same function as pose nodes.

---

### Step 5: Update Model for 77 Nodes

**File:** `hybrid_classifier/models/hybrid_gcn_hands.py`

```python
class HybridGCNWithHands(nn.Module):
    def __init__(self, num_node_features, num_hybrid_features, num_classes, hidden_dim=96):
        super().__init__()
        # 77 nodes instead of 35
        self.node_embedding = nn.Embedding(77, 4)
        
        # 3 GCN layers (slightly deeper due to richer graph)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features + 4, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Attention pooling (learnable)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        
        # Hybrid MLP kept but REDUCED (weak auxiliary signal, not dominant)
        self.hybrid_mlp = nn.Sequential(
            nn.Linear(num_hybrid_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Classifier
        self.fc1 = nn.Linear(hidden_dim + 32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.6)
```

**Why keep hybrid MLP here (unlike Plan A):**
With hand landmarks, the GCN has enough structural signal to **not be misled** by the hybrid branch. The hybrid features become a weak auxiliary signal rather than a dominant prior. We keep them because:
- They provide a coarse "pose family" signal
- The hand landmarks override them on left/right decisions
- Removing them entirely would waste the already-extracted 33-dim feature computation

---

### Step 6: Regenerate All Feature Tensors

**Required commands (must run in order):**
```bash
# 1. Regenerate reference templates (for completeness)
python hybrid_classifier/1_extract_reference_features.py

# 2. Regenerate train/test node features WITH hands
python hybrid_classifier/2b_generate_node_hybrid_features.py --viewpoint front

# 3. Regenerate synthetic features (hand landmarks need perturbation too)
python hybrid_classifier/2c_generate_synthetic_features.py --train_factor 5
```

**Estimated time:** ~2 hours total (hands detection adds ~50ms per image × 2,182 images).

---

### Step 7: Train + Evaluate

**Command:**
```bash
python hybrid_classifier/4e_train_hybrid_gcn_v2_with_synthetic.py \
    --viewpoint front \
    --synthetic_factor 5 \
    --epochs 150
```

**Use updated hyperparameters from Plan E:**
| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `HIDDEN_DIM` | 96 | Moderate increase for 77 nodes |
| `NUM_LAYERS` | 3 | Hand subgraph needs deeper propagation |
| `DROPOUT` | 0.6 | From Plan E grid search |
| `WEIGHT_DECAY` | 1e-3 | From Plan E |
| Loss | FocalLoss(gamma=1.5) | From Plan E |

**Metrics to report:**
- Overall validation accuracy
- Per-class recall (especially left-side classes)
- Binary left/right accuracy
- Binary block/thrust accuracy
- Confusion matrix for left/right pairs (class 1 vs 6, 2 vs 7, 3 vs 8, 4 vs 9, 5 vs 10)

---

## Dependencies Between Steps

```
Step 1 (Hands extraction) ───┐
                             ├── Step 4 (synthetic gen) ──┐
Step 2 (graph edges) ────────┤                            ├── Step 6 (regenerate .pt) ── Step 7 (train)
Step 3 (hand features) ──────┘                            │
                                                          │
Step 5 (model 77 nodes) ──────────────────────────────────┘
```

Steps 1–3 are parallel. Step 4 depends on 1–3. Step 5 is independent (model code). Step 6 depends on 4 and 5. Step 7 depends on 6.

## Estimated Effort
| Task | Time |
|------|------|
| Add MediaPipe Hands extraction | 2 hours |
| Update graph edges (hand internal + dynamic stick) | 1 hour |
| Update node feature normalization for hands | 1 hour |
| Modify synthetic generation for hand perturbation | 1 hour |
| Create `HybridGCNWithHands` model | 1 hour |
| Regenerate all `.pt` files (train, test, synthetic) | 2 hours |
| Retrain + evaluate | 2 hours |
| **Total** | **~10 hours** |

## Success Criteria
- **Val accuracy ≥ 65%**
- **Left/right recall ≥ 60%**
- **Block vs. thrust accuracy ≥ 75%**

## Risk: What If Plan B Doesn't Deliver?

If Plan B achieves only 60–62%, the bottleneck is likely:
1. **Hand detection failure rate too high** in front view (hands occluded by stick/body)
2. **Graph too large** (77 nodes) for 1,997 training samples → underfitting
3. **GCN message passing doesn't propagate hand signal** effectively to the global classifier

**Mitigation:**
- If hand detection fails >30% of the time, lower `min_detection_confidence` to 0.2
- If underfitting, reduce to 2 GCN layers and hidden_dim=64
- If propagation fails, add **skip connections** from hand nodes directly to the attention pooling layer

## Decision Trigger

**Plan B should only be started if Plan A + Plan E together plateau below 65%.**
Plan B is the highest-ceiling option but requires the most work (re-extracting all features). It is the "nuclear option" after the lower-effort plans are exhausted.
