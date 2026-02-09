# Standalone Desktop App: Arnis Pose Classifier

Deploy the Hybrid GCN V2 specialist models (Front/Left/Right) as an offline desktop application for real-time pose classification.

## User Review Required

> [!IMPORTANT]
> **Critical Clarifications Needed**
> 
> 1. **Database Purpose**: You mentioned "offline with DB" - what data needs to be stored?
>    - User profiles and training history?
>    - Pose classification logs/timestamps?
>    - Reference pose templates?
>    - Video recordings?
> 
> 2. **Viewpoint Selection**: How does the user specify which specialist model to use?
>    - Manual dropdown menu (Front/Left/Right)?
>    - Auto-detect viewpoint using a lightweight classifier?
>    - Fixed camera position (always use one model)?
> 
> 3. **Action Recognition Logic**: You said "user poses in front of cam only" - what's the expected UX?
>    - Real-time feedback showing current pose classification?
>    - "Hold pose for N seconds" validation?
>    - Sequence detection (e.g., "perform 5 correct thrusts in a row")?
> 
> 4. **Packaging Preference**: For standalone desktop apps, which would you prefer?
>    - **PyInstaller** (single .exe, ~500MB, slower startup)
>    - **Electron + Python backend** (modern UI, ~200MB, requires Node.js knowledge)
>    - **PyQt/Tkinter + embedded Python** (traditional desktop app, ~300MB)

> [!WARNING]
> **Performance Bottleneck Identified**
> 
> Your YOLO stick detector (`best.pt` = 6.12MB) is the main performance bottleneck:
> - **Current**: YOLOv8-Pose (likely YOLOv8n or YOLOv8s)
> - **CPU Inference**: ~50-80ms per frame on Intel Core 7 150U
> - **Combined Pipeline**: MediaPipe (30ms) + YOLO (60ms) + GCN (5ms) = **~95ms = 10 FPS**
> 
> **Proposed Solutions** (pick one):
> 1. **Quantize YOLO to INT8** (2-3x faster, ~3MB, minimal accuracy loss)
> 2. **Switch to YOLO11n-pose** (newer, 40% faster than YOLOv8n)
> 3. **Use MediaPipe-only mode** (fallback when FPS < 20, disable stick features)
> 4. **Frame skipping** (run YOLO every 3rd frame, interpolate stick positions)

---

## Proposed Changes

### Component 1: Inference Engine

#### [NEW] [inference_engine.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis-ML/deployment/inference_engine.py)

**Purpose**: Real-time pose classification pipeline

**Key Features**:
- Load 3 specialist models (Front/Left/Right) at startup
- Unified inference API: `classify_pose(frame, viewpoint) -> (class_name, confidence)`
- Feature extraction pipeline:
  - MediaPipe Pose (33 keypoints)
  - YOLO Stick Detector (2 keypoints: grip, tip)
  - Node feature computation (35 nodes × 6 features)
  - Hybrid feature computation (30 similarity scores)
- Performance optimizations:
  - Model caching (load once, reuse)
  - Batch size = 1 (real-time mode)
  - CPU-optimized inference (`torch.set_num_threads(4)`)

**Dependencies**:
- Reuse existing `2b_generate_node_hybrid_features.py` feature extraction logic
- Load models from `hybrid_classifier/models/hybrid_gcn_v2_{front|left|right}.pth`

---

#### [NEW] [model_loader.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis-ML/deployment/model_loader.py)

**Purpose**: Lazy model loading and caching

**Key Features**:
- Load models on-demand (only load selected viewpoint)
- Model state validation (check architecture compatibility)
- Graceful fallback if model file corrupted

---

### Component 2: Desktop Application

#### [NEW] [app_main.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis-ML/deployment/app_main.py)

**Purpose**: Main application entry point

**Framework**: **PyQt6** (recommended for desktop apps)

**UI Components**:
1. **Camera Feed Panel**: Live webcam display with skeleton overlay
2. **Classification Panel**: 
   - Current pose name (large text)
   - Confidence score (progress bar)
   - Top-3 predictions
3. **Control Panel**:
   - Viewpoint selector (Front/Left/Right dropdown)
   - Start/Stop camera button
   - FPS counter (real-time performance monitoring)
4. **Settings Panel**:
   - Confidence threshold slider (default: 0.7)
   - Enable/disable stick detection toggle
   - Database logging toggle

**Performance Monitoring**:
- Display FPS in real-time
- Show warning if FPS < 20: "⚠️ Low FPS detected. Consider disabling stick detection."

---

#### [NEW] [camera_handler.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis-ML/deployment/camera_handler.py)

**Purpose**: Webcam capture and frame preprocessing

**Key Features**:
- OpenCV VideoCapture wrapper
- Frame buffering (avoid dropped frames)
- Resolution: 640×480 (balance quality vs speed)
- Auto-exposure adjustment

---

### Component 3: Database Layer

#### [NEW] [database.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis-ML/deployment/database.py)

**Purpose**: SQLite database for offline storage

**Schema** (pending user clarification):
```sql
CREATE TABLE pose_logs (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    viewpoint TEXT,  -- 'front', 'left', 'right'
    predicted_class TEXT,
    confidence REAL,
    fps REAL
);

-- Additional tables TBD based on user requirements
```

**Operations**:
- `log_prediction(viewpoint, class_name, confidence, fps)`
- `get_session_history(limit=100)`
- `export_to_csv(start_date, end_date)`

---

### Component 4: Performance Optimizations

#### [NEW] [optimize_models.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis-ML/deployment/optimize_models.py)

**Purpose**: Model quantization and optimization

**Optimizations**:
1. **PyTorch JIT Compilation**:
   ```python
   model = torch.jit.script(model)  # 10-20% speedup
   ```

2. **Dynamic Quantization** (INT8):
   ```python
   model = torch.quantization.quantize_dynamic(
       model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
   )
   # Result: 1.35MB → 400KB, 1.5-2x faster
   ```

3. **ONNX Export** (optional, for future GPU support):
   ```python
   torch.onnx.export(model, dummy_input, "model.onnx")
   ```

**YOLO Optimization**:
- Export to ONNX with INT8 quantization:
  ```python
  yolo_model.export(format="onnx", int8=True)
  # 6.12MB → 2-3MB, 2-3x faster on CPU
  ```

---

### Component 5: Packaging

#### [NEW] [build_standalone.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis-ML/deployment/build_standalone.py)

**Purpose**: Build standalone .exe using PyInstaller

**Configuration**:
```python
# pyinstaller.spec
a = Analysis(
    ['app_main.py'],
    pathex=['deployment'],
    binaries=[],
    datas=[
        ('hybrid_classifier/models/*.pth', 'models'),
        ('runs/pose/arnis_stick_detector/weights/best.pt', 'weights'),
    ],
    hiddenimports=['torch', 'mediapipe', 'ultralytics'],
    ...
)
```

**Output**:
- Single executable: `ArnisClassifier.exe` (~400-500MB)
- Includes embedded Python 3.11 runtime
- No external dependencies required

---

#### [NEW] [requirements_deployment.txt](file:///c:/Users/HP/Documents/GitHub/TuroArnis-ML/deployment/requirements_deployment.txt)

**Locked Versions** (compatible with your current environment):
```
torch==2.10.0+cpu
torch-geometric==2.4.0
mediapipe==0.10.14
ultralytics==8.3.252
opencv-python==4.13.0.92
numpy==1.26.4
PyQt6==6.8.0
pyinstaller==6.12.0
```

---

## Verification Plan

### Automated Tests

#### 1. Unit Tests: Feature Extraction
**File**: `tests/test_feature_extraction.py`

**Command**:
```bash
python -m pytest tests/test_feature_extraction.py -v
```

**Coverage**:
- Test node feature extraction with synthetic keypoints
- Verify output shape: [35, 6]
- Test hybrid feature computation against known templates

---

#### 2. Integration Test: End-to-End Inference
**File**: `tests/test_inference_pipeline.py`

**Command**:
```bash
python -m pytest tests/test_inference_pipeline.py -v
```

**Coverage**:
- Load a sample image from `dataset_augmented/test/front/neutral_stance/`
- Run full pipeline: MediaPipe → YOLO → GCN
- Assert output is valid class name with confidence > 0.0

---

#### 3. Performance Benchmark
**File**: `tests/benchmark_fps.py`

**Command**:
```bash
python tests/benchmark_fps.py --frames 100
```

**Expected Output**:
```
MediaPipe: 33.2 FPS (30ms/frame)
YOLO Stick: 16.7 FPS (60ms/frame)
GCN Inference: 200 FPS (5ms/frame)
Full Pipeline: 10.5 FPS (95ms/frame)

⚠️ Target: 30 FPS | Current: 10.5 FPS
Recommendation: Enable YOLO quantization
```

---

### Manual Verification

#### 1. Real-Time Classification Test

**Steps**:
1. Run the application:
   ```bash
   python deployment/app_main.py
   ```
2. Select "Front" viewpoint from dropdown
3. Perform "Neutral Stance" in front of webcam
4. **Expected**: UI shows "neutral_stance" with confidence > 0.7
5. Perform "Right Temple Block"
6. **Expected**: UI updates to "right_temple_block_correct" within 1 second

**Success Criteria**:
- FPS counter shows ≥ 20 FPS (or displays warning if < 20)
- Predictions update in real-time (< 200ms latency)
- No crashes or freezes during 5-minute session

---

#### 2. Viewpoint Switching Test

**Steps**:
1. Start with "Front" viewpoint
2. Perform "Crown Thrust"
3. Switch dropdown to "Left" viewpoint
4. Perform same "Crown Thrust" pose
5. **Expected**: Different confidence scores (specialist models behave differently)

---

#### 3. Database Logging Test

**Steps**:
1. Enable "Database Logging" toggle
2. Perform 10 different poses
3. Close application
4. Open SQLite database:
   ```bash
   sqlite3 deployment/pose_logs.db
   SELECT * FROM pose_logs ORDER BY timestamp DESC LIMIT 10;
   ```
5. **Expected**: 10 rows with correct timestamps, class names, and FPS values

---

#### 4. Standalone .exe Test

**Steps**:
1. Build executable:
   ```bash
   python deployment/build_standalone.py
   ```
2. Copy `dist/ArnisClassifier.exe` to a **clean Windows machine** (no Python installed)
3. Run `ArnisClassifier.exe`
4. **Expected**: Application launches without errors, camera works, predictions run

**Success Criteria**:
- .exe size < 600MB
- Startup time < 10 seconds
- No "DLL not found" errors

---

## Post-Implementation Enhancements (Optional)

1. **Auto-Viewpoint Detection**:
   - Train a lightweight CNN to classify camera angle (Front/Left/Right)
   - Auto-switch specialist models based on detected viewpoint

2. **Action Sequence Recognition**:
   - Implement temporal smoothing (e.g., "hold pose for 2 seconds")
   - Add sequence validation (e.g., "perform 5 correct blocks in a row")

3. **Export to Mobile**:
   - Convert models to TensorFlow Lite for Android/iOS
   - Use MediaPipe's mobile SDK (no YOLO, use fallback features)
