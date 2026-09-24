# Face Locking

Face Locking is a local webcam face recognition and identity tracking project. It detects faces and five facial landmarks with YuNet, aligns each face to 112 x 112 pixels, extracts a 512-dimensional ArcFace embedding, compares it with enrolled identities, and locks onto a target face across video frames with smile, blink, and position error detection.

## Quick Start & Installation

### 1. Prerequisites
- **Operating System:** Windows, Linux, or macOS with desktop GUI and camera permissions.
- **Python:** 64-bit Python 3.11 or 3.12 (Python 3.13 is also supported).
- **Webcam:** Integrated or USB webcam.

### 2. Environment Setup (Windows PowerShell / CMD)
Run these commands from the `FaceLocking` folder:

```powershell
# Create a fresh Python virtual environment
py -3.12 -m venv .venv

# Activate the virtual environment or run directly using its interpreter
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

*Note: On Linux/macOS, run `python3 -m venv .venv`, activate with `source .venv/bin/activate`, and run `pip install -r requirements.txt`.*

### 3. Download Required Models
Download YuNet face detector and ArcFace ResNet100 embedding models:

```powershell
.\.venv\Scripts\python.exe init_project.py --download-models
```

This fetches and verifies:
- `models/detector_yunet.onnx` (YuNet face detector)
- `models/embedder_arcface.onnx` (ArcFace ResNet100 feature extractor)

### 4. Basic Pipeline Steps

#### Step A: Test Camera
```powershell
.\.venv\Scripts\python.exe -m src.camera
```
*(Press **Q** or **Escape** to close the camera window)*

#### Step B: Enroll a Target Identity
```powershell
.\.venv\Scripts\python.exe -m src.enroll --name "Your Name" --samples 8
```
- Keep your face visible under good lighting.
- Press **Space** to capture each sample when prompted (**Ready** status).
- Captures are saved to `data/db/face_db.json`.

#### Step C: Live Face Recognition
```powershell
.\.venv\Scripts\python.exe -m src.recognize --threshold 0.45 --margin 0.05
```
Displays bounding boxes, 5-point landmarks, similarity scores, and recognized names in real-time.

#### Step D: Lock & Track Target Person (Part 2 Pipeline)
```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --target "Your Name"
```
Locks onto the specified target name, ignores non-target faces, tracks position with EMA smoothing, detects smile/blink/eye-closed states, and calculates normalized horizontal/vertical position errors (`error_x`, `error_y`).

---

## Model Setup & Preprocessing

```sh
python init_project.py --download-models
```

This downloads YuNet (232,589 bytes) and ArcFace ResNet100 (261,036,388 bytes), verifies their SHA-256 checksums, and installs them under `models/`. Internet is needed only for installing dependencies and downloading models. Existing verified models are reused.

| File | Source |
| --- | --- |
| `models/detector_yunet.onnx` | [OpenCV Zoo YuNet 2023mar](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `models/embedder_arcface.onnx` | [ONNX Model Zoo ArcFace ResNet100](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface) |

Default preprocessing is **RGB float32, NCHW, 0-255** (`--preprocessing raw`) for this Model Zoo export. For a compatible external ArcFace model that expects `(RGB - 127.5) / 127.5`, explicitly pass `--preprocessing normalized` to enrollment, evaluation, and recognition.

---

## Enrolling an Identity

```sh
python -m src.enroll --name "Kaliza" --samples 8
python -m src.enroll --name "Isaac" --camera 1
```

1. Keep exactly one person in view, with their whole face visible and good front lighting.
2. Wait for **Ready**, then press **Space** to capture each sample. Change expression and head angle slightly between captures.
3. After all samples are captured, enrollment saves automatically and closes the camera.

The default is eight captures, at least half a second apart. Faces must be at least 90 pixels across and pass a blur check. At least three samples are required. **Q**, **Escape**, or closing the main window cancels without saving partial enrollment.

To replace an existing identity sample set:

```sh
python -m src.enroll --name "Kaliza" --replace
```

Names are case-sensitive; `Unknown` and `Too small` are reserved. Embeddings are stored in `data/db/face_db.json`.

---

## Live Recognition

```sh
python -m src.recognize --camera 0 --threshold 0.45 --margin 0.05
```

The window displays face boxes, five landmark dots, names, cosine similarity, and processing FPS. Press **Q** or **Escape** to quit. Multiple faces can be processed simultaneously.

---

## Face Tracking with Identity Lock (Part 2)

```sh
python -m src.face_tracking --target "Kaliza"
```

Use the exact enrolled name. This reuses YuNet detection, five-point alignment, ArcFace model, JSON database, and similarity threshold/margin as `src.recognize`. Unknown faces and other enrolled names cannot acquire the lock.

Only the target gets a box; the window shows status (`SEARCHING`, `LOCKED`, `UNCERTAIN`, or `LOST`), position smoothing, normalized horizontal/vertical errors (`error_x`, `error_y`), smile status, eye state, and blink counts.

### Key Features:
- **Lock Contract & States:** `SEARCHING`, `LOCKED`, `UNCERTAIN`, and `LOST`. Missing detections hold state temporarily before returning to `SEARCHING`.
- **Periodic Verification:** `--verify-every 10` verifies identity periodically while using IoU geometric association on intermediate frames.
- **Position Error & Smoothing:** Exponential moving average (`--ema-alpha 0.30`) reduces bounding box center jitter. Centered region dead zone (`--dead-zone 0.07`) prevents rapid oscillation.
- **Facial Signals:** MediaPipe FaceMesh calculates Eye Aspect Ratio (EAR) for blink detection and sustained eye closure, and mouth ratio for smile detection with hysteresis (`--smile-on 0.38`, `--smile-off 0.35`).
- **Session Logging:** Logs tracking states, facial signals, position errors, and timestamps to CSV files under `data/logs/`.

---

## Inspect Individual Pipeline Stages

```sh
python -m src.camera       # Webcam capture & FPS test
python -m src.detect       # Haar box detector demo
python -m src.landmarks    # YuNet boxes & 5 landmarks preview
python -m src.alignment    # 112x112 similarity transform alignment preview
python -m src.embed        # ONNX model feature vector extraction
```

---

## Evaluate with Held-Out Photos

```csv
path,label
validation/belise_01.jpg,belise
validation/isaac_01.jpg,isaac
validation/visitor_01.jpg,Unknown
```

Run accuracy evaluation against a manifest:

```sh
python -m src.evaluate --manifest data/evaluation.csv --threshold 0.45
```

---

## Project Structure

```text
FaceLocking/
  init_project.py        Directory setup and verified model downloads
  requirements.txt      Runtime dependencies
  models/               YuNet and ArcFace ONNX weights
  data/db/              Local enrollment database
  data/logs/            Session tracking CSV logs
  src/
    camera.py           Webcam video feed demo
    detect.py           Haar face detector demo
    landmarks.py        YuNet detector, face records, and landmark drawing
    align.py            Five-point similarity transform alignment
    embed.py            ONNX ArcFace inference & feature normalization
    database.py         Storage, model compatibility, and cosine similarity matching
    enroll.py           Quality-gated webcam identity enrollment
    recognize.py        Real-time webcam face recognition
    face_tracking.py    Target identity lock, position error tracking, and main HUD
    face_signals.py     MediaPipe FaceMesh EAR blink count & smile detection
    event_log.py        Structured session logging to CSV
    evaluate.py         Held-out image dataset evaluation
    harr_5pt.py         Landmark and alignment preview
    config.py           Project paths configuration
    workflow.py         CLI helpers and camera cleanup
  tests/                Automated unit tests
```

---

## Running Tests

Run the complete automated unit test suite:

```sh
python -m unittest discover -s tests -v
```

Tests cover similarity transformations, database operations, target candidate selection, lost state timeouts, distractor rejection, smile hysteresis, EAR blink counting, and camera resource cleanup.

---

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `No Python at ...` | Recreate `.venv` using an installed Python (`py -3.12 -m venv .venv`). |
| Missing/empty model | Run `python init_project.py --download-models`. |
| Camera cannot open | Close other camera apps (Teams, Zoom), check camera permissions, or try `--camera 1`. |
| No window / GUI error | Run in a desktop session and verify `opencv-python` / `opencv-contrib-python` is installed. |
| Target is Unknown | Verify lighting, re-enroll with clean front-facing samples, or tune `--threshold`. |
| Distractor takes lock | Ensure target identity name is exact; identity verification prevents lock transfer. |

---

## Reference & Acknowledgments

This project is built following the identity lock, tracking, and facial gesture architecture described in *Tracking with Identity Lock, Smile, Blink and Position Detection (Part 2)* by Gabriel Baziramwabo (Benax Technologies Ltd & Rwanda Coding Academy).
