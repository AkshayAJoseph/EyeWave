# EyeWave — Eye-Tracking AAC Keyboard

A gaze-controlled virtual keyboard designed for people with motor disabilities.
Uses 3D eyeball-sphere tracking (MediaPipe FaceMesh + PCA head pose) with a
full precision gaze pipeline.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

---

## Folder Structure

```
EyeWave/
├── assets/
│   ├── click.wav             # Key-activation sound (optional)
│   └── face_landmarker.task  # MediaPipe face landmarker model
├── src/
│   ├── __init__.py           # Public package API
│   ├── config.py             # All constants and tuning parameters
│   ├── interface.py          # LayoutManager + EyeKeyboard (GUI)
│   ├── utils.py              # Math helpers, CalibrationManager, GazeDataCollector
│   └── visionc.py            # Vision pipeline + debug orbit view
├── .gitignore
├── calibration.json          # Auto-generated on first full calibration
├── gaze_data.csv             # Auto-generated training data (grows with use)
├── main.py                   # Entry point — main loop
├── README.md
└── requirements.txt
```

---

## Calibration Procedure

### Step 1 — Eye Sphere Calibration
Look straight ahead at the keyboard. Press **C**.

- If a saved calibration exists, it loads automatically.
- Otherwise, eye spheres are locked from the current frame.

### Step 2 — 4-Point Corner Calibration
After Step 1, look at each corner and press the corresponding key, then SPACE:

| Key | Corner       |
|-----|--------------|
| `1` | Top-Left     |
| `2` | Top-Right    |
| `3` | Bottom-Left  |
| `4` | Bottom-Right |

On completion, calibration is **saved** to `calibration.json`.
Next session: just press **C** once — done in under 10 seconds.

---

## Selection Modes

| Mode    | How it works                                              |
|---------|-----------------------------------------------------------|
| **SCAN** (default) | Rows highlight automatically. Blink to enter row, blink again to select column. Gaze jumps the scanner to your looked-at row/column. |
| **GAZE** | Hold gaze on a key for ~1.3 s (dwell) to activate. Blink also activates the hovered key instantly. |

Press **M** to switch modes at any time.

---

## Keyboard Shortcuts

| Key       | Action                                          |
|-----------|-------------------------------------------------|
| `C`       | Calibrate / load saved calibration              |
| `1-4`     | Start corner calibration step                   |
| `SPACE`   | Confirm calibration point                       |
| `TAB`     | Switch layout (QWERTY ↔ AAC)                   |
| `M`       | Switch mode (Scanning ↔ Gaze-dwell)             |
| `B`       | Toggle blink selection on/off                   |
| `X`       | Drop debug gaze marker on monitor plane         |
| `F7`      | Toggle OS mouse cursor control                  |
| `J/L`     | Orbit debug camera: yaw left/right              |
| `I/K`     | Orbit debug camera: pitch up/down               |
| `[/]`     | Orbit zoom out/in                               |
| `R`       | Reset orbit camera                              |
| `Q`       | Quit                                            |

---

## Gaze Pipeline

```
Webcam frame
    │
    ▼
MediaPipe FaceMesh  (refine_landmarks=True → iris landmarks 468-477)
    │
    ▼
pca_orientation(NOSE_IDX)  →  head_center, R_final
    │
    ▼
Eye sphere update  (locked offset × scale ratio)
    │
    ▼
avg_combined = mean(left_gaze_dir, right_gaze_dir)
    │
    ▼
ray_plane_ab(eye_midpoint, avg_combined, monitor_plane)
    │  → (a_raw, b_raw)  ← mirror horizontal + vertical
    ▼
AdaptiveGazeFilter  (velocity-aware 2-D EMA in (a,b) space)
    │  → (a_f, b_f, speed)
    ▼
FixationDetector  (I-DT dispersion threshold)
    │  → is_fixating, centroid_a/b
    ▼
MultiPointCalib.correct(centroid_a, centroid_b)
    │  → (a_corr, b_corr)  ← homography-corrected
    ▼
SmartDwellController  OR  ScanningController
    │  → activated (row, col)
    ▼
EyeKeyboard.activate_key()  →  text update + TTS + data logging
```

---

## Gaze Data & Future ML Model

Every key activation logs one row to `gaze_data.csv`:
- Iris 3D positions (left + right)
- Head pose (roll, pitch, yaw)
- Raw gaze position (a, b)
- Confirmed key (ground truth)

After ~500+ samples, a personal neural network can be trained on this data
to replace the geometry pipeline — adapting to your eye geometry, your
webcam, and your typical head position for much higher accuracy.

---

## Roadmap

| Phase | Feature | Impact |
|-------|---------|--------|
| Next | IR LED ring (£10-15 hardware mod) | Biggest accuracy jump |
| After 500 samples | Train personal gaze MLP | Replaces geometry |
| Later | N-gram AAC language model | -60% keystrokes |
| Later | Per-user profiles | Multi-user support |

---

## Assets

### `assets/click.wav`
Optional short click sound played on key activation.
Any short WAV file works. Leave absent to disable sound.

### `assets/face_landmarker.task`
MediaPipe face landmarker model file (for future upgrade to Task API).
Currently the script uses `mp.solutions.face_mesh` directly.
Download from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
