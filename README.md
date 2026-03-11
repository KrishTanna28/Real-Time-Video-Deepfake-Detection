# Real-Time Video Deepfake Detection

A real-time deepfake detection system that works as a Chrome browser extension. It captures frames from any video playing in the browser, sends them to a local Python backend, and classifies them as REAL or FAKE using a trained EfficientNet-B0 model combined with six frame-level forensic signals and a temporal voting mechanism.

The system was designed around a single principle: make deepfake detection accessible to anyone watching video in a browser, without requiring specialized knowledge or cloud services. Everything runs locally on the user's machine.

---

## Table of Contents

- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Training Configuration](#training-configuration)
- [Training Metrics and Results](#training-metrics-and-results)
- [Inference Pipeline](#inference-pipeline)
- [Face Detection](#face-detection)
- [Frame-Level Forensic Analysis](#frame-level-forensic-analysis)
- [Temporal Voting System](#temporal-voting-system)
- [Backend API](#backend-api)
- [Chrome Extension](#chrome-extension)
- [Performance Benchmarks](#performance-benchmarks)
- [Testing](#testing)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Requirements](#requirements)
- [License](#license)

---

## How It Works

When a user clicks "Start Detection" in the extension popup, the content script finds the video element on the page, captures a frame via an HTML5 Canvas, encodes it as JPEG (quality 0.85), and sends it to the Flask backend at localhost:5000. The backend runs face detection first. If a face is found, it is aligned with MTCNN, preprocessed with CLAHE contrast enhancement, resized to 224x224, normalized with ImageNet statistics, and passed through the EfficientNet-B0 model to produce a fake probability score. Simultaneously, the frame goes through six forensic analysis signals. The face model probability carries 70% weight and the forensic score carries 30% when both are available. If no face is detected, the system falls back to forensic-only analysis. Every frame's classification feeds into a temporal voting window that accumulates 10 consecutive verdicts and takes a majority vote to produce a stable REAL or FAKE label. The result is displayed in real time on an overlay injected directly onto the video page and in the extension popup dashboard.

---

## System Architecture

```
+----------------------------------------------------------------------+
|                     CHROME BROWSER EXTENSION                         |
|                                                                      |
|  popup.js           content.js             overlay-script.js         |
|  (Control panel)    (Captures video        (Live overlay on video    |
|  Start/Stop,         frames via Canvas,     showing verdict,         |
|  Dashboard)          sends to backend)      confidence, metrics)     |
|                          |                                           |
+--------------------------|-------------------------------------------+
                           | HTTP POST (localhost:5000/analyze)
                           v
+----------------------------------------------------------------------+
|                  FLASK BACKEND SERVER (Python)                        |
|                                                                      |
|  Endpoints: /analyze, /reset, /health, /stats                        |
|                          |                                           |
|                          v                                           |
|  +----------------------------------------------------------------+  |
|  |                  DEEPFAKE DETECTOR                              |  |
|  |                                                                 |  |
|  |  Face Detection          Frame Forensic Analysis                |  |
|  |  (OpenCV DNN SSD         (6 signals: FFT, Noise, ELA,          |  |
|  |   + Haar fallback)        Edge, Color, Temporal)                |  |
|  |       |                        |                                |  |
|  |       v                        |                                |  |
|  |  Face Analysis                 |                                |  |
|  |  CLAHE -> MTCNN ->             |                                |  |
|  |  EfficientNet-B0 ->            |                                |  |
|  |  sigmoid -> probability        |                                |  |
|  |       |                        |                                |  |
|  |       +----------+-------------+                                |  |
|  |                  |                                               |  |
|  |                  v                                               |  |
|  |       Temporal Tracker (Voting System)                          |  |
|  |       10-frame majority vote -> REAL / FAKE / UNCERTAIN         |  |
|  +----------------------------------------------------------------+  |
+----------------------------------------------------------------------+
```

---

## Model Details

### Architecture: EfficientNet-B0

EfficientNet-B0 was chosen for its best-in-class accuracy-to-parameter ratio. It achieves 77.1% ImageNet Top-1 accuracy with only 5.3 million parameters, compared to ResNet-50 which needs 25.6 million parameters for 76.0% accuracy. For a real-time system that must process frames within 200ms, this efficiency matters.

EfficientNet uses compound scaling, simultaneously adjusting network depth, width, and resolution:

- depth: d = 1.2^phi
- width: w = 1.1^phi
- resolution: r = 1.15^phi

For B0 the scaling coefficient phi = 1 (the baseline).

The backbone consists of 16 MBConv (Mobile Inverted Bottleneck Convolution) blocks organized in 7 stages. Each block contains a 1x1 expansion convolution, a depthwise separable convolution (3x3 or 5x5 kernel), a Squeeze-and-Excitation attention module, a 1x1 projection convolution, and a skip connection when dimensions match. The activation function is SiLU (Swish) throughout, and all layers use BatchNorm.

### Classification Head

The 1280-dimensional feature vector from the backbone feeds into a custom classification head:

```
AdaptiveAvgPool2d(1) -> Flatten
-> Dropout(0.5) -> Linear(1280, 512) -> BatchNorm1d(512) -> ReLU
-> Dropout(0.35) -> Linear(512, 256) -> BatchNorm1d(256) -> ReLU
-> Dropout(0.25) -> Linear(256, 1) -> Sigmoid -> [0, 1]
```

The output is a single scalar: the probability that the input face is fake.

### Parameter Count

| Component             | Parameters   | Status    |
|-----------------------|-------------|-----------|
| Stem + Blocks 0-2     | ~990,000    | Frozen    |
| Blocks 3-15           | ~3,010,000  | Trainable |
| Classification Head    | ~1,288,548  | Trainable |
| **Total**             | **5,288,548** | --      |
| Trainable             | ~4,300,000  | 81%       |
| Frozen                | ~990,000    | 19%       |

Freezing the early layers preserves general-purpose low-level feature extractors (edges, textures, color patterns) learned from ImageNet, while letting the deeper layers and head specialize for deepfake artifacts.

### Model Comparison

| Model            | Params | ImageNet Top-1 | Inference Speed | Suitability          |
|-----------------|--------|----------------|-----------------|----------------------|
| VGG-16           | 138M   | 71.5%          | Slow            | Too large, too slow   |
| ResNet-50        | 25.6M  | 76.0%          | Medium          | 5x more params        |
| MobileNet-V2     | 3.4M   | 72.0%          | Very fast       | Lower feature quality |
| **EfficientNet-B0** | **5.3M** | **77.1%** | **Fast**        | **Best tradeoff**     |
| EfficientNet-B7  | 66M    | 84.3%          | Slow            | Too slow for real-time|

### What the Model Learns to Detect

The model picks up on artifacts that deepfake generation methods leave behind:

- Blending boundary artifacts at the face-background seam
- Unnatural skin texture smoothness from GAN upsampling
- Inconsistent lighting between the swapped face and original scene
- Warping distortions from affine face alignment during synthesis
- Compression double-encoding signatures
- Frequency domain anomalies from neural network upsampling (checkerboard patterns)

---

## Dataset

### FaceForensics++ C23

The model is trained on the FaceForensics++ dataset at C23 (medium) compression, which is the standard benchmark for face manipulation detection research.

| Split   | Real Samples | Fake Samples | Total   | Fake Ratio |
|---------|-------------|-------------|---------|------------|
| Train   | 12,750      | 76,395      | 89,145  | 85.7%      |
| Val     | ~2,250      | ~13,488     | ~15,738 | 85.7%      |

The fake samples come from five manipulation methods: DeepFakes, Face2Face, FaceSwap, FaceShifter, and NeuralTextures. Training on multiple methods helps the model generalize rather than overfit to a single forgery technique.

The class imbalance (85.7% fake vs 14.3% real) is handled through WeightedRandomSampler during training, which oversamples the minority class (real) so the model sees approximately equal numbers of real and fake examples per epoch (~31,800 samples per epoch).

### DFDC Support

The project also includes scripts for downloading and processing the Deepfake Detection Challenge (DFDC) dataset from Kaggle. The download script performs balanced sampling (equal real and fake), and the processing pipeline is fully resumable with progress tracking.

---

## Training Configuration

The training pipeline went through five iterations. The final stable version (v5) resolved oscillation issues from earlier versions by reducing the learning rate, fixing Focal Loss alpha, switching model selection from F1 to Balanced Accuracy, and disabling Mixup/CutMix (which were destroying the subtle deepfake artifact signals the model needs to learn).

| Hyperparameter            | Value                                           |
|--------------------------|------------------------------------------------|
| Epochs                    | 30 (early stopping with patience 10)            |
| Batch Size                | 32 (effective 64 via 2-step gradient accumulation) |
| Max Learning Rate         | 1.5e-4                                          |
| Backbone Learning Rate    | 0.1x of head LR (1.5e-5)                       |
| Weight Decay              | 0.01                                            |
| Optimizer                 | AdamW                                           |
| Scheduler                 | OneCycleLR (10% warmup, cosine annealing)       |
| Loss Function             | Focal Loss (gamma=2.0, alpha=0.5)               |
| Dropout Rates             | 0.5, 0.35, 0.25 (decreasing through head layers)|
| Gradient Clipping         | max_norm=1.0                                    |
| Mixed Precision           | Enabled (torch.cuda.amp)                        |
| EMA Decay                 | 0.999                                           |
| Frozen Layers             | Stem + first 3 MBConv blocks (19% of backbone)  |
| Sampling                  | WeightedRandomSampler, 2x minority class        |
| Model Selection Metric    | Balanced Accuracy                                |

### Data Augmentation

| Augmentation              | Parameter                              |
|--------------------------|----------------------------------------|
| Random Resized Crop       | 224x224                                |
| Horizontal Flip           | p=0.5                                  |
| Color Jitter              | brightness, contrast, saturation, hue  |
| Random Rotation           | degrees                                |
| Random Affine             | translate, scale, shear                |
| Random Perspective        | distortion                             |
| Gaussian Blur             | kernel size                            |
| Random Erasing            | p varies                               |
| JPEG Compression          | quality 30-80, p=0.3                   |
| Gaussian Noise            | sigma 0.01-0.04, p=0.15               |

### Why Focal Loss

Standard cross-entropy treats all examples equally. Focal Loss down-weights well-classified examples and focuses training on hard, misclassified ones:

```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```

With gamma=2.0, examples classified with >0.9 confidence contribute 100x less gradient than ambiguous examples near 0.5. This prevents the model from coasting on easy examples and forces it to learn the subtle boundaries between real and fake.

### Why Balanced Accuracy over F1

Earlier versions selected the best model checkpoint based on F1 score. The problem: with 85.7% fake data, a model that predicts everything as FAKE achieves an F1 of 0.925, which looks excellent but is completely useless. Balanced Accuracy is the arithmetic mean of per-class recall, so both classes must be predicted well. A model that predicts all-FAKE gets a Balanced Accuracy of 50%, immediately revealing the failure.

---

## Training Metrics and Results

Training was conducted on Google Colab with GPU runtime. The model went through multiple iterations. The current (new) model represents a significant improvement over the old model in both accuracy and bias balance.

### Current Model Performance

| Metric              | Old Model | New Model | Improvement |
|---------------------|-----------|-----------|-------------|
| Balanced Accuracy   | 86.30%    | 91.85%    | +5.55%      |
| Real Accuracy       | 90.20%    | 91.60%    | +1.40%      |
| Fake Accuracy       | 82.40%    | 92.10%    | +9.70%      |
| Bias Gap            | 7.80%     | 0.50%     | -7.30%      |
| Bias Direction      | Real-biased | Balanced | --        |

### Mean Predictions by Class

| Class                        | Old Model | New Model | Ideal |
|------------------------------|-----------|-----------|-------|
| Real images (should be ~0.0) | 0.1824    | 0.0716    | 0.000 |
| Fake images (should be ~1.0) | 0.6412    | 0.9143    | 1.000 |

The new model scores real images much closer to 0.0 (0.0716 vs 0.1824) and fake images much closer to 1.0 (0.9143 vs 0.6412), showing substantially better separation between classes. The old model had a 7.80% accuracy gap favoring real images, meaning it was biased toward classifying inputs as real. The new model reduces that gap to just 0.50%, making it effectively balanced across both classes.

### Training Progression (Old Model Baseline)

| Metric             | Epoch 1 | Best (Epoch 8) |
|--------------------|---------|-----------------|
| Training Loss      | 0.7066  | 0.1064          |
| Training Accuracy  | 52.2%   | 85.0%           |
| Validation Loss    | 0.7626  | 0.1948          |
| Validation Accuracy| 27.8%   | 86.6%           |
| Validation F1      | 0.2973  | 0.9270          |
| Validation AUC-ROC | 0.5606  | 0.8445          |
| Epoch Duration     | ~4.3 hrs| --              |

### Training Log

The full training history is stored in `weights/training_log.json`. Model weights are saved as:

- `weights/best_model.pt` -- Best checkpoint (state dict)
- `weights/best_model.pth` -- Alternative format
- `weights/training.pth` -- Resume checkpoint with full optimizer/scheduler/RNG state

---

## Inference Pipeline

Each frame goes through the following pipeline:

1. **Receive frame** via /analyze endpoint (JPEG/PNG/BMP accepted)
2. **Face detection** using OpenCV DNN SSD (ResNet-10 backbone, 300x300 input, 5-15ms)
3. **If face found:**
   - Apply CLAHE contrast enhancement to the face crop
   - Align face using MTCNN (5 facial landmarks: eyes, nose, mouth corners)
   - Resize to 224x224
   - Normalize with ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
   - Run through EfficientNet-B0, apply sigmoid to get fake probability
   - Small faces (width or height < 80px) get a +0.10 fake probability bias (low resolution increases unreliability)
4. **Frame forensic analysis** runs in parallel on every frame (6 signals, see below)
5. **Score fusion:**
   - Face detected: 70% face model score + 30% forensic score
   - No face: 100% forensic score
6. **Temporal voting:** Classification (FAKE if score > 0.55, REAL otherwise) enters the voting queue
7. **Majority vote** over last 10 classifications produces the final verdict

---

## Face Detection

### Primary: OpenCV DNN SSD

- Model: ResNet-10 backbone SSD trained on face detection
- Weights: `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- Input resolution: 300x300 BGR
- Confidence threshold: 0.5
- Latency: 5-15ms per frame
- Returns bounding boxes as (x, y, width, height)

### Fallback: Haar Cascade

- Classifier: `haarcascade_frontalface_default.xml`
- Scale factor: 1.1, minimum neighbors: 5, minimum size: 30x30
- Latency: 10-30ms per frame
- Activates automatically if DNN detection fails

### Face Alignment: MTCNN

- Detects 5 facial landmarks (left eye, right eye, nose, left mouth corner, right mouth corner)
- Computes affine transformation to align face to canonical frontal position
- Used during inference preprocessing to normalize face orientation

---

## Frame-Level Forensic Analysis

Six independent forensic signals analyze each frame for manipulation artifacts. Each signal produces a score between 0 (likely real) and 1 (likely fake), and they are combined using a weighted sum.

### Signal 1: Frequency Domain Analysis (FFT) -- Weight: 0.25

Converts the frame to frequency domain using Fast Fourier Transform. Deepfakes often show reduced high-frequency content (GAN smoothing) and abnormal frequency distributions.

| Indicator                        | Fake Score Contribution |
|----------------------------------|------------------------|
| Low high-frequency ratio (<0.18) | +0.40                  |
| High mid-frequency CV (>0.6)     | +0.25                  |
| Abnormal mid-to-high ratio       | +0.15                  |

### Signal 2: Noise Pattern Consistency -- Weight: 0.20

Analyzes the noise residual pattern across the frame. Authentic images have consistent sensor noise; deepfakes show inconsistent noise from different source images or generation artifacts.

| Indicator                        | Fake Score Contribution |
|----------------------------------|------------------------|
| High noise CoV (>0.7)            | +0.50                  |
| Low mean noise (<1.0)            | +0.30                  |

### Signal 3: Error Level Analysis (ELA) -- Weight: 0.20

Recompresses the image at quality 90 and measures the pixel-level difference. Manipulated regions show different error levels than untouched regions due to double compression.

| Indicator                        | Fake Score Contribution |
|----------------------------------|------------------------|
| High ELA CoV (>0.9)              | +0.50                  |
| High mean ELA error (>15)        | +0.20                  |

### Signal 4: Edge Coherence Analysis -- Weight: 0.15

Examines edge density and sharpness consistency. Deepfakes often have unnaturally smooth regions or inconsistent edge patterns around blending boundaries.

| Indicator                        | Fake Score Contribution |
|----------------------------------|------------------------|
| Low edge density (<0.02)         | +0.35                  |
| Low Laplacian variance (<50)     | +0.30                  |

### Signal 5: Color Space Analysis -- Weight: 0.10

Checks for unnatural color distributions. GAN-generated faces sometimes have limited color palettes or suspiciously uniform saturation/brightness.

| Indicator                        | Fake Score Contribution |
|----------------------------------|------------------------|
| Uniform saturation (std<15)      | +0.30                  |
| Uniform brightness (std<15)      | +0.25                  |
| Limited hue palette (<30 hues)   | +0.25                  |

### Signal 6: Temporal Consistency -- Weight: 0.10

Tracks frame-to-frame differences over time. Deepfakes can produce erratic jumps or suspiciously static regions between frames.

| Indicator                        | Fake Score Contribution |
|----------------------------------|------------------------|
| Erratic differences (CV>1.5)     | +0.40                  |
| Near-zero differences (<0.3)     | +0.30                  |

### Combined Forensic Score

```
forensic_score = SUM(weight_i * signal_i)    clipped to [0, 1]
```

Full analysis (all 6 signals): 10-15ms per frame.
Fast analysis mode (FFT + Temporal + Edge only): 3-5ms per frame.

---

## Temporal Voting System

Single-frame predictions are noisy. The temporal tracker smooths them into stable verdicts.

| Parameter          | Value                          |
|--------------------|--------------------------------|
| History window     | 60 frames (~2 seconds at 30fps)|
| Voting window      | 10 frames                      |
| Detection threshold| 0.55                           |
| Tie-breaking rule  | REAL (conservative default)    |

### How Voting Works

1. Each frame produces a classification: FAKE (score > 0.55) or REAL (score <= 0.55)
2. The last 10 classifications are collected
3. If more than 5 are FAKE, the verdict is FAKE
4. If more than 5 are REAL, the verdict is REAL
5. On a 5-5 tie, the verdict defaults to REAL (to avoid false positives)
6. Fewer than 10 accumulated frames: verdict is UNCERTAIN

The system also computes:
- **Temporal average**: Running mean of fake probability scores over the 60-frame history window
- **Stability score**: Measures prediction consistency; >0.9 means very consistent predictions, <0.5 means oscillating. Returns 0.0 if fewer than 10 frames have been processed.

---

## Backend API

The Flask server runs on port 5000 with CORS enabled.

### POST /analyze

The main analysis endpoint. Accepts a video frame and returns classification results.

**Request:** multipart/form-data with a `frame` field containing an image file (JPEG, PNG, or BMP).

**Rate limit:** 100ms minimum between requests. Requests arriving faster receive a 429 response.

**Response (200):**

```json
{
  "success": true,
  "analysis_mode": "face+frame",
  "faces_detected": 1,
  "fake_probability": 0.42,
  "face_probability": 0.38,
  "frame_forensic_probability": 0.51,
  "confidence_level": "REAL",
  "temporal_average": 0.44,
  "stability_score": 0.87,
  "frame_count": 15,
  "processing_time_ms": 127.3,
  "face_bbox": {"x": 120, "y": 80, "width": 200, "height": 220}
}
```

| Field                       | Type    | Description                                           |
|----------------------------|---------|-------------------------------------------------------|
| success                     | bool    | Whether analysis completed without error              |
| analysis_mode               | string  | "face+frame" when face found, "frame_only" otherwise  |
| faces_detected              | int     | Number of faces detected in the frame                 |
| fake_probability            | float   | Combined probability the frame is fake [0, 1]         |
| face_probability            | float   | Model-only probability (present when face detected)   |
| frame_forensic_probability  | float   | Forensic-only probability from 6 signals              |
| confidence_level            | string  | Temporal verdict: "FAKE", "REAL", or "UNCERTAIN"      |
| temporal_average            | float   | Running mean of fake probabilities (60-frame window)  |
| stability_score             | float   | Prediction consistency score [0, 1]                   |
| frame_count                 | int     | Total frames analyzed in this session                 |
| processing_time_ms          | float   | End-to-end processing time for this frame             |
| face_bbox                   | object  | Bounding box of detected face (null if no face)       |

### GET /health

Returns server status, model loading state, device info, and capability flags.

**Response (200):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "frame_count": 0,
  "capabilities": {}
}
```

### POST /reset

Clears the temporal tracker, frame count, and all forensic state. Call this when switching videos or starting a new detection session.

**Response (200):**

```json
{
  "success": true
}
```

### GET /stats

Returns current session statistics without analyzing a new frame.

**Response (200):**

```json
{
  "frame_count": 42,
  "temporal_average": 0.61,
  "stability_score": 0.93,
  "confidence_level": "FAKE",
  "voting_stats": {}
}
```

---

## Chrome Extension

The extension is built on Manifest V3 and consists of four main components.

### Content Script (content.js)

Injected into every web page. Finds the video element on the page (including inside iframes), captures frames by drawing the video onto an HTML5 canvas using `canvas.toDataURL('image/jpeg', 0.85)`, and sends them to the background service worker at a configurable interval (default: 1000ms). Also creates and manages the overlay iframe that displays results on top of the video.

### Background Service Worker (background.js)

Manages communication between the content script and the backend server. Forwards frame data to `localhost:5000/analyze` as multipart form-data, handles tab lifecycle cleanup, injects the content script on SPA navigation, and manages rate limiting with exponential backoff.

### Popup (popup.html / popup.js)

The control panel accessible from the extension toolbar icon. Provides Start/Stop buttons, displays all detection metrics in real time (classification, confidence percentage, temporal average, stability score, frames analyzed, processing speed, analysis mode), and offers settings for the backend URL and capture interval. Settings are persisted via `chrome.storage.local`.

### Overlay (overlay.html / overlay-script.js)

An iframe injected into the page next to the video. Shows the live detection verdict with color-coded status badges: red for FAKE, green for REAL, yellow for UNCERTAIN, and gray for disconnected. Displays confidence, temporal average, stability, frame count, and processing speed. Includes a stop button and a close button.

### Extension Settings

| Setting           | Default              | Description                          |
|-------------------|---------------------|--------------------------------------|
| Backend URL       | http://localhost:5000| Address of the Flask backend server  |
| Capture Interval  | 1000ms              | Time between frame captures          |

---

## Performance Benchmarks

These are the performance targets enforced by the test suite. Actual performance will vary with hardware.

### Latency Targets

| Operation                  | Target      | Notes                              |
|---------------------------|-------------|-------------------------------------|
| Face Detection (DNN SSD)   | < 100ms     | Typically 5-15ms                    |
| Face Detection (Haar)      | < 100ms     | Typically 10-30ms (fallback only)   |
| Model Inference (GPU)      | < 200ms     | Single frame through EfficientNet   |
| Model Inference (CPU)      | < 500ms     | Fallback when no GPU available      |
| Full Forensic Analysis     | < 50ms      | All 6 signals                       |
| Fast Forensic Analysis     | < 20ms      | FFT + Temporal + Edge only          |
| Frequency Feature Extract  | < 30ms      | FFT computation                     |
| API /analyze (end-to-end)  | < 1500ms    | Full pipeline including network      |
| API /health                | < 50ms      | Health check response               |
| API /reset                 | < 50ms      | State reset response                |

### Resource Targets

| Metric              | Target   |
|---------------------|----------|
| Model Parameters    | < 8M     |
| Weight File Size    | < 50MB   |

### Throughput

Batch inference is faster per frame than single-frame inference due to GPU parallelism. The extension default capture interval of 1000ms is conservative; the backend can handle faster rates depending on GPU capacity.

---

## Testing

The project includes 95 tests organized across four test files, covering functional correctness, algorithmic accuracy, performance compliance, and reliability under adverse conditions.

### Test Summary

| Test File              | Tests | Focus                                        |
|------------------------|-------|----------------------------------------------|
| test_functional.py     | 36    | Model loading, face detection, forensics, temporal tracker, API endpoints |
| test_algorithm.py      | 19    | Threshold behavior, voting accuracy, forensic signal logic, stability scores |
| test_performance.py    | 12    | Latency targets, parameter counts, weight file size, batch throughput |
| test_reliability.py    | 28    | Malformed input handling, resolution variance, determinism, format support, rate limiting, continuous operation |

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_functional.py

# Run a specific test
pytest tests/test_algorithm.py::TestThresholdBehavior::test_above_threshold_is_fake

# Run slow tests (marked with @pytest.mark.slow)
pytest -m slow

# Run all tests via the custom runner
python tests/run_all_tests.py
```

### What the Tests Verify

**Functional tests** confirm that the model weight file exists and loads with zero missing keys, that the forward pass produces the correct output shape, that softmax outputs are valid probabilities, that face detection returns properly formatted bounding boxes, that all six forensic signals produce scores in the valid range, that the temporal tracker follows voting rules (including the tie-to-REAL rule), and that all API endpoints return correct status codes and response schemas.

**Algorithm tests** verify that scores above the threshold produce FAKE verdicts and below produce REAL, that slim-majority voting works correctly (6 vs 4 in either direction), that forensic signal weights combine correctly, that frequency features have the expected shape and dtype, and that stability scores are high (>0.9) for consistent predictions and low (<0.5) for oscillating ones.

**Performance tests** enforce all latency targets listed in the benchmarks section above, verify the model stays under 8 million parameters, and confirm the weight file is under 50MB.

**Reliability tests** throw malformed inputs at every component (None values, 1D arrays, single-pixel frames, garbage bytes, empty files) and verify nothing crashes. They test across resolution ranges from 160x120 to 1920x1080, confirm deterministic outputs for identical inputs, validate all image format acceptance (JPEG, PNG, BMP), exercise rate limiting, verify that reset fully clears all state, and run 50 consecutive frames and 200 temporal updates to confirm stability over extended operation.

### Test Configuration

Tests are configured in `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers = slow: marks tests as slow
```

---

## Installation

### Prerequisites

- Python 3.11 or later
- pip
- Google Chrome (for the browser extension)
- NVIDIA GPU with CUDA support (recommended; CPU fallback available)

### Backend Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Real-Time-Video-Deepfake-Detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:

| Package            | Minimum Version | Purpose                                |
|--------------------|----------------|----------------------------------------|
| torch              | 1.9.0          | Deep learning framework                |
| torchvision        | 0.10.0         | EfficientNet-B0 pretrained model       |
| facenet-pytorch    | 2.5.2          | MTCNN face alignment                   |
| opencv-python      | 4.5.3          | Face detection, image preprocessing    |
| Pillow             | 8.0.0          | Image loading and manipulation         |
| numpy              | any            | Numerical operations                   |
| albumentations     | 1.3.0          | Training data augmentation             |
| scikit-learn       | 0.24.0         | Metrics computation                    |
| tqdm               | 4.62.0         | Progress bars                          |
| pyyaml             | 5.4.0          | Configuration parsing                  |
| flask              | 2.0.0          | Backend web server                     |
| flask-cors         | 3.0.10         | Cross-origin request support           |
| mss                | 6.1.0          | Screen capture support                 |
| grad-cam           | 1.3.0          | Optional: model visualization          |

3. Verify the model weights exist:

```bash
ls weights/best_model.pt
```

If weights are not included in the repository, you will need to train the model (see Training section below).

### Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in the top right)
3. Click "Load unpacked"
4. Select the `extension/` folder from this repository
5. The extension icon should appear in the Chrome toolbar

---

## Usage

### Starting Detection

1. Start the backend server:

```bash
python backend_server.py
```

The server will load the model and start listening on port 5000. You should see a health status indicating whether the model loaded on CPU or GPU.

2. Open any web page with a video in Chrome (YouTube, social media, etc.)

3. Click the extension icon in the Chrome toolbar

4. Click "Start Detection"

5. The extension will begin capturing frames and displaying results in both the popup dashboard and the on-page overlay

### Stopping Detection

Click "Stop Detection" in either the popup or the overlay. This sends a reset command to the backend, clearing all temporal state so the next session starts fresh.

### Training the Model

If you need to train from scratch or fine-tune:

```bash
# Start training (auto-resumes if checkpoint exists)
python train.py

# Force a fresh start (ignores existing checkpoints)
python train.py --fresh

# Custom settings
python train.py --epochs 30 --batch_size 16 --lr 3e-4
```

Training requires the FaceForensics++ C23 face crops in `dataset/ff_face_crops/` organized as:

```
dataset/ff_face_crops/
  train/
    real/    (12,750 images)
    fake/    (76,395 images)
  val/
    real/    (~2,250 images)
    fake/    (~13,488 images)
```

Training is fully resumable. If interrupted (Ctrl+C, shutdown, power loss), simply run the same command again and it will pick up from the last completed epoch. The resume checkpoint at `weights/training.pth` stores the epoch counter, optimizer state, scheduler state, gradient scaler state, best metrics, and RNG states.

### Downloading DFDC Dataset (Optional)

```bash
python download_dfdc.py
```

Downloads DFDC parts from Kaggle with balanced sampling (equal real and fake), exponential backoff on rate limits, and full resumability.

### Processing DFDC Dataset (Optional)

```bash
python process_dfdc.py
```

Processes downloaded DFDC data one part at a time (fits in 80GB), with progress tracking via `dataset/dfdc_progress.json`.

---

## Project Structure

```
Real-Time-Video-Deepfake-Detection/
|
|-- backend_server.py           Flask API server (endpoints: /analyze, /reset, /health, /stats)
|-- deepfake_detection.py       DeepfakeDetector class and TemporalTracker
|-- face_detection.py           OpenCV DNN SSD + Haar Cascade face detection
|-- frame_analysis.py           6 forensic signal analyzers (FFT, noise, ELA, edge, color, temporal)
|-- model.py                    DeepfakeEfficientNet model definition
|-- train.py                    Full training pipeline with checkpointing and resumability
|-- download_dfdc.py            DFDC dataset downloader with balanced sampling
|-- process_dfdc.py             DFDC dataset processor
|-- requirements.txt            Python dependencies
|-- pytest.ini                  Test configuration
|-- PROJECT_DOCUMENTATION.md    Detailed technical documentation
|
|-- weights/
|   |-- best_model.pt           Best model checkpoint (state dict)
|   |-- best_model.pth          Alternative checkpoint format
|   |-- training.pth            Full resume checkpoint (optimizer, scheduler, RNG)
|   |-- training_log.json       Epoch-by-epoch training metrics
|
|-- extension/
|   |-- manifest.json           Chrome Extension Manifest V3 configuration
|   |-- background.js           Service worker for backend communication
|   |-- content.js              Frame capture and overlay management
|   |-- popup.html              Extension popup layout
|   |-- popup.js                Popup control logic and metrics display
|   |-- popup.css               Popup styling
|   |-- overlay.html            On-page overlay layout
|   |-- overlay-script.js       Overlay update logic
|   |-- overlay.css             Overlay styling (dark theme, color-coded verdicts)
|   |-- icons/                  Extension icons (16, 48, 128px)
|
|-- tests/
|   |-- test_functional.py      36 functional correctness tests
|   |-- test_algorithm.py       19 algorithm accuracy tests
|   |-- test_performance.py     12 performance benchmark tests
|   |-- test_reliability.py     28 reliability and edge-case tests
|   |-- run_all_tests.py        Custom test runner
|
|-- dataset/
|   |-- ff_face_crops/          FaceForensics++ face crops (train/val, real/fake)
|   |-- dfdc_videos/            DFDC videos (real/fake)
|   |-- FaceForensics++_C23/    Raw FF++ data and metadata CSVs
```

---

## Technology Stack

| Layer              | Technology                                          |
|--------------------|-----------------------------------------------------|
| Deep Learning      | PyTorch, torchvision EfficientNet-B0                |
| Face Detection     | OpenCV DNN SSD (ResNet-10), Haar Cascade fallback   |
| Face Alignment     | MTCNN via facenet-pytorch                           |
| Forensic Analysis  | OpenCV, NumPy, FFT                                  |
| Backend Server     | Flask, Flask-CORS                                   |
| Browser Extension  | Chrome Extension Manifest V3, JavaScript            |
| Training Platform  | Google Colab (GPU runtime)                          |
| Dataset            | FaceForensics++ C23, DFDC (optional)                |
| Testing            | pytest                                              |
| Language           | Python 3.11 (backend), JavaScript (extension)       |

---

## Requirements

The full dependency list from `requirements.txt`:

```
torch>=1.9.0
torchvision>=0.10.0
facenet-pytorch>=2.5.2
opencv-python>=4.5.3
grad-cam>=1.3.0
Pillow>=8.0.0
mss>=6.1.0
numpy
albumentations>=1.3.0
scikit-learn>=0.24.0
tqdm>=4.62.0
pyyaml>=5.4.0
flask>=2.0.0
flask-cors>=3.0.10
```

### Hardware Recommendations

| Component | Minimum          | Recommended                          |
|-----------|------------------|--------------------------------------|
| GPU       | None (CPU works) | NVIDIA GPU with 4GB+ VRAM and CUDA  |
| RAM       | 8GB              | 16GB                                 |
| Storage   | 500MB (weights)  | 100GB+ (if training with datasets)   |

---

## License

See the repository for license information.
