# Benchmarks & Validation Suites

This directory contains benchmarking suites and performance profiling tools for the Digital Health 360 Human Interaction module.

---

## 1. Functional Face Detection Validation (`validate_detection.py`)

Exercises the YuNet face detector on standard non-identifying reference fixtures ([tests/fixtures/sample_face.jpg](file:///Users/apurbobiswas/Library/CloudStorage/OneDrive-AmericanInternationalUniversity-Bangladesh%282%29/KBCon/dh360-human-interaction/tests/fixtures/sample_face.jpg)) to verify detection accuracy, bounding box bounds, 5-point facial landmark localization, and confidence statistics.

### Usage
```bash
# Run functional validation over 100 animated fixture frames
python benchmarks/validate_detection.py --frames 100
```

### Metrics Reported
- **Input Resolution & Detector Resolution**
- **Total Frames Processed & Frames with Detected Faces**
- **Detection Success Rate (%)**
- **Confidence Statistics**: Mean, Min, Max, p50
- **Sample Face Coordinates**: Bounding box `(x, y, w, h)` and landmark coordinates
- **Throughput & Latency Distribution**: Average FPS, p50, p95, p99 (ms)

---

## 2. Detection + Tracking Combined Benchmark (`benchmark_tracking.py`)

Measures combined detection and multi-object face tracking throughput and per-stage latency distributions (`face_detection` vs. `face_tracking`).

### Usage
```bash
# Benchmark on animated fixture video (150 frames)
python benchmarks/benchmark_tracking.py --frames 150

# Benchmark on video file
python benchmarks/benchmark_tracking.py --video data/test_video.mp4 --frames 150
```

---

## 3. Face Quality Gate Validation (`validate_quality_gate.py`)

Exercises the `FaceQualityGate` pipeline over detected and tracked faces to evaluate sharpness (Laplacian variance), bounding box size, detector confidence, and head pose frontality (roll tilt and yaw asymmetry) to verify rejection/acceptance metrics and latency.

### Usage
```bash
# Run quality gate validation over 100 animated fixture frames
python benchmarks/validate_quality_gate.py --frames 100
```

---

## 4. Face Alignment & Embedding Model Benchmark (`benchmark_embedder.py`)

Measures 5-point similarity transform alignment latency, embedding inference latency, throughput (FPS), and percentile latency distributions across candidate face feature embedding models (SFace FP32 vs. SFace INT8).

### Usage
```bash
# Benchmark all embedding candidate models (100 frames)
python benchmarks/benchmark_embedder.py --model all --frames 100

# Benchmark specific model
python benchmarks/benchmark_embedder.py --model sface --frames 100
python benchmarks/benchmark_embedder.py --model sface_int8 --frames 100
```

---

## 5. Offline Face Verification & Threshold Calibration (`evaluate_embeddings_dataset.py`)

Constructs paired genuine (same identity, varying conditions) and impostor (different identity) trials to evaluate cosine similarity distributions, False Accept Rate (FAR), False Reject Rate (FRR), Equal Error Rate (EER), and threshold calibration without hardcoded assumptions.

### Usage
```bash
# Evaluate verification metrics and threshold operating points for SFace FP32
python benchmarks/evaluate_embeddings_dataset.py --model sface

# Evaluate SFace INT8 quantized
python benchmarks/evaluate_embeddings_dataset.py --model sface_int8
```

---

## 6. End-to-End Recognition & Identity State Benchmark (`benchmark_recognition.py`)

Measures per-stage latency percentiles and overall pipeline throughput across the complete integrated vision chain:
`Detection -> Tracking -> Quality Gate -> Face Alignment -> SFace Feature Embedding -> Gallery Matching -> Identity State Machine`.

### Usage
```bash
# Benchmark full end-to-end recognition pipeline (100 frames)
python benchmarks/benchmark_recognition.py --frames 100
```

---

## 7. Throughput Performance Benchmark (`benchmark_detector.py`)

Measures pipeline throughput, sliding-window rolling FPS, and latency percentiles under sustained execution on video files, synthetic streams, or webcam input.

### Usage
```bash
# Benchmark using synthetic frame stream (200 frames)
python benchmarks/benchmark_detector.py --synthetic --frames 200

# Benchmark using prerecorded video
python scripts/generate_test_video.py --output data/test_video.mp4 --frames 200
python benchmarks/benchmark_detector.py --video data/test_video.mp4 --frames 200

# Benchmark live webcam (if available)
python benchmarks/benchmark_detector.py --webcam 0 --frames 100
```

---

> [!NOTE]
> **Performance Disclaimer**:
> Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance.
> The `> 15 FPS` requirement is a future deployment target on Raspberry Pi.
