# Digital Health 360: Human Interaction Module (`dh360-human-interaction`)

A modular edge-AI and conversational interaction system for Digital Health 360, combining real-time face detection, tracking, recognition, FastAPI backend services, and LLM-based interaction for a semi-humanoid robot.

---

## 📌 Project Overview & Purpose

The Human Interaction Module is responsible for human perception and dialogue management in healthcare environments:
- **Vision Pipeline**: Real-time face detection, multi-person tracking, face quality filtering, feature embedding extraction, and local gallery matching.
- **Edge Runtime**: Target selection, interaction session supervision, event transport, and robot physical interface commands.
- **Backend Services**: Asynchronous FastAPI endpoints, conversation history, LLM orchestration, and WebSocket streaming.

---

## 🚀 Current Development Status

- **Current Milestone**: **`M2 — Recognition (Tracking, Quality Gate, Alignment & Embeddings Sub-Steps)`**
- **Status**: Implemented & Verified
- **Approach**: Built **incrementally** milestone by milestone. Core software foundation (M0), vision base & YuNet detection (M1), modular face tracking (M2.1), face quality gating (M2.2), and face alignment + embedding evaluation (M2.3) are established. Gallery matching, recognition decision logic, backend services, or hardware integrations will follow incrementally.

### Implemented Capabilities
- **Hardware-Independent Frame Sources**: `FrameSource` interface with `VideoFileFrameSource`, `WebcamFrameSource`, and `SyntheticFrameSource`.
- **Thin Face Detector Interface**: `BaseFaceDetector` interface with `YuNetFaceDetector` backed by OpenCV DNN and official YuNet ONNX weights.
- **Modular Multi-Object Face Tracker**: `BaseFaceTracker` interface with `FaceTracker` implementing bipartite IoU matching, spatial centroid fallbacks, configurable track expiration, and bounding box smoothing.
- **Face Quality Gate**: `BaseFaceQualityGate` and `FaceQualityGate` evaluating detector confidence, bounding box size, image sharpness (Laplacian variance), and 5-point facial pose frontality (roll tilt and yaw asymmetry) with configurable thresholds and rejection reasons.
- **Face Alignment Module**: `BaseFaceAligner` and `FaceAligner` providing deterministic 5-point similarity transformation to canonical dimensions (default 112x112) with bounding-box fallbacks.
- **Face Feature Embedders**: `BaseFaceEmbedder` and `SFaceEmbedder` extracting L2-normalized 128-D embeddings using OpenCV DNN (supporting FP32 and INT8 quantized variants).
- **Reproducible Model Acquisition**: `scripts/download_models.py` with SHA-256 cryptographic verification and model cards documentation in `models/MODEL_CARDS.md`.
- **Monotonic Benchmark Utility**: `PipelineBenchmark` measuring frame count, average/rolling FPS, and latency percentiles (mean, min, p50, p90, p95, p99, max).
- **Benchmarking & Validation Suites**: `benchmarks/validate_detection.py`, `benchmarks/benchmark_tracking.py`, `benchmarks/validate_quality_gate.py`, `benchmarks/benchmark_embedder.py`, and `benchmarks/benchmark_detector.py`.

---

## 🏛 Architecture & Design Records

- **Architecture Overview**: [docs/architecture.md](docs/architecture.md)
- **Model Specifications**: [models/MODEL_CARDS.md](models/MODEL_CARDS.md)
- **Architectural Decision Records**:
  - [ADR 0001: Initial Architecture and Design Decisions](docs/adr/0001-initial-architecture.md)

---

## 🛠 Development & Tooling

This project requires **Python 3.11+** (preferring Python 3.11 for edge compatibility).

### 1. Setup Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Download Pretrained Models
```bash
python scripts/download_models.py
```

### 3. Run Code Quality Checks & Tests
```bash
# Linting & code style
ruff check .
ruff format --check .

# Static type checking
mypy

# Unit & integration tests
pytest
```

### 4. Run Functional Face Detection Validation & Benchmark
```bash
# 1. Functional Face Detection Validation (exercises detector on reference fixture)
python benchmarks/validate_detection.py --frames 100

# 2. Performance Throughput Benchmark (pure throughput profiling)
python benchmarks/benchmark_detector.py --synthetic --frames 200
```

> [!NOTE]
> **Performance Disclaimer**:
> Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance.
> The `> 15 FPS` requirement is a future deployment target on Raspberry Pi.

---

## 🔐 Security & Privacy
- Face embeddings represent sensitive biometric information and are handled with strict privacy-by-design standards.
- Real API keys, credentials, local captures, logs, and pretrained binary models are strictly excluded from version control (see `.gitignore` and `.env.example`).
