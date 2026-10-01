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

- **Current Milestone**: **`M2 — Recognition (Completed M2.1 Tracking, M2.2 Quality Gate, M2.3 Alignment & Embeddings, M2.4 Recognition & Identity State Machine, M2.5a SQLite Persistence, M2.5b Face Enrollment, M2.5c Person Lifecycle, M2.6 Identity Stability & Two-Rate Pipeline)`**
- **Status**: Implemented & Verified
- **Approach**: Built **incrementally** milestone by milestone. Core software foundation (M0), vision base & YuNet detection (M1), modular face tracking (M2.1), face quality gating (M2.2), face alignment + embedding evaluation (M2.3), real-time recognition + identity state machine (M2.4), SQLite persistence + Person Service (M2.5a), camera-based face enrollment (M2.5b), complete person lifecycle operations (M2.5c), and two-rate perception pipeline with bounded async worker and identity stability (M2.6) are established. Session supervision, target selection, and backend/robot services will follow incrementally.

### Implemented Capabilities
- **Hardware-Independent Frame Sources**: `FrameSource` interface with `VideoFileFrameSource`, `WebcamFrameSource`, and `SyntheticFrameSource`.
- **Thin Face Detector Interface**: `BaseFaceDetector` interface with `YuNetFaceDetector` backed by OpenCV DNN and official YuNet ONNX weights.
- **Modular Multi-Object Face Tracker**: `BaseFaceTracker` interface with `FaceTracker` implementing bipartite IoU matching, spatial centroid fallbacks, configurable track expiration, and bounding box smoothing.
- **Face Quality Gate & Quality Tiers**: `BaseFaceQualityGate` and `FaceQualityGate` evaluating confidence, size, sharpness, and 5-point pose frontality, categorizing crops into operational tiers (`FaceQualityTier`: `REJECTED`, `LOW`, `MEDIUM`, `HIGH`).
- **Face Alignment Module**: `BaseFaceAligner` and `FaceAligner` providing deterministic 5-point similarity transformation to canonical dimensions (default 112x112) with bounding-box fallbacks.
- **Face Feature Embedders**: `BaseFaceEmbedder` and `SFaceEmbedder` extracting L2-normalized 128-D embeddings using OpenCV DNN (supporting FP32 and INT8 quantized variants).
- **In-Memory Face Recognition & Gallery**: `BaseFaceRecognizer`, `FaceRecognizer`, and `InMemoryGallery` providing cosine similarity matching, multi-embedding aggregation (`max`/`mean`), top-1 vs. top-2 ambiguity margin gating, and model/version compatibility validation.
- **Per-Track Identity State Machine & ID Swap Protection**: `IdentityStateManager` managing track lifecycle transitions (`NEW` → `COLLECTING` → `IDENTIFIED` / `UNREGISTERED` / `UNCERTAIN` → `LOST`) with sliding-window temporal consistency voting, hysteresis conflict patience, spatial crossing detection (`check_track_crossings`), and track identity invalidation.
- **Bounded Asynchronous Embedding Worker**: `BaseEmbeddingWorker` and `BoundedEmbeddingWorker` decoupling expensive embedding inference from the real-time frame loop with latest-wins replacement per track and bounded queue capacity.
- **Two-Rate Perception Pipeline Orchestrator**: `TwoRatePerceptionPipeline` integrating real-time 30+ FPS detection, tracking, quality tiers, and target focus with decoupled background recognition worker.
- **Multi-Person Interaction Target Selector**: `BaseTargetSelector` and `InteractionTargetSelector` providing deterministic composite scoring (centrality, face size, identity state, dwell time) with stickiness hysteresis to prevent target flickering.
- **Temporal Presence Debouncer**: `PresenceDebouncer` smoothing entry (`PERSON_ENTERED`) and departure (`PERSON_LEFT`) transitions to eliminate detection dropout flicker.
- **SQLite Database Layer & Migrations**: `DatabaseManager` providing connection lifecycle, schema version tracking, thread-safe single-writer locking, foreign key constraints (`PRAGMA foreign_keys = ON;`), and `WAL` journal mode without external ORM dependencies.
- **SQLite Person & Embedding Repository**: `BasePersonRepository` and `SQLitePersonRepository` providing transactional CRUD operations for person records and binary 128-D float32 embeddings, atomic embedding replacement (`replace_embeddings_for_person`), and foreign key cascade deletions.
- **Person Service Layer**: `BasePersonService` and `PersonService` managing person lifecycle semantics, validation (UUIDs, required names, status and consent enums), profile abstractions without exposing raw feature vectors, and biometric resets.
- **Camera-Based Face Enrollment & Re-Registration**: `BaseFaceEnrollment` and `FaceEnrollmentController` supporting both `ADD_SAMPLES` (with stored vector deduplication) and `REPLACE` (staged replacement with zero-data-loss rollback protection), single dominant face locking, strict quality filtering, sample diversity deduplication (cosine similarity $< 0.95$), transactional batch persistence, and instant in-memory gallery reloading.
- **In-Memory Gallery Synchronization**: `load_gallery_from_repository` extracting active, consented, model-compatible embeddings into an isolated `InMemoryGallery` for real-time recognition without database querying on camera frames.
- **Reproducible Model Acquisition**: `scripts/download_models.py` with SHA-256 cryptographic verification and model cards documentation in `models/MODEL_CARDS.md`.
- **Monotonic Benchmark Utility**: `PipelineBenchmark` measuring frame count, average/rolling FPS, and latency percentiles (mean, min, p50, p90, p95, p99, max).
- **Benchmarking & Validation Suites**: `benchmarks/benchmark_m2_6_pipeline.py`, `benchmarks/validate_detection.py`, `benchmarks/benchmark_tracking.py`, `benchmarks/validate_quality_gate.py`, `benchmarks/benchmark_embedder.py`, `benchmarks/benchmark_recognition.py`, and `benchmarks/benchmark_detector.py`.

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
