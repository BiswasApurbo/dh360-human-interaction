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

- **Current Milestone**: **`M2 — Recognition, Session Supervision & Persistent Audit History (Completed M2.1 Tracking, M2.2 Quality Gate, M2.3 Alignment & Embeddings, M2.4 Recognition & Identity State Machine, M2.5a SQLite Persistence, M2.5b Face Enrollment, M2.5c Person Lifecycle, M2.6 Identity Stability & Two-Rate Pipeline, M2.7 Session Lifecycle Manager & Structured Event Bus, M2.8 Edge Interaction Runtime & Frame Pump, M2.9 Interaction Context Coordinator & Application Event Bridge, M2.10 Unified Application Service & Composition Root, M2.11 Persistent Interaction Session Store & Historical Audit Repository, M2.12 Session Analytics & Historical Query Layer, M2.13 Application Integration Boundary & Transport-Neutral Contracts, M2.14 Unified Configuration & Local Operator CLI)`**
- **Status**: Implemented & Verified
- **Approach**: Built **incrementally** milestone by milestone. Core software foundation (M0), vision base & YuNet detection (M1), modular face tracking (M2.1), face quality gating (M2.2), face alignment + embedding evaluation (M2.3), real-time recognition + identity state machine (M2.4), SQLite persistence + Person Service (M2.5a), camera-based face enrollment (M2.5b), complete person lifecycle operations (M2.5c), two-rate perception pipeline with bounded async worker and identity stability (M2.6), session lifecycle manager with structured in-process event bus (M2.7), edge interaction runtime with continuous frame pump (M2.8), interaction context coordinator with application event bridge (M2.9), unified application service with composition root (M2.10), persistent interaction session store with historical audit repository (M2.11), session analytics with historical query layer (M2.12), application integration boundary with transport-neutral contracts (M2.13), and unified configuration loader with local operator CLI (M2.14) are established. Backend service endpoints and robot behavioral interfaces will follow incrementally.


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
- **Typed Domain Event Contracts**: `BaseEvent`, `PersonEnteredEvent`, `SessionStartedEvent`, `PersonIdentifiedEvent`, `PersonLostIdentityEvent`, `PersonLeftEvent`, `SessionEndedEvent` in `dh360_hi.contracts.events` encapsulating pure domain semantics without vision tensor leakage.
- **Synchronous In-Process Event Bus**: `EventBus` in `dh360_hi.edge.event_bus` providing deterministic type-filtered dispatch, subscriber exception isolation, and unsubscription without background queues or external brokers.
- **Session Lifecycle Supervision**: `SessionManager` in `dh360_hi.edge.session_manager` managing primary interaction target session state (`NO_SESSION` ↔ `ACTIVE_SESSION`), target switching stability, deduplicated identity changes, and unregistered person interactions.
- **Edge Interaction Runtime Orchestrator**: `EdgeInteractionRuntime` in `dh360_hi.edge.runtime` supervising the end-to-end edge pipeline (`FrameSource` → `TwoRatePerceptionPipeline` → `SessionManager` → `EventBus`) with lifecycle management (`start`, `stop`, `pause`, `resume`, `step_once`), constant-rate FPS pacing, transient error resilience, and operational metrics telemetry (`RuntimeMetrics`).
- **Interaction Context Coordinator & Application Boundary**: `InteractionCoordinator` in `dh360_hi.application.coordinator` maintaining queryable, thread-safe application context (`InteractionContext`), profile enrichment via `PersonService`, finalized historical session records (`InteractionSummary`), and listener hooks for downstream services.
- **Unified Application Service & Composition Root**: `HumanInteractionService` and `create_interaction_service` in `dh360_hi.application` providing a single typed entrypoint for subsystem assembly, lifecycle supervision, context/telemetry queries, and gallery hot-swapping.
- **Persistent Interaction Session Store & Audit Repository**: `BaseSessionRepository`, `SQLiteSessionRepository`, and `SessionPersistenceWorker` in `dh360_hi.backend.session_repository` storing completed `InteractionSummary` records via Schema v2 migrations, providing non-blocking asynchronous queueing, zero record drops, and chronological querying (`get_session`, `list_sessions`, `count_sessions`).
- **Session Analytics & Historical Interaction Query Layer**: `SessionQueryFilter`, `SessionAggregateAnalytics`, and `PersonEngagementStats` in `dh360_hi.backend` providing parameter-filtered session queries with pagination, SQL-side aggregations (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`), and person-level engagement summaries without raw biometric leakage or real-time perception loop interference.
- **Application Integration Boundary & Transport-Neutral Contracts**: `dh360_hi.contracts.dto`, `integration`, `errors`, and `health` providing frozen JSON-serializable DTOs (`PersonProfileDTO`, `InteractionContextDTO`, `InteractionSummaryDTO`, `InteractionSessionDTO`, `StepResultDTO`), external integration events with stripped vision internals (`track_id`, `frame_id`), structured error taxonomy (`ApplicationError`), and non-throwing health diagnostics (`ServiceHealthSnapshot`).

- **Reproducible Model Acquisition**: `scripts/download_models.py` with SHA-256 cryptographic verification and model cards documentation in `models/MODEL_CARDS.md`.
- **Monotonic Benchmark Utility**: `PipelineBenchmark` measuring frame count, average/rolling FPS, and latency percentiles (mean, min, p50, p90, p95, p99, max).
- **Benchmarking & Validation Suites**: `benchmarks/benchmark_m2_8_runtime.py`, `benchmarks/benchmark_m2_6_pipeline.py`, `benchmarks/validate_detection.py`, `benchmarks/benchmark_tracking.py`, `benchmarks/validate_quality_gate.py`, `benchmarks/benchmark_embedder.py`, `benchmarks/benchmark_recognition.py`, and `benchmarks/benchmark_detector.py`.

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

### Operator Profile API (M3.2.1)

The person-profile endpoints require an operator bearer token and explicit scopes. Configure the token only through the environment; configuration serialization and logs must not contain it. Generate a new local token without printing it to the terminal:

```bash
export DH360_HI_API__OPERATOR_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export DH360_HI_API__OPERATOR_SCOPES="persons:read,persons:create,persons:update,persons:delete,biometrics:reset,consent:manage,enrollment:manage,interaction:read,authorization:manage"
export DH360_HI_DB_PATH="dh360_hi.db"
uvicorn --factory dh360_hi.api.app:create_app --host 127.0.0.1 --port 8000
```

Send the token only in the `Authorization: Bearer` header. The profile API fails closed if the token or scopes are not configured. M3.1 health, runtime, interaction, and session routes retain their existing access behavior and are not made authenticated by M3.2.1. Keep the service bound to loopback unless a separately secured network deployment is configured.

M3.2.2 adds `persons:update`, `persons:delete`, `biometrics:reset`, and `consent:manage`. Consent changes require both `persons:update` and `consent:manage`. Profile updates, person deletion, and biometric reset briefly pause frame processing at a completed frame boundary, quiesce recognition work, reconcile the in-memory gallery and active identity/session state, then restore the prior runtime state. Run one live service owner per database; process-local gallery/runtime state is not synchronized across multiple API/CLI processes. Person deletion detaches historical session person IDs but retains their display names and identity-known history.

### 4. Run Functional Face Detection Validation & Benchmark
```bash
# 1. Functional Face Detection Validation (exercises detector on reference fixture)
python benchmarks/validate_detection.py --frames 100

# 2. Performance Throughput Benchmark (pure throughput profiling)
python benchmarks/benchmark_detector.py --synthetic --frames 200
```

### 5. Application Service Usage Example
```python
from dh360_hi.application import (
    HumanInteractionConfig,
    SessionQueryFilter,
    create_interaction_service,
)
from dh360_hi.contracts import CreatePersonCommand
from dh360_hi.vision.frame import WebcamFrameSource

# 1. Configure subsystem with models and SQLite persistence
config = HumanInteractionConfig(
    db_path="data/dh360.db",
    detector_model_path="models/face_detection_yunet_2023mar.onnx",
    embedder_model_path="models/face_recognition_sface_2021dec.onnx",
)

# 2. Instantiate composed service using composition root
service = create_interaction_service(
    config=config,
    frame_source=WebcamFrameSource(device_index=0),
)

# 3. Create a person profile via typed command
person = service.create_person(
    CreatePersonCommand(display_name="Dr. Jane Doe", preferred_name="Jane")
)
print(f"Created person profile: {person.display_name} (ID: {person.person_id})")

# 4. Subscribe to sanitized integration events
sub_id = service.add_integration_event_listener(
    lambda ev: print(f"Integration event [{ev.event_type}]: {ev.to_dict()}")
)

# 5. Start background frame pump and query active interaction context DTO
service.start(async_thread=True)

context_dto = service.get_interaction_context()
print(f"Current State: {context_dto.state}, Person: {context_dto.display_name}")

# 6. Filtered historical session queries & database-side analytics DTOs
query_filter = SessionQueryFilter(is_identified=True, min_duration_seconds=5.0, limit=10)
filtered_sessions = service.query_sessions(query_filter)
for s in filtered_sessions:
    print(f"Identified Session {s.session_id}: {s.duration_seconds:.1f}s ({s.display_name})")

analytics = service.get_aggregate_analytics()
print(
    f"Total Sessions: {analytics.total_sessions}, "
    f"Avg Duration: {analytics.average_duration_seconds:.1f}s, "
    f"Identified Sessions: {analytics.identified_sessions_count}"
)

# 7. Query defensive subsystem health snapshot
health = service.get_health_snapshot()
print(f"Overall Service Health: {health.status.value}")

# 8. Graceful shutdown
service.stop()
```

### 6. Local Operator CLI (`dh360-hi`)

The repository installs a standalone operator CLI `dh360-hi` for runtime management, diagnostics, face enrollment, and profile administration:

```bash
# 1. Show active resolved configuration & presets
dh360-hi config show --preset production
dh360-hi config validate --config-file config.toml

# 2. Inspect subsystem operational health
dh360-hi health --preset development
dh360-hi health --json

# 3. Manage enrolled patient/staff profiles
dh360-hi person create --display-name "Nurse Joy" --preferred-name "Joy" --status active --consent-status granted
dh360-hi person list
dh360-hi person get <person_id>

# 4. Perform guided face enrollment session
dh360-hi enroll <person_id> --source webcam

# 5. Inspect historical interaction sessions & analytics
dh360-hi session list
dh360-hi session query --is-identified --min-duration 5.0
dh360-hi session analytics
dh360-hi session stats <person_id>

# 6. Launch continuous edge runtime
dh360-hi run --source webcam --target-fps 30.0 --json-events
```

> [!NOTE]
> **Performance Disclaimer**:
> Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance.
> The `> 15 FPS` requirement is a future deployment target on Raspberry Pi.

---

## 🍓 Raspberry Pi 5 Deployment & Setup Guide

This guide details setting up and running `dh360-human-interaction` on a fresh **Raspberry Pi 5 (8 GB RAM, 64-bit OS / Debian Bookworm)** from a fresh Git clone.

### 1. Install Linux System Packages
Install required system libraries for Python virtual environments, OpenCV headless execution, and USB camera tooling:
```bash
sudo apt update && sudo apt install -y \
    python3-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    v4l-utils
```
> [!NOTE]
> `libgl1` and `libglib2.0-0` are essential shared libraries for `opencv-python` on Debian Linux. `v4l-utils` provides the `v4l2-ctl` diagnostic utility for USB UVC cameras.

Ensure your user account has access to video devices:
```bash
sudo usermod -aG video $USER
# Apply group membership immediately without logging out:
newgrp video
```

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/BiswasApurbo/dh360-human-interaction.git
cd dh360-human-interaction

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```
*(Optional: use `pip install -e ".[dev]"` if you plan to run unit tests and linters on the Pi).*

### 3. Acquire Pretrained Vision Models
The repository includes a reproducible downloader with SHA-256 cryptographic verification:
```bash
python scripts/download_models.py
```
This fetches:
- `models/weights/face_detection_yunet_2023mar.onnx` (YuNet Face Detector)
- `models/weights/face_recognition_sface_2021dec.onnx` (SFace FP32 Recognizer)
- `models/weights/face_recognition_sface_2021dec_int8.onnx` (SFace INT8 Quantized)
- `tests/fixtures/sample_face.jpg` (Test Fixture)

### 4. Connect & Identify USB UVC Webcam
Plug in your USB UVC camera and identify its Video4Linux2 device node:
```bash
v4l2-ctl --list-devices
```
Example output:
```text
USB Camera (usb-xhci-hcd.0-1):
    /dev/video0
    /dev/video1
```
The numeric index at the end of `/dev/video0` corresponds to `--camera-index 0`.

### 5. Configure Operator Authentication Token
The REST API and Demo UI require an operator Bearer token with appropriate scopes:
```bash
export DH360_HI_API__OPERATOR_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export DH360_HI_API__OPERATOR_SCOPES="persons:read,persons:create,persons:update,persons:delete,biometrics:reset,consent:manage,enrollment:manage,interaction:read,authorization:manage"
export DH360_HI_DB_PATH="dh360_hi.db"

# Print token to copy into the browser Demo UI:
echo "Generated Operator Token: $DH360_HI_API__OPERATOR_TOKEN"
```

### 6. Start the Unified Server & Interactive Demo UI
Launch the unified live perception pipeline, authorization controller, and API server:
```bash
# To allow accessing the Demo UI from another machine on your local network:
dh360-hi serve --source webcam --camera-index 0 --host 0.0.0.0 --port 8000
```
Open a browser and navigate to:
- Locally on Pi: `http://127.0.0.1:8000/demo`
- Across LAN: `http://<raspberry-pi-ip>:8000/demo`

Paste your operator token into the **Operator Bearer Token** field in the UI. You can now:
- Enroll new identities using guided head poses (FRONT, RIGHT, LEFT).
- Perform live face recognition and authorization.
- Inspect and delete registered persons.

### 7. Run Verification Checks (Optional)
```bash
pytest tests/ -q
```

---

## 🔐 Security & Privacy
- Face embeddings represent sensitive biometric information and are handled with strict privacy-by-design standards.
- Real API keys, credentials, local captures, logs, and pretrained binary models are strictly excluded from version control (see `.gitignore` and `.env.example`).
