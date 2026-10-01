# Architecture: Digital Health 360 Human Interaction Module

## 1. Project Purpose

The **Digital Health 360 Human Interaction Module** (`dh360-human-interaction`) provides a modular, real-time edge vision and conversational interaction system for a semi-humanoid robot in digital healthcare environments.

The system is designed to:
- Detect, track, and recognize human faces in real-time.
- Identify individuals against a local gallery.
- Manage interaction sessions and presence tracking.
- Orchestrate context-aware conversational dialogues via an asynchronous Large Language Model (LLM) backend.
- Command robot behaviors and feedback mechanisms based on user presence and conversation state.

---

## 2. High-Level Architecture Overview

The system is strictly partitioned into three main domains: **Edge Runtime**, **Backend Service**, and **Shared Contracts**.

```
+-----------------------------------------------------------------------------------+
|                                    EDGE LAYER                                     |
|                                                                                   |
|  [ FrameSource ]                                                                  |
|        │                                                                          |
|        ▼                                                                          |
|  [ Face Detector ]                                                                |
|        │                                                                          |
|        ▼                                                                          |
|  [ Tracker ]                                                                      |
|        │                                                                          |
|        ▼                                                                          |
|  [ Face Quality Gate ]                                                            |
|        │                                                                          |
|        ▼                                                                          |
|  [ Embedding Worker ] ───► [ Local Gallery Matching ]                             |
|                                     │                                             |
|                                     ▼                                             |
|                             [ Identity Decision ]                                 |
|                                     │                                             |
|                                     ▼                                             |
|                             [ Target Selection ]                                  |
|                                     │                                             |
|                                     ▼                                             |
|                       [ Presence / Session Supervisor ]                           |
|                                     │                                             |
|                                     ▼                                             |
|                             [ Event Transport ]                                   |
|                                     │                                             |
|                                     ▼                                             |
|                             [ Robot Interface ]                                   |
+─────────────────────────────────────┼─────────────────────────────────────────────+
                                      │ (Asynchronous Network Transport / Events)
                                      ▼
+-----------------------------------------------------------------------------------+
|                                   BACKEND LAYER                                   |
|                                                                                   |
|                                [ FastAPI REST API ]                               |
|                                         │                                         |
|                 ┌───────────────────────┼────────────────────────┐                |
|                 ▼                       ▼                        ▼                |
|           [ Database ]      [ Conversation Manager ]      [ WebSocket Hub ]       |
|                                         │                        │                |
|                                         ▼                        ▼                |
|                                   [ LLM Adapter ]     [ Robot/Edge Responses ]    |
+-----------------------------------------------------------------------------------+
                                          ▲
                                          │
+─────────────────────────────────────────┴─────────────────────────────────────────+
|                                 SHARED CONTRACTS                                  |
|                                                                                   |
|  • schemas/events.py (PresenceEvent, RecognitionEvent, InteractionStateEvent)      |
|  • schemas/messages.py (UserMessage, AssistantMessage, DialogueContext)           |
|  • schemas/robot.py (GazeCommand, ExpressionCommand, MotorAction)                 |
+───────────────────────────────────────────────────────────────────────────────────+
```

---

## 3. Strict Edge / Backend Separation

To ensure system stability, predictability, and safety in real-time robotic operations:

1. **Decoupled Real-Time Loop**: The backend is **never** placed in the synchronous camera frame processing loop. Frame acquisition, detection, and tracking operate completely decoupled on the edge device.
2. **Non-Blocking LLM Integration**: LLM inference and external network operations are asynchronous and event-driven. Slow or fluctuating LLM response times will never drop vision frame rates or block tracking.
3. **Hardware Independence**: Core computer vision algorithms, mathematical vector operations, and business logic are written to be hardware-agnostic. Platform-specific capture (e.g., V4L2/libcamera on Linux vs. AVFoundation on macOS) and hardware acceleration layers are isolated within edge adapters.

---

## 4. Mac Development to Raspberry Pi Deployment Strategy

Development follows a two-tier strategy:

- **MacBook M4 Air (Development Tier)**:
  - Rapid algorithm development, unit testing, schema design, and modular benchmarking.
  - Video stream simulation, test harnesses, and mocked sensor feeds.
  - Backend development, prompt engineering, and conversational testing.

- **Raspberry Pi (Deployment Tier)**:
  - Target edge platform hosting the real-time vision pipeline and robot hardware interfaces.
  - Execution optimized using lightweight ONNX Runtime / CPU-friendly pretrained models.
  - **Performance Requirement**: Sustained **`> 15 FPS`** real-time vision processing throughput under continuous operation.

---

## 5. Major System Components

### 5.1 Edge Components

| Component | Responsibility |
| :--- | :--- |
| **FrameSource** | Hardware-abstracted frame capture interface (webcam, CSI camera, RTSP, video file). |
| **Face Detector** | Lightweight, high-speed face detection producing bounding boxes and 5-point facial landmarks (`YuNetFaceDetector`). |
| **Tracker** | Multi-object tracking associating bounding boxes across frames with persistent track IDs (`FaceTracker`). |
| **Face Quality Gate** | Evaluates detector confidence, bounding box size, sharpness (Laplacian variance), and pose frontality (`FaceQualityGate`). |
| **Face Aligner** | Warps and crops faces via 5-point similarity transformation to canonical 112x112 aligned crops (`FaceAligner`). |
| **Embedding Worker** | Extracts L2-normalized 128-D feature embeddings using lightweight pretrained models (`SFaceEmbedder`). |
| **Local Gallery Matching** | Evaluates cosine similarity against registered in-memory identities with model compatibility checks (`FaceRecognizer`, `InMemoryGallery`). |
| **Identity State Machine** | Applies multi-observation temporal smoothing and hysteresis patience to stabilize track identities (`IdentityStateManager`). |
| **Target Selection** | Determines primary user of interest when multiple persons are visible. |
| **Presence / Session Supervisor** | Tracks user entry, dwell time, active engagement, and departure states. |
| **Event Transport** | Emits structured edge events to the backend over reliable, low-overhead protocols. |
| **Robot Interface** | Translates interaction commands (gaze tracking, audio cues, LED/screen expressions) to robot hardware. |

#### 5.1.1 Recognition & Gallery Architecture

The recognition module (`FaceRecognizer`) provides modular in-memory identity matching:
- **`InMemoryGallery`**: Thread-safe in-memory store associating `person_id` and `display_name` with one or more normalized 128-D feature vectors (`GalleryEntry`).
- **Cosine Similarity Matching**: Uses dot products of unit L2-normalized vectors (`compute_cosine_similarity`).
- **Multi-Embedding Aggregation**: Configurable aggregation strategies (`max` or `mean` similarity across enrolled samples per identity).
- **Top-1 vs. Top-2 Margin Gating**: Requires both `top_similarity >= similarity_threshold` (default `0.65`) and `(top_similarity - second_similarity) >= min_similarity_margin` (default `0.08`). Ambiguous candidate pairs return `UNCERTAIN` instead of guessing.
- **Model Version Compatibility**: Rejects or ignores gallery vectors produced by incompatible model versions.

#### 5.1.2 Identity State Machine Lifecycle & Hysteresis

To prevent identity flickering caused by momentary face occlusions, lighting transients, or profile views, the `IdentityStateManager` manages a state machine per track ID:

```
                  ┌──────────────┐
                  │     NEW      │
                  └──────┬───────┘
                         │ (First observation)
                         ▼
                  ┌──────────────┐
        ┌────────►│  COLLECTING  │◄────────┐
        │         └──────┬───────┘         │
        │                │                 │
        │ (Contradiction │ (Consistent     │ (Sustained
        │  / Ambiguity)  │  Evidence)      │  Conflict)
        │                ▼                 │
  ┌─────┴────────┐┌──────────────┐┌────────┴──────┐
  │  UNCERTAIN   ││  IDENTIFIED  ││ UNREGISTERED  │
  └──────────────┘└──────┬───────┘└───────────────┘
                         │
                         │ (Track expired / max missed frames)
                         ▼
                  ┌──────────────┐
                  │     LOST     │
                  └──────────────┘
```

- **States**:
  - `NEW`: Track initialized; awaiting initial recognition observations.
  - `COLLECTING`: Accumulating observations over a sliding evidence window (default: 7 observations, minimum 3 to decide).
  - `IDENTIFIED`: Track confirmed as a registered identity based on consistent thresholded matches (`consistency_ratio >= 0.70`).
  - `UNREGISTERED`: Track consistently identified as unknown / below recognition threshold.
  - `UNCERTAIN`: Conflicting identities in window or top candidates within the ambiguity margin.
  - `LOST`: Track has been missing for more than `max_missed_frames_to_lost` (default: 30 frames).
- **Hysteresis Conflict Patience**: Once a track reaches `IDENTIFIED`, isolated conflicting observations (e.g. single blurry frame) do not immediately flip the state. The state machine requires sustained conflicting observations (`hysteresis_patience_observations >= 4`) before demoting the identity to `UNCERTAIN`.

#### 5.1.3 Camera-Based Face Enrollment Subsystem

The enrollment subsystem (`FaceEnrollmentController`, `BaseFaceEnrollment`) provides a safe, operator-controlled workflow capturing multiple high-quality face samples from the robot camera and persisting their embeddings without per-user model fine-tuning:

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Camera Frame   │────►│ Face Detection   │────►│ Single Dominant  │
└─────────────────┘     │     (YuNet)      │     │    Face Check    │
                        └──────────────────┘     └────────┬─────────┘
                                                          │ Exactly 1 face
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ SFace Embedding │◄────│  Face Alignment  │◄────│ FaceQualityGate  │
│ (128-D float32) │     │ (112x112 affine) │     │ (Confidence, Size│
└────────┬────────┘     └──────────────────┘     │  Sharpness, Pose)│
         │                                       └──────────────────┘
         ▼
┌──────────────────────────────────────────┐
│        Diversity & Duplicate Check       │
│ (Cosine Similarity < 0.95 vs prev saved) │
└────────┬─────────────────────────────────┘
         │ Passed
         ▼
┌──────────────────────────────────────────┐
│     In-Memory Session Sample Buffer      │
│         (5–10 Samples Target)            │
└────────┬─────────────────────────────────┘
         │ Operator Commit (>= 5 samples)
         ▼
┌──────────────────────────────────────────┐
│   Transactional Batch Persistence &      │
│   Atomic FaceRecognizer Gallery Reload   │
└──────────────────────────────────────────┘
```

- **Single Dominant Face Lock**: Requires strictly one detected face in the frame (`FaceDetection`). Frames with zero faces (`NO_FACE_DETECTED`) or multiple faces (`MULTIPLE_FACES_DETECTED`) are rejected to avoid accidental multi-subject blending.
- **Strict Quality Gating**: Every candidate face must pass `FaceQualityGate` thresholds (confidence $\ge 0.80$, bounding box $\ge 80\times 80$, sharpness $\ge 50.0$, roll $\le 15^\circ$, yaw $\le 2.0$).
- **Diversity & Duplicate Protection**: Consecutive camera frames often have nearly identical poses and expressions. The controller compares the candidate embedding against all previously buffered session embeddings; samples with cosine similarity $\ge 0.95$ are rejected (`DUPLICATE_SAMPLE`) to enforce visual diversity (angles, expressions, subtle lighting variations).
- **Consent & Active Status Validation**: Starting a session verifies with `PersonService` that the person exists, is `ACTIVE`, and has `consent_status == GRANTED`.
- **Single Active Session Lock**: Only one enrollment session may be open at any given time, preventing concurrency hazards on the shared camera feed.
- **Transactional Batch Persistence**: On commit, all collected samples are persisted within a single transaction with a shared `enrollment_batch_id`. If the session is cancelled or aborted prior to reaching `minimum_samples` (default: 5), zero partial records are committed to the database.
- **Atomic Gallery Synchronization**: Immediately upon successful transactional commit, the controller triggers `PersonService.load_recognition_gallery()`, atomically swapping the active `InMemoryGallery` in `FaceRecognizer` so the newly enrolled user is recognized instantly without service restarts.

### 5.2 Backend Components

| Component | Responsibility |
| :--- | :--- |
| **FastAPI Layer** | High-performance asynchronous REST endpoints for management, enrolment, and telemetry. |
| **Database Layer** | Thread-safe SQLite persistence engine with schema versioning, WAL journal mode, and foreign keys (`DatabaseManager`). |
| **Repository Layer** | Persistence abstraction providing transactional CRUD operations for person records and binary feature embeddings (`SQLitePersonRepository`). |
| **Person Service** | Business logic and lifecycle supervisor managing profile validation, consent tracking, and embedding counts without exposing raw vectors (`PersonService`). |
| **Gallery Loader** | Extracts, validates, and normalizes persisted embeddings into an isolated in-memory recognition gallery (`load_gallery_from_repository`). |
| **Conversation Manager** | Maintains dialogue state, session history, intent resolution, and turn-taking logic. |
| **LLM Adapter** | Asynchronously interfaces with LLM providers with latency-tolerant fallback routines. |
| **WebSocket Hub** | Real-time bidirectional event streaming with edge nodes and UI dashboards. |
| **Robot/Edge Responses** | Dispatches high-level behavioral directives and synthesized responses back to edge actuators. |

#### 5.2.1 Persistence & Repository Architecture

The persistent storage tier uses standard library `sqlite3` without external ORM overhead:

```
┌─────────────────────────────────────────────────────────────┐
│                       Person Service                        │
│             (Domain Validation & Privacy Boundary)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  SQLite Person Repository                   │
│        (Transactional Persistence & Cascade Handling)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Database Manager                       │
│     (Single-Writer Lock, WAL Mode, Foreign Keys, Pragma)     │
└──────────────────────┬──────────────────────────────┬───────┘
                       ▼                              ▼
             ┌──────────────────┐           ┌──────────────────┐
             │     persons      │           │ face_embeddings  │
             │   (UUID PK)      │◄──────────┤ (FK ON DELETE    │
             └──────────────────┘  CASCADE  │  CASCADE, BLOB)  │
                                            └──────────────────┘
```

- **Schema & Versioning**: `schema_version` table with atomic migration runner ensuring safe schema evolution.
- **Referential Integrity**: `PRAGMA foreign_keys = ON;` strictly enforced. Deleting a person automatically cascades to remove all associated face embeddings, preventing orphaned biometric vectors.
- **Binary Embedding Storage**: 128-D feature vectors are stored as float32 binary `BLOB` buffers (512 bytes) with shape and finiteness validation on both read and write.
- **Single-Writer Concurrency**: A thread-safe reentrant lock (`threading.RLock`) synchronizes write transactions with `BEGIN IMMEDIATE` and `WAL` journal mode for unhindered concurrent reads.

#### 5.2.2 Gallery Loading & Synchronization

To guarantee real-time vision performance (`> 15 FPS` deployment target), the camera loop never queries SQLite directly:

```
┌─────────────────────────┐
│     SQLite Database     │
│ (persons & embeddings)  │
└────────────┬────────────┘
             │
             │ load_gallery_from_repository()
             ▼
┌─────────────────────────┐
│    InMemoryGallery      │ (Atomic Gallery Replacement)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     FaceRecognizer      │ (Real-Time In-Memory Cosine Similarity)
└─────────────────────────┘
```

1. **Model & Metadata Filtering**: Loads only embeddings compatible with the configured detector/embedder (`model_name="SFace"`, `model_version="2021dec"`).
2. **Status & Consent Filtering**: By default, only identities with `status = ACTIVE` and `consent_status = GRANTED` are admitted into the active recognition gallery.
3. **Atomic Replacement**: A fresh `InMemoryGallery` is constructed in the background before assignment to `FaceRecognizer`, ensuring lock-free, zero-downtime gallery updates.

### 5.3 Shared Contracts


- Type-safe, versioned Pydantic models defining:
  - Identity and detection events (`PersonSeen`, `PersonIdentified`, `PersonLeft`).
  - Conversation payloads (`UserUtterance`, `SystemResponse`).
  - Actuation and robot commands (`LookAt`, `Speak`, `SetExpression`).


---

## 6. Integration with Digital Health 360

The Human Interaction module serves as the primary perceptual and social front-end for the broader Digital Health 360 ecosystem:
- **EHR & Patient Records**: Feeds verified session logs and interaction summaries into clinical workflows.
- **Triage & Assessment**: Supplies conversational insights and patient engagement telemetry to healthcare dashboards.
- **Safety & Auditability**: Ensures traceable timestamps, deterministic identity verification, and privacy compliance across all hospital or clinic interactions.

---

## 7. Security, Privacy & Biometrics

- **Biometric Protection**: Face embeddings constitute sensitive biometric data. Raw embeddings are stored with strict access controls, hashed references, and never exposed over unauthenticated channels.
- **Pretrained Models**: Model weights are downloaded and verified with cryptographic checksums; no model binaries are committed to version control.
- **Zero-Secret Commits**: All configuration secrets, API tokens, and private endpoints are loaded strictly via environment variables.
- **Non-Authentication Scope / Disclaimer**: Face recognition within this module is designed strictly for **conversational personalization, natural user engagement, and presence awareness**. It is **NOT** a secure biometric authentication mechanism, cryptographic credential, or physical access control system. The pipeline does not incorporate presentation attack detection (PAD) or 3D liveness verification; clinical authorization or sensitive EHR data access must always rely on external verified authentication mechanisms.
