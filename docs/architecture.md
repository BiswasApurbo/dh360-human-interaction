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
- **Encapsulated Gallery Replacement**: `FaceRecognizer` owns its active gallery instance; external components (such as `PersonService` and `FaceEnrollmentController`) update the gallery exclusively via `FaceRecognizer.replace_gallery(new_gallery)`.
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

#### 5.1.4 Two-Rate Processing Model & Bounded Embedding Worker

To achieve sustained high frame rates (`> 15 FPS` Pi target, `~56 FPS` measured on Mac) without dropping camera frames during expensive neural network embedding inference, the system implements a **Two-Rate Architecture**:

```
[ Camera Frame (30+ FPS) ]
        │
        ▼
[ Face Detection (YuNet) ] ──► [ Tracker ] ──► [ Quality Gate & Tiers ] ──► [ Crossing Detection ]
                                                                                   │
        ┌──────────────────────────────────────────────────────────────────────────┘
        │ (Policy: should_request_embedding)
        ▼
┌──────────────────────────────────────────────┐
│  Bounded Asynchronous Embedding Worker       │
│  • Latest-Wins Queueing per Track ID         │
│  • Max Bounded Capacity (e.g. 8 tasks)       │
│  • Dedicated Background Worker Thread        │
└──────────────────────┬───────────────────────┘
                       │ (Non-blocking Result Polling)
                       ▼
[ Identity State Machine ] ──► [ Target Selector ] ──► [ Presence Debouncer ]
```

- **Fast-Rate Path (Per-Frame Synchronous, ~5ms)**:
  - Frame acquisition $\rightarrow$ YuNet face detection $\rightarrow$ Multi-face tracking $\rightarrow$ Face quality gating with discrete operational tiers (`FaceQualityTier`: `REJECTED`, `LOW`, `MEDIUM`, `HIGH`) $\rightarrow$ Spatial crossing & proximity checks $\rightarrow$ Non-blocking result polling $\rightarrow$ Target selection $\rightarrow$ Presence debouncing.
- **Slow-Path / Event-Driven Path (Asynchronous Background Thread)**:
  - Alignment (5-point affine transform) $\rightarrow$ 128-D SFace feature embedding $\rightarrow$ Gallery cosine similarity matching.
- **Latest-Wins Bounded Queue & Results Buffer**:
  - `BoundedEmbeddingWorker` manages a bounded queue with latest-wins replacement per `track_id`. If a track produces a newer, higher-quality frame while an earlier request is pending, the older request is updated in-place, eliminating stale backlogs.
  - If pending task queue capacity (`max_queue_size`, default: 8) is exceeded, oldest pending tracks are evicted so camera loop execution never blocks.
  - Completed recognition results are buffered in a thread-safe bounded buffer (`max_results_queue_size`, default: 32) where oldest unpolled results are discarded if polling lags, preventing unbounded memory growth.
  - Calling `worker.stop()` gracefully terminates the background thread and safely discards any remaining unstarted pending tasks.
- **Decoupled Backend/LLM**: Backend REST, WebSocket, and LLM dialogues operate strictly out-of-band and never sit on the real-time vision loop.

#### 5.1.5 Per-Track Identity Stability & Track-ID Swap Protection

- **Identity Associated with Track State, Not Raw Track ID**:
  - Identity state is managed by `IdentityStateManager` over a sliding temporal evidence window (`TrackIdentityState`), preventing raw tracker ID mutations from immediately corrupting recognition decisions.
- **Threading Contract**:
  - `IdentityStateManager` is designed for single-threaded execution on the primary perception loop thread. External lifecycle operations (such as profile deletion or biometric reset) are invoked when the loop is paused or between frame processing steps.
- **Current-Frame Spatial Crossing & ID Swap Protection**:
  - `check_track_crossings()` measures pairwise bounding box IoU and centroid Euclidean distance across all active tracks.
  - The `is_spatial_ambiguous` flag is re-evaluated per frame: reset to `False` for all active tracks at the beginning of each frame's check, and set to `True` only for tracks currently involved in crossing/overlap ($\le 80\text{px}$ or $\text{IoU} \ge 0.08$).
  - Spatially ambiguous tracks require fresh, high-quality recognition evidence and cannot blindly inherit prior momentum.
- **Late Result Discard & Ghost Track Protection**:
  - When an old track disappears and is pruned, any late-arriving asynchronous recognition results for the missing `track_id` are safely discarded without recreating ghost track states.
- **Track Disappearance and Reappearance**:
  - When an old track disappears and is pruned, any newly spawned track begins strictly in `NEW` state and must gather independent evidence across the sliding observation window before achieving `IDENTIFIED` state.
- **Hysteresis Protection**:
  - A confirmed `IDENTIFIED` subject is resilient to temporary occlusions, blur, or noisy matches. Sustained conflicting evidence (`hysteresis_patience_observations >= 4`) is required before demoting to `UNCERTAIN`.

#### 5.1.6 Multi-Person Interaction Target Selection

The `InteractionTargetSelector` (`BaseTargetSelector`) deterministically selects a single primary interaction target when multiple people are present:

- **Composite Scoring Function**:
  $$\text{Score} = w_{\text{central}} \cdot S_{\text{central}} + w_{\text{size}} \cdot S_{\text{size}} + w_{\text{id}} \cdot S_{\text{id}} + w_{\text{dwell}} \cdot S_{\text{dwell}}$$
  - **Centrality ($w=0.35$)**: Normalized Euclidean proximity of face center to camera frame center.
  - **Size / Proximity ($w=0.35$)**: Relative bounding box area ratio (closer people score higher).
  - **Identity State ($w=0.20$)**: Bonus for `IDENTIFIED` registered users ($1.0$) vs `COLLECTING`/`UNREGISTERED` ($0.5$) vs `UNCERTAIN` ($0.2$).
  - **Dwell Time ($w=0.10$)**: Continuous track persistence in scene.
- **Deterministic Ordering**: Sort key uses `(-score, track_id)` to eliminate dictionary iteration nondeterminism.
- **Stickiness Hysteresis**: To prevent rapid gaze/target flickering between two equidistant people, an existing target is retained unless a rival candidate exceeds its score by at least `hysteresis_switch_margin` (default `0.15`).

#### 5.1.7 Temporal Presence Debouncing

The `PresenceDebouncer` eliminates boundary flicker and brief 1-2 frame detection dropouts:
- **Entry Debounce**: Requires `enter_patience_frames` (default: 3) consecutive visible frames before emitting `PERSON_ENTERED`.
- **Leave Debounce**: Requires `leave_patience_frames` (default: 10) consecutive missed frames before emitting `PERSON_LEFT`.
- **Transient Recovery**: Brief detection dropouts recover back to `PRESENT` without emitting spurious departure/re-entry events.

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
3. **Atomic Encapsulated Replacement**: A fresh `InMemoryGallery` is constructed in the background and applied atomically via `FaceRecognizer.replace_gallery(new_gallery)`, ensuring lock-free, zero-downtime gallery updates without mutating private attributes across package boundaries.

#### 5.2.3 Person Lifecycle Management & Biometric Operations

The `PersonService` provides complete lifecycle operations ensuring strict privacy boundaries and referential integrity:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Person Lifecycle Operations                      │
├──────────────────┬────────────────────────────────────────────────────────┤
│ Operation        │ Semantics & Safety Behavior                            │
├──────────────────┼────────────────────────────────────────────────────────┤
│ View / List      │ Returns sanitized PersonProfile models with embedding  │
│                  │ counts and metadata. Raw vectors are never exposed.    │
├──────────────────┼────────────────────────────────────────────────────────┤
│ Edit Profile     │ Updates basic demographics (name, preferred name,      │
│                  │ notes, status). Preserves original consent timestamp.  │
├──────────────────┼────────────────────────────────────────────────────────┤
│ ADD_SAMPLES      │ Appends new camera captures to existing person.        │
│                  │ Rejects duplicates vs in-session AND stored vectors.   │
├──────────────────┼────────────────────────────────────────────────────────┤
│ REPLACE          │ Staged replacement: captures new batch in memory.      │
│ (Re-register)    │ Atomically replaces old records on commit; if aborted, │
│                  │ old active biometric records remain 100% intact.       │
├──────────────────┼────────────────────────────────────────────────────────┤
│ Delete           │ Transactionally removes person; SQLite ON DELETE       │
│                  │ CASCADE wipes embeddings; automatically reloads the   │
│                  │ recognition gallery and invalidates active tracks.     │
└──────────────────┴────────────────────────────────────────────────────────┘
```

- **Privacy Boundary**: `PersonProfile` and `EmbeddingMetadata` provide user-facing views containing counts and metadata. Raw binary BLOBs and float32 embedding vectors are strictly internal to the database repository and in-memory gallery.
- **Explicit Re-registration Modes**:
  - `EnrollmentMode.ADD_SAMPLES`: Pre-loads existing stored embeddings for the subject and deduplicates incoming camera samples against both the active session buffer and existing stored vectors (cosine similarity $\ge 0.95$ rejected).
  - `EnrollmentMode.REPLACE`: Collects and validates candidate samples in-memory without deleting pre-existing database records. On explicit commit, executes `SQLitePersonRepository.replace_embeddings_for_person()` within an atomic SQLite transaction (`DELETE` old + `INSERT` new batch). If the session fails or is cancelled, old embeddings are preserved completely intact.
- **Biometric Reset**: `delete_person_embeddings()` permits clearing biometric data (face vectors) without deleting the user's demographic record.
- **Cascade Deletion & Active Track Invalidation**: Calling `delete_person(person_id, recognizer, identity_manager)` transactionally removes the person from SQLite (which cascades to all associated embeddings), reloads `FaceRecognizer` gallery, and invokes `IdentityStateManager.invalidate_person(person_id)`, transitioning any matching active camera tracks immediately to `UNREGISTERED` and resetting their observation history.

### 5.3 Edge Session Supervision & Structured Event Bus (M2.7)

Milestone M2.7 introduces a clean domain boundary that converts raw perception outputs (`TwoRateFrameResult`) into stable, typed interaction events, decoupling computer vision internals from downstream dialogue and robotic behavior controllers.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      Two-Rate Perception Pipeline (Vision)                       │
│        (YuNet Detect → Track → Quality → Async SFace → Target Selector)          │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ TwoRateFrameResult
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                SessionManager                                    │
│       (Primary Interaction Target Lifecycle & State Machine Supervision)         │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ list[BaseEvent]
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                   EventBus                                       │
│          (Synchronous In-Process Dispatch with Subscriber Isolation)             │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  ▼                      ▼                      ▼
        [ PersonEnteredEvent ] [ SessionStartedEvent ] [ PersonIdentifiedEvent ]
        [ PersonLeftEvent ]    [ SessionEndedEvent ]   [ PersonLostIdentityEvent ]
```

#### 5.3.1 Domain Event Contracts (`dh360_hi.contracts.events`)

All events are frozen, immutable dataclasses carrying only domain-level semantics:
- **Zero Vision Leakage**: Events contain no embeddings, bounding boxes, landmark points, numpy arrays, or camera frame buffers.
- **Opaque Identification**: Subjects are referenced by opaque `person_id: str | None` and `track_id: int`. Unregistered individuals are first-class citizens represented with `person_id=None`.
- **Core Event Hierarchy**:
  - `BaseEvent`: Base class with `event_id` (UUID4 string), `frame_id`, and `timestamp`.
  - `PersonEnteredEvent`: Emitted when the primary interaction target's presence is confirmed after entry debounce.
  - `SessionStartedEvent`: Emitted exactly once when a new primary interaction session begins (`session_id: str`, `person_id: str | None`).
  - `PersonIdentifiedEvent`: Emitted when the active target transitions to `IDENTIFIED` state (`session_id`, `person_id: str`).
  - `PersonLostIdentityEvent`: Emitted when the active target loses a previously stable identity without terminating the interaction session.
  - `PersonLeftEvent`: Emitted when the active target's departure is confirmed after leave debounce.
  - `SessionEndedEvent`: Emitted exactly once when the primary session terminates (`session_id`, `final_person_id: str | None`).

#### 5.3.2 In-Process Synchronous Event Bus (`dh360_hi.edge.event_bus`)

`EventBus` provides deterministic, lightweight in-process event delivery:
- **No Global Singleton**: Callers instantiate and wire `EventBus` explicitly.
- **Synchronous & In-Line**: Dispatches directly on the caller's thread without queues, worker threads, or background event loops.
- **Subscriber Isolation**: Exceptions in subscriber callbacks are caught and logged; failure in one handler cannot abort event delivery to remaining subscribers.
- **Deterministic Order & Unsubscription**: Subscribers execute in exact registration order and receive unique subscription IDs for clean unregistration.

#### 5.3.3 Session Lifecycle State Machine (`dh360_hi.edge.session_manager`)

`SessionManager` manages the lifecycle of the single **primary interaction target**:

```
                  ┌──────────────────────────────────────────────┐
                  │                 NO_SESSION                   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         │ confirmed PERSON_ENTERED
                                         │ & primary target selected
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               ACTIVE_SESSION                 │
                  │   - SessionStartedEvent(person_id=None/ID)   │
                  │   - [PersonIdentifiedEvent]                  │
                  │   - [PersonLostIdentityEvent]                │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         │ confirmed PERSON_LEFT
                                         │ (active target departs)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │                 NO_SESSION                   │
                  │   - PersonLeftEvent                          │
                  │   - SessionEndedEvent                        │
                  └──────────────────────────────────────────────┘
```

#### 5.3.4 Primary Target Switching & Multi-Person Semantics

1. **Target Ranking Stability**: If the target selector switches candidate rankings while the active target is still present in the scene, `SessionManager` **retains the current session**. No fake `PersonLeftEvent`, `SessionEndedEvent`, or secondary session is emitted.
2. **True Departure Transition**: Only when the active session's subject physically leaves (confirmed by `PresenceDebouncer` after departure patience frames) does the active session end with `PersonLeftEvent → SessionEndedEvent`.
3. **Subsequent Target Promotion**: If another person was present in the background and becomes the primary target after the previous subject leaves, a new session starts cleanly (`PersonEnteredEvent → SessionStartedEvent`).
4. **Background Person Isolation**: Background faces that are not selected as the primary interaction target do not generate sessions, entry, identification, or departure domain events.
5. **Deterministic Event Ordering**:
   - **Entry**: `PersonEnteredEvent` → `SessionStartedEvent` → [`PersonIdentifiedEvent`]
   - **Departure**: `PersonLeftEvent` → `SessionEndedEvent`
   - **Mid-Session Identity Transition**: `PersonLostIdentityEvent` → `PersonIdentifiedEvent`

### 5.4 Edge Interaction Runtime & Continuous Frame Pump (`dh360_hi.edge.runtime`)

The `EdgeInteractionRuntime` is the top-level orchestration component responsible for the continuous execution loop and operational supervision of the complete edge pipeline:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             EdgeInteractionRuntime                               │
│                                                                                  │
│   ┌───────────────┐     ┌──────────────────────────┐     ┌───────────────────┐   │
│   │  FrameSource  │ ──► │ TwoRatePerceptionPipeline│ ──► │  SessionManager   │   │
│   └───────────────┘     └──────────────────────────┘     └─────────┬─────────┘   │
│                                                                    │             │
│                                                                    ▼ list[Event] │
│                                                          ┌───────────────────┐   │
│                                                          │     EventBus      │   │
│                                                          └───────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

#### 5.4.1 Core Responsibilities & Dependency Ownership

1. **Top-Level Lifecycle Supervision**: Manages continuous frame acquisition, perception execution, session state progression, and domain event dispatch under thread-safe state machine control (`STOPPED`, `RUNNING`, `PAUSED`, `ERROR`).
2. **Explicit Dependency Ownership**:
   - Manages the underlying `FrameSource` lifecycle (`open()` on start, `release()` on stop).
   - Manages the `TwoRatePerceptionPipeline` lifecycle (`pipeline.start()` / `pipeline.stop(timeout)`), ensuring clean shutdown of the background `BoundedEmbeddingWorker` thread.
   - Forwards perception results to `SessionManager` and ensures any returned domain events are published to `EventBus`.
3. **Execution Modes**:
   - **Asynchronous Mode (`start(async_thread=True)`)**: Launches a dedicated background frame pump thread (`EdgeInteractionRuntimeThread`) executing continuous acquisition, processing, and event emission.
   - **Synchronous Mode (`start(async_thread=False)`)**: Executes the continuous loop on the caller thread until stream termination or external stop.
   - **Single-Step Execution (`step_once()`)**: Executes exactly one complete deterministic frame-to-event cycle, returning `(TwoRateFrameResult, list[BaseEvent])` for testing, diagnostics, and headless simulation.

#### 5.4.2 Constant-Rate FPS Pacing & Error Isolation

- **FPS Pacing**: Regulates continuous frame pump rate toward `target_fps` using monotonic deadlines. If frame processing falls behind schedule, deadlines are dynamically reset to avoid burst processing or lagging frame cascades.
- **Transient Error Resilience**: Individual frame processing exceptions (e.g., corrupted frame buffers, temporary driver glitches) are logged and recorded without terminating the runtime loop.
- **Consecutive Error Threshold**: If consecutive errors exceed `max_consecutive_frame_errors` (default: 10), the runtime transitions safely to `RuntimeState.ERROR`, recording the root cause for telemetry and external supervision.
- **Operational Metrics Snapshot (`RuntimeMetrics`)**: Exposes thread-safe immutable telemetry including `frames_processed`, `current_fps` (rolling window), `average_latency_ms`, `last_error`, and `uptime_seconds`.

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
