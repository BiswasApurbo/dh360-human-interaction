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

#### 5.2.1 Persistence & Repository Architecture (Schema v2)

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

- **Schema & Versioning (v2)**: `schema_version` table with atomic migration runner ensuring safe schema evolution:
  - **Schema v1**: `persons` demographic and consent table + `face_embeddings` float32 biometric vectors.
  - **Schema v2**: `interaction_sessions` historical audit table indexed by `session_id`, `person_id`, `started_at`, `ended_at`, and `created_at`.
- **Automatic Migration & Data Safety**: Existing v1 databases automatically upgrade to v2 preserving all registered persons and embeddings without data loss. Fresh databases initialize directly at v2.
- **Referential Integrity & Privacy**: `PRAGMA foreign_keys = ON;` strictly enforced.
  - Deleting a person cascades (`ON DELETE CASCADE`) to permanently wipe all associated 128-D face embeddings.
  - Deleting a person sets session references to NULL (`ON DELETE SET NULL`), preserving historical interaction duration and timestamps while severing biometric linkage.
- **Binary Embedding Storage**: 128-D feature vectors are stored as float32 binary `BLOB` buffers (512 bytes) with shape and finiteness validation on both read and write.
- **Single-Writer Concurrency**: A thread-safe reentrant lock (`threading.RLock`) synchronizes write transactions with `BEGIN IMMEDIATE` and `WAL` journal mode for unhindered concurrent reads.
- **Non-Blocking Asynchronous Persistence (`SessionPersistenceWorker`)**:
  - `InteractionCoordinator` delegates completed `InteractionSummary` records to `SessionPersistenceWorker`.
  - Uses a bounded queue (`max_queue_size = 128`) on a dedicated daemon thread, preventing disk write stalls from hitching real-time vision frame processing.
  - If the queue saturates, executes a bounded wait followed by a synchronous write fallback, guaranteeing zero lost audit records.
  - Gracefully flushes and drains all pending records on service shutdown before database connection closure.


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

#### 5.2.4 Historical Interaction Query Layer & SQL-Side Analytics

Milestone M2.12 extends the persistent session store with structured filtering, pagination, and database-side analytics:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             HumanInteractionService                              │
│  • query_sessions(filter)                 • count_query_sessions(filter)         │
│  • get_aggregate_analytics(filter)        • get_person_engagement_stats(pid)     │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Delegated to BaseSessionRepository
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            SQLiteSessionRepository                               │
│  • Parameterized SQL queries ("?") with indexed columns                          │
│  • SQL-side aggregations: COUNT, SUM, AVG, MIN, MAX, CASE WHEN                   │
│  • Zero full-table in-memory Python materialization                              │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ PRAGMA foreign_keys = ON;
                                         ▼
                             ┌───────────────────────┐
                             │ interaction_sessions  │
                             │ (Schema v2, SQLite)   │
                             └───────────────────────┘
```

1. **Structured Query Filtering (`SessionQueryFilter`)**:
   - Provides optional filtering across `person_id`, `is_identified` (boolean), `started_after`, `started_before`, `ended_after`, `ended_before`, `min_duration_seconds`, and `max_duration_seconds`.
   - Includes built-in boundary validation (`limit > 0`, `offset >= 0`, `min_duration <= max_duration`, chronological timestamp ordering).
   - Dynamically compiles into parameterized SQL (`?`) with zero raw string interpolation or injection hazards.
2. **SQL-Side Aggregate Analytics (`SessionAggregateAnalytics`)**:
   - Computes summary metrics entirely inside SQLite using `COUNT(*)`, `COALESCE(SUM(duration_seconds), 0.0)`, `COALESCE(AVG(duration_seconds), 0.0)`, `COALESCE(MIN(duration_seconds), 0.0)`, `COALESCE(MAX(duration_seconds), 0.0)`, and `COUNT(CASE WHEN identity_was_known = 1 THEN 1 END)`.
   - On empty result sets, cleanly returns zero counts/durations with `None` timestamps without throwing errors.
3. **Person Engagement Summaries (`PersonEngagementStats`)**:
   - Aggregates interaction history for an individual (`total_sessions`, `total_duration_seconds`, `average_duration_seconds`, `first_interaction_at`, `last_interaction_at`, `display_name`).
   - Returns `None` if the person has zero recorded sessions or if the person record was deleted.
4. **Referential Integrity & Historical Retention**:
   - When a person is removed (`delete_person`), SQLite's `ON DELETE SET NULL` clears `person_id` in `interaction_sessions` while permanently retaining the historical record, `display_name`, and `identity_was_known=True`.
   - Global queries and aggregate analytics continue to account for historical interactions accurately while person-specific queries (`get_person_engagement_stats`) return `None`.
5. **Real-Time Decoupling & Zero Biometric Leakage**:
   - Historical querying is completely isolated from the real-time frame-processing path and executes out-of-band.
   - Query models and analytics contain zero biometric vectors, face crops, landmark arrays, or medical data.

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

### 5.5 Interaction Context Coordinator & Application Boundary (`dh360_hi.application.coordinator`)

The `InteractionCoordinator` creates a clean, decoupled application-level boundary immediately following the in-process `EventBus`:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Edge Perception & Runtime                            │
│           (FrameSource → TwoRatePerceptionPipeline → SessionManager)             │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ list[BaseEvent]
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                   EventBus                                       │
│                 (In-Process Synchronous Event Distribution)                      │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ BaseEvent Subscriptions
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             InteractionCoordinator                               │
│              (Application Context Aggregation & Profile Resolution)              │
│                                        │                                         │
│                ┌───────────────────────┴───────────────────────┐                 │
│                ▼                                               ▼                 │
│      [ PersonService Lookup ]                     [ Context Listener Callbacks ] │
│      (Public PersonProfile)                       (Thread-Safe Immutable Events) │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼ Queryable Application Contracts
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      Application Boundary (Contracts)                            │
│                                                                                  │
│   • InteractionContext (session_id, person_id, display_name, profile, state)    │
│   • InteractionSummary (session_id, timestamps, duration, identity transitions)  │
│   • InteractionState (IDLE ↔ GREETING ↔ ENGAGED ↔ FAREWELL)                      │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   ▼                     ▼                     ▼
        [ Future FastAPI / REST ] [ Future LLM Dialogue ] [ Future Robot Actions ]
```

#### 5.5.1 Architectural Invariants & Separation of Concerns

1. **SessionManager Remains Authoritative**: `SessionManager` retains exclusive ownership of target presence tracking, departure debouncing, and session ID generation. `InteractionCoordinator` consumes authoritative lifecycle events without duplicating vision state machines.
2. **Context Aggregator & Enricher**: `InteractionCoordinator` maintains the active `InteractionContext` snapshot, resolving opaque `person_id` values into rich `PersonProfile` metadata via `PersonService` (display name, preferred name, consent status).
3. **No Vision Data Leakage**: Application contracts (`InteractionContext`, `InteractionSummary`, `InteractionState`) are pure Python frozen dataclasses containing zero image arrays, bounding boxes, landmarks, 128-D biometric embeddings, or database connection handles.
4. **Thread-Safe Snapshot Access**: Context queries (`get_context()`, `get_last_summary()`) return immutable dataclasses protected by internal reentrant locking (`threading.RLock`), supporting safe concurrent queries from future HTTP or async worker threads.
5. **Deterministic Summary Finalization**: When a `SessionEndedEvent` is received, `InteractionCoordinator` finalizes a historical `InteractionSummary` and resets the active context back to `IDLE` state.

### 5.6 Unified Application Service & Composition Root (`dh360_hi.application.service` & `factory`)

The `HumanInteractionService` and its composition root `create_interaction_service` package the complete multi-layer subsystem into a cohesive, production-ready application facade:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         Future Application / Transport Layer                     │
│               [ FastAPI REST API ]   [ CLI Runner ]   [ Robot Adapters ]         │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Calls Application Service Facade
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             HumanInteractionService                              │
│       • Lifecycle: start(), stop(), pause(), resume(), step_once()               │
│       • Context Queries: get_interaction_context(), get_last_summary()           │
│       • Session History: get_session(), list_sessions(), count_sessions()        │
│       • Query & Analytics: query_sessions(), count_query_sessions(),             │
│                            get_aggregate_analytics(), get_person_engagement_stats()│
│       • Person Queries: get_person(), list_persons()                             │
│       • Metrics & State: get_runtime_metrics(), get_runtime_state()              │
│       • Management: reload_gallery(), add_context_listener(), add_event_listener()│
└───────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┘
        │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│DatabaseManager││PersonService ││  GallerySync ││ EdgeRuntime  ││Coordinator   │
│& Repositories││  (Profiles)  ││  (Recognizer)││(Frame Pump)  ││(Persist Sync)│
└──────────────┘└──────────────┘└──────────────┘└──────────────┘└──────────────┘
```

#### 5.6.1 Core Responsibilities & Dependency Direction

1. **Thin Application Facade**: `HumanInteractionService` contains zero SQL, zero OpenCV image manipulation, and zero event-generation logic. It acts strictly as an in-process coordinator delegating to `EdgeInteractionRuntime`, `InteractionCoordinator`, `PersonService`, `BaseSessionRepository`, and `DatabaseManager`.
2. **Composition Root (`create_interaction_service`)**: Builds and validates the entire dependency graph from a typed `HumanInteractionConfig` model in the correct architectural order (Database → Person/Session Repositories → PersonService → GalleryLoader → Recognizer → Perception Pipeline → SessionManager → EventBus → Runtime → PersistenceWorker → Coordinator → Service).
3. **Safe, Deadlock-Free Teardown Sequence**: Enforces graceful, deterministic shutdown ordering on `service.stop()`:
   - Signals and joins the `EdgeInteractionRuntime` frame pump thread.
   - Signals and joins the background `BoundedEmbeddingWorker` thread.
   - Drains and terminates the background `SessionPersistenceWorker`, ensuring all pending interaction summaries are safely committed to SQLite.
   - Unsubscribes coordinator listeners from the `EventBus`.
   - Closes the `DatabaseManager` SQLite connection.
4. **Deterministic Simulation & Test Step (`step_once()`)**: Executes a single frame cycle through perception and event dispatch, returning a sanitized `StepResultDTO` without leaking raw vision tensors or internal tracker IDs.
5. **Zero-Leakage Privacy Boundary**: All external interactions traverse pure transport-neutral DTOs (`InteractionContextDTO`, `InteractionSummaryDTO`, `InteractionSessionDTO`, `PersonProfileDTO`, `RuntimeMetricsDTO`). Raw biometric feature vectors, model weights, OpenCV arrays, internal tracker IDs (`track_id`), frame sequence IDs (`frame_id`), and database cursors remain strictly encapsulated.

### 5.7 Application Integration Boundary & Transport-Neutral Contracts (`dh360_hi.contracts.dto`, `integration`, `errors`, & `health`)

Milestone M2.13 establishes the formal application integration boundary, decoupling the core perception and domain subsystem from all transport and presentation layers (REST APIs, WebSockets, dialogue managers, robotics, UI dashboards):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        External Orchestrators & Transports                             │
│       [ FastAPI REST / WebSockets ]    [ Dialogue Manager / LLM ]    [ Robot UI ]      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Operates Strictly Across DTO Boundary
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Application Integration Boundary (M2.13)                        │
├─────────────────────────┬──────────────────────────┬───────────────────────────────────┤
│ Transport-Neutral DTOs  │ External Integration     │ Application Error Taxonomy &      │
│ • PersonProfileDTO      │ Events                   │ Health Diagnostics                │
│ • CreatePersonCommand   │ • SessionStarted         │ • ApplicationError                │
│ • UpdatePersonCommand   │ • PersonIdentified       │ • EntityNotFoundError             │
│ • InteractionContextDTO │ • PersonLostIdentity     │ • ValidationAppError              │
│ • InteractionSummaryDTO │ • SessionEnded           │ • ConflictAppError                │
│ • InteractionSessionDTO │ • ContextUpdated         │ • ServiceUnavailableError         │
│ • Analytics DTOs        │ (Zero tracker/frame IDs, │ • EnrollmentAppError              │
│ • StepResultDTO         │  ISO 8601 UTC times)     │ • ServiceHealthSnapshot           │
└─────────────────────────┴─────────────┬────────────┴───────────────────────────────────┘
                                        │ Typed Facade Delegation
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                HumanInteractionService                                 │
│  • get_interaction_context() -> InteractionContextDTO                                  │
│  • get_last_summary() -> InteractionSummaryDTO | None                                  │
│  • get_person() / list_persons() -> PersonProfileDTO / list[PersonProfileDTO]          │
│  • create_person() / update_person() / delete_person() / reset_person_biometrics()     │
│  • start_enrollment() / process_enrollment_frame() / commit_enrollment()               │
│  • query_sessions() / get_aggregate_analytics() / get_person_engagement_stats()        │
│  • get_health_snapshot() -> ServiceHealthSnapshot                                      │
│  • add_integration_event_listener() / step_once() -> StepResultDTO                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 5.7.1 Transport-Neutral DTO Architecture

1. **Frozen & JSON-Serializable**: All DTOs are frozen dataclasses (`@dataclass(frozen=True)`) equipped with `.to_dict()` methods, producing JSON-compliant dictionary trees with zero OpenCV matrices, NumPy ndarrays, or database cursor dependencies.
2. **Standardized ISO 8601 UTC Timestamps**: Float epoch timestamps are converted to UTC strings formatted according to ISO 8601 (e.g. `2026-10-01T12:00:00+00:00`), ensuring transport interoperability across web and backend clients.
3. **Strict Privacy Invariants**:
   - `PersonProfileDTO` exposes only public demographic metadata (`person_id`, `display_name`, `preferred_name`, `status`, `consent_status`, `consent_timestamp`, `created_at`, `updated_at`). Internal biometric implementation details (raw embeddings, vector counts, model names) and free-form `notes` are completely omitted.
   - `StepResultDTO` packages `InteractionContextDTO` and sanitized `list[BaseIntegrationEvent]`, stripping internal vision tracker IDs (`track_id`) and camera frame IDs (`frame_id`).

#### 5.7.2 External Integration Events & Domain Event Mapper

- **Sanitized Event Bridge (`map_domain_event_to_integration`)**: Translates internal perception domain events (`SessionStartedEvent`, `PersonIdentifiedEvent`, `PersonLostIdentityEvent`, `SessionEndedEvent`) into external integration events (`SessionStartedIntegrationEvent`, `PersonIdentifiedIntegrationEvent`, `PersonLostIdentityIntegrationEvent`, `SessionEndedIntegrationEvent`).
- **Internal Perception Event Encapsulation**: Low-level vision tracking events (`PersonEnteredEvent`, `PersonLeftEvent`) remain internal to the perception pipeline and return `None` during integration mapping.
- **Integration Event Subscribers**: `add_integration_event_listener(callback)` and `remove_integration_event_listener(listener_id)` provide a clean in-process subscription mechanism for downstream application adapters.

#### 5.7.3 Application Error Hierarchy & Health Diagnostics

- **Structured Application Exceptions**:
  - `ApplicationError`: Base class providing structured error codes (`code`) and diagnostic context dictionaries (`details`), easily serializable via `.to_dict()`.
  - `EntityNotFoundError`: Raised on missing domain entities (e.g. person or session).
  - `ValidationAppError`: Raised on invalid command or query filter parameters.
  - `ConflictAppError`: Raised on concurrent state conflicts or duplicate keys.
  - `ServiceUnavailableError`: Raised when unconfigured or offline subsystems are invoked.
  - `EnrollmentAppError`: Raised on face enrollment failures or timeouts.
- **Defensive Health Diagnostics (`evaluate_service_health`)**:
  - Inspects `runtime`, `database`, `persistence_worker`, `recognizer`, and `enrollment` subsystems defensively without throwing exceptions.
  - Produces a `ServiceHealthSnapshot` with operational health tiers (`HEALTHY`, `DEGRADED`, `UNHEALTHY`, `DISABLED`) and real-time performance telemetry (`RuntimeMetricsDTO`).

### 5.8 Unified Configuration & Local Operator CLI

The local operator CLI (`dh360-hi`) and unified configuration loader provide a production-ready operational front-end for local runtime execution, administration, diagnostics, and testing.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Local Operator CLI                             │
│       dh360-hi [run | health | person | enroll | session | config]     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Unified Configuration Loader                        │
│   Precedence: Defaults ──► Preset ──► TOML File ──► Env ──► Overrides  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      HumanInteractionService                           │
│     (Application Integration Facade & Transport-Neutral DTOs)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Domain, Vision & Backend Subsystems                   │
└────────────────────────────────────────────────────────────────────────┘
```

#### 5.8.1 Configuration Precedence & Composition Architecture

Configuration loading follows a strict 5-tier deterministic precedence hierarchy:
1. **Built-in Defaults**: Canonical baseline specified in `HumanInteractionConfig`.
2. **Configuration Presets (`get_preset`)**:
   - `development`: Targets local rapid prototyping (`dh360_dev.db`, 30 FPS).
   - `testing`: Headless in-memory execution (`:memory:`, 60 FPS, 1-frame patience).
   - `production`: High-reliability production deployments (`dh360_hi.db`, 30 FPS, strict error tolerance).
3. **TOML Configuration File**: Parsed via Python standard library `tomllib` (`--config-file` or `DH360_HI_CONFIG_FILE`).
4. **Environment Variables**: Prefixed with `DH360_HI_*` supporting flat keys and double-underscore nesting (e.g. `DH360_HI_RUNTIME__TARGET_FPS=30.0`).
5. **CLI / Programmatic Overrides**: Explicit arguments supplied via command line.

#### 5.8.2 Operator CLI Subcommands (`dh360-hi`)

- **`run`**: Launches continuous edge perception runtime with selectable frame source (`synthetic`, `webcam`, `video`), graceful signal handling (`SIGINT`, `SIGTERM`), and live integration event streaming.
- **`health`**: Evaluates operational subsystem health (`runtime`, `database`, `persistence`, `recognizer`, `enrollment`) with human-readable table or `--json` formatting.
- **`person`**: Manages patient/user profiles (`list`, `get`, `create`, `update`, `delete`, `reset-biometrics`).
- **`enroll`**: Guided interactive camera enrollment session with real-time feedback and transactional commit.
- **`session`**: Historical session inspection (`list`, `get`, `query`, `analytics`, `stats`).
- **`config`**: Inspects resolved configurations (`show`, `validate`, `presets`).

#### 5.8.3 Exit Code Specification

- **`0`**: Successful command execution.
- **`1`**: Expected domain/application/validation error (`ApplicationError`, `EntityNotFoundError`, `ValidationAppError`, `EnrollmentAppError`).
- **`2`**: Unexpected runtime/system failure or command-line syntax error.
- **`130`**: Clean exit on operator interrupt (`SIGINT` / `Ctrl+C`).

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
