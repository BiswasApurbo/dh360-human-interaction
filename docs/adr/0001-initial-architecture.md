# 1. Initial Architecture and Design Decisions

Date: 2026-10-01  
Status: Accepted  

## Context

The **Digital Health 360 Human Interaction Module** is designed to power perceptual and conversational capabilities for a semi-humanoid robot operating in healthcare contexts. The system must perform real-time face detection, tracking, identification, session management, and conversational interaction via LLMs.

Initial development occurs on Apple Silicon (MacBook M4 Air), with eventual deployment targeting a resource-constrained edge device (Raspberry Pi) connected to robot actuators and a centralized or local FastAPI backend.

The system requires clear architectural boundaries from inception to avoid high-risk coupling, performance bottlenecks, and security pitfalls.

---

## Decision

We have agreed upon the following architectural decisions:

### 1. Decoupled Edge / Backend Separation
- The **Edge Layer** runs the vision perception loop, frame processing, track management, quality filtering, local identity matching, and hardware actuators.
- The **Backend Layer** runs the FastAPI REST services, database persistence, session tracking, conversation management, and LLM adapters.
- Under no circumstances is the backend or an LLM call allowed inside the synchronous camera frame loop.

### 2. Hardware-Independent Vision Layer
- Core computer vision logic (detection processing, tracking mathematics, quality assessment, cosine vector similarity) must remain strictly hardware-agnostic.
- Camera I/O, OS-level video capture, and platform accelerators are isolated behind edge adapter interfaces.

### 3. Mac-First Development with Raspberry Pi Deployment Target
- Primary development, prototyping, and automated testing are conducted on macOS (MacBook M4 Air).
- Target edge deployment is Raspberry Pi. Edge code must adhere to lightweight standards and avoid Mac-exclusive dependencies.

### 4. Real-Time Performance Target (> 15 FPS on Raspberry Pi)
- The edge vision pipeline must achieve and sustain `> 15 FPS` on the Raspberry Pi target hardware.
- Computationally expensive operations (such as embedding extraction) are triggered selectively via the Face Quality Gate and tracker state rather than executed indiscriminately on every frame.

### 5. Pretrained Models Only
- We will not train face recognition models from scratch. Standard, lightweight, production-proven pretrained model weights (e.g., SCRFD / YuNet for detection, ArcFace / MobileFaceNet for recognition) will be utilized.
- Model weight binaries will not be tracked in Git.

### 6. Event-Driven Asynchronous Transport
- Communication between Edge, Backend, and Robot Hardware is event-driven and type-safe, defined using shared Pydantic contracts (`dh360_hi.contracts`).

### 7. Biometric Data Privacy and Security First
- Face embeddings constitute sensitive biometric data. All storage, serialization, and transmission of biometric features must adhere to privacy-by-design principles, secure hashing, encrypted storage, and strict access controls.

---

## Consequences

### Positive
- **Predictable Real-Time Latency**: Frame rates remain stable even during complex LLM dialogues or network hiccups.
- **Portability**: The vision layer can be tested with unit tests and synthetic video inputs on any OS without physical camera or robot hardware.
- **Maintainability**: Clear schema contracts prevent tight coupling and drift between edge and backend teams.
- **Incremental Delivery**: Enables milestone-by-milestone development without premature optimization or dependency bloat.

### Negative / Trade-offs
- Requires maintaining explicit event schemas and serialization contracts between edge and backend.
- Requires careful model profiling to ensure the selected pretrained neural networks meet the `> 15 FPS` budget on Raspberry Pi.
