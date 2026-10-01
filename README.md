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

- **Current Milestone**: **`M0 — Foundation`**
- **Status**: Active Development
- **Approach**: Built **incrementally** milestone by milestone. Core software foundation, tooling, CI, quality checks, and architectural specifications are established first before introducing vision models, backend services, or hardware integrations.

---

## 🏛 Architecture & Design Records

- **Architecture Overview**: [docs/architecture.md](docs/architecture.md)
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

### 2. Run Code Quality Checks
```bash
# Linting & code style
ruff check .
ruff format --check .

# Static type checking
mypy

# Unit & integration tests
pytest
```

---

## 🔐 Security & Privacy
- Face embeddings represent sensitive biometric information and are handled with strict privacy-by-design standards.
- Real API keys, credentials, local captures, logs, and pretrained binary models are strictly excluded from version control (see `.gitignore` and `.env.example`).
