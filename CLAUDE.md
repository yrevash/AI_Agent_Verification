# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated KYC (Know Your Customer) verification pipeline that processes Aadhaar identity documents and selfies using AI/ML models to make approval/rejection/review decisions. Built as a distributed system with two core components communicating over HTTP.

## Running the System

```bash
# Prerequisites: Ollama, Redis, and CUDA (optional) must be running
ollama serve                          # Start Ollama in background
ollama pull qwen3-vl:8b-instruct     # Pull the vision model (first time)
sudo systemctl start redis            # Start Redis

# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start the Local AI Server (must start first)
python batch.py                       # FastAPI server on :8101

# Terminal 2: Start the Dispatcher
python batch_dispatcher.py            # Orchestrator client

# Or launch multiple agents at once
python agent_manager.py               # Starts agents on ports 8100+

# Health check
curl http://localhost:8101/health
```

There is no test suite or linter configured in this project.

## Architecture

### Two-Process Distributed Model

```
Central Backend (qoneqt.com/v1/api)
        │
        ▼
batch_dispatcher.py (Dispatcher/Client)
  - Locks batches of users from backend
  - Downloads images into memory (BytesIO)
  - Sends to local AI server
  - Pushes results back to backend
  - Logs to: SQLite, Google Sheets, JSON files
  - Caches results in Redis (24h TTL)
        │
        ▼
batch.py (Local AI Server, FastAPI :8101)
  - Hosts all AI models
  - Endpoint: /verification/verify/agent/{user_id}
  - Returns JSON with scores, extracted data, decision
```

### Verification Pipeline

Qwen handles document OCR only. YOLOv12 + DeepFace handles all face/identity verification via embeddings (gender mismatches are inherently caught by embedding comparison).

```
Step 1a: Extract Aadhaar FRONT via Qwen (aadharnumber, name, dob, gender, vid, is_masked)
Step 1b: Extract Aadhaar BACK via Qwen (aadharnumber, address, pincode, vid, is_masked)
Step 1c: Cross-check front/back numbers (Aadhaar + VID) → rejection if mismatch
Step 1d: Masked Aadhaar check → rejection if masked
Step 2:  Face embedding verification (YOLOv12 + DeepFace)
         - Face detection + embedding extraction from selfie and Aadhaar
         - Cosine similarity (threshold 0.55)
         - Duplicate check against stored embeddings (threshold 0.85)
         - Save embedding if verified + not duplicate
Step 3:  Score calculation (weighted scoring + critical failures)
Step 4:  Build response (enriched extracted_data)
```

### Extraction Pipeline (with fallbacks)

1. **Primary — Ollama Qwen Agent**: `qwen3-vl:8b-instruct` via local Ollama for document OCR. Separate front/back extraction for Aadhaar with cross-checking. Images resized to <800px for token efficiency.
2. **Secondary — Entity Agent** (`app/entity.py`): YOLO detection (`models/best4.pt`, `models/best.pt`) + Tesseract OCR. Supports Hindi, Telugu, Bengali.
3. **Tertiary — Qwen Fallback** (`app/qwen_fallback.py`): HuggingFace Qwen2-VL-2B-Instruct transformer, only triggers if primary extraction fails. Controlled by `ENABLE_QWEN_FALLBACK` env var.

### Verification Checks (after extraction)

- **Masked Aadhaar detection** — X's or \*'s in number → auto-reject
- **Aadhaar front/back cross-check** — Aadhaar number and VID must match between front and back images
- **Face embedding verification** (`app/faceEmbeddings.py`) — YOLOv12 face detection + DeepFace Facenet embeddings, cosine similarity threshold 0.55. Inherently catches gender mismatches (different-gender faces won't match embeddings).
- **Duplicate face detection** (`app/faceEmbeddings.py`) — Compares selfie embedding against stored embeddings database (threshold 0.85)
- **Data matching** — DOB year, Aadhaar digit count (12)

### Scoring (`scoring.py`)

Weighted scoring (total 100): Aadhaar validity (30pts), DOB/Age (30pts), face similarity (40pts). No separate gender checks — face embeddings inherently reject mismatched identities. Thresholds: >60 APPROVED, 40-60 REVIEW, <40 REJECTED. Critical failures (masked Aadhaar, underage, cross-check mismatch, duplicate face) cause auto-rejection regardless of score.

## Key Files

| File | Role |
|---|---|
| `batch.py` | Core FastAPI server — hosts all AI models, processes verification requests |
| `batch_dispatcher.py` | Orchestrator — fetches batches from backend, coordinates processing, pushes results |
| `app/entity.py` | YOLO + Tesseract entity extraction (largest module, ~35KB) |
| `app/faceEmbeddings.py` | YOLOv12 face detection + DeepFace embedding verification + duplicate detection |
| `app/qwen_fallback.py` | Qwen2-VL transformer fallback for failed extractions |
| `scoring.py` | Weighted scoring engine and decision logic |
| `config.py` | Pydantic-based configuration (reads from .env) |
| `redis_cache.py` | Redis caching interface |
| `agent_manager.py` | Multi-agent launcher (spawns multiple server instances) |
| `monitoring.py` | Metrics collection (approval rates, p95 latency, cache hits) |

## Configuration

Configuration is managed via `config.py` (Pydantic) reading from `.env` file. See `.env.example` for all options. Key settings: `APP_PORT` (default 8101), `USE_GPU`, `CUDA_VISIBLE_DEVICES`, `REDIS_*`, model paths (`MODEL1_PATH`, `MODEL2_PATH`), `ENABLE_QWEN_FALLBACK`.

## Storage & Logging

- **SQLite** (`local_kyc_data.db`) — audit trail for all verification results, WAL mode
- **JSON logs** — `logs/agent_{id}/{session_id}/user_{id}.json`
- **Google Sheets** — optional remote logging via gspread
- **Redis** — caching verified users to prevent reprocessing
- **Face embeddings** — `app/extracted_data/face_embeddings.pkl` (pickle database of user embeddings for duplicate detection)
- **Rotating file logs** — configured in `logging_config.py` (50MB app, 20MB error, 100MB verification)

## Important Conventions

- All image processing is done in-memory using `io.BytesIO` (no disk I/O for images)
- The dispatcher uses exponential backoff for 502 errors from the backend
- Models are initialized once at server startup via FastAPI lifespan (including face detector via lazy init)
- GPU memory is explicitly managed with `torch.cuda.empty_cache()` after model loading
- The `legacy/` directory contains deprecated code kept for reference only
- Face detector (`app/faceEmbeddings.py`) uses lazy initialization — call `initialize_face_detector()` at startup or rely on `get_face_detector()` for auto-init
