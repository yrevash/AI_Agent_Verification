# AI Agent Verification System

## 📖 Overview
The **AI Agent Verification System** is an automated pipeline designed to verify user identities given a set of documents (Aadhaar, PAN) and a selfie. It uses advanced Large Language Models (LLMs) and Computer Vision to extracting data, detecting fraudulent/masked documents, and verifying face ownership.

The system is built as a **Distributed Architecture** consisting of two main components:
1.  **Batch Dispatcher (`batch_dispatcher.py`)**: The "Client" that acts as an agent. It fetches batches of users from a central backend, locks them for processing, and sends them to the local AI server.
2.  **Local AI Server (`batch.py`)**: The "Worker" that runs the heavy AI models (Qwen-VL, Face Verification) to process the images and return a decision.

---

## 🏗️ Architecture & Pipeline

### End-to-End Flow
1.  **Job Assignment**: The Dispatcher (`batch_dispatcher.py`) contacts the main backend (`qoneqt.com`) to "lock" a batch of pending KYC requests for a specific Agent ID.
2.  **Data Retrieval**: The Dispatcher downloads the user's uploaded images (Selfie, Aadhaar Front/Back, PAN) into memory.
3.  **Local Processing**: The data is sent to the Local AI Server (`batch.py`) running on `localhost:8101`.
4.  **AI Analysis**:
    *   **Document Extraction**: `qwen3-vl:8b-instruct` (via Ollama) reads the Aadhaar/PAN cards.
    *   **Logic Check**: The system checks for **Masked Aadhaar** cards (illegal for this specific flow) and rejects them.
    *   **Data Matching**: Extracted text (Name, DOB, Gender) is compared against the user's input.
    *   **Face Verification**: The selfie gender is detected and matched against the document gender.
5.  **Result Submission**: The final decision (APPROVED/REJECTED/REVIEW) and extracted metadata are pushed back to the central backend by the Dispatcher.

---

## 🛠️ Components Deep Dive

### 1. The Dispatcher (`batch_dispatcher.py`)
This script manages the lifecycle of a verification job.
*   **Locking Mechanism**: Uses `/admin/kyc-lock-batch` to ensure no other agent processes the same users.
*   **Resiliency**: Implements exponential backoff for 502 Server Errors (Server Down/Overloaded).
*   **Logging**:
    *   **Local SQLite**: Saves every record to `local_kyc_data.db` for audit trails.
    *   **Google Sheets**: Optionally logs stats to a Google Sheet.
    *   **JSON Logs**: detailed logs in `logs/agent_{id}/`.
*   **Redis Caching**: Caches repeatedly accessed data to save bandwidth.

### 2. The Local AI Server (`batch.py`)
A FastAPI application that hosts the intelligence.
*   **Ollama Integration**: Connects to a local Ollama instance running `qwen3-vl:8b-instruct`.
*   **Prompt Engineering**: Uses specific prompts to extract fields like Name, DOB, and strictly identifiy Masked Aadhaars.
*   **Image Optimization**: Resizes images to <800px to ensure fast processing and low token usage.
*   **In-Memory Processing**: Uses `io.BytesIO` to handle images in RAM, avoiding slow disk I/O.

---

## 🚀 Setup & Installation

### Prerequisites
*   **Operating System**: Linux / MacOS (Recommended for performant I/O)
*   **Python**: 3.10 or higher
*   **Ollama**: Installed and running.
*   **Redis**: Installed and running (default port 6379).
*   **Models**:
    *   Pull the vision model: `ollama pull qwen3-vl:8b-instruct`

### Installation Steps
1.  **Clone/Setup Directory**:
    Ensure you are in the `AI_Agent_Verification` folder.

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Check `config.py` or `.env` file (if applicable) for API keys and Backend URLs.
    *   `batch_dispatcher.py` has constants like `ADMIN_ID`, `AGENT_IDS` (77, 78, 79, 80).

---

## 🚦 Usage Guide

### Step 1: Start the Local AI Server
This must be running *before* you start the dispatcher.
```bash
python batch.py
```
*   **Health Check**: Open `http://localhost:8101/health` in your browser. It should say `"status": "healthy"`.

### Step 2: Start the Dispatcher
In a separate terminal window:
```bash
python batch_dispatcher.py
```
This will:
1.  Connect to the backend.
2.  Lock a batch of users (default 20).
3.  Start processing them one by one.
4.  Print status logs (e.g., `✅ User 123: Processed & Pushed (APPROVED)`).

---

## 🛡️ Verification Policies

The system enforces strict rules. A user is **REJECTED** if:
1.  **Masked Aadhaar**: The Aadhaar number is hidden with 'X' or '*'. *This is a strict rejection criteria.*
2.  **Document Not Found**: The AI cannot find a valid Aadhaar card in the image.
3.  **Gender Mismatch**: The gender detected in the selfie does not match the Aadhaar gender.

A user is sent to **REVIEW** if:
1.  **Low Confidence**: The AI is unsure about the data extraction (rare).
2.  **Partial Match**: Specific fields match but others are ambiguous.

---

## 📂 File Structure
*   `batch.py`: **Core AI Server**.
*   `batch_dispatcher.py`: **Core Orchestrator**.
*   `app/`: Helper modules (Gender detection, Entity definitions).
*   `legacy/`: Old/Unused files (`main.py`, `scoring.py`) - kept for reference.
*   `logs/`: Execution logs.
*   `redis_cache.py`: Redis interface.

---

## ❓ Troubleshooting

*   **"Qwen agent not initialized"**: cancel the script, make sure `ollama serve` is running, and generic model access is working.
*   **502 Bad Gateway**: The main backend server is down. The dispatcher will auto-retry (exponential backoff).
*   **Redis Connection Error**: Ensure Redis is running (`sudo systemctl start redis`).
