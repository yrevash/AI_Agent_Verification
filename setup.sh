#!/bin/bash
set -e

# ==============================================================================
#  KYC AI Agent — Full Server Setup
#  Run once: chmod +x setup.sh && ./setup.sh
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_step()  { echo -e "\n${CYAN}[STEP]${NC} $1"; }
print_ok()    { echo -e "  ${GREEN}[OK]${NC} $1"; }
print_warn()  { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
print_fail()  { echo -e "  ${RED}[FAIL]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================================================================="
echo "  KYC AI Agent Verification — Server Setup"
echo "  Project: $PROJECT_DIR"
echo "=============================================================================="

# ---------- 1. System packages ----------
print_step "1/9 — Checking system packages"

for cmd in python3 pip3 redis-cli curl; do
    if command -v $cmd &>/dev/null; then
        print_ok "$cmd found"
    else
        print_fail "$cmd not found — install it first"
        if [ "$cmd" = "redis-cli" ]; then
            echo "       sudo apt install redis-server"
        fi
    fi
done

# Tesseract (needed for entity.py OCR fallback)
if command -v tesseract &>/dev/null; then
    print_ok "tesseract found"
else
    print_warn "tesseract not found — installing..."
    sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-ben
fi

# ---------- 2. Python venv ----------
print_step "2/9 — Setting up Python virtual environment"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_ok "Created venv"
else
    print_ok "venv already exists"
fi

source venv/bin/activate
print_ok "Activated venv ($(python --version))"

# ---------- 3. Install Python dependencies ----------
print_step "3/9 — Installing Python dependencies"

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
print_ok "All Python packages installed"

# ---------- 4. Create directories ----------
print_step "4/9 — Creating project directories"

mkdir -p data/encrypted_logs
mkdir -p logs
mkdir -p temp
mkdir -p batch_temp
mkdir -p models
mkdir -p app/model
mkdir -p app/extracted_data

print_ok "All directories created"

# ---------- 5. YOLO models ----------
print_step "5/9 — Checking AI models"

if [ -f "models/best4.pt" ]; then
    print_ok "models/best4.pt ($(du -h models/best4.pt | cut -f1))"
else
    print_fail "models/best4.pt MISSING — copy your YOLO detection model here"
fi

if [ -f "models/best.pt" ]; then
    print_ok "models/best.pt ($(du -h models/best.pt | cut -f1))"
else
    print_fail "models/best.pt MISSING — copy your YOLO extraction model here"
fi

# YOLOv12 face model — auto-downloads at startup, but we can grab it now
FACE_MODEL="app/model/yolov12l-face.pt"
if [ -f "$FACE_MODEL" ]; then
    print_ok "$FACE_MODEL ($(du -h $FACE_MODEL | cut -f1))"
else
    print_warn "Downloading YOLOv12 face model..."
    curl -L -o "$FACE_MODEL" "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt"
    if [ -f "$FACE_MODEL" ]; then
        print_ok "Downloaded $FACE_MODEL"
    else
        print_warn "Download failed — it will auto-download on first server start"
    fi
fi

# ---------- 6. Ollama + Qwen model ----------
print_step "6/9 — Checking Ollama + Qwen vision model"

if command -v ollama &>/dev/null; then
    print_ok "ollama found"

    # Check if ollama is running
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        print_ok "ollama is running"
    else
        print_warn "ollama is not running — starting..."
        nohup ollama serve &>/dev/null &
        sleep 3
    fi

    # Check if qwen model is pulled
    if ollama list 2>/dev/null | grep -q "qwen3-vl:8b-instruct"; then
        print_ok "qwen3-vl:8b-instruct model present"
    else
        print_warn "Pulling qwen3-vl:8b-instruct (this may take a while)..."
        ollama pull qwen3-vl:8b-instruct
        print_ok "qwen3-vl:8b-instruct pulled"
    fi
else
    print_fail "ollama not found — install from https://ollama.ai"
    echo "       curl -fsSL https://ollama.ai/install.sh | sh"
fi

# ---------- 7. Redis ----------
print_step "7/9 — Checking Redis"

if redis-cli ping &>/dev/null; then
    print_ok "Redis is running"
else
    print_warn "Redis not running — starting..."
    sudo systemctl start redis-server 2>/dev/null || sudo systemctl start redis 2>/dev/null || print_fail "Could not start Redis"
    if redis-cli ping &>/dev/null; then
        print_ok "Redis started"
    else
        print_warn "Redis not available — system will work without caching"
    fi
fi

# ---------- 8. Encryption key ----------
print_step "8/9 — Setting up encryption key"

# Generate a fresh Fernet key
NEW_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Write to key file
echo -n "$NEW_KEY" > data/encryption.key
chmod 600 data/encryption.key
print_ok "Key file: data/encryption.key (chmod 600)"

# Update .env if it exists
if [ -f ".env" ]; then
    # Replace existing ENCRYPTION_KEY line or append
    if grep -q "^ENCRYPTION_KEY=" .env; then
        sed -i "s|^ENCRYPTION_KEY=.*|ENCRYPTION_KEY=$NEW_KEY|" .env
        print_ok "Updated ENCRYPTION_KEY in .env"
    else
        echo "" >> .env
        echo "ENCRYPTION_KEY=$NEW_KEY" >> .env
        print_ok "Added ENCRYPTION_KEY to .env"
    fi
    chmod 600 .env
    print_ok ".env locked (chmod 600)"
else
    print_warn "No .env file found — creating from .env.example"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        sed -i "s|^ENCRYPTION_KEY=.*|ENCRYPTION_KEY=$NEW_KEY|" .env
        chmod 600 .env
        print_ok "Created .env from .env.example with key"
    fi
fi

# Lock encrypted logs directory
chmod 700 data/encrypted_logs
print_ok "data/encrypted_logs/ locked (chmod 700)"

# ---------- 9. Verify setup ----------
print_step "9/9 — Verifying setup"

# Quick Python import check
python -c "
import sys
errors = []
try:
    import fastapi; print('  [OK] fastapi')
except: errors.append('fastapi')
try:
    import uvicorn; print('  [OK] uvicorn')
except: errors.append('uvicorn')
try:
    import torch; gpu = torch.cuda.is_available(); print(f'  [OK] torch (CUDA: {gpu})')
except: errors.append('torch')
try:
    from cryptography.fernet import Fernet; print('  [OK] cryptography')
except: errors.append('cryptography')
try:
    import redis; print('  [OK] redis')
except: errors.append('redis')
try:
    from ultralytics import YOLO; print('  [OK] ultralytics (YOLO)')
except: errors.append('ultralytics')
try:
    from deepface import DeepFace; print('  [OK] deepface')
except: errors.append('deepface')
try:
    import aiohttp; print('  [OK] aiohttp')
except: errors.append('aiohttp')
try:
    from scoring import VerificationScorer; print('  [OK] scoring.py')
except: errors.append('scoring')
try:
    from encrypted_logger import EncryptedUserLogger; print('  [OK] encrypted_logger.py')
except: errors.append('encrypted_logger')

if errors:
    print(f'\n  [WARN] Failed imports: {errors}')
    sys.exit(1)
else:
    print('\n  All imports OK')
"

echo ""
echo "=============================================================================="
echo "  SETUP COMPLETE"
echo "=============================================================================="
echo ""
echo "  Your encryption key:"
echo "  $NEW_KEY"
echo ""
echo "  SAVE THIS KEY. If you lose it, encrypted logs cannot be decrypted."
echo "  Key is stored in: data/encryption.key"
echo "  Key is also in:   .env (ENCRYPTION_KEY=...)"
echo ""
echo "  To start the server:"
echo "    source venv/bin/activate"
echo "    python batch.py"
echo ""
echo "  To start the dispatcher (in another terminal):"
echo "    source venv/bin/activate"
echo "    python batch_dispatcher.py"
echo ""
echo "  Health check:"
echo "    curl http://localhost:8101/health"
echo ""
echo "=============================================================================="
