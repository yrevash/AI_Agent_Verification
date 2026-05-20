import os
import sys
import asyncio
import aiohttp
import aiofiles
import cloudscraper
import gc
import time
import traceback
import json
import glob
import tempfile  # Added for temporary file handling
import re
from pathlib import Path
from typing import Optional, Union, List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
print("new")
# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import requests
import base64
from PIL import Image
import io
from scoring import VerificationScorer

# --- Ollama Qwen Agent ---
class OllamaQwenAgent:
    """Ollama-based Qwen agent for document extraction using qwen3-vl:8b-instruct"""

    def __init__(self, ollama_url: str = "http://localhost:8000/v1/chat/completions", model: str = "Qwen/Qwen3-VL-8B-Instruct"):
        self.ollama_url = ollama_url
        self.model = model
        print(f"Initialized Qwen Agent (vLLM) with model: {model}")

    def _image_to_base64(self, image_input: Union[str, io.BytesIO]) -> str:
        """
        Convert image to base64 string with OPTIMIZED RESIZING.
        Accepts file path (str) or in-memory file (io.BytesIO).
        """
        try:
            # Load Image from Path or Memory
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return ""
                img = Image.open(image_input)
            else:
                # Assuming io.BytesIO or bytes
                if isinstance(image_input, io.BytesIO):
                    image_input.seek(0)
                img = Image.open(image_input)

            # --- OPTIMIZATION 1: RESIZING ---
            # Resize if too large (Max 800px) to reduce token count drastically
            max_size = 800
            width, height = img.size
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB (standardize format)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save to buffer as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"Error processing image for base64: {e}")
            return ""

    def _call_ollama(self, prompt: str, images: list) -> str:
        """Call vLLM OpenAI-compatible API with vision model"""
        content = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "temperature": 0.1
        }

        response = requests.post(self.ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def validate_aadhaar_front(self, image_input: Union[str, io.BytesIO]) -> dict:
        """Check if the image is an Aadhaar card front side."""
        try:
            img_b64 = self._image_to_base64(image_input)
            if not img_b64:
                return {"is_aadhaar_front": False, "reason": "Image processing failed"}

            prompt = """Look at this image carefully. Is this the FRONT side of an Indian Aadhaar card?

The front of an Aadhaar card typically contains:
- A photo of the person
- Name
- Date of Birth
- Gender
- Aadhaar number (12 digits)
- Government of India / UIDAI logo

Return ONLY in this exact JSON format:
{
  "is_aadhaar_front": true,
  "reason": "short reason"
}

Do not include any explanation, only return the JSON."""

            response_text = self._call_ollama(prompt, [img_b64])
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"is_aadhaar_front": False, "reason": "Failed to parse validation response"}
        except Exception as e:
            return {"is_aadhaar_front": False, "reason": str(e)}

    def validate_aadhaar_back(self, image_input: Union[str, io.BytesIO]) -> dict:
        """Check if the image is an Aadhaar card back side."""
        try:
            img_b64 = self._image_to_base64(image_input)
            if not img_b64:
                return {"is_aadhaar_back": False, "reason": "Image processing failed"}

            prompt = """Look at this image carefully. Is this the BACK side of an Indian Aadhaar card?

The back of an Aadhaar card typically contains:
- Address
- QR code
- Aadhaar number (12 digits)
- VID number (may or may not be present)

Return ONLY in this exact JSON format:
{
  "is_aadhaar_back": true,
  "reason": "short reason"
}

Do not include any explanation, only return the JSON."""

            response_text = self._call_ollama(prompt, [img_b64])
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"is_aadhaar_back": False, "reason": "Failed to parse validation response"}
        except Exception as e:
            return {"is_aadhaar_back": False, "reason": str(e)}

    def extract_aadhaar_data(self, front_image: Union[str, io.BytesIO], back_image: Union[str, io.BytesIO]) -> dict:
        """Extract Aadhaar data from front and back images (legacy combined method)"""
        try:
            # Convert images to base64
            front_b64 = self._image_to_base64(front_image)
            back_b64 = self._image_to_base64(back_image)

            if not front_b64: return {"error": "Front image processing failed"}

            prompt = """Analyze these Aadhaar card images (front and back) and extract the following information:

1. Aadhaar Number (12 digits)
2. Name
3. Date of Birth (DOB)
4. Gender/Sex
5. Address
6. Pincode
7. Is this a Masked Aadhaar? (Look for 'X' or '*' in the first 8 digits of the number)

Return ONLY in this exact JSON format:
{
  "aadharnumber": "123456789012",
  "name": "Full Name",
  "dob": "DD/MM/YYYY",
  "gender": "Male or Female",
  "address": "Full Address",
  "pincode": "123456",
  "is_masked": true/false
}

If any field is not clearly visible, use empty string "". Do not include any explanation, only return the JSON."""

            response_text = self._call_ollama(prompt, [front_b64, back_b64] if back_b64 else [front_b64])

            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data
            else:
                return {"error": "Failed to parse response"}

        except Exception as e:
            return {"error": str(e)}

    def extract_aadhaar_front(self, front_image: Union[str, io.BytesIO]) -> dict:
        """Extract data from Aadhaar FRONT image only: number, name, dob, gender, vid, is_masked"""
        try:
            front_b64 = self._image_to_base64(front_image)
            if not front_b64:
                return {"error": "Front image processing failed"}

            prompt = """Analyze this Aadhaar card FRONT image and extract the following information:

1. Aadhaar Number (12 digits, usually at the bottom of the card)
2. Name
3. Date of Birth (DOB)
4. Gender/Sex
5. VID Number (16 digits, if visible - Virtual ID printed below or near Aadhaar number)
6. Is this a Masked Aadhaar? (Look for 'X' or '*' in the first 8 digits of the number)

Return ONLY in this exact JSON format:
{
  "aadharnumber": "123456789012",
  "name": "Full Name",
  "dob": "DD/MM/YYYY",
  "gender": "Male or Female",
  "vid_number": "1234567890123456",
  "is_masked": false
}

If any field is not clearly visible, use empty string "". Do not include any explanation, only return the JSON."""

            response_text = self._call_ollama(prompt, [front_b64])

            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Failed to parse front response"}

        except Exception as e:
            return {"error": str(e)}

    def extract_aadhaar_back(self, back_image: Union[str, io.BytesIO]) -> dict:
        """Extract data from Aadhaar BACK image only: number, address, pincode, vid, is_masked"""
        try:
            back_b64 = self._image_to_base64(back_image)
            if not back_b64:
                return {"error": "Back image processing failed"}

            prompt = """Analyze this Aadhaar card BACK image and extract the following information:

1. Aadhaar Number (12 digits, usually at the bottom of the card)
2. Address (full address text)
3. Pincode (6 digits, at the end of the address)
4. VID Number (16 digits, if visible - Virtual ID printed below or near Aadhaar number)
5. Is this a Masked Aadhaar? (Look for 'X' or '*' in the first 8 digits of the number)

Return ONLY in this exact JSON format:
{
  "aadharnumber": "123456789012",
  "address": "Full Address",
  "pincode": "123456",
  "vid_number": "1234567890123456",
  "is_masked": false
}

If any field is not clearly visible, use empty string "". Do not include any explanation, only return the JSON."""

            response_text = self._call_ollama(prompt, [back_b64])

            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Failed to parse back response"}

        except Exception as e:
            return {"error": str(e)}

def _digit_diff_count(a: str, b: str) -> int:
    """Count how many digit positions differ between two equal-length strings."""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)

def cross_check_aadhaar(front_data: dict, back_data: dict) -> tuple[dict, bool, list]:
    """
    Merge front and back Aadhaar data and cross-check Aadhaar numbers and VID numbers.
    Allows up to 2 digit differences to tolerate OCR misreads.

    Returns:
        tuple: (merged_data, cross_check_passed, failure_list)
    """
    OCR_TOLERANCE = 2  # max digit differences allowed
    failures = []

    # Merge: front provides name/dob/gender, back provides address/pincode
    merged = {
        "aadharnumber": front_data.get("aadharnumber", ""),
        "name": front_data.get("name", ""),
        "dob": front_data.get("dob", ""),
        "gender": front_data.get("gender", ""),
        "address": back_data.get("address", "") if back_data else "",
        "pincode": back_data.get("pincode", "") if back_data else "",
        "vid_number": front_data.get("vid_number", ""),
        "is_masked": front_data.get("is_masked", False),
    }

    # If back data has is_masked set, propagate it
    if back_data and back_data.get("is_masked", False):
        merged["is_masked"] = True

    if not back_data:
        # No back image — skip cross-check
        return merged, True, failures

    # Cross-check Aadhaar numbers
    front_num = front_data.get("aadharnumber", "").replace(" ", "")
    back_num = back_data.get("aadharnumber", "").replace(" ", "")

    if front_num and back_num:
        # Both are 12-digit numbers
        front_is_valid = len(front_num) == 12 and front_num.isdigit()
        back_is_valid = len(back_num) == 12 and back_num.isdigit()

        if front_is_valid and back_is_valid:
            diff = _digit_diff_count(front_num, back_num)
            if diff > OCR_TOLERANCE:
                failures.append(f"Aadhaar Number Mismatch (Front: {front_num} vs Back: {back_num}, {diff} digits differ)")
        # If one side is invalid/empty, we don't fail cross-check — just use what we have

    # Cross-check VID numbers
    front_vid = front_data.get("vid_number", "").replace(" ", "")
    back_vid = back_data.get("vid_number", "").replace(" ", "") if back_data else ""

    if front_vid and back_vid:
        front_vid_valid = len(front_vid) == 16 and front_vid.isdigit()
        back_vid_valid = len(back_vid) == 16 and back_vid.isdigit()

        if front_vid_valid and back_vid_valid:
            diff = _digit_diff_count(front_vid, back_vid)
            if diff > OCR_TOLERANCE:
                failures.append(f"VID Number Mismatch (Front: {front_vid} vs Back: {back_vid}, {diff} digits differ)")

    # Use back VID if front VID is empty
    if not merged["vid_number"] and back_vid:
        merged["vid_number"] = back_vid

    cross_check_passed = len(failures) == 0
    return merged, cross_check_passed, failures


# --- Global State ---
qwen_agent = None
http_session = None
verification_scorer = None
face_verifier_ready = False
encrypted_logger = None

TEMP_DIR = Path("batch_temp")
PROGRESS_FILE = "batch_progress.json"
RESULTS_FILE = "batch_results.json"

# Batch processing state
batch_processing = False
batch_results = []
batch_progress = {
    "total": 0,
    "processed": 0,
    "current_batch": "",
    "current_user": "",
    "status": "idle"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global qwen_agent, http_session, verification_scorer, face_verifier_ready, encrypted_logger

    # Startup
    TEMP_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print("    BATCH VERIFICATION SERVER - QWEN POWERED (OPTIMIZED)")
    print("=" * 70)

    # Create persistent HTTP session
    connector = aiohttp.TCPConnector(
        limit=50,
        limit_per_host=10,
        ttl_dns_cache=300,
        force_close=True,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    # Initialize Qwen Agent (primary extraction method - OLLAMA)
    try:
        print("Initializing Qwen Agent (vLLM) for document extraction...")
        qwen_agent = OllamaQwenAgent(
            ollama_url="http://localhost:8000/v1/chat/completions",
            model="Qwen/Qwen3-VL-8B-Instruct"
        )
        print("Qwen Agent initialized via vLLM (Qwen/Qwen3-VL-8B-Instruct)")
    except Exception as e:
        print(f"Ollama Qwen Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        qwen_agent = None

    # Initialize VerificationScorer
    try:
        print("Initializing VerificationScorer...")
        verification_scorer = VerificationScorer()
        print("VerificationScorer initialized")
    except Exception as e:
        print(f"VerificationScorer initialization failed: {e}")
        verification_scorer = None

    # Initialize Encrypted User Logger
    try:
        from encrypted_logger import EncryptedUserLogger
        from config import get_config
        cfg = get_config()
        encrypted_logger = EncryptedUserLogger(
            logs_dir=cfg.app.encrypted_logs_dir,
            key_path=cfg.app.encryption_key_path,
        )
        print("Encrypted User Logger initialized")
    except Exception as e:
        print(f"Encrypted User Logger initialization failed: {e}")
        encrypted_logger = None

    # Initialize Face Detector (YOLOv12 for face embeddings)
    try:
        print("Initializing Face Detector (YOLOv12) for face embedding verification...")
        from app.faceEmbeddings import initialize_face_detector
        initialize_face_detector()
        face_verifier_ready = True
        print("Face Detector initialized")
    except Exception as e:
        print(f"Face Detector initialization failed: {e}")
        face_verifier_ready = False

    print("=" * 70)
    print("BATCH SERVER READY")
    print("=" * 70)

    yield

    # Shutdown
    if http_session:
        await http_session.close()
        await asyncio.sleep(0.25)

    print("--- BATCH SERVER SHUTDOWN ---")

app = FastAPI(lifespan=lifespan)

# --- Mahafraxn JSON Logger ---
class MahafraxnLogger:
    """Writes per-user JSON logs to logs/mahafraxn-logs/{agent_name}/{date}/{user_id}.json"""

    BASE_DIR = Path("logs/mahafraxn-logs")

    def log(self, agent_name: str, user_id: str, record: dict) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_dir = self.BASE_DIR / agent_name / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = str(user_id).replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{safe_id}.json"
        out_path.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
        return out_path

    def read(self, agent_name: str, date: str, user_id: str) -> dict:
        safe_id = str(user_id).replace("/", "_").replace("\\", "_")
        path = self.BASE_DIR / agent_name / date / f"{safe_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No log: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_agents(self) -> list[str]:
        if not self.BASE_DIR.exists():
            return []
        return sorted(p.name for p in self.BASE_DIR.iterdir() if p.is_dir())

    def list_dates(self, agent_name: str) -> list[str]:
        agent_dir = self.BASE_DIR / agent_name
        if not agent_dir.exists():
            return []
        return sorted(p.name for p in agent_dir.iterdir() if p.is_dir())

    def list_users(self, agent_name: str, date: str) -> list[str]:
        date_dir = self.BASE_DIR / agent_name / date
        if not date_dir.exists():
            return []
        return sorted(p.stem for p in date_dir.glob("*.json"))


mahafraxn_logger = MahafraxnLogger()

# --- Request Models ---
class BatchVerifyRequest(BaseModel):
    dataset_root: str
    output_dir: Optional[str] = "batch_outputs"

class VerifyRequest(BaseModel):
    user_id: Union[int, str]
    dob: Optional[str] = None
    passport_first: str
    passport_old: str
    selfie_photo: str
    gender: Optional[str] = None



# --- Helper Functions ---
def save_progress(data: dict):
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed to save progress: {e}")

def load_progress() -> Optional[dict]:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load progress: {e}")
    return None

def save_results(results: list):
    try:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Failed to save results: {e}")

def load_results() -> list:
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load results: {e}")
    return []

def find_images_in_folder(folder_path: Path) -> dict:
    """Find Aadhaar and Selfie images in a user folder."""
    images = {
        "aadhar_front": None,
        "aadhar_back": None,
        "selfie": None
    }

    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_files = []
    for ext in exts:
        all_files.extend(folder_path.glob(ext))

    for file_path in all_files:
        fname = file_path.name.lower()
        if "aadhar_front" in fname or "aadhaar_front" in fname:
            images["aadhar_front"] = str(file_path)
        elif "aadhar_back" in fname or "aadhaar_back" in fname:
            images["aadhar_back"] = str(file_path)
        elif "selfie" in fname or "profile" in fname:
            images["selfie"] = str(file_path)

    return images

# --- OPTIMIZATION 2: IN-MEMORY FETCH ---
async def fetch_file(session: aiohttp.ClientSession, source: str) -> Optional[io.BytesIO]:
    """
    Download file from URL or load from path DIRECTLY INTO MEMORY.
    Returns io.BytesIO object instead of writing to disk.
    """
    try:
        if str(source).startswith(('http://', 'https://')):
            loop = asyncio.get_event_loop()

            def download_with_cloudscraper(url: str) -> bytes:
                scraper = cloudscraper.create_scraper(
                    browser={'browser': 'chrome', 'platform': 'darwin', 'mobile': False}
                )
                try:
                    response = scraper.get(url, timeout=30)
                    response.raise_for_status()
                    return response.content
                finally:
                    if hasattr(scraper, 'close'):
                        scraper.close()

            content = await loop.run_in_executor(None, download_with_cloudscraper, str(source))
            return io.BytesIO(content)
        else:
            source_path = Path(source)
            if source_path.exists():
                with open(source_path, 'rb') as f:
                    return io.BytesIO(f.read())
    except Exception as e:
        print(f"Error fetching {source}: {e}")
    return None

async def verify_single_user(user_id: str, images: dict, expected_dob: Optional[str] = None, expected_gender: Optional[str] = None, product: str = "qoneqt") -> dict:
    """
    Verify a single user.
    Images dict can contain file paths (str) or in-memory objects (io.BytesIO).
    """
    user_record = {
        "user_id": user_id,
        "status": "PROCESSING",
        "final_decision": "PENDING",
        "status_code": 0,
        "score": 0,
        "aadhaar_found": False,
        "aadhaar_number": "", "aadhaar_name": "", "aadhaar_dob": "", "aadhaar_gender": "", "aadhaar_address": "", "aadhaar_pincode": "",
        "extracted_data": {},
        "input_data": { "dob": expected_dob },
        "rejection_reasons": [], "error_log": ""
    }

    if not qwen_agent:
        user_record.update({"status": "ERROR", "final_decision": "SYSTEM_ERROR", "status_code": -1, "reason": "Qwen agent not initialized"})
        return user_record

    loop = asyncio.get_event_loop()

    # ========================================================================
    # Step 1: Extract Aadhaar data — separate front/back with cross-check
    # ========================================================================
    cross_check_failures = []
    vid_number = ""
    aadhaar_cross_check_passed = True

    # Both front and back are required
    if not images.get("aadhar_front") or not images.get("aadhar_back"):
        missing = []
        if not images.get("aadhar_front"): missing.append("aadhar_front")
        if not images.get("aadhar_back"): missing.append("aadhar_back")
        user_record.update({
            "status": "REJECTED", "final_decision": "REJECTED", "status_code": 1,
            "rejection_reasons": [f"missing_{m}" for m in missing],
            "error_log": f"Missing required images: {', '.join(missing)}"
        })
        print(f"[{user_id}] REJECTED: Missing {', '.join(missing)}")
        return user_record

    try:
        # Validate FRONT — is this actually an Aadhaar front?
        print(f"[{user_id}] Validating Aadhaar FRONT image...")
        front_validation = await loop.run_in_executor(
            None,
            qwen_agent.validate_aadhaar_front,
            images["aadhar_front"]
        )
        if not front_validation.get("is_aadhaar_front"):
            reason = front_validation.get("reason", "Not an Aadhaar front")
            user_record.update({
                "status": "REJECTED", "final_decision": "REJECTED", "status_code": 1,
                "rejection_reasons": ["not_aadhaar_front"],
                "error_log": f"Image is not Aadhaar FRONT: {reason}"
            })
            print(f"[{user_id}] REJECTED: Image is not Aadhaar FRONT — {reason}")
            return user_record

        # Validate BACK — is this actually an Aadhaar back?
        print(f"[{user_id}] Validating Aadhaar BACK image...")
        back_validation = await loop.run_in_executor(
            None,
            qwen_agent.validate_aadhaar_back,
            images["aadhar_back"]
        )
        if not back_validation.get("is_aadhaar_back"):
            reason = back_validation.get("reason", "Not an Aadhaar back")
            user_record.update({
                "status": "REJECTED", "final_decision": "REJECTED", "status_code": 1,
                "rejection_reasons": ["not_aadhaar_back"],
                "error_log": f"Image is not Aadhaar BACK: {reason}"
            })
            print(f"[{user_id}] REJECTED: Image is not Aadhaar BACK — {reason}")
            return user_record

        # Extract FRONT
        print(f"[{user_id}] Extracting Aadhaar FRONT with Qwen...")
        front_data = await loop.run_in_executor(
            None,
            qwen_agent.extract_aadhaar_front,
            images["aadhar_front"]
        )
        if not front_data or "error" in front_data:
            error_msg = front_data.get("error", "Unknown error") if front_data else "No response"
            user_record.update({
                "status": "REJECTED", "final_decision": "REJECTED", "status_code": 1,
                "rejection_reasons": ["aadhaar_front_extraction_failed"],
                "error_log": f"Aadhaar FRONT extraction failed: {error_msg}"
            })
            print(f"[{user_id}] REJECTED: Aadhaar FRONT extraction failed")
            return user_record

        # Extract BACK
        print(f"[{user_id}] Extracting Aadhaar BACK with Qwen...")
        back_data = await loop.run_in_executor(
            None,
            qwen_agent.extract_aadhaar_back,
            images["aadhar_back"]
        )
        if not back_data or "error" in back_data:
            error_msg = back_data.get("error", "Unknown error") if back_data else "No response"
            user_record.update({
                "status": "REJECTED", "final_decision": "REJECTED", "status_code": 1,
                "rejection_reasons": ["aadhaar_back_extraction_failed"],
                "error_log": f"Aadhaar BACK extraction failed: {error_msg}"
            })
            print(f"[{user_id}] REJECTED: Aadhaar BACK extraction failed")
            return user_record

        # Cross-check front and back
        aadhaar_data, aadhaar_cross_check_passed, cross_check_failures = cross_check_aadhaar(
            front_data, back_data
        )

        if cross_check_failures:
            user_record["rejection_reasons"].extend(
                [f"cross_check: {f}" for f in cross_check_failures]
            )
            print(f"[{user_id}] Cross-check failures: {cross_check_failures}")

        # Check for Masked Aadhaar
        is_masked = aadhaar_data.get("is_masked", False)
        aadhar_num = aadhaar_data.get("aadharnumber", "")
        if "X" in aadhar_num.upper() or "*" in aadhar_num:
            is_masked = True

        vid_number = aadhaar_data.get("vid_number", "")

        if is_masked:
            user_record["status"] = "REJECTED"
            user_record["final_decision"] = "REJECTED"
            user_record["status_code"] = 1
            user_record["rejection_reasons"].append("aadhaar_masked")
            user_record["aadhaar_found"] = True
            user_record["error_log"] += "Masked Aadhaar detected; "
            print(f"[{user_id}] REJECTED: Masked Aadhaar detected")
            return user_record
        else:
            user_record["aadhaar_found"] = True
            user_record["aadhaar_number"] = aadhar_num
            user_record["aadhaar_name"] = aadhaar_data.get("name", "")
            user_record["aadhaar_dob"] = aadhaar_data.get("dob", "")
            user_record["aadhaar_gender"] = aadhaar_data.get("gender", "")
            user_record["aadhaar_address"] = aadhaar_data.get("address", "")
            user_record["aadhaar_pincode"] = aadhaar_data.get("pincode", "")
            print(f"[{user_id}] Aadhaar extracted - Gender: {user_record['aadhaar_gender']}")

    except Exception as e:
        user_record.update({
            "status": "REJECTED", "final_decision": "REJECTED", "status_code": 1,
            "rejection_reasons": ["aadhaar_processing_error"],
            "error_log": f"Aadhaar exception: {str(e)}"
        })
        print(f"[{user_id}] REJECTED: Aadhaar exception: {e}")
        return user_record

    # ========================================================================
    # Step 2: Face Embedding Verification (YOLOv12 + DeepFace)
    # ========================================================================
    face_verification_result = None
    if images.get("selfie") and images.get("aadhar_front") and face_verifier_ready:
        try:
            print(f"[{user_id}] Running face embedding verification (YOLOv12 + DeepFace)...")
            from app.faceEmbeddings import verify_faces_memory

            # Seek to beginning of BytesIO objects before passing
            selfie_input = images["selfie"]
            aadhaar_input = images["aadhar_front"]
            if isinstance(selfie_input, io.BytesIO):
                selfie_input.seek(0)
            if isinstance(aadhaar_input, io.BytesIO):
                aadhaar_input.seek(0)

            face_verification_result = await loop.run_in_executor(
                None,
                lambda s, a, u, p: verify_faces_memory(s, a, u, p),
                selfie_input,
                aadhaar_input,
                str(user_id),
                product
            )

            face_sim = face_verification_result.get("similarity_score", 0.0)
            face_verified = face_verification_result.get("verified", False)
            is_duplicate = face_verification_result.get("is_duplicate", False)
            face_error = face_verification_result.get("error")

            print(f"[{user_id}] Face similarity: {face_sim}, verified: {face_verified}, duplicate: {is_duplicate}")

            if is_duplicate:
                dup_user = face_verification_result.get("duplicate_user_id", "unknown")
                user_record["rejection_reasons"].append(f"duplicate_face_matches_{dup_user}")
                print(f"[{user_id}] DUPLICATE FACE detected (matches {dup_user})")

            if face_error and not face_verified:
                user_record["error_log"] += f"Face verification: {face_error}; "

        except Exception as e:
            print(f"[{user_id}] Face embedding verification failed: {e}")
            user_record["error_log"] += f"Face embedding exception: {str(e)}; "

    # ========================================================================
    # Step 3: Calculate Score using VerificationScorer
    # ========================================================================
    if verification_scorer:
        try:
            # Prepare entity_data in the format expected by scoring.py
            entity_data = {
                "aadharnumber": user_record["aadhaar_number"],
                "name": user_record["aadhaar_name"],
                "dob": user_record["aadhaar_dob"],
                "gender": user_record["aadhaar_gender"],
                "address": user_record["aadhaar_address"],
                "age_status": "age_approved"  # We don't have age validation yet, assume approved if DOB extracted
            }

            # Calculate score using VerificationScorer
            scoring_result = verification_scorer.calculate_score(
                entity_data=entity_data,
                expected_dob=expected_dob,
                face_verification_result=face_verification_result,
                cross_check_failures=cross_check_failures if cross_check_failures else None
            )

            # Extract results
            final_score = scoring_result["total_score"]
            final_status = scoring_result["status"]
            breakdown = scoring_result["breakdown"]
            rejection_reasons = scoring_result["rejection_reasons"]

            # Update user_record with scoring results
            user_record["score"] = final_score
            user_record["status"] = final_status
            user_record["final_decision"] = final_status

            # Map status to status_code
            status_code_map = {"APPROVED": 2, "REVIEW": 0, "REJECTED": 1}
            user_record["status_code"] = status_code_map.get(final_status, 0)

            # Add rejection reasons
            user_record["rejection_reasons"].extend(rejection_reasons)

            print(f"[{user_id}] -> {final_status} (Score: {final_score})")
            if rejection_reasons:
                print(f"[{user_id}] Rejection Reasons: {', '.join(rejection_reasons)}")

            # Post-scoring overrides
            if final_status != "REJECTED":
                if face_verification_result and not face_verification_result.get("verified", False):
                    face_error = face_verification_result.get("error")
                    similarity = float(face_verification_result.get("similarity_score", 0.0))

                    if face_error and "No face detected" in face_error:
                        # No face found in one of the images
                        user_record["status"] = "REVIEW"
                        user_record["final_decision"] = "REVIEW"
                        user_record["status_code"] = 0
                        user_record["rejection_reasons"].append("no_face_detected")
                        print(f"[{user_id}] -> REVIEW (No face detected)")
                    elif similarity >= 0.4:
                        # Close to threshold — send for human review (e.g. age gap, facial hair)
                        user_record["status"] = "REVIEW"
                        user_record["final_decision"] = "REVIEW"
                        user_record["status_code"] = 0
                        user_record["rejection_reasons"].append(f"face_similarity_low_{similarity:.4f}")
                        print(f"[{user_id}] -> REVIEW (Face similarity {similarity:.4f} below threshold but close)")
                    else:
                        # Very low similarity — reject
                        user_record["status"] = "REJECTED"
                        user_record["final_decision"] = "REJECTED"
                        user_record["status_code"] = 1
                        user_record["rejection_reasons"].append(f"face_not_verified_{similarity:.4f}")
                        print(f"[{user_id}] -> REJECTED (Face similarity {similarity:.4f} too low)")

        except Exception as e:
            print(f"[{user_id}] Scoring exception: {e}")
            user_record["error_log"] += f"Scoring exception: {str(e)}; "
            user_record.update({"status": "REVIEW", "final_decision": "REVIEW", "status_code": 0, "score": 0})
            user_record["rejection_reasons"].append("scoring_error")
    else:
        # Fallback if scorer not initialized
        print(f"[{user_id}] VerificationScorer not initialized, using fallback")
        user_record.update({"status": "REVIEW", "final_decision": "REVIEW", "status_code": 0, "score": 0})
        user_record["rejection_reasons"].append("scorer_not_initialized")

    # ========================================================================
    # Step 4: Build extracted_data (enriched)
    # ========================================================================
    user_record["extracted_data"] = {
        "aadhaar": user_record["aadhaar_number"],
        "name": user_record["aadhaar_name"],
        "dob": user_record["aadhaar_dob"],
        "address": user_record["aadhaar_address"],
        "gender": user_record["aadhaar_gender"],
    }
    user_record["_internal"] = {
        "vid_number": vid_number,
        "face_similarity": float(face_verification_result.get("similarity_score", 0.0)) if face_verification_result else 0.0,
        "face_verified": bool(face_verification_result.get("verified", False)) if face_verification_result else False,
        "is_duplicate_face": bool(face_verification_result.get("is_duplicate", False)) if face_verification_result else False,
        "aadhaar_cross_check_passed": bool(aadhaar_cross_check_passed),
    }

    return user_record

# --- API Endpoints ---
@app.post("/batch/verify")
async def start_batch_verification(req: BatchVerifyRequest, background_tasks: BackgroundTasks):
    global batch_processing, batch_progress
    if batch_processing:
        return {"status": "error", "message": "Batch processing already in progress", "progress": batch_progress}

    if not os.path.exists(req.dataset_root):
        return {"status": "error", "message": f"Dataset directory not found: {req.dataset_root}"}

    background_tasks.add_task(process_batch, req.dataset_root, req.output_dir)
    return {"status": "started", "message": "Batch verification started", "dataset_root": req.dataset_root}

@app.post("/verification/verify")
async def verify_user(req: VerifyRequest):
    """Verify a single user - Returns CLEAN JSON (Production)"""
    return await _run_verification(req, debug=False)

@app.post("/verification/verify/debug")
async def verify_user_debug(req: VerifyRequest):
    """Verify a single user - Returns FULL VERBOSE JSON (Debug)"""
    return await _run_verification(req, debug=True)

@app.post("/verification/verify/agent/")
async def verify_user_production(req: VerifyRequest):
    """Production Endpoint - Returns CLEAN JSON"""
    return await _run_verification(req, debug=False)

async def _run_verification(req: VerifyRequest, debug: bool = False):
    """Internal helper to run verification and strictly filter response if not debug"""
    user_id = str(req.user_id)

    try:
        # Download files directly to memory (No disk I/O)
        print(f"[{user_id}] Fetching images to memory...")
        download_tasks = [
            fetch_file(http_session, req.selfie_photo),
            fetch_file(http_session, req.passport_first),
            fetch_file(http_session, req.passport_old)
        ]

        results = await asyncio.gather(*download_tasks, return_exceptions=True)

        selfie_img, front_img, back_img = results[0], results[1], results[2]

        # Check if mandatory files exist (all three required)
        missing = []
        if not selfie_img: missing.append("selfie")
        if not front_img: missing.append("aadhar_front")
        if not back_img: missing.append("aadhar_back")
        if missing:
            return {
                "user_id": user_id, "status": "FAILED", "final_decision": "REJECTED",
                "status_code": 1, "score": 0, "reason": f"File Retrieval Failed: {', '.join(missing)}",
                "rejection_reasons": [f"file_download_failed_{m}" for m in missing]
            }

        # Construct images dict with in-memory BytesIO objects
        images = {
            "selfie": selfie_img,
            "aadhar_front": front_img,
            "aadhar_back": back_img
        }

        # Verify user passing memory objects
        result = await verify_single_user(user_id, images, req.dob, req.gender)

        # Encrypted audit log
        if encrypted_logger:
            try:
                encrypted_logger.log_user(user_id, result)
            except Exception as log_err:
                print(f"[{user_id}] Encrypted logging failed: {log_err}")

        # Filter response if not debug
        if not debug:
            return _filter_response(result)

        return result

    except Exception as e:
        print(f"Error in single verification: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e), "user_id": user_id}
    finally:
        gc.collect()

# --- Mahafraxn Verification Helpers ---
async def _run_mahafraxn_verification(images: dict, user_id: str, dob: Optional[str], gender: Optional[str], debug: bool = False, agent_name: str = "default"):
    """Internal helper for Mahafraxn verification pipeline"""
    try:
        has_selfie = images.get("selfie") is not None
        has_front = images.get("aadhar_front") is not None
        has_back = images.get("aadhar_back") is not None

        # Selfie-only mode: just run duplicate check against mahafraxn DB
        if has_selfie and not has_front and not has_back:
            print(f"[mahafraxn][{user_id}] Selfie-only mode — running duplicate check...")
            from app.faceEmbeddings import check_selfie_duplicate

            loop = asyncio.get_event_loop()
            selfie_input = images["selfie"]
            if isinstance(selfie_input, io.BytesIO):
                selfie_input.seek(0)

            dup_result = await loop.run_in_executor(
                None,
                lambda: check_selfie_duplicate(selfie_input, str(user_id), product="mahafraxn")
            )

            result = {
                "user_id": user_id,
                "mode": "selfie_only_duplicate_check",
                "selfie_face_detected": dup_result["selfie_face_detected"],
                "is_duplicate": dup_result["is_duplicate"],
                "duplicate_user_id": dup_result.get("duplicate_user_id"),
                "duplicate_similarity": dup_result.get("duplicate_similarity", 0.0),
                "status": "REJECTED" if dup_result["is_duplicate"] else ("FAILED" if dup_result.get("error") and not dup_result["selfie_face_detected"] else "OK"),
                "final_decision": "REJECTED" if dup_result["is_duplicate"] else ("FAILED" if dup_result.get("error") and not dup_result["selfie_face_detected"] else "NOT_DUPLICATE"),
                "status_code": 1 if dup_result["is_duplicate"] else 0,
                "rejection_reasons": [f"duplicate_face_matches_{dup_result['duplicate_user_id']}"] if dup_result["is_duplicate"] else [],
                "error": dup_result.get("error"),
                "product": "mahafraxn",
            }

            try:
                mahafraxn_logger.log(agent_name, user_id, result)
            except Exception as log_err:
                print(f"[mahafraxn][{user_id}] JSON logging failed: {log_err}")

            if encrypted_logger:
                try:
                    encrypted_logger.log_user(f"mahafraxn_{user_id}", result)
                except Exception:
                    pass

            return result

        # Full verification mode: selfie + front + back required
        if not has_selfie:
            return {
                "user_id": user_id, "status": "FAILED", "final_decision": "REJECTED",
                "status_code": 1, "score": 0, "product": "mahafraxn",
                "reason": "Selfie is required",
                "rejection_reasons": ["missing_selfie"]
            }

        missing = []
        if not has_front: missing.append("aadhar_front")
        if not has_back: missing.append("aadhar_back")
        if missing:
            return {
                "user_id": user_id, "status": "FAILED", "final_decision": "REJECTED",
                "status_code": 1, "score": 0, "product": "mahafraxn",
                "reason": f"Image processing failed: {', '.join(missing)}",
                "rejection_reasons": [f"missing_{m}" for m in missing]
            }

        result = await verify_single_user(user_id, images, dob, gender, product="mahafraxn")

        # JSON log to logs/mahafraxn-logs/{agent_name}/{date}/{user_id}.json
        try:
            mahafraxn_logger.log(agent_name, user_id, result)
        except Exception as log_err:
            print(f"[mahafraxn][{user_id}] JSON logging failed: {log_err}")

        # Encrypted audit log with mahafraxn prefix
        if encrypted_logger:
            try:
                encrypted_logger.log_user(f"mahafraxn_{user_id}", result)
            except Exception as log_err:
                print(f"[mahafraxn][{user_id}] Encrypted logging failed: {log_err}")

        # Add product tag to response
        result["product"] = "mahafraxn"

        if not debug:
            filtered = _filter_response(result)
            filtered["product"] = "mahafraxn"
            return filtered

        return result

    except Exception as e:
        print(f"[mahafraxn] Error in verification: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e), "user_id": user_id, "product": "mahafraxn"}
    finally:
        gc.collect()

def _filter_response(full_record: dict) -> dict:
    """Filter the response to only include essential fields for the API (Production)"""
    return {
        "user_id": full_record.get("user_id"),
        "status": full_record.get("status"),
        "final_decision": full_record.get("final_decision"),
        "status_code": full_record.get("status_code"),
        "score": full_record.get("score"),
        "rejection_reasons": full_record.get("rejection_reasons", []),
        "extracted_data": full_record.get("extracted_data", {})
    }

# --- Mahafraxn Endpoint ---
@app.post("/mahafraxn/verify")
async def mahafraxn_verify(
    user_id: str = Form(...),
    agent_name: str = Form("default"),
    documents_aadhaar_front: UploadFile = File(None),
    documents_aadhaar_back: UploadFile = File(None),
    documents_selfie: UploadFile = File(None),
    aadhar_front_url: Optional[str] = Form(None),
    aadhar_back_url: Optional[str] = Form(None),
    selfie_url: Optional[str] = Form(None),
    dob: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
):
    """
    Unified Mahafraxn verification endpoint.
    Accepts raw file uploads (multipart/form-data) OR presigned URLs.
    Priority: raw file upload first, falls back to URL if file not provided or empty.
    Logs are written to logs/mahafraxn-logs/{agent_name}/{date}/{user_id}.json
    """
    try:
        print(f"[mahafraxn][{agent_name}][{user_id}] Resolving images (upload > URL fallback)...")

        async def resolve_image(upload: Optional[UploadFile], url: Optional[str], label: str) -> Optional[io.BytesIO]:
            # Try raw file upload first
            if upload and upload.size and upload.size > 0:
                raw = await upload.read()
                if raw:
                    print(f"[mahafraxn][{agent_name}][{user_id}] {label}: loaded from upload ({len(raw)} bytes)")
                    return io.BytesIO(raw)
            # Fallback to presigned URL / CDN link
            if url:
                print(f"[mahafraxn][{agent_name}][{user_id}] {label}: fetching from URL...")
                img = await fetch_file(http_session, url)
                if img:
                    return img
                print(f"[mahafraxn][{agent_name}][{user_id}] {label}: URL fetch failed")
            return None

        front_img, back_img, selfie_img = await asyncio.gather(
            resolve_image(documents_aadhaar_front, aadhar_front_url, "aadhar_front"),
            resolve_image(documents_aadhaar_back, aadhar_back_url, "aadhar_back"),
            resolve_image(documents_selfie, selfie_url, "selfie"),
        )

        images = {
            "aadhar_front": front_img,
            "aadhar_back": back_img,
            "selfie": selfie_img,
        }

        return await _run_mahafraxn_verification(images, user_id, dob, gender, agent_name=agent_name)

    except Exception as e:
        print(f"[mahafraxn] Endpoint error: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e), "user_id": user_id, "product": "mahafraxn"}

@app.post("/mahafraxn/verify/debug")
async def mahafraxn_verify_debug(
    user_id: str = Form(...),
    agent_name: str = Form("default"),
    documents_aadhaar_front: UploadFile = File(None),
    documents_aadhaar_back: UploadFile = File(None),
    documents_selfie: UploadFile = File(None),
    aadhar_front_url: Optional[str] = Form(None),
    aadhar_back_url: Optional[str] = Form(None),
    selfie_url: Optional[str] = Form(None),
    dob: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
):
    """Mahafraxn verification - Returns FULL VERBOSE JSON (Debug). Same input as /mahafraxn/verify."""
    try:
        print(f"[mahafraxn][{agent_name}][{user_id}] Resolving images (debug, upload > URL fallback)...")

        async def resolve_image(upload: Optional[UploadFile], url: Optional[str], label: str) -> Optional[io.BytesIO]:
            if upload and upload.size and upload.size > 0:
                raw = await upload.read()
                if raw:
                    return io.BytesIO(raw)
            if url:
                img = await fetch_file(http_session, url)
                if img:
                    return img
            return None

        front_img, back_img, selfie_img = await asyncio.gather(
            resolve_image(documents_aadhaar_front, aadhar_front_url, "aadhar_front"),
            resolve_image(documents_aadhaar_back, aadhar_back_url, "aadhar_back"),
            resolve_image(documents_selfie, selfie_url, "selfie"),
        )

        images = {
            "aadhar_front": front_img,
            "aadhar_back": back_img,
            "selfie": selfie_img,
        }

        return await _run_mahafraxn_verification(images, user_id, dob, gender, debug=True, agent_name=agent_name)

    except Exception as e:
        print(f"[mahafraxn] Debug endpoint error: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e), "user_id": user_id, "product": "mahafraxn"}

@app.get("/batch/progress")
async def get_batch_progress():
    return batch_progress

@app.get("/batch/results")
async def get_batch_results():
    return {
        "results": batch_results,
        "total": len(batch_results),
        "summary": {
            "approved": sum(1 for r in batch_results if r.get("final_decision") == "APPROVED"),
            "rejected": sum(1 for r in batch_results if r.get("final_decision") == "REJECTED"),
            "review": sum(1 for r in batch_results if r.get("final_decision") == "REVIEW"),
            "error": sum(1 for r in batch_results if r.get("final_decision") == "SYSTEM_ERROR")
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "products": ["qoneqt", "mahafraxn"],
        "components": {
            "qwen_agent": qwen_agent is not None,
            "face_verifier": face_verifier_ready,
            "encrypted_logger": encrypted_logger is not None
        },
        "batch_processing": batch_processing
    }

# --- Audit Endpoints ---
@app.get("/audit/user/{user_id}")
async def audit_user(user_id: str):
    """Decrypt and return a user's encrypted verification log."""
    if not encrypted_logger:
        return {"status": "error", "message": "Encrypted logger not initialized"}
    try:
        record = encrypted_logger.read_user_log(user_id)
        return {"status": "ok", "user_id": user_id, "record": record}
    except FileNotFoundError:
        return {"status": "not_found", "message": f"No log for user {user_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/audit/users")
async def audit_list_users():
    """List all user IDs with encrypted logs."""
    if not encrypted_logger:
        return {"status": "error", "message": "Encrypted logger not initialized"}
    users = encrypted_logger.list_logged_users()
    return {"status": "ok", "count": len(users), "user_ids": users}

# --- Mahafraxn Audit Endpoints ---
@app.get("/audit/mahafraxn")
async def mahafraxn_audit_agents():
    """List all agent names that have Mahafraxn logs."""
    agents = mahafraxn_logger.list_agents()
    return {"status": "ok", "count": len(agents), "agents": agents}

@app.get("/audit/mahafraxn/{agent_name}")
async def mahafraxn_audit_dates(agent_name: str):
    """List all dates that have Mahafraxn logs for a given agent."""
    dates = mahafraxn_logger.list_dates(agent_name)
    return {"status": "ok", "agent_name": agent_name, "count": len(dates), "dates": dates}

@app.get("/audit/mahafraxn/{agent_name}/{date}")
async def mahafraxn_audit_users(agent_name: str, date: str):
    """List all user IDs logged under a specific agent and date."""
    users = mahafraxn_logger.list_users(agent_name, date)
    return {"status": "ok", "agent_name": agent_name, "date": date, "count": len(users), "user_ids": users}

@app.get("/audit/mahafraxn/{agent_name}/{date}/{user_id}")
async def mahafraxn_audit_user(agent_name: str, date: str, user_id: str):
    """Return the JSON verification log for a specific Mahafraxn user."""
    try:
        record = mahafraxn_logger.read(agent_name, date, user_id)
        return {"status": "ok", "agent_name": agent_name, "date": date, "user_id": user_id, "record": record}
    except FileNotFoundError:
        return {"status": "not_found", "message": f"No log for {agent_name}/{date}/{user_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- Background Processing ---
async def process_batch(dataset_root: str, output_dir: str):
    """Background task to process entire batch dataset."""
    global batch_processing, batch_results, batch_progress

    batch_processing = True
    batch_results = []

    try:
        # Find all batch folders
        batch_folders = sorted([f for f in glob.glob(os.path.join(dataset_root, "batch_*")) if os.path.isdir(f)])

        if not batch_folders:
            print(f"No batch folders found in {dataset_root}")
            batch_processing = False
            return

        # Count total users
        total_users = 0
        for batch_path in batch_folders:
            user_folders = [f for f in glob.glob(os.path.join(batch_path, "*")) if os.path.isdir(f)]
            total_users += len(user_folders)

        batch_progress["total"] = total_users
        batch_progress["processed"] = 0
        batch_progress["status"] = "processing"

        print(f"Starting batch processing: {len(batch_folders)} batches, {total_users} users")

        # Process each batch
        for batch_path in batch_folders:
            batch_name = os.path.basename(batch_path)
            batch_progress["current_batch"] = batch_name

            user_folders = [f for f in glob.glob(os.path.join(batch_path, "*")) if os.path.isdir(f)]

            for user_path in user_folders:
                user_id = os.path.basename(user_path)
                batch_progress["current_user"] = f"{batch_name}/{user_id}"

                try:
                    # Find images (returns file paths as strings)
                    images = find_images_in_folder(Path(user_path))

                    # Verify user
                    result = await verify_single_user(f"{batch_name}_{user_id}", images, None, None)
                    batch_results.append(result)

                    # Encrypted audit log
                    if encrypted_logger:
                        try:
                            encrypted_logger.log_user(f"{batch_name}_{user_id}", result)
                        except Exception as log_err:
                            print(f"[{batch_name}_{user_id}] Encrypted logging failed: {log_err}")

                    # Save individual user result immediately
                    user_result_file = Path(output_dir) / "individual_results" / f"{batch_name}_{user_id}.json"
                    user_result_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(user_result_file, 'w') as f:
                            json.dump(result, f, indent=2)
                    except Exception as e:
                        print(f"Failed to save individual result for {user_id}: {e}")

                    # Update progress
                    batch_progress["processed"] += 1
                    batch_progress["last_processed"] = f"{batch_name}_{user_id}"
                    batch_progress["timestamp"] = datetime.now().isoformat()

                    if batch_progress["processed"] % 5 == 0:
                        save_progress(batch_progress)
                        save_results(batch_results)

                    if batch_progress["processed"] % 20 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"Error processing user {user_id}: {e}")
                    batch_results.append({
                        "user_id": f"{batch_name}_{user_id}",
                        "status": "ERROR",
                        "final_decision": "SYSTEM_ERROR",
                        "error_log": str(e)
                    })
                    batch_progress["processed"] += 1

        # Save final results
        batch_progress["status"] = "completed"
        batch_progress["completion_time"] = datetime.now().isoformat()
        save_progress(batch_progress)
        save_results(batch_results)

        if batch_results:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            comprehensive_data = {
                "metadata": {"total_users": len(batch_results), "timestamp": timestamp},
                "users": batch_results
            }

            with open(output_path / f"batch_results_complete_{timestamp}.json", 'w') as f:
                json.dump(comprehensive_data, f, indent=2)

            df = pd.DataFrame(batch_results)
            df.to_csv(output_path / f"batch_results_{timestamp}.csv", index=False)

        gc.collect()

    except Exception as e:
        print(f"Batch processing error: {e}")
        batch_progress["status"] = "error"
        batch_progress["error"] = str(e)
    finally:
        batch_processing = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101, workers=1)
