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
from pathlib import Path
from typing import Optional, Union, List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
print("new")
# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- IMPORTS ---
try:
    from app.gender_pipeline import GenderPipeline
except ImportError as e:
    print(f"ImportError: {e}")
    # allowing to fail if not present, but logged
    pass

import requests
import base64
from PIL import Image
import io
from scoring import VerificationScorer

# --- Ollama Qwen Agent ---
class OllamaQwenAgent:
    """Ollama-based Qwen agent for document extraction using qwen3-vl:8b-instruct"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "qwen3-vl:8b-instruct"):
        self.ollama_url = ollama_url
        self.model = model
        print(f"Initialized Ollama Qwen Agent with model: {model}")
    
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
        """Call Ollama API with vision model"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512
            }
        }
        
        response = requests.post(self.ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    
    def extract_aadhaar_data(self, front_image: Union[str, io.BytesIO], back_image: Union[str, io.BytesIO]) -> dict:
        """Extract Aadhaar data from front and back images"""
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
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                import json
                data = json.loads(json_match.group())
                return data
            else:
                return {"error": "Failed to parse response"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def extract_pancard_data(self, image_input: Union[str, io.BytesIO]) -> dict:
        """Extract PAN card data from image"""
        try:
            # Convert image to base64
            img_b64 = self._image_to_base64(image_input)
            
            if not img_b64: return {"error": "Image processing failed"}
            
            prompt = """Analyze this PAN card image and extract the following information:

1. PAN Number (10 characters)
2. Name
3. Father's Name
4. Date of Birth

Return ONLY in this exact JSON format:
{
  "pan_number": "ABCDE1234F",
  "name": "Full Name",
  "father_name": "Father Name",
  "dob": "DD/MM/YYYY"
}

If any field is not clearly visible, use empty string "". Do not include any explanation, only return the JSON."""

            response_text = self._call_ollama(prompt, [img_b64])
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                import json
                data = json.loads(json_match.group())
                return data
            else:
                return {"error": "Failed to parse response"}
                
        except Exception as e:
            return {"error": str(e)}

# --- Global State ---
qwen_agent = None
gender_pipeline = None
http_session = None
verification_scorer = None

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
    global qwen_agent, gender_pipeline, http_session, verification_scorer
    
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
        print("Initializing Ollama Qwen Agent for document extraction...")
        qwen_agent = OllamaQwenAgent(
            ollama_url="http://localhost:11434/api/generate",
            model="qwen3-vl:8b-instruct"
        )
        print("✅ Ollama Qwen Agent initialized (qwen3-vl:8b-instruct)")
    except Exception as e:
        print(f"⚠️ Ollama Qwen Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        qwen_agent = None
    
    # Initialize Gender Pipeline (for face gender detection)
    try:
        print("Initializing GenderPipeline for face verification...")
        from app.gender_pipeline import GenderPipeline # Import here to ensure it uses the mock if needed
        gender_pipeline = GenderPipeline()
        print("✅ GenderPipeline initialized")
    except Exception as e:
        print(f"⚠️ GenderPipeline initialization failed: {e}")
        gender_pipeline = None
    
    # Initialize VerificationScorer
    try:
        print("Initializing VerificationScorer...")
        verification_scorer = VerificationScorer()
        print("✅ VerificationScorer initialized")
    except Exception as e:
        print(f"⚠️ VerificationScorer initialization failed: {e}")
        verification_scorer = None
    
    print("=" * 70)
    print("✅ BATCH SERVER READY")
    print("=" * 70)
    
    yield
    
    # Shutdown
    if http_session:
        await http_session.close()
        await asyncio.sleep(0.25)
    
    print("--- BATCH SERVER SHUTDOWN ---")

app = FastAPI(lifespan=lifespan)

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
    """Find Aadhaar, PAN, and Selfie images in a user folder."""
    images = {
        "aadhar_front": None,
        "aadhar_back": None,
        "pancard": None,
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
        elif "pancard" in fname or "pan_card" in fname or fname.startswith("pan"):
            images["pancard"] = str(file_path)
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

async def verify_single_user(user_id: str, images: dict, expected_dob: Optional[str] = None, expected_gender: Optional[str] = None) -> dict:
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
        "gender_match": False,
        "extracted_data": {},
        "input_data": { "dob": expected_dob, "gender": expected_gender },
        "rejection_reasons": [], "error_log": ""
    }
    
    if not qwen_agent:
        user_record.update({"status": "ERROR", "final_decision": "SYSTEM_ERROR", "status_code": -1, "reason": "Qwen agent not initialized"})
        return user_record
    
    loop = asyncio.get_event_loop()
    
    # Step 1: Extract Aadhaar data using Qwen (Handles paths or BytesIO internally now)
    if images.get("aadhar_front"):
        try:
            print(f"[{user_id}] Extracting Aadhaar with Qwen...")
            aadhaar_data = await loop.run_in_executor(
                None,
                qwen_agent.extract_aadhaar_data,
                images["aadhar_front"],
                images.get("aadhar_back") # Pass back image if exists
            )
            
            if aadhaar_data and "error" not in aadhaar_data:
                # Check for Masked Aadhaar
                is_masked = aadhaar_data.get("is_masked", False)
                
                # Double check for masking in the number string itself
                aadhar_num = aadhaar_data.get("aadharnumber", "")
                if "X" in aadhar_num.upper() or "*" in aadhar_num:
                    is_masked = True

                if is_masked:
                     user_record["status"] = "REJECTED"
                     user_record["final_decision"] = "REJECTED"
                     user_record["status_code"] = 1
                     user_record["rejection_reasons"].append("aadhaar_masked")
                     user_record["aadhaar_found"] = True # It was found, just rejected
                     user_record["error_log"] += "Masked Aadhaar detected; "
                     print(f"[{user_id}] ❌ REJECTED: Masked Aadhaar detected")
                else:
                    user_record["aadhaar_found"] = True
                    user_record["aadhaar_number"] = aadhar_num
                    user_record["aadhaar_name"] = aadhaar_data.get("name", "")
                    user_record["aadhaar_dob"] = aadhaar_data.get("dob", "")
                    user_record["aadhaar_gender"] = aadhaar_data.get("gender", "")
                    user_record["aadhaar_address"] = aadhaar_data.get("address", "")
                    user_record["aadhaar_pincode"] = aadhaar_data.get("pincode", "")
                    print(f"[{user_id}] ✅ Aadhaar extracted - Gender: {user_record['aadhaar_gender']}")
            else:
                error_msg = aadhaar_data.get("error", "Unknown error") if aadhaar_data else "No response"
                user_record["error_log"] += f"Aadhaar extraction failed: {error_msg}; "
                user_record["rejection_reasons"].append("aadhaar_extraction_failed")
        except Exception as e:
            user_record["error_log"] += f"Aadhaar exception: {str(e)}; "
            user_record["rejection_reasons"].append("aadhaar_processing_error")
            print(f"[{user_id}] ❌ Aadhaar exception: {e}")
    
    # Step 2: Extract PAN data using Qwen (if available)
    if images.get("pancard"):
        try:
            print(f"[{user_id}] Extracting PAN with Qwen...")
            pan_data = await loop.run_in_executor(
                None,
                qwen_agent.extract_pancard_data,
                images["pancard"]
            )
            
            if pan_data and "error" not in pan_data and pan_data.get("pan_number"):
                user_record["pan_found"] = True
                user_record["pan_number"] = pan_data.get("pan_number", "")
                user_record["pan_name"] = pan_data.get("name", "")
                user_record["pan_father_name"] = pan_data.get("father_name", "")
                user_record["pan_dob"] = pan_data.get("dob", "")
                print(f"[{user_id}] ✅ PAN extracted")
            else:
                error_msg = pan_data.get("error", "Unknown error") if pan_data else "No response"
                user_record["error_log"] += f"PAN extraction failed: {error_msg}; "
        except Exception as e:
            user_record["error_log"] += f"PAN exception: {str(e)}; "

    
    # Step 3: Qwen-based Selfie Gender Detection (before scoring)
    face_gender_match = None
    if images.get("selfie") and qwen_agent and user_record["aadhaar_gender"]:
        try:
            print(f"[{user_id}] Detecting gender from selfie using Qwen...")
            
            selfie_input = images["selfie"]
            gender_prompt = """Analyze this selfie image and determine the person's gender.
            
Return ONLY in this exact JSON format:
{
  "gender": "Male or Female",
  "confidence": "High or Medium or Low"
}

Do not include any explanation, only return the JSON."""
            
            selfie_b64 = qwen_agent._image_to_base64(selfie_input)
            
            if selfie_b64:
                response_text = qwen_agent._call_ollama(gender_prompt, [selfie_b64])
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    import json
                    gender_data = json.loads(json_match.group())
                    selfie_gender = gender_data.get("gender", "").strip()
                    confidence = gender_data.get("confidence", "Low")
                    
                    print(f"[{user_id}] 📷 Selfie gender: {selfie_gender} (confidence: {confidence})")
                    
                    aadhaar_gender = user_record["aadhaar_gender"].lower()
                    selfie_gender_normalized = selfie_gender.lower()
                    
                    if aadhaar_gender and selfie_gender_normalized in ['male', 'female']:
                        if aadhaar_gender == selfie_gender_normalized:
                            face_gender_match = True
                            print(f"[{user_id}] ✅ Selfie gender matches Aadhaar")
                        else:
                            face_gender_match = False
                            print(f"[{user_id}] ⚠️ Selfie gender mismatch")
                
        except Exception as e:
            print(f"[{user_id}] ⚠️ Selfie gender detection failed: {e}")
            user_record["error_log"] += f"Selfie gender detection failed: {str(e)}; "
    
    # Step 3b: Qwen Face Similarity
    face_similarity_score = 0
    if images.get("selfie") and images.get("aadhar_front") and qwen_agent:
        try:
            print(f"[{user_id}] Checking face similarity...")
            prompt = """Compare these images. Same person? Return JSON: {"similarity_percentage": 0-100}"""
            s_b64 = qwen_agent._image_to_base64(images["selfie"])
            a_b64 = qwen_agent._image_to_base64(images["aadhar_front"])
            if s_b64 and a_b64:
                resp = qwen_agent._call_ollama(prompt, [s_b64, a_b64])
                import re, json
                m = re.search(r'\{[^}]+\}', resp, re.DOTALL)
                if m:
                    face_similarity_score = json.loads(m.group()).get("similarity_percentage", 0)
                    print(f"[{user_id}] Similarity: {face_similarity_score}%")
        except Exception as e:
            print(f"[{user_id}] Similarity check failed: {e}")
    
    # Step 3: Calculate Score using VerificationScorer
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
            
            # Face data (not implemented yet, set to 0)
            face_data = {"score": 0}
            
            # Calculate score using VerificationScorer
            scoring_result = verification_scorer.calculate_score(
                face_data=face_data,
                entity_data=entity_data,
                expected_gender=expected_gender,
                expected_dob=expected_dob,
                qwen_face_result=None,
                face_gender_match=face_gender_match  # Pass selfie gender match result
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
            
            # Update gender_match based on breakdown
            if "gender_score" in breakdown:
                user_record["gender_match"] = breakdown["gender_score"] > 0
            
            print(f"[{user_id}] → {final_status} (Score: {final_score})")
            if rejection_reasons:
                print(f"[{user_id}] Rejection Reasons: {', '.join(rejection_reasons)}")
            
            # Override: Gender mismatch or low similarity → REVIEW
            if final_status != "REJECTED":
                if face_gender_match == False:
                    user_record["status"] = "REVIEW"
                    user_record["final_decision"] = "REVIEW"
                    user_record["status_code"] = 0
                    user_record["rejection_reasons"].append("gender_mismatch")
                    print(f"[{user_id}] → REVIEW (Gender mismatch)")
                elif face_gender_match == True and 0 < face_similarity_score < 60:
                    user_record["status"] = "REVIEW"
                    user_record["final_decision"] = "REVIEW"
                    user_record["status_code"] = 0
                    user_record["rejection_reasons"].append(f"low_similarity_{face_similarity_score}%")
                    print(f"[{user_id}] → REVIEW (Low similarity)")
                
        except Exception as e:
            print(f"[{user_id}] ❌ Scoring exception: {e}")
            user_record["error_log"] += f"Scoring exception: {str(e)}; "
            user_record.update({"status": "REVIEW", "final_decision": "REVIEW", "status_code": 0, "score": 0})
            user_record["rejection_reasons"].append("scoring_error")
    else:
        # Fallback if scorer not initialized
        print(f"[{user_id}] ⚠️ VerificationScorer not initialized, using fallback")
        user_record.update({"status": "REVIEW", "final_decision": "REVIEW", "status_code": 0, "score": 0})
        user_record["rejection_reasons"].append("scorer_not_initialized")
    
    user_record["extracted_data"] = {
        "aadhaar": user_record["aadhaar_number"],
        "name": user_record["aadhaar_name"],
        "dob": user_record["aadhaar_dob"],
        "address": user_record["aadhaar_address"],
        "gender": user_record["aadhaar_gender"]
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
        
        # Check if mandatory files exist
        if not (selfie_img and front_img): 
            return {
                "user_id": user_id, "status": "FAILED", "final_decision": "REJECTED", 
                "status_code": 1, "score": 0, "reason": "File Retrieval Failed",
                "rejection_reasons": ["file_download_failed"]
            }
        
        # Construct images dict with in-memory BytesIO objects
        images = {
            "selfie": selfie_img,
            "aadhar_front": front_img,
            "aadhar_back": back_img
        }
        
        # Verify user passing memory objects
        result = await verify_single_user(user_id, images, req.dob, req.gender)
        
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
        "components": {
            "qwen_agent": qwen_agent is not None,
            "gender_pipeline": gender_pipeline is not None
        },
        "batch_processing": batch_processing
    }

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
        print(f"❌ Batch processing error: {e}")
        batch_progress["status"] = "error"
        batch_progress["error"] = str(e)
    finally:
        batch_processing = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101, workers=1)