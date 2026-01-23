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
from pathlib import Path
from typing import Optional, Union, List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
from datetime import datetime

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- IMPORTS ---
try:
    from app.gender_pipeline import GenderPipeline
except ImportError as e:
    print(f"ImportError: {e}")
    raise

import requests
import base64
from PIL import Image
import io

# --- Ollama Qwen Agent ---
class OllamaQwenAgent:
    """Ollama-based Qwen agent for document extraction using qwen3-vl:8b-instruct"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "qwen3-vl:8b-instruct"):
        self.ollama_url = ollama_url
        self.model = model
        print(f"Initialized Ollama Qwen Agent with model: {model}")
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
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
    
    def extract_aadhaar_data(self, front_image_path: str, back_image_path: str) -> dict:
        """Extract Aadhaar data from front and back images"""
        try:
            # Convert images to base64
            front_b64 = self._image_to_base64(front_image_path)
            back_b64 = self._image_to_base64(back_image_path)
            
            prompt = """Analyze these Aadhaar card images (front and back) and extract the following information:

1. Aadhaar Number (12 digits)
2. Name
3. Date of Birth (DOB)
4. Gender/Sex
5. Address
6. Pincode

Return ONLY in this exact JSON format:
{
  "aadharnumber": "123456789012",
  "name": "Full Name",
  "dob": "DD/MM/YYYY",
  "gender": "Male or Female",
  "address": "Full Address",
  "pincode": "123456"
}

If any field is not clearly visible, use empty string "". Do not include any explanation, only return the JSON."""

            response_text = self._call_ollama(prompt, [front_b64, back_b64])
            
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
    
    def extract_pancard_data(self, image_path: str) -> dict:
        """Extract PAN card data from image"""
        try:
            # Convert image to base64
            img_b64 = self._image_to_base64(image_path)
            
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
    global qwen_agent, gender_pipeline, http_session
    
    # Startup
    TEMP_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print("    BATCH VERIFICATION SERVER - QWEN POWERED")
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
        gender_pipeline = GenderPipeline()
        print("✅ GenderPipeline initialized")
    except Exception as e:
        print(f"⚠️ GenderPipeline initialization failed: {e}")
        gender_pipeline = None
    
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
    """Save batch progress to file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed to save progress: {e}")

def load_progress() -> Optional[dict]:
    """Load batch progress from file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load progress: {e}")
    return None

def save_results(results: list):
    """Save batch results to file."""
    try:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Failed to save results: {e}")

def load_results() -> list:
    """Load batch results from file."""
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

async def fetch_file(session: aiohttp.ClientSession, source: str, destination: Path) -> bool:
    """Download file from URL or copy from local path."""
    try:
        if str(source).startswith(('http://', 'https://')):
            loop = asyncio.get_event_loop()
            
            def download_with_cloudscraper(url: str) -> bytes:
                scraper = cloudscraper.create_scraper(
                    browser={
                        'browser': 'chrome',
                        'platform': 'darwin',
                        'mobile': False
                    }
                )
                try:
                    response = scraper.get(url, timeout=30)
                    response.raise_for_status()
                    return response.content
                finally:
                    if hasattr(scraper, 'close'):
                        scraper.close()
            
            content = await loop.run_in_executor(None, download_with_cloudscraper, str(source))
            
            async with aiofiles.open(destination, 'wb') as f:
                await f.write(content)
            
            return True
        else:
            source_path = Path(source)
            if source_path.exists():
                import shutil
                shutil.copy2(source_path, destination)
                return True
    except Exception as e:
        print(f"Error fetching {source}: {e}")
    return False

async def verify_single_user(user_id: str, images: dict, expected_dob: Optional[str] = None, expected_gender: Optional[str] = None) -> dict:
    """
    Verify a single user using Qwen for extraction and gender matching only.
    
    Process:
    1. Extract Aadhaar data using Qwen (get gender from document)
    2. Extract PAN data using Qwen (if available)
    3. Detect gender from selfie using GenderPipeline
    4. Compare genders - if mismatch, send for REVIEW
    """
    user_record = {
        "user_id": user_id,
        "status": "PROCESSING",
        "final_decision": "PENDING",
        "status_code": 0,
        "score": 0,
        # Aadhaar Fields
        "aadhaar_found": False,
        "aadhaar_number": "",
        "aadhaar_name": "",
        "aadhaar_dob": "",
        "aadhaar_gender": "",
        "aadhaar_address": "",
        "aadhaar_pincode": "",
        # PAN Fields
        "pan_found": False,
        "pan_number": "",
        "pan_name": "",
        "pan_father_name": "",
        "pan_dob": "",
        # Gender Verification
        "selfie_gender": "",
        "gender_match": False,
        "gender_confidence": 0.0,
        # Data returned
        "extracted_data": {},
        "input_data": {
            "dob": expected_dob,
            "gender": expected_gender
        },
        "rejection_reasons": [],
        "breakdown": {},
        "error_log": ""  # Initialize error_log field
    }
    
    if not qwen_agent:
        user_record["status"] = "ERROR"
        user_record["final_decision"] = "SYSTEM_ERROR"
        user_record["status_code"] = -1
        user_record["score"] = 0
        user_record["reason"] = "Qwen agent not initialized"
        user_record["rejection_reasons"] = ["system_initialization_failed"]
        return user_record
    
    loop = asyncio.get_event_loop()
    
    # Step 1: Extract Aadhaar data using Qwen
    if images.get("aadhar_front") and images.get("aadhar_back"):
        try:
            print(f"[{user_id}] Extracting Aadhaar with Qwen...")
            aadhaar_data = await loop.run_in_executor(
                None,
                qwen_agent.extract_aadhaar_data,
                images["aadhar_front"],
                images["aadhar_back"]
            )
            
            if aadhaar_data and "error" not in aadhaar_data:
                user_record["aadhaar_found"] = True
                user_record["aadhaar_number"] = aadhaar_data.get("aadharnumber", "")
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
                print(f"[{user_id}] ⚠️ Aadhaar extraction failed: {error_msg}")
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
                print(f"[{user_id}] ⚠️ PAN extraction failed: {error_msg}")
        except Exception as e:
            user_record["error_log"] += f"PAN exception: {str(e)}; "
            print(f"[{user_id}] ❌ PAN exception: {e}")
    
    # Step 3: Gender Verification (ONLY verification logic)
    if images.get("selfie") and gender_pipeline:
        try:
            print(f"[{user_id}] Detecting gender from selfie...")
            gender_result = await loop.run_in_executor(
                None,
                gender_pipeline.detect_gender,
                images["selfie"]
            )
            
            if hasattr(gender_result, 'face_detected') and gender_result.face_detected:
                detected_gender = gender_result.gender.capitalize() if hasattr(gender_result, 'gender') else 'Unknown'
                confidence = gender_result.confidence if hasattr(gender_result, 'confidence') else 0.0
                
                user_record["selfie_gender"] = detected_gender
                user_record["gender_confidence"] = confidence
                
                # Compare with Aadhaar gender
                aadhaar_gender = user_record["aadhaar_gender"].lower()
                selfie_gender = detected_gender.lower()
                
                if aadhaar_gender and selfie_gender in ['male', 'female']:
                    if aadhaar_gender == selfie_gender:
                        user_record["gender_match"] = True
                        print(f"[{user_id}] ✅ Gender MATCH: {detected_gender}")
                    else:
                        user_record["gender_match"] = False
                        user_record["rejection_reasons"].append(f"gender_mismatch_aadhaar_{aadhaar_gender}_selfie_{selfie_gender}")
                        print(f"[{user_id}] ⚠️ Gender MISMATCH: Aadhaar={aadhaar_gender}, Selfie={selfie_gender}")
                else:
                    user_record["error_log"] += "Gender comparison failed (missing data); "
                    print(f"[{user_id}] ⚠️ Cannot compare genders (missing data)")
            else:
                user_record["error_log"] += "No face detected in selfie; "
                user_record["rejection_reasons"].append("no_face_detected_in_selfie")
                print(f"[{user_id}] ⚠️ No face detected in selfie")
        except Exception as e:
            user_record["error_log"] += f"Gender detection exception: {str(e)}; "
            print(f"[{user_id}] ❌ Gender detection exception: {e}")
    
    # Step 4: Final Decision
    if not user_record["aadhaar_found"]:
        user_record["status"] = "REJECTED"
        user_record["final_decision"] = "REJECTED"
        user_record["status_code"] = 1
        user_record["score"] = 0
        user_record["rejection_reasons"].append("aadhaar_not_found")
    elif not user_record["gender_match"] and user_record["selfie_gender"]:
        # Gender mismatch - send for REVIEW
        user_record["status"] = "REVIEW"
        user_record["final_decision"] = "REVIEW"
        user_record["status_code"] = 0
        user_record["score"] = 50
        print(f"[{user_id}] → Sending for REVIEW (Gender Mismatch)")
    elif user_record["aadhaar_found"] and user_record["gender_match"]:
        # Everything good - APPROVED
        user_record["status"] = "APPROVED"
        user_record["final_decision"] = "APPROVED"
        user_record["status_code"] = 2
        user_record["score"] = 100
        print(f"[{user_id}] → APPROVED")
    else:
        # Incomplete verification - REVIEW
        user_record["status"] = "REVIEW"
        user_record["final_decision"] = "REVIEW"
        user_record["status_code"] = 0
        user_record["score"] = 50
        user_record["rejection_reasons"].append("incomplete_verification")
    
    # Populate extracted_data in same format as main.py
    user_record["extracted_data"] = {
        "aadhaar": user_record["aadhaar_number"],
        "dob": user_record["aadhaar_dob"],
        "gender": user_record["aadhaar_gender"]
    }
    
    # Add breakdown
    user_record["breakdown"] = {
        "gender_match": user_record["gender_match"],
        "aadhaar_extracted": user_record["aadhaar_found"],
        "pan_extracted": user_record["pan_found"]
    }
    
    return user_record

# --- API Endpoints ---
@app.post("/batch/verify")
async def start_batch_verification(req: BatchVerifyRequest, background_tasks: BackgroundTasks):
    """Start batch verification process in background."""
    global batch_processing, batch_progress
    
    if batch_processing:
        return {
            "status": "error",
            "message": "Batch processing already in progress",
            "progress": batch_progress
        }
    
    if not os.path.exists(req.dataset_root):
        return {
            "status": "error",
            "message": f"Dataset directory not found: {req.dataset_root}"
        }
    
    # Start batch processing in background
    background_tasks.add_task(process_batch, req.dataset_root, req.output_dir)
    
    return {
        "status": "started",
        "message": "Batch verification started",
        "dataset_root": req.dataset_root,
        "output_dir": req.output_dir
    }

@app.post("/verification/verify")
async def verify_user(req: VerifyRequest):
    """Verify a single user with provided image paths/URLs."""
    user_id = str(req.user_id)
    task_dir = TEMP_DIR / user_id
    
    # Cleanup and create directory
    if task_dir.exists():
        import shutil
        shutil.rmtree(task_dir, ignore_errors=True)
    task_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare file paths
        files = {
            "selfie": task_dir / f"{user_id}_selfie.jpg",
            "aadhar_front": task_dir / f"{user_id}_aadhar_front.jpg",
            "aadhar_back": task_dir / f"{user_id}_aadhar_back.jpg"
        }
        
        # Download/copy files
        download_tasks = [
            fetch_file(http_session, req.selfie_photo, files["selfie"]),
            fetch_file(http_session, req.passport_first, files["aadhar_front"]),
            fetch_file(http_session, req.passport_old, files["aadhar_back"])
        ]
        
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        if not all(r is True for r in results[:3]):  # First 3 are required
            return {
                "user_id": user_id,
                "status": "FAILED",
                "final_decision": "REJECTED",
                "status_code": 1,
                "score": 0,
                "reason": "File Retrieval Failed",
                "rejection_reasons": ["file_download_failed"]
            }
        
        # Convert to dict with string paths
        image_paths = {k: str(v) for k, v in files.items() if v.exists()}
        
        # Verify user
        result = await verify_single_user(user_id, image_paths, req.dob, req.gender)
        
        return result
        
    except Exception as e:
        print(f"Error in single verification: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "user_id": user_id
        }
    finally:
        # Cleanup
        if task_dir.exists():
            import shutil
            shutil.rmtree(task_dir, ignore_errors=True)
        gc.collect()

@app.post("/verification/verify/agent/")
async def verify_user_production(req: VerifyRequest):
    """Production endpoint - reuse logic from verify_user."""
    return await verify_user(req)

@app.get("/batch/progress")
async def get_batch_progress():
    """Get current batch processing progress."""
    return batch_progress

@app.get("/batch/results")
async def get_batch_results():
    """Get batch processing results."""
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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "qwen_agent": qwen_agent is not None,
            "gender_pipeline": gender_pipeline is not None
        },
        "batch_processing": batch_processing,
        "batch_progress": batch_progress
    }

# --- Background Processing ---
async def process_batch(dataset_root: str, output_dir: str):
    """Background task to process entire batch dataset."""
    global batch_processing, batch_results, batch_progress
    
    batch_processing = True
    batch_results = []
    
    try:
        # Find all batch folders
        batch_folders = sorted([
            f for f in glob.glob(os.path.join(dataset_root, "batch_*"))
            if os.path.isdir(f)
        ])
        
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
                    # Find images
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
                    
                    # Save progress and results every 5 users
                    if batch_progress["processed"] % 5 == 0:
                        save_progress(batch_progress)
                        save_results(batch_results)
                        print(f"Progress: {batch_progress['processed']}/{total_users} | Last: {user_id}")
                    
                    # Garbage collection every 20 users (more frequent)
                    if batch_progress["processed"] % 20 == 0:
                        gc.collect()
                        print(f"🗑️ Garbage collection performed")
                    
                except Exception as e:
                    print(f"Error processing user {user_id}: {e}")
                    traceback.print_exc()
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
        
        # Export to CSV/Excel/JSON
        if batch_results:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = output_path / f"batch_results_{timestamp}.csv"
            excel_file = output_path / f"batch_results_{timestamp}.xlsx"
            json_file = output_path / f"batch_results_complete_{timestamp}.json"
            
            # Save comprehensive JSON with all user data
            comprehensive_data = {
                "metadata": {
                    "total_users": len(batch_results),
                    "timestamp": timestamp,
                    "completion_time": datetime.now().isoformat(),
                    "summary": {
                        "approved": sum(1 for r in batch_results if r.get("final_decision") == "APPROVED"),
                        "rejected": sum(1 for r in batch_results if r.get("final_decision") == "REJECTED"),
                        "review": sum(1 for r in batch_results if r.get("final_decision") == "REVIEW"),
                        "error": sum(1 for r in batch_results if r.get("final_decision") == "SYSTEM_ERROR")
                    }
                },
                "users": batch_results
            }
            
            with open(json_file, 'w') as f:
                json.dump(comprehensive_data, f, indent=2)
            print(f"✅ Comprehensive JSON saved to {json_file}")
            
            # Save CSV
            df = pd.DataFrame(batch_results)
            df.to_csv(csv_file, index=False)
            print(f"✅ CSV saved to {csv_file}")
            
            # Save Excel
            try:
                df.to_excel(excel_file, index=False)
                print(f"✅ Excel saved to {excel_file}")
            except Exception as e:
                print(f"⚠️ Excel export failed: {e}")
        
        # Final garbage collection
        gc.collect()
        print(f"✅ Batch processing complete: {batch_progress['processed']} users processed")
        print(f"📊 Results: {comprehensive_data['metadata']['summary']}")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        traceback.print_exc()
        batch_progress["status"] = "error"
        batch_progress["error"] = str(e)
    finally:
        batch_processing = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101, workers=1)
