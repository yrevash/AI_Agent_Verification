import os
import sys
import shutil
import asyncio
import aiohttp
import aiofiles
import cloudscraper
import gc
import weakref
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Union
from contextlib import asynccontextmanager

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- IMPORT AGENTS ---
try:
    from app.face_sim import FaceAgent
    # DocAgent is no longer needed because EntityAgent now handles card detection & extraction internally
    # from app.fb_detect import DocAgent 
    from app.entity import EntityAgent
    from app.gender_pipeline import GenderPipeline
    from scoring import VerificationScorer
    # --- ADDED: Import Redis Cache ---
    from redis_cache import get_cache 
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"Current directory: {current_dir}")
    print(f"sys.path: {sys.path}")
    raise

# --- Global State ---
face_agent = None
entity_agent = None
gender_pipeline = None
scorer = None
http_session = None
redis_cache = None

TEMP_DIR = Path("temp")

# Track active sessions with weak references to prevent memory leaks
active_sessions = weakref.WeakSet()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global face_agent, entity_agent, gender_pipeline, scorer, http_session, redis_cache
    
    # Startup
    TEMP_DIR.mkdir(exist_ok=True)
    print("--- STARTING SYSTEM INITIALIZATION ---")
    
    # Create persistent HTTP session with stricter connection limits
    connector = aiohttp.TCPConnector(
        limit=50,
        limit_per_host=10,
        ttl_dns_cache=300,
        force_close=True,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    active_sessions.add(http_session)
    
    # Initialize Agents
    face_agent = FaceAgent()
    
    # Initialize EntityAgent with both models (Detection + Extraction)
    # This matches the new flow where EntityAgent does everything including Qwen fallback
    entity_agent = EntityAgent(
        model1_path="models/best4.pt", 
        model2_path="models/best.pt",
        enable_qwen_fallback=True
    )
    
    gender_pipeline = GenderPipeline()
    scorer = VerificationScorer()
    
    # --- ADDED: Initialize Redis ---
    try:
        redis_cache = get_cache()
        if redis_cache.enabled:
            print(f"✅ Redis Cache Initialized via lifespan")
    except Exception as e:
        print(f"⚠️ Redis Init Warning: {e}")

    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    print("--- SYSTEM READY ---")
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    if http_session:
        await http_session.close()
        await asyncio.sleep(0.25)  # Give time for connections to close
    
    if redis_cache and redis_cache.enabled:
        redis_cache.close()

    # Final cleanup of temp directory
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        except Exception as e:
            print(f"Final cleanup error: {e}")
    
    print("--- SYSTEM SHUTDOWN ---")

app = FastAPI(lifespan=lifespan)

async def periodic_cleanup():
    """Background task to clean up old temp directories every 2 minutes."""
    import time
    while True:
        try:
            await asyncio.sleep(120)  # Run every 2 minutes
            
            if not TEMP_DIR.exists():
                continue
            
            current_time = time.time()
            cleaned_count = 0
            
            # Delete directories older than 5 minutes
            for user_dir in TEMP_DIR.iterdir():
                if not user_dir.is_dir():
                    continue
                
                dir_age = current_time - user_dir.stat().st_mtime
                if dir_age > 300:  # 5 minutes
                    try:
                        force_cleanup_directory(user_dir)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Periodic cleanup failed for {user_dir}: {e}")
            
            if cleaned_count > 0:
                print(f"Periodic cleanup: Removed {cleaned_count} old temp directories")
                gc.collect()  # Force garbage collection after cleanup
                
        except Exception as e:
            print(f"Error in periodic cleanup: {e}")

def force_cleanup_directory(directory: Path, max_retries: int = 3):
    """Aggressively cleanup a directory with retries."""
    for attempt in range(max_retries):
        try:
            # Force garbage collection to close file handles
            gc.collect()
            
            # First try: normal removal
            if attempt == 0:
                shutil.rmtree(directory, ignore_errors=False)
                return
            
            # Second try: ignore errors
            if attempt == 1:
                shutil.rmtree(directory, ignore_errors=True)
                if not directory.exists():
                    return
            
            # Final try: manual file-by-file deletion
            if directory.exists():
                for item in directory.rglob('*'):
                    try:
                        if item.is_file():
                            item.unlink(missing_ok=True)
                        elif item.is_dir():
                            try:
                                item.rmdir()
                            except:
                                pass
                    except Exception:
                        pass
                try:
                    directory.rmdir()
                except:
                    pass
            
            if not directory.exists():
                return
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to cleanup {directory} after {max_retries} attempts: {e}")
            else:
                import time
                time.sleep(0.1 * (attempt + 1))  # Progressive backoff

# --- INPUT MODEL ---
class VerifyRequest(BaseModel):
    user_id: Union[int, str]
    dob: Optional[str] = None
    passport_first: str
    passport_old: str
    selfie_photo: str
    gender: Optional[str] = None

async def fetch_file(session: aiohttp.ClientSession, source: str, destination: Path) -> bool:
    """
    Hybrid Fetcher with proper resource management.
    """
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
            
            print(f"Downloaded: {source} ({len(content)} bytes)")
            return True
            
        else:
            source_path = Path(source)
            if source_path.exists():
                shutil.copy2(source_path, destination)
                return True
            else:
                print(f"Local file not found: {source}")
    except Exception as e:
        print(f"Error fetching {source}: {e}")
    return False

async def setup_files(user_id: str, req: VerifyRequest):
    """Sets up temp environment with aggressive cleanup."""
    task_dir = TEMP_DIR / str(user_id)
    
    if task_dir.exists():
        force_cleanup_directory(task_dir)
    
    task_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "selfie": task_dir / f"{user_id}_selfie.jpg",
        "front": task_dir / f"{user_id}_aadhar_front.jpg",
        "back": task_dir / f"{user_id}_aadhar_back.jpg"
    }

    global http_session
    results = await asyncio.gather(
        fetch_file(http_session, req.selfie_photo, paths["selfie"]),
        fetch_file(http_session, req.passport_first, paths["front"]),
        fetch_file(http_session, req.passport_old, paths["back"]),
        return_exceptions=True
    )

    if all(r is True for r in results):
        return {k: str(v) for k, v in paths.items()}, task_dir
    else:
        force_cleanup_directory(task_dir)
        return None, None

@app.post("/verification/verify")
async def verify_user(req: VerifyRequest):
    """Full verification endpoint with detailed breakdown."""
    user_id = str(req.user_id)
    print(f"[{user_id}] New Verification Request")

    # --- ADDED: Check Redis Cache First ---
    if redis_cache and redis_cache.enabled:
        cached_data = redis_cache.get_verification_result(user_id)
        if cached_data:
            print(f"[{user_id}] ✅ Returning Cached Result")
            # Return the stored 'verification_result' portion
            return cached_data.get('verification_result')

    files, task_dir = await setup_files(user_id, req)
    if not files:
        return {
            "user_id": user_id,
            "status": "FAILED", 
            "reason": "File Retrieval Failed"
        }

    try:
        loop = asyncio.get_event_loop()
        
        # 1. Face Comparison
        face_task = loop.run_in_executor(None, face_agent.compare, files['selfie'], files['front'])

        # 2. Entity Pipeline (Detection + Extraction)
        # We use EntityAgent.process_images which now handles everything including Qwen fallback
        # Arguments: image_paths_list, user_id, task_id, threshold
        entity_task_args = [files['front'], files['back']]
        
        async def run_entity_pipeline():
            entity_res = await loop.run_in_executor(
                None, 
                entity_agent.process_images, 
                entity_task_args, 
                user_id, 
                "main_process", 
                0.15 # Confidence threshold
            )
            return entity_res

        # 3. Gender Fallback
        gender_task = loop.run_in_executor(None, gender_pipeline.detect_gender, files['front'])

        # Execute all tasks in parallel
        face_score, entity_res, gender_res = await asyncio.gather(face_task, run_entity_pipeline(), gender_task)

        # --- VALIDATE ENTITY RESULT (Front/Back Detection Check) ---
        if not entity_res.get('success'):
            # This handles cases where no cards were found
            error_msg = entity_res.get('error', 'Document Verification Failed')
            if error_msg == 'no_aadhar_detected':
                reason = "Aadhaar Card Not Detected (Front/Back Missing)"
            else:
                reason = f"Processing Error: {error_msg}"
                
            response = {
                "user_id": user_id,
                "status": "REJECTED",
                "score": 0,
                "reason": reason
            }
            return response

        # Retrieve extracted data
        merged_data = entity_res.get('data', {})
        
        # --- Gender Logic (Merging OCR + Pipeline) ---
        ocr_gender = merged_data.get('gender', 'Other')
        
        if ocr_gender in ['Other', 'Not Detected', None] and gender_res.face_detected:
            detected_gender = gender_res.gender.capitalize()
            if detected_gender in ['Male', 'Female']:
                print(f"[{user_id}] OCR Gender '{ocr_gender}' -> Fallback: {detected_gender}")
                merged_data['gender'] = detected_gender
                merged_data['gender_source'] = 'pipeline'
            else:
                merged_data['gender_source'] = 'ocr_failed'
        else:
            merged_data['gender_source'] = 'ocr'

        # --- Scoring ---
        expected_gender = req.gender.lower() if req.gender else None
        expected_dob = req.dob if req.dob else None
        
        final_result = scorer.calculate_score(
            face_data=face_score,
            entity_data=merged_data,
            expected_gender=expected_gender,
            expected_dob=expected_dob
        )

        status_code_map = {"APPROVED": 2, "REJECTED": 1, "REVIEW": 0}
        status_code = status_code_map.get(final_result['status'], 0)
        
        final_response = {
            "user_id": user_id,
            "final_decision": final_result['status'],
            "status_code": status_code,
            "score": final_result['total_score'],
            "breakdown": final_result['breakdown'],
            "extracted_data": {
                "aadhaar": merged_data.get('aadharnumber'),
                "dob": merged_data.get('dob'),
                "gender": merged_data.get('gender')
            },
            "input_data": {
                "dob": req.dob,
                "gender": req.gender
            },
            "rejection_reasons": final_result['rejection_reasons']
        }

        # --- ADDED: Store in Redis ---
        if redis_cache and redis_cache.enabled:
            # We store the final_response directly
            redis_cache.store_verification_result(user_id, final_response)
            print(f"[{user_id}] 💾 Result Cached")

        return final_response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "user_id": user_id,
            "status": "ERROR", 
            "message": str(e)
        }

    finally:
        # Immediate cleanup
        if task_dir and task_dir.exists():
            force_cleanup_directory(task_dir)
            print(f"[{user_id}] Cleanup Complete")
        
        # Force garbage collection
        gc.collect()

@app.post("/verification/verify/agent/")
async def verify_user_production(req: VerifyRequest):
    """Production endpoint - reuse logic from verify_user."""
    # Since the logic is identical and we want consistent caching, we just call verify_user
    return await verify_user(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101, workers=1) 
