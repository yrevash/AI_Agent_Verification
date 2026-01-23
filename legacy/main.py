import os
import sys
import shutil
import asyncio
import aiohttp
import aiofiles
import cloudscraper
import gc
import weakref
import time
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Union
from contextlib import asynccontextmanager

# Set PyTorch memory management environment variable to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Production imports
try:
    from logging_config import setup_production_logging, get_verification_logger
    from monitoring import get_monitor
    PRODUCTION_MODE = True
except ImportError:
    # Fallback if production modules not available
    import logging
    logging.basicConfig(level=logging.INFO)
    PRODUCTION_MODE = False

# === GPU Detection ===
USE_GPU = False
GPU_AVAILABLE = False

# Check if GPU exists
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        GPU_AVAILABLE = True
        USE_GPU = True
        print("✅ NVIDIA GPU detected and enabled")
except:
    GPU_AVAILABLE = False
    USE_GPU = False
    print("⚠️ No NVIDIA GPU detected - using CPU mode")

# Configure environment for CPU/GPU mode
if not GPU_AVAILABLE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("CPU mode configured")

# Setup production logging if available
if PRODUCTION_MODE:
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_production_logging(log_level=log_level, log_dir="logs")
    verification_logger = get_verification_logger()
    monitor = get_monitor()
    print("✅ Production logging and monitoring enabled")
else:
    verification_logger = None
    monitor = None

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- IMPORT AGENTS ---
try:
    from app.face_sim import FaceAgent
    from app.entity import EntityAgent
    from app.gender_pipeline import GenderPipeline
    from app.qwen_fallback import QwenFallbackAgent
    from scoring import VerificationScorer
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
qwen_agent = None
scorer = None
http_session = None
redis_cache = None

TEMP_DIR = Path("temp")

# Track active sessions with weak references to prevent memory leaks
active_sessions = weakref.WeakSet()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global face_agent, entity_agent, gender_pipeline, qwen_agent, scorer, http_session, redis_cache
    
    # Startup
    TEMP_DIR.mkdir(exist_ok=True)
    print("--- STARTING SYSTEM INITIALIZATION ---")
    print(f"GPU Mode: {USE_GPU}")
    
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
    
    # === Error-resistant agent initialization ===
    try:
        print("Initializing FaceAgent...")
        face_agent = FaceAgent()
        print("✅ FaceAgent initialized")
    except Exception as e:
        print(f"⚠️ FaceAgent initialization failed: {e}")
        face_agent = None
    
    try:
        print("Initializing EntityAgent...")
        entity_agent = EntityAgent(
            model1_path="models/best4.pt", 
            model2_path="models/best.pt",
            enable_qwen_fallback=True
        )
        print("✅ EntityAgent initialized")
    except Exception as e:
        print(f"⚠️ EntityAgent initialization failed: {e}")
        entity_agent = None
    
    try:
        print("Initializing GenderPipeline...")
        gender_pipeline = GenderPipeline()
        print("✅ GenderPipeline initialized")
    except Exception as e:
        print(f"⚠️ GenderPipeline initialization failed: {e}")
        gender_pipeline = None
    
    try:
        print("Initializing QwenFallbackAgent...")
        qwen_agent = QwenFallbackAgent()
        print("✅ QwenFallbackAgent initialized")
    except Exception as e:
        print(f"⚠️ QwenFallbackAgent initialization failed: {e}")
        qwen_agent = None
    
    scorer = VerificationScorer()
    
    # Initialize Redis
    try:
        redis_cache = get_cache()
        if redis_cache.enabled:
            print(f"✅ Redis Cache Initialized")
    except Exception as e:
        print(f"⚠️ Redis Init Warning: {e}")
        redis_cache = None

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
        await asyncio.sleep(0.25)
    
    if redis_cache and redis_cache.enabled:
        redis_cache.close()

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
            await asyncio.sleep(120)
            
            if not TEMP_DIR.exists():
                continue
            
            current_time = time.time()
            cleaned_count = 0
            
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
                gc.collect()
                
        except Exception as e:
            print(f"Error in periodic cleanup: {e}")

def force_cleanup_directory(directory: Path, max_retries: int = 3):
    """Aggressively cleanup a directory with retries."""
    for attempt in range(max_retries):
        try:
            gc.collect()
            
            if attempt == 0:
                shutil.rmtree(directory, ignore_errors=False)
                return
            
            if attempt == 1:
                shutil.rmtree(directory, ignore_errors=True)
                if not directory.exists():
                    return
            
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
                time.sleep(0.1 * (attempt + 1))

class VerifyRequest(BaseModel):
    user_id: Union[int, str]
    dob: Optional[str] = None
    passport_first: str
    passport_old: str
    selfie_photo: str
    gender: Optional[str] = None

async def fetch_file(session: aiohttp.ClientSession, source: str, destination: Path) -> bool:
    """Hybrid Fetcher with proper resource management."""
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
    """Full verification endpoint with enhanced error handling."""
    user_id = str(req.user_id)
    start_time = time.time()
    
    # Log request if production mode
    if PRODUCTION_MODE and verification_logger:
        verification_logger.log_request(user_id, bool(req.dob), bool(req.gender))
    
    print(f"[{user_id}] New Verification Request")

    # Check agent availability
    if not all([face_agent, entity_agent, gender_pipeline, scorer]):
        return {
            "user_id": user_id,
            "status": "ERROR",
            "final_decision": "SYSTEM_ERROR",
            "status_code": -1,
            "score": 0,
            "reason": "System components not initialized properly",
            "rejection_reasons": ["system_initialization_failed"]
        }

    # Check Redis Cache First
    if redis_cache and redis_cache.enabled:
        try:
            cached_data = redis_cache.get_verification_result(user_id)
            if cached_data:
                print(f"[{user_id}] ✅ Returning Cached Result")
                if PRODUCTION_MODE and verification_logger:
                    verification_logger.log_cache_hit(user_id)
                if PRODUCTION_MODE and monitor:
                    monitor.record_cache_hit()
                return cached_data.get('verification_result', cached_data)
            else:
                if PRODUCTION_MODE and monitor:
                    monitor.record_cache_miss()
        except Exception as e:
            print(f"[{user_id}] Redis cache read failed: {e}")
            if PRODUCTION_MODE and monitor:
                monitor.record_cache_miss()

    files, task_dir = await setup_files(user_id, req)
    if not files:
        return {
            "user_id": user_id,
            "status": "FAILED",
            "final_decision": "REJECTED",
            "status_code": 1,
            "score": 0,
            "reason": "File Retrieval Failed",
            "rejection_reasons": ["file_download_failed"]
        }

    try:
        loop = asyncio.get_event_loop()
        
        # Face Comparison with error handling
        async def safe_face_compare():
            try:
                return await loop.run_in_executor(None, face_agent.compare, files['selfie'], files['front'])
            except Exception as e:
                print(f"[{user_id}] Face comparison failed: {e}")
                return {
                    'face_detected': False,
                    'similarity': 0.0,
                    'error': str(e)
                }

        # Entity Pipeline with error handling
        async def safe_entity_process():
            try:
                entity_task_args = [files['front'], files['back']]
                return await loop.run_in_executor(
                    None, 
                    entity_agent.process_images, 
                    entity_task_args, 
                    user_id, 
                    "main_process", 
                    0.15
                )
            except Exception as e:
                print(f"[{user_id}] Entity processing failed: {e}")
                return {
                    'success': False,
                    'error': f'entity_processing_failed: {str(e)}',
                    'data': {}
                }

        # Gender Detection with error handling
        async def safe_gender_detect():
            try:
                return await loop.run_in_executor(None, gender_pipeline.detect_gender, files['front'])
            except Exception as e:
                print(f"[{user_id}] Gender detection failed: {e}")
                return type('obj', (object,), {
                    'face_detected': False,
                    'gender': 'Unknown',
                    'confidence': 0.0
                })()

        # Execute all tasks in parallel
        face_score, entity_res, gender_res = await asyncio.gather(
            safe_face_compare(), 
            safe_entity_process(), 
            safe_gender_detect()
        )

        # Check if entity processing failed critically
        if not entity_res.get('success'):
            error_msg = entity_res.get('error', 'Document Verification Failed')
            
            if 'no_aadhar_detected' in error_msg.lower():
                reason = "Aadhaar Card Not Detected (Front/Back Missing)"
            elif 'security' in error_msg.lower():
                reason = "Security Check Failed - Suspicious Document Detected"
            elif 'processing_failed' in error_msg.lower():
                reason = "Document Processing Failed - Poor Image Quality"
            else:
                reason = f"Verification Error: {error_msg}"
                
            response = {
                "user_id": user_id,
                "status": "REJECTED",
                "final_decision": "REJECTED",
                "status_code": 1,
                "score": 0,
                "reason": reason,
                "rejection_reasons": [error_msg],
                "extracted_data": {},
                "input_data": {
                    "dob": req.dob,
                    "gender": req.gender
                }
            }
            
            if redis_cache and redis_cache.enabled:
                try:
                    redis_cache.store_verification_result(user_id, response)
                except Exception as e:
                    print(f"[{user_id}] Failed to cache result: {e}")
            
            return response

        # Retrieve extracted data
        merged_data = entity_res.get('data', {})
        
        # === QWEN FACE VERIFICATION FALLBACK ===
        # Trigger when face similarity is very low (<20%)
        face_sim_score = face_score.get("score", 0)
        qwen_face_result = None
        
        if face_sim_score < 20 and qwen_agent:
            print(f"[{user_id}] ⚠️ Face similarity very low ({face_sim_score:.2f}%) - Triggering Qwen Face Verification")
            try:
                qwen_face_result = await loop.run_in_executor(
                    None,
                    qwen_agent.verify_face_with_qwen,
                    files['selfie'],
                    files['front'],
                    req.gender
                )
                print(f"[{user_id}] 🤖 Qwen Face Verification: {qwen_face_result.get('decision')} - {qwen_face_result.get('reason')}")
            except Exception as e:
                print(f"[{user_id}] Qwen face verification failed: {e}")
                qwen_face_result = None
        
        # Gender Logic
        ocr_gender = merged_data.get('gender', 'Other')
        
        if ocr_gender in ['Other', 'Not Detected', None] and hasattr(gender_res, 'face_detected') and gender_res.face_detected:
            detected_gender = gender_res.gender.capitalize() if hasattr(gender_res, 'gender') else 'Unknown'
            if detected_gender in ['Male', 'Female']:
                print(f"[{user_id}] OCR Gender '{ocr_gender}' -> Fallback: {detected_gender}")
                merged_data['gender'] = detected_gender
                merged_data['gender_source'] = 'pipeline'
            else:
                merged_data['gender_source'] = 'ocr_failed'
        else:
            merged_data['gender_source'] = 'ocr'

        # Scoring
        expected_gender = req.gender.lower() if req.gender else None
        expected_dob = req.dob if req.dob else None
        
        final_result = scorer.calculate_score(
            face_data=face_score,
            entity_data=merged_data,
            expected_gender=expected_gender,
            expected_dob=expected_dob,
            qwen_face_result=qwen_face_result
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
            "rejection_reasons": final_result.get('rejection_reasons', [])
        }
        
        # Record metrics
        if PRODUCTION_MODE and monitor:
            processing_time = time.time() - start_time
            monitor.record_verification(final_result['status'], processing_time)
        
        # Log result
        if PRODUCTION_MODE and verification_logger:
            processing_time = time.time() - start_time
            verification_logger.log_result(
                user_id, 
                final_result['status'], 
                final_result['total_score'],
                final_result['breakdown'].get('face_score', 0),
                processing_time
            )

        # Store in Redis
        if redis_cache and redis_cache.enabled:
            try:
                redis_cache.store_verification_result(user_id, final_response)
                print(f"[{user_id}] 💾 Result Cached")
            except Exception as e:
                print(f"[{user_id}] Failed to cache result: {e}")

        return final_response

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[{user_id}] CRITICAL ERROR:")
        print(error_trace)
        
        # Record error
        if PRODUCTION_MODE and monitor:
            monitor.record_error("system_error")
        
        if PRODUCTION_MODE and verification_logger:
            verification_logger.log_error(user_id, "system_error", str(e))
        
        return {
            "user_id": user_id,
            "status": "ERROR",
            "final_decision": "SYSTEM_ERROR",
            "status_code": -1,
            "score": 0,
            "message": str(e),
            "reason": "Internal Processing Error",
            "rejection_reasons": ["system_error"],
            "extracted_data": {},
            "input_data": {
                "dob": req.dob,
                "gender": req.gender
            }
        }

    finally:
        if task_dir and task_dir.exists():
            force_cleanup_directory(task_dir)
            print(f"[{user_id}] Cleanup Complete")
        
        gc.collect()

@app.post("/verification/verify/agent/")
async def verify_user_production(req: VerifyRequest):
    """Production endpoint - reuse logic from verify_user."""
    return await verify_user(req)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify system status."""
    base_health = {
        "status": "healthy",
        "gpu_enabled": USE_GPU,
        "components": {
            "face_agent": face_agent is not None,
            "entity_agent": entity_agent is not None,
            "gender_pipeline": gender_pipeline is not None,
            "scorer": scorer is not None,
            "redis": redis_cache is not None and redis_cache.enabled if redis_cache else False
        }
    }
    
    # Add monitoring data if available
    if PRODUCTION_MODE and monitor:
        base_health["monitoring"] = monitor.get_health_status()
    
    return base_health


@app.get("/metrics")
async def get_metrics():
    """Get application metrics (production only)."""
    if not PRODUCTION_MODE or not monitor:
        return {"error": "Metrics not available (production mode disabled)"}
    
    return {
        "system": monitor.get_system_stats(),
        "application": monitor.get_application_metrics()
    }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101, workers=1)


# import os
# import sys
# import shutil
# import asyncio
# import aiohttp
# import aiofiles
# import cloudscraper
# import gc
# import weakref
# import time
# import traceback
# from fastapi import FastAPI
# from pydantic import BaseModel
# from pathlib import Path
# from typing import Optional, Union
# from contextlib import asynccontextmanager

# # Set PyTorch memory management environment variable to reduce fragmentation
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# # Production imports
# try:
#     from logging_config import setup_production_logging, get_verification_logger
#     from monitoring import get_monitor
#     PRODUCTION_MODE = True
# except ImportError:
#     # Fallback if production modules not available
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     PRODUCTION_MODE = False

# # === CRITICAL FIX: Check GPU BEFORE setting environment ===
# USE_GPU = True
# GPU_AVAILABLE = False

# # Check if GPU exists before doing anything
# try:
#     import subprocess
#     result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
#     if result.returncode == 0:
#         GPU_AVAILABLE = True
#         print("✅ NVIDIA GPU detected")
# except:
#     GPU_AVAILABLE = False
#     print("No NVIDIA GPU detected")

# # Only force CPU if explicitly needed or if no GPU
# if not GPU_AVAILABLE:
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''
#     print("Using CPU mode (no GPU available)")
# else:
#     # Let GPU be visible
#     print("GPU mode enabled - initializing with GPU support")

# # Now import TensorFlow AFTER setting the environm
# # Setup production logging if available
# if PRODUCTION_MODE:
#     log_level = os.getenv("LOG_LEVEL", "INFO")
#     setup_production_logging(log_level=log_level, log_dir="logs")
#     verification_logger = get_verification_logger()
#     monitor = get_monitor()
#     print("✅ Production logging and monitoring enabled")
# else:
#     verification_logger = None
#     monitor = None

# # Ensure the current directory is in the Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# # --- IMPORT AGENTS ---
# try:
#     from app.face_sim import FaceAgent
#     from app.entity import EntityAgent
#     from app.gender_pipeline import GenderPipeline
#     from app.qwen_fallback import QwenFallbackAgent
#     from scoring import VerificationScorer
#     from redis_cache import get_cache 
# except ImportError as e:
#     print(f"ImportError: {e}")
#     print(f"Current directory: {current_dir}")
#     print(f"sys.path: {sys.path}")
#     raise

# # --- Global State ---
# face_agent = None
# entity_agent = None
# gender_pipeline = None
# qwen_agent = None
# scorer = None
# http_session = None
# redis_cache = None

# TEMP_DIR = Path("temp")

# # Track active sessions with weak references to prevent memory leaks
# active_sessions = weakref.WeakSet()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Lifespan context manager for startup and shutdown."""
#     global face_agent, entity_agent, gender_pipeline, qwen_agent, scorer, http_session, redis_cache
    
#     # Startup
#     TEMP_DIR.mkdir(exist_ok=True)
#     print("--- STARTING SYSTEM INITIALIZATION ---")
#     print(f"GPU Mode: {USE_GPU}")
    
#     # Create persistent HTTP session with stricter connection limits
#     connector = aiohttp.TCPConnector(
#         limit=50,
#         limit_per_host=10,
#         ttl_dns_cache=300,
#         force_close=True,
#         enable_cleanup_closed=True
#     )
#     timeout = aiohttp.ClientTimeout(total=30, connect=10)
#     http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
#     active_sessions.add(http_session)
    
#     # === Error-resistant agent initialization ===
#     try:
#         print("Initializing FaceAgent...")
#         face_agent = FaceAgent()
#         print("✅ FaceAgent initialized")
#     except Exception as e:
#         print(f"⚠️ FaceAgent initialization failed: {e}")
#         face_agent = None
    
#     try:
#         print("Initializing EntityAgent...")
#         entity_agent = EntityAgent(
#             model1_path="models/best4.pt", 
#             model2_path="models/best.pt",
#             enable_qwen_fallback=True
#         )
#         print("✅ EntityAgent initialized")
#     except Exception as e:
#         print(f"⚠️ EntityAgent initialization failed: {e}")
#         entity_agent = None
    
#     try:
#         print("Initializing GenderPipeline...")
#         gender_pipeline = GenderPipeline()
#         print("✅ GenderPipeline initialized")
#     except Exception as e:
#         print(f"⚠️ GenderPipeline initialization failed: {e}")
#         gender_pipeline = None
    
#     try:
#         print("Initializing QwenFallbackAgent...")
#         qwen_agent = QwenFallbackAgent()
#         print("✅ QwenFallbackAgent initialized")
#     except Exception as e:
#         print(f"⚠️ QwenFallbackAgent initialization failed: {e}")
#         qwen_agent = None
    
#     scorer = VerificationScorer()
    
#     # Initialize Redis
#     try:
#         redis_cache = get_cache()
#         if redis_cache.enabled:
#             print(f"✅ Redis Cache Initialized")
#     except Exception as e:
#         print(f"⚠️ Redis Init Warning: {e}")
#         redis_cache = None

#     # Start background cleanup task
#     cleanup_task = asyncio.create_task(periodic_cleanup())
    
#     print("--- SYSTEM READY ---")
    
#     yield
    
#     # Shutdown
#     cleanup_task.cancel()
#     try:
#         await cleanup_task
#     except asyncio.CancelledError:
#         pass
    
#     if http_session:
#         await http_session.close()
#         await asyncio.sleep(0.25)
    
#     if redis_cache and redis_cache.enabled:
#         redis_cache.close()

#     if TEMP_DIR.exists():
#         try:
#             shutil.rmtree(TEMP_DIR, ignore_errors=True)
#         except Exception as e:
#             print(f"Final cleanup error: {e}")
    
#     print("--- SYSTEM SHUTDOWN ---")

# app = FastAPI(lifespan=lifespan)

# async def periodic_cleanup():
#     """Background task to clean up old temp directories every 2 minutes."""
#     import time
#     while True:
#         try:
#             await asyncio.sleep(120)
            
#             if not TEMP_DIR.exists():
#                 continue
            
#             current_time = time.time()
#             cleaned_count = 0
            
#             for user_dir in TEMP_DIR.iterdir():
#                 if not user_dir.is_dir():
#                     continue
                
#                 dir_age = current_time - user_dir.stat().st_mtime
#                 if dir_age > 300:  # 5 minutes
#                     try:
#                         force_cleanup_directory(user_dir)
#                         cleaned_count += 1
#                     except Exception as e:
#                         print(f"Periodic cleanup failed for {user_dir}: {e}")
            
#             if cleaned_count > 0:
#                 print(f"Periodic cleanup: Removed {cleaned_count} old temp directories")
#                 gc.collect()
                
#         except Exception as e:
#             print(f"Error in periodic cleanup: {e}")

# def force_cleanup_directory(directory: Path, max_retries: int = 3):
#     """Aggressively cleanup a directory with retries."""
#     for attempt in range(max_retries):
#         try:
#             gc.collect()
            
#             if attempt == 0:
#                 shutil.rmtree(directory, ignore_errors=False)
#                 return
            
#             if attempt == 1:
#                 shutil.rmtree(directory, ignore_errors=True)
#                 if not directory.exists():
#                     return
            
#             if directory.exists():
#                 for item in directory.rglob('*'):
#                     try:
#                         if item.is_file():
#                             item.unlink(missing_ok=True)
#                         elif item.is_dir():
#                             try:
#                                 item.rmdir()
#                             except:
#                                 pass
#                     except Exception:
#                         pass
#                 try:
#                     directory.rmdir()
#                 except:
#                     pass
            
#             if not directory.exists():
#                 return
                
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 print(f"Failed to cleanup {directory} after {max_retries} attempts: {e}")
#             else:
#                 import time
#                 time.sleep(0.1 * (attempt + 1))

# class VerifyRequest(BaseModel):
#     user_id: Union[int, str]
#     dob: Optional[str] = None
#     passport_first: str
#     passport_old: str
#     selfie_photo: str
#     gender: Optional[str] = None

# async def fetch_file(session: aiohttp.ClientSession, source: str, destination: Path) -> bool:
#     """Hybrid Fetcher with proper resource management."""
#     try:
#         if str(source).startswith(('http://', 'https://')):
#             loop = asyncio.get_event_loop()
            
#             def download_with_cloudscraper(url: str) -> bytes:
#                 scraper = cloudscraper.create_scraper(
#                     browser={
#                         'browser': 'chrome',
#                         'platform': 'darwin',
#                         'mobile': False
#                     }
#                 )
#                 try:
#                     response = scraper.get(url, timeout=30)
#                     response.raise_for_status()
#                     return response.content
#                 finally:
#                     if hasattr(scraper, 'close'):
#                         scraper.close()
            
#             content = await loop.run_in_executor(None, download_with_cloudscraper, str(source))
            
#             async with aiofiles.open(destination, 'wb') as f:
#                 await f.write(content)
            
#             print(f"Downloaded: {source} ({len(content)} bytes)")
#             return True
            
#         else:
#             source_path = Path(source)
#             if source_path.exists():
#                 shutil.copy2(source_path, destination)
#                 return True
#             else:
#                 print(f"Local file not found: {source}")
#     except Exception as e:
#         print(f"Error fetching {source}: {e}")
#     return False

# async def setup_files(user_id: str, req: VerifyRequest):
#     """Sets up temp environment with aggressive cleanup."""
#     task_dir = TEMP_DIR / str(user_id)
    
#     if task_dir.exists():
#         force_cleanup_directory(task_dir)
    
#     task_dir.mkdir(parents=True, exist_ok=True)

#     paths = {
#         "selfie": task_dir / f"{user_id}_selfie.jpg",
#         "front": task_dir / f"{user_id}_aadhar_front.jpg",
#         "back": task_dir / f"{user_id}_aadhar_back.jpg"
#     }

#     global http_session
#     results = await asyncio.gather(
#         fetch_file(http_session, req.selfie_photo, paths["selfie"]),
#         fetch_file(http_session, req.passport_first, paths["front"]),
#         fetch_file(http_session, req.passport_old, paths["back"]),
#         return_exceptions=True
#     )

#     if all(r is True for r in results):
#         return {k: str(v) for k, v in paths.items()}, task_dir
#     else:
#         force_cleanup_directory(task_dir)
#         return None, None

# @app.post("/verification/verify")
# async def verify_user(req: VerifyRequest):
#     """Full verification endpoint with enhanced error handling."""
#     user_id = str(req.user_id)
#     start_time = time.time()
    
#     # Log request if production mode
#     if PRODUCTION_MODE and verification_logger:
#         verification_logger.log_request(user_id, bool(req.dob), bool(req.gender))
    
#     print(f"[{user_id}] New Verification Request")

#     # Check agent availability
#     if not all([face_agent, entity_agent, gender_pipeline, scorer]):
#         return {
#             "user_id": user_id,
#             "status": "ERROR",
#             "final_decision": "SYSTEM_ERROR",
#             "status_code": -1,
#             "score": 0,
#             "reason": "System components not initialized properly",
#             "rejection_reasons": ["system_initialization_failed"]
#         }

#     # Check Redis Cache First
#     if redis_cache and redis_cache.enabled:
#         try:
#             cached_data = redis_cache.get_verification_result(user_id)
#             if cached_data:
#                 print(f"[{user_id}] ✅ Returning Cached Result")
#                 if PRODUCTION_MODE and verification_logger:
#                     verification_logger.log_cache_hit(user_id)
#                 if PRODUCTION_MODE and monitor:
#                     monitor.record_cache_hit()
#                 return cached_data.get('verification_result', cached_data)
#             else:
#                 if PRODUCTION_MODE and monitor:
#                     monitor.record_cache_miss()
#         except Exception as e:
#             print(f"[{user_id}] Redis cache read failed: {e}")
#             if PRODUCTION_MODE and monitor:
#                 monitor.record_cache_miss()

#     files, task_dir = await setup_files(user_id, req)
#     if not files:
#         return {
#             "user_id": user_id,
#             "status": "FAILED",
#             "final_decision": "REJECTED",
#             "status_code": 1,
#             "score": 0,
#             "reason": "File Retrieval Failed",
#             "rejection_reasons": ["file_download_failed"]
#         }

#     try:
#         loop = asyncio.get_event_loop()
        
#         # Face Comparison with error handling
#         async def safe_face_compare():
#             try:
#                 return await loop.run_in_executor(None, face_agent.compare, files['selfie'], files['front'])
#             except Exception as e:
#                 print(f"[{user_id}] Face comparison failed: {e}")
#                 return {
#                     'face_detected': False,
#                     'similarity': 0.0,
#                     'error': str(e)
#                 }

#         # Entity Pipeline with error handling
#         async def safe_entity_process():
#             try:
#                 entity_task_args = [files['front'], files['back']]
#                 return await loop.run_in_executor(
#                     None, 
#                     entity_agent.process_images, 
#                     entity_task_args, 
#                     user_id, 
#                     "main_process", 
#                     0.15
#                 )
#             except Exception as e:
#                 print(f"[{user_id}] Entity processing failed: {e}")
#                 return {
#                     'success': False,
#                     'error': f'entity_processing_failed: {str(e)}',
#                     'data': {}
#                 }

#         # Gender Detection with error handling
#         async def safe_gender_detect():
#             try:
#                 return await loop.run_in_executor(None, gender_pipeline.detect_gender, files['front'])
#             except Exception as e:
#                 print(f"[{user_id}] Gender detection failed: {e}")
#                 return type('obj', (object,), {
#                     'face_detected': False,
#                     'gender': 'Unknown',
#                     'confidence': 0.0
#                 })()

#         # Execute all tasks in parallel
#         face_score, entity_res, gender_res = await asyncio.gather(
#             safe_face_compare(), 
#             safe_entity_process(), 
#             safe_gender_detect()
#         )

#         # Check if entity processing failed critically
#         if not entity_res.get('success'):
#             error_msg = entity_res.get('error', 'Document Verification Failed')
            
#             if 'no_aadhar_detected' in error_msg.lower():
#                 reason = "Aadhaar Card Not Detected (Front/Back Missing)"
#             elif 'security' in error_msg.lower():
#                 reason = "Security Check Failed - Suspicious Document Detected"
#             elif 'processing_failed' in error_msg.lower():
#                 reason = "Document Processing Failed - Poor Image Quality"
#             else:
#                 reason = f"Verification Error: {error_msg}"
                
#             response = {
#                 "user_id": user_id,
#                 "status": "REJECTED",
#                 "final_decision": "REJECTED",
#                 "status_code": 1,
#                 "score": 0,
#                 "reason": reason,
#                 "rejection_reasons": [error_msg],
#                 "extracted_data": {},
#                 "input_data": {
#                     "dob": req.dob,
#                     "gender": req.gender
#                 }
#             }
            
#             if redis_cache and redis_cache.enabled:
#                 try:
#                     redis_cache.store_verification_result(user_id, response)
#                 except Exception as e:
#                     print(f"[{user_id}] Failed to cache result: {e}")
            
#             return response

#         # Retrieve extracted data
#         merged_data = entity_res.get('data', {})
        
#         # === QWEN FACE VERIFICATION FALLBACK ===
#         # Trigger when face similarity is very low (<20%)
#         face_sim_score = face_score.get("score", 0)
#         qwen_face_result = None
        
#         if face_sim_score < 20 and qwen_agent:
#             print(f"[{user_id}] ⚠️ Face similarity very low ({face_sim_score:.2f}%) - Triggering Qwen Face Verification")
#             try:
#                 qwen_face_result = await loop.run_in_executor(
#                     None,
#                     qwen_agent.verify_face_with_qwen,
#                     files['selfie'],
#                     files['front'],
#                     req.gender
#                 )
#                 print(f"[{user_id}] 🤖 Qwen Face Verification: {qwen_face_result.get('decision')} - {qwen_face_result.get('reason')}")
#             except Exception as e:
#                 print(f"[{user_id}] Qwen face verification failed: {e}")
#                 qwen_face_result = None
        
#         # Gender Logic
#         ocr_gender = merged_data.get('gender', 'Other')
        
#         if ocr_gender in ['Other', 'Not Detected', None] and hasattr(gender_res, 'face_detected') and gender_res.face_detected:
#             detected_gender = gender_res.gender.capitalize() if hasattr(gender_res, 'gender') else 'Unknown'
#             if detected_gender in ['Male', 'Female']:
#                 print(f"[{user_id}] OCR Gender '{ocr_gender}' -> Fallback: {detected_gender}")
#                 merged_data['gender'] = detected_gender
#                 merged_data['gender_source'] = 'pipeline'
#             else:
#                 merged_data['gender_source'] = 'ocr_failed'
#         else:
#             merged_data['gender_source'] = 'ocr'

#         # Scoring
#         expected_gender = req.gender.lower() if req.gender else None
#         expected_dob = req.dob if req.dob else None
        
#         final_result = scorer.calculate_score(
#             face_data=face_score,
#             entity_data=merged_data,
#             expected_gender=expected_gender,
#             expected_dob=expected_dob,
#             qwen_face_result=qwen_face_result
#         )

#         status_code_map = {"APPROVED": 2, "REJECTED": 1, "REVIEW": 0}
#         status_code = status_code_map.get(final_result['status'], 0)
        
#         final_response = {
#             "user_id": user_id,
#             "final_decision": final_result['status'],
#             "status_code": status_code,
#             "score": final_result['total_score'],
#             "breakdown": final_result['breakdown'],
#             "extracted_data": {
#                 "aadhaar": merged_data.get('aadharnumber'),
#                 "dob": merged_data.get('dob'),
#                 "gender": merged_data.get('gender')
#             },
#             "input_data": {
#                 "dob": req.dob,
#                 "gender": req.gender
#             },
#             "rejection_reasons": final_result.get('rejection_reasons', [])
#         }
        
#         # Record metrics
#         if PRODUCTION_MODE and monitor:
#             processing_time = time.time() - start_time
#             monitor.record_verification(final_result['status'], processing_time)
        
#         # Log result
#         if PRODUCTION_MODE and verification_logger:
#             processing_time = time.time() - start_time
#             verification_logger.log_result(
#                 user_id, 
#                 final_result['status'], 
#                 final_result['total_score'],
#                 final_result['breakdown'].get('face_score', 0),
#                 processing_time
#             )

#         # Store in Redis
#         if redis_cache and redis_cache.enabled:
#             try:
#                 redis_cache.store_verification_result(user_id, final_response)
#                 print(f"[{user_id}] 💾 Result Cached")
#             except Exception as e:
#                 print(f"[{user_id}] Failed to cache result: {e}")

#         return final_response

#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"[{user_id}] CRITICAL ERROR:")
#         print(error_trace)
        
#         # Record error
#         if PRODUCTION_MODE and monitor:
#             monitor.record_error("system_error")
        
#         if PRODUCTION_MODE and verification_logger:
#             verification_logger.log_error(user_id, "system_error", str(e))
        
#         return {
#             "user_id": user_id,
#             "status": "ERROR",
#             "final_decision": "SYSTEM_ERROR",
#             "status_code": -1,
#             "score": 0,
#             "message": str(e),
#             "reason": "Internal Processing Error",
#             "rejection_reasons": ["system_error"],
#             "extracted_data": {},
#             "input_data": {
#                 "dob": req.dob,
#                 "gender": req.gender
#             }
#         }

#     finally:
#         if task_dir and task_dir.exists():
#             force_cleanup_directory(task_dir)
#             print(f"[{user_id}] Cleanup Complete")
        
#         gc.collect()

# @app.post("/verification/verify/agent/")
# async def verify_user_production(req: VerifyRequest):
#     """Production endpoint - reuse logic from verify_user."""
#     return await verify_user(req)

# @app.get("/health")
# async def health_check():
#     """Health check endpoint to verify system status."""
#     base_health = {
#         "status": "healthy",
#         "gpu_enabled": USE_GPU,
#         "components": {
#             "face_agent": face_agent is not None,
#             "entity_agent": entity_agent is not None,
#             "gender_pipeline": gender_pipeline is not None,
#             "scorer": scorer is not None,
#             "redis": redis_cache is not None and redis_cache.enabled if redis_cache else False
#         }
#     }
    
#     # Add monitoring data if available
#     if PRODUCTION_MODE and monitor:
#         base_health["monitoring"] = monitor.get_health_status()
    
#     return base_health


# @app.get("/metrics")
# async def get_metrics():
#     """Get application metrics (production only)."""
#     if not PRODUCTION_MODE or not monitor:
#         return {"error": "Metrics not available (production mode disabled)"}
    
#     return {
#         "system": monitor.get_system_stats(),
#         "application": monitor.get_application_metrics()
#     }
    

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8101, workers=1)
