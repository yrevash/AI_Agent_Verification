import asyncio
import aiohttp
import time
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import gspread
import cloudscraper
from dotenv import load_dotenv

# Import Redis cache
from redis_cache import get_cache


def _backend_post(url: str, payload: dict, headers: dict, timeout: int = 25) -> dict:
    """Make a POST request to the backend using cloudscraper to bypass Cloudflare."""
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'darwin', 'mobile': False}
    )
    try:
        # Update scraper's session headers so they persist across redirects
        scraper.headers.update(headers)
        resp = scraper.post(url, json=payload, headers=headers, timeout=timeout)
        return {"status": resp.status_code, "body": resp.text}
    finally:
        if hasattr(scraper, 'close'):
            scraper.close()

# Load environment variables
load_dotenv()

BASE_URL = "https://qoneqt.com/v1/api"
ADMIN_ID = 27  
AGENT_IDS = [77, 78, 79, 80]  
BATCH_SIZE = 20
TTL_HOURS = 12

LOCAL_AI_URL = "http://localhost:8101/verification/verify/agent/"

# Retry configuration for 502 errors
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
AGENT_CYCLE_DELAY = 3  # delay between agents
FULL_CYCLE_DELAY = 10  # delay between full cycles

# Set to False to suppress detailed 502 HTML error logs
LOG_502_DETAILS = False

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("agent_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "X-sign-Secret": "QONEQT1607_AI",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://qoneqt.com",
    "Referer": "https://qoneqt.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"'
}


current_session = {
    "session_id": None,
    "start_time": None,
    "batch_number": 0,
    "users_processed": [],
    "agent_id": None
}

class GoogleSheetLogger:
    def __init__(self, creds_file, sheet_name):
        """
        Initialize Google Sheets logger
        :param creds_file: Path to Google Service Account JSON credentials
        :param sheet_name: Name of the Google Sheet to log to
        """
        try:
            self.gc = gspread.service_account(filename=creds_file)
            self.sh = self.gc.open(sheet_name)
            self.worksheet = self.sh.sheet1
            
            # Check if headers exist, if not create them
            headers = self.worksheet.row_values(1)
            if not headers:
                self._create_headers()
                logger.info("✅ Created Google Sheets headers")
            else:
                logger.info("✅ Connected to existing Google Sheet")
                
        except FileNotFoundError:
            logger.error(f"❌ Google Sheets credentials file not found: {creds_file}")
            raise
        except gspread.exceptions.SpreadsheetNotFound:
            logger.error(f"❌ Google Sheet '{sheet_name}' not found. Please create it or share it with the service account.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Sheets: {e}")
            raise
    
    def _create_headers(self):
        """Create column headers for the sheet"""
        headers = [
            "Timestamp",
            "User ID",
            "Agent ID",
            "Batch Number",
            "Final Decision",
            "Status Code",
            "Score",
            "Face Similarity",
            "Age Verification",
            "Gender Match",
            "DOB Match",
            "Input DOB",
            "Input Gender",
            "Extracted DOB",
            "Extracted Gender",
            "Extracted Aadhaar",
            "Rejection Reasons",
            "Total Processing Time (s)",
            "AI Processing Time (s)",
            "Has Errors",
            "Error Count",
            "Session ID"
        ]
        self.worksheet.append_row(headers)
        # Format headers (bold)
        self.worksheet.format('A1:V1', {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
        })

    def log(self, user_log):
        """
        Log user processing data to Google Sheets
        :param user_log: Dictionary containing user processing information
        """
        try:
            final_result = user_log.get("final_result", {})
            breakdown = final_result.get("breakdown", {})
            extracted_data = final_result.get("extracted_data", {})
            timing = user_log.get("timing", {})
            input_data = user_log.get("input_data", {})
            
            row_data = [
                user_log.get("timestamp", ""),
                user_log.get("user_id", ""),
                user_log.get("agent_id", ""),
                user_log.get("batch_number", ""),
                final_result.get("final_decision", ""),
                final_result.get("status_code", ""),
                final_result.get("score", 0),
                breakdown.get("face_similarity", 0),
                breakdown.get("age_verification", 0),
                breakdown.get("gender_match", 0),
                breakdown.get("dob_match", 0),
                input_data.get("dob", ""),
                input_data.get("gender", ""),
                extracted_data.get("dob", ""),
                extracted_data.get("gender", ""),
                extracted_data.get("aadhaar", ""),
                ", ".join(final_result.get("rejection_reasons", [])),
                round(timing.get("total_time", 0), 2),
                round(timing.get("ai_processing_time", 0), 2),
                "Yes" if len(user_log.get("errors", [])) > 0 else "No",
                len(user_log.get("errors", [])),
                user_log.get("session_id", "")
            ]
            
            self.worksheet.append_row(row_data)
            logger.info(f"📊 Logged User {user_log.get('user_id')} to Google Sheets")
            
        except Exception as e:
            logger.error(f"❌ Failed to log to Google Sheets: {e}")
            # Don't raise - we don't want to break the pipeline if logging fails

# Global Google Sheets logger instance (will be initialized in main)
google_sheet_logger = None

import sqlite3
import shutil

class LocalDBLogger:
    def __init__(self, db_name="local_kyc_data.db"):
        """Initialize Local SQLite Database with Production Settings"""
        self.db_name = db_name
        self._backup_db() # 1. Auto-backup on startup
        self._init_db()

    def _backup_db(self):
        """Create a safety backup on restart if DB exists"""
        if os.path.exists(self.db_name):
            try:
                backup_name = f"{self.db_name}.backup"
                shutil.copy2(self.db_name, backup_name)
                logger.info(f"🛡️  Created startup backup: {backup_name}")
            except Exception as e:
                logger.error(f"⚠️ Failed to backup DB: {e}")

    def _init_db(self):
        """Create table and Enable WAL mode"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                # 2. Enable Write-Ahead Logging (WAL) for concurrency/speed
                conn.execute("PRAGMA journal_mode=WAL;")
                
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS kyc_records (
                        user_id TEXT PRIMARY KEY,
                        agent_id TEXT,
                        final_decision TEXT,
                        status_code INTEGER,
                        score REAL,
                        aadhaar_number TEXT,
                        timestamp DATETIME,
                        full_json_log TEXT
                    )
                """)
                
                # 3. Add Indices for faster querying later
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent ON kyc_records(agent_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision ON kyc_records(final_decision);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON kyc_records(timestamp);")
                
                conn.commit()
                logger.info("✅ Local SQLite Database initialized (WAL Mode)")
        except Exception as e:
            logger.error(f"❌ Failed to init Local DB: {e}")

    def save_record(self, user_log):
        """Save user log to SQLite"""
        try:
            # Extract key fields for columns
            user_id = str(user_log.get("user_id"))
            agent_id = str(user_log.get("agent_id"))
            timestamp = user_log.get("timestamp")
            
            final_res = user_log.get("final_result", {})
            decision = final_res.get("final_decision", "UNKNOWN")
            status_code = final_res.get("status_code", 0)
            score = final_res.get("score", 0)
            
            # Extract aadhaar specifically if available
            extracted = final_res.get("extracted_data", {})
            aadhaar = extracted.get("aadhaar", "") or extracted.get("aadhaar_number", "")

            # Convert full dict to JSON string
            full_json = json.dumps(user_log, ensure_ascii=False)

            with sqlite3.connect(self.db_name, timeout=10) as conn: # 10s timeout to wait for locks
                cursor = conn.cursor()
                # UPSERT (Insert or Replace if user_id exists)
                # This ensures if you re-run a user, it updates the record instead of crashing
                cursor.execute("""
                    INSERT OR REPLACE INTO kyc_records 
                    (user_id, agent_id, final_decision, status_code, score, aadhaar_number, timestamp, full_json_log)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, agent_id, decision, status_code, score, aadhaar, timestamp, full_json))
                conn.commit()
                logger.info(f"💾 Saved User {user_id} to Local DB")
        except Exception as e:
            logger.error(f"❌ Database Save Error for {user_id}: {e}")

# Initialize Global DB Instance
local_db = LocalDBLogger()


def create_session_log_file(agent_id):
    """Create a new session log file for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_session["session_id"] = f"agent_{agent_id}_{timestamp}"
    current_session["start_time"] = datetime.now().isoformat()
    current_session["batch_number"] = 0
    current_session["users_processed"] = []
    current_session["agent_id"] = agent_id
    
    # Create agent-specific directory
    agent_dir = LOGS_DIR / f"agent_{agent_id}"
    agent_dir.mkdir(exist_ok=True)
    
    # Create session directory
    session_dir = agent_dir / current_session["session_id"]
    session_dir.mkdir(exist_ok=True)
    
    return session_dir

def save_user_log(user_id, log_data, session_dir):
    """Save detailed log for a single user"""
    user_log_file = session_dir / f"user_{user_id}.json"
    
    with open(user_log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved detailed log for User {user_id}")

def save_batch_summary(session_dir, batch_data):
    """Save summary of the current batch"""
    current_session["batch_number"] += 1
    batch_file = session_dir / f"batch_{current_session['batch_number']}_summary.json"
    
    summary = {
        "batch_number": current_session["batch_number"],
        "timestamp": datetime.now().isoformat(),
        "agent_id": current_session["agent_id"],
        "total_users": len(batch_data),
        "users": batch_data
    }
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved batch summary: {batch_file.name}")

def save_session_summary(session_dir):
    """Save overall session summary at the end"""
    session_file = session_dir / "session_summary.json"
    
    summary = {
        "session_id": current_session["session_id"],
        "agent_id": current_session["agent_id"],
        "start_time": current_session["start_time"],
        "end_time": datetime.now().isoformat(),
        "total_batches": current_session["batch_number"],
        "total_users_processed": len(current_session["users_processed"]),
        "user_ids": current_session["users_processed"],
        "statistics": calculate_session_statistics(session_dir)
    }
    
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved session summary: {session_file.name}")

def calculate_session_statistics(session_dir):
    """Calculate detailed statistics from all user logs in the session"""
    stats = {
        "approved": 0,
        "rejected": 0,
        "review": 0,
        "failed": 0,
        "total_processing_time": 0,
        "total_score": 0,
        "rejection_reasons": [],
        "avg_score_approved": 0,
        "avg_score_rejected": 0,
        "breakdown_summary": {
            "face_similarity": [],
            "age_verification": [],
            "gender_match": [],
            "dob_match": []
        }
    }
    
    approved_scores = []
    rejected_scores = []
    
    # Read all user log files
    for user_file in session_dir.glob("user_*.json"):
        try:
            with open(user_file, 'r') as f:
                user_data = json.load(f)
                
            final_result = user_data.get("final_result", {})
            decision = final_result.get("final_decision", "FAILED")
            score = final_result.get("score", 0)
            breakdown = final_result.get("breakdown", {})
            
            if decision == "APPROVED":
                stats["approved"] += 1
                approved_scores.append(score)
            elif decision == "REJECTED":
                stats["rejected"] += 1
                rejected_scores.append(score)
                # Collect rejection reasons
                stats["rejection_reasons"].extend(final_result.get("rejection_reasons", []))
            elif decision == "REVIEW":
                stats["review"] += 1
            else:
                stats["failed"] += 1
            
            stats["total_processing_time"] += user_data.get("timing", {}).get("total_time", 0)
            stats["total_score"] += score
            
            # Collect breakdown data
            if breakdown:
                for key in stats["breakdown_summary"].keys():
                    if key in breakdown:
                        stats["breakdown_summary"][key].append(breakdown[key])
        except Exception as e:
            logger.error(f"Error processing stats for {user_file}: {e}")
            pass
    
    total_users = stats["approved"] + stats["rejected"] + stats["review"] + stats["failed"]
    
    if total_users > 0:
        stats["avg_processing_time"] = stats["total_processing_time"] / total_users
        stats["avg_score"] = stats["total_score"] / total_users
    else:
        stats["avg_processing_time"] = 0
        stats["avg_score"] = 0
    
    if approved_scores:
        stats["avg_score_approved"] = sum(approved_scores) / len(approved_scores)
    if rejected_scores:
        stats["avg_score_rejected"] = sum(rejected_scores) / len(rejected_scores)
    
    # Count rejection reasons
    from collections import Counter
    if stats["rejection_reasons"]:
        reason_counts = Counter(stats["rejection_reasons"])
        stats["top_rejection_reasons"] = dict(reason_counts.most_common(10))
    else:
        stats["top_rejection_reasons"] = {}
    
    # Calculate average breakdown scores
    for key, values in stats["breakdown_summary"].items():
        if values:
            stats["breakdown_summary"][key] = {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        else:
            stats["breakdown_summary"][key] = {"avg": 0, "min": 0, "max": 0, "count": 0}
    
    return stats

async def lock_batch(session, agent_id, retry_count=0):
    """Step 1: Lock a batch of users for this agent with retry logic"""
    url = f"{BASE_URL}/admin/kyc-lock-batch"
    payload = {
        "admin_id": ADMIN_ID,
        "target_admin_id": agent_id,
        "batch_size": BATCH_SIZE,
        "ttl_hours": TTL_HOURS
    }

    try:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _backend_post, url, payload, HEADERS)
        status = raw["status"]
        body = raw["body"]

        if status == 200:
            data = json.loads(body)
            if data.get("success"):
                logger.info(f"🔒 Locked {data.get('locked_count')} users for Agent {agent_id}")
                return data.get("kyc_ids", [])
            else:
                logger.error(f"Failed to lock batch: {data.get('message')}")
        elif status == 502 and retry_count < MAX_RETRIES:
            wait_time = 2 ** retry_count * INITIAL_RETRY_DELAY
            logger.warning(f"⚠️ 502 Bad Gateway for Agent {agent_id}. Retrying in {wait_time}s... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait_time)
            return await lock_batch(session, agent_id, retry_count + 1)
        else:
            if LOG_502_DETAILS or status != 502:
                logger.error(f"API Error {status} for Agent {agent_id}: {body[:200]}")
            else:
                logger.error(f"API Error {status} for Agent {agent_id} - Server unavailable")
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Timeout locking batch for Agent {agent_id}")
    except Exception as e:
        logger.error(f"❌ Lock Batch Exception for Agent {agent_id}: {e}")
    return []

async def release_batch(session, agent_id, retry_count=0):
    """Release the batch for an agent after processing with retry logic"""
    url = f"{BASE_URL}/admin/kyc-release-batch"
    payload = {
        "admin_id": ADMIN_ID,
        "target_admin_id": agent_id
    }

    try:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _backend_post, url, payload, HEADERS)
        status = raw["status"]
        body = raw["body"]

        if status == 200:
            data = json.loads(body)
            if data.get("success"):
                logger.info(f"🔓 Released batch for Agent {agent_id}")
                return True
            else:
                logger.error(f"Failed to release batch for Agent {agent_id}: {data.get('message')}")
        elif status == 502 and retry_count < MAX_RETRIES:
            wait_time = 2 ** retry_count * INITIAL_RETRY_DELAY
            logger.warning(f"⚠️ 502 Bad Gateway releasing batch for Agent {agent_id}. Retrying in {wait_time}s... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait_time)
            return await release_batch(session, agent_id, retry_count + 1)
        else:
            if LOG_502_DETAILS or status != 502:
                logger.error(f"Release Batch API Error {status} for Agent {agent_id}: {body[:200]}")
            else:
                logger.error(f"Release Batch API Error {status} for Agent {agent_id} - Server unavailable")
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Timeout releasing batch for Agent {agent_id}")
    except Exception as e:
        logger.error(f"❌ Release Batch Exception for Agent {agent_id}: {e}")
    return False

async def fetch_user_details(session, agent_id, retry_count=0):
    """Step 2: Get details for the assigned users with retry logic"""
    url = f"{BASE_URL}/admin/kyc-my-batch"
    payload = {
        "admin_id": agent_id,
        "page": 1,
        "only_pending": True,
        "limit": BATCH_SIZE,
        "statusFilter": "2",
        "isFullSearch": True,
        "offset": 0,
        "type": "0",
        "assignment_filter": "assigned"
    }

    try:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _backend_post, url, payload, HEADERS)
        status = raw["status"]
        body = raw["body"]

        if status == 200:
            data = json.loads(body)
            users = data.get("data", [])
            logger.info(f"📋 Fetched details for {len(users)} users (Agent {agent_id})")
            return users
        elif status == 502 and retry_count < MAX_RETRIES:
            wait_time = 2 ** retry_count * INITIAL_RETRY_DELAY
            logger.warning(f"⚠️ 502 Bad Gateway fetching users for Agent {agent_id}. Retrying in {wait_time}s... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait_time)
            return await fetch_user_details(session, agent_id, retry_count + 1)
        else:
            if LOG_502_DETAILS or status != 502:
                logger.error(f"Fetch Details Error {status} for Agent {agent_id}: {body[:200]}")
            else:
                logger.error(f"Fetch Details Error {status} for Agent {agent_id} - Server unavailable")
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Timeout fetching user details for Agent {agent_id}")
    except Exception as e:
        logger.error(f"❌ Fetch Details Exception for Agent {agent_id}: {e}")
    return []

async def process_single_user(session, user, session_dir, agent_id):
    """Step 3 & 4: Process via Local AI and Push Result"""
    user_id = user.get("user_id")
    
    user_log = {
        "user_id": user_id,
        "agent_id": agent_id,
        "session_id": current_session["session_id"],
        "batch_number": current_session["batch_number"] + 1,
        "timestamp": datetime.now().isoformat(),
        "input_data": {},
        "processing_steps": [],
        "timing": {},
        "api_calls": [],
        "final_result": {},
        "errors": []
    }
    
    start_time = time.time()
    
    try:
        # Store original input data
        user_log["input_data"] = {
            "user_id": user_id,
            "dob": user.get("dob"),
            "gender": user.get("gender"),
            "passport_first": user.get("passport_first"),
            "passport_old": user.get("passport_old"),
            "selfie_photo": user.get("selfie_photo"),
            "full_user_data": user  # Store complete user object
        }
        
        user_log["processing_steps"].append({
            "step": "1_input_received",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "User data received from main server"
        })
        
        # 1. Prepare Request for Local AI
        ai_request = {
            "user_id": user_id,
            "dob": user.get("dob"),
            "passport_first": user.get("passport_first"),
            "passport_old": user.get("passport_old"),
            "selfie_photo": user.get("selfie_photo"),
            "gender": user.get("gender")
        }
        
        # Handle image paths
        CDN_BASE = "https://cdn.qoneqt.com/" 
        for key in ["passport_first", "passport_old", "selfie_photo"]:
            if ai_request[key] and not ai_request[key].startswith("http"):
                ai_request[key] = CDN_BASE + ai_request[key]
        
        user_log["processing_steps"].append({
            "step": "2_image_paths_resolved",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Image URLs prepared",
            "resolved_urls": {
                "passport_first": ai_request["passport_first"],
                "passport_old": ai_request["passport_old"],
                "selfie_photo": ai_request["selfie_photo"]
            }
        })

        # 2. Call Local AI
        ai_start_time = time.time()
        try:
            async with session.post(LOCAL_AI_URL, json=ai_request, timeout=120) as resp:
                ai_elapsed = time.time() - ai_start_time
                
                api_call_log = {
                    "endpoint": LOCAL_AI_URL,
                    "method": "POST",
                    "request": ai_request,
                    "status_code": resp.status,
                    "elapsed_time": ai_elapsed,
                    "timestamp": datetime.now().isoformat()
                }
                
                if resp.status != 200:
                    error_text = await resp.text()
                    api_call_log["error"] = error_text
                    user_log["api_calls"].append(api_call_log)
                    user_log["errors"].append(f"Local AI Failed: Status {resp.status}")
                    user_log["processing_steps"].append({
                        "step": "3_local_ai_processing",
                        "timestamp": datetime.now().isoformat(),
                        "status": "failed",
                        "message": f"AI processing failed with status {resp.status}",
                        "error": error_text
                    })
                    logger.error(f" User {user_id}: Local AI Failed ({resp.status})")
                    user_log["final_result"] = {
                        "final_decision": "FAILED",
                        "status_code": -1,
                        "reason": "Local AI processing failed",
                        "rejection_reasons": [f"Local AI processing failed with status {resp.status}"],
                        "extracted_data": {}
                    }
                    save_user_log(user_id, user_log, session_dir)
                    return user_log
                
                ai_result = await resp.json()
                api_call_log["response"] = ai_result
                user_log["api_calls"].append(api_call_log)
                
                user_log["processing_steps"].append({
                    "step": "3_local_ai_processing",
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "message": "AI processing completed successfully",
                    "processing_time": ai_elapsed,
                    "ai_decision": ai_result.get("final_decision"),
                    "ai_score": ai_result.get("score", 0),
                    "ai_breakdown": ai_result.get("breakdown", {}),
                    "ai_rejection_reasons": ai_result.get("rejection_reasons", []),
                    "ai_extracted_data": ai_result.get("extracted_data", {}),
                    "ai_result": ai_result
                })
                
                # Log detailed decision information
                decision_details = f"Decision: {ai_result.get('final_decision')} | Score: {ai_result.get('score', 0)}"
                if ai_result.get('rejection_reasons'):
                    decision_details += f" | Reasons: {', '.join(ai_result.get('rejection_reasons', []))}"
                logger.info(f"  User {user_id}: {decision_details}")
                
        except asyncio.TimeoutError:
            ai_elapsed = time.time() - ai_start_time
            user_log["errors"].append("Local AI timeout after 120 seconds")
            user_log["processing_steps"].append({
                "step": "3_local_ai_processing",
                "timestamp": datetime.now().isoformat(),
                "status": "timeout",
                "message": "AI processing timed out",
                "elapsed_time": ai_elapsed
            })
            logger.error(f" User {user_id}: Local AI Timeout")
            user_log["final_result"] = {
                "final_decision": "FAILED",
                "status_code": -1,
                "reason": "Local AI timeout",
                "rejection_reasons": ["AI processing timeout after 120 seconds"],
                "extracted_data": {}
            }
            save_user_log(user_id, user_log, session_dir)
            return user_log
            
        # 3. Push Result to Main Server
        push_url = f"{BASE_URL}/ai-agent"
        push_payload = {
            "user_id": user_id,
            "agent_id": str(agent_id),
            "final_decision": ai_result.get("final_decision", "REJECTED"),
            "status_code": ai_result.get("status_code", 0),
            "extracted_data": ai_result.get("extracted_data", {}),
            "rejection_reasons": ai_result.get("rejection_reasons", [])
        }
        submission_headers = {
            "User-Agent": HEADERS["User-Agent"],
            "Accept": HEADERS["Accept"],
            "X-sign-Secret": "QONEQT1607_AI",
            "Accept-Language": HEADERS["Accept-Language"],
            "Accept-Encoding": HEADERS["Accept-Encoding"],
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Origin": HEADERS["Origin"],
            "Referer": HEADERS["Referer"],
            "Sec-Fetch-Dest": HEADERS["Sec-Fetch-Dest"],
            "Sec-Fetch-Mode": HEADERS["Sec-Fetch-Mode"],
            "Sec-Fetch-Site": HEADERS["Sec-Fetch-Site"],
            "sec-ch-ua": HEADERS["sec-ch-ua"],
            "sec-ch-ua-mobile": HEADERS["sec-ch-ua-mobile"],
            "sec-ch-ua-platform": HEADERS["sec-ch-ua-platform"],
        }
        
        push_start_time = time.time()
        push_api_response = None
        push_success = False

        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, _backend_post, push_url, push_payload, submission_headers
            )
            push_elapsed = time.time() - push_start_time
            push_status = raw["status"]
            push_body = raw["body"]

            push_call_log = {
                "endpoint": push_url,
                "method": "POST",
                "request": push_payload,
                "status_code": push_status,
                "elapsed_time": push_elapsed,
                "timestamp": datetime.now().isoformat()
            }

            # Try to parse response JSON
            try:
                push_result = json.loads(push_body)
                push_api_response = push_result
                push_call_log["response"] = push_result
            except (json.JSONDecodeError, ValueError):
                push_api_response = {
                    "status_code": push_status,
                    "error": push_body,
                    "message": push_body
                }
                push_call_log["response"] = push_api_response

            if push_status == 200:
                push_result = push_call_log.get("response", {})
                user_log["api_calls"].append(push_call_log)
                push_success = True

                user_log["processing_steps"].append({
                    "step": "4_result_pushed",
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "message": "Result successfully pushed to main server",
                    "push_time": push_elapsed,
                    "server_response": push_result,
                    "pushed_data": {
                        "final_decision": push_payload['final_decision'],
                        "status_code": push_payload['status_code'],
                        "rejection_reasons": push_payload['rejection_reasons'],
                        "extracted_data": push_payload['extracted_data']
                    }
                })

                decision_log = f"{push_payload['final_decision']} (Code: {push_payload['status_code']})"
                if push_payload['rejection_reasons']:
                    decision_log += f" | Reasons: {', '.join(push_payload['rejection_reasons'])}"
                logger.info(f"✅ User {user_id}: Processed & Pushed ({decision_log})")

            else:
                push_result = push_call_log.get("response", {})
                user_log["api_calls"].append(push_call_log)
                user_log["errors"].append(f"Push failed: Status {push_status}")

                user_log["processing_steps"].append({
                    "step": "4_result_pushed",
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "message": f"Failed to push result to main server",
                    "status_code": push_status,
                    "api_response": push_result,
                    "error": push_result.get("error") if isinstance(push_result, dict) else str(push_result)
                })

                api_error_msg = push_result.get("message") if isinstance(push_result, dict) else str(push_result)[:100]
                logger.error(f" User {user_id}: Failed to Push Result {push_status} - {api_error_msg}")

        except Exception as push_error:
            push_elapsed = time.time() - push_start_time
            user_log["errors"].append(f"Push exception: {str(push_error)}")
            push_api_response = {
                "status_code": -1,
                "error": str(push_error),
                "message": f"Exception during push: {str(push_error)}"
            }
            user_log["processing_steps"].append({
                "step": "4_result_pushed",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": "Exception during result push",
                "error": str(push_error),
                "elapsed_time": push_elapsed
            })
            logger.error(f" User {user_id}: Push Exception - {push_error}")
        
        # Store final result with detailed breakdown and API response
        user_log["final_result"] = {
            "final_decision": ai_result.get("final_decision", "REJECTED"),
            "status_code": ai_result.get("status_code", 0),
            "score": ai_result.get("score", 0),
            "breakdown": ai_result.get("breakdown", {}),
            "extracted_data": ai_result.get("extracted_data", {}),
            "rejection_reasons": ai_result.get("rejection_reasons", []),
            "api_response": push_api_response,  # Store API response here
            "api_push_success": push_success,   # Track if push was successful
            "decision_summary": {
                "is_approved": ai_result.get("final_decision") == "APPROVED",
                "is_rejected": ai_result.get("final_decision") == "REJECTED",
                "is_review": ai_result.get("final_decision") == "REVIEW",
                "total_issues": len(ai_result.get("rejection_reasons", [])),
                "extracted_aadhaar": ai_result.get("extracted_data", {}).get("aadhaar"),
                "extracted_dob": ai_result.get("extracted_data", {}).get("dob"),
                "extracted_gender": ai_result.get("extracted_data", {}).get("gender")
            }
        }

    except Exception as e:
        user_log["errors"].append(f"Critical exception: {str(e)}")
        user_log["processing_steps"].append({
            "step": "error",
            "timestamp": datetime.now().isoformat(),
            "status": "critical_error",
            "message": "Unhandled exception during processing",
            "error": str(e)
        })
        user_log["final_result"] = {
            "final_decision": "FAILED",
            "status_code": -1,
            "reason": f"Exception: {str(e)}",
            "rejection_reasons": [f"Critical exception: {str(e)}"],
            "extracted_data": {}
        }
        logger.error(f" User {user_id} Exception: {e}")
    
    # Calculate timing
    total_time = time.time() - start_time
    user_log["timing"] = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_time": total_time,
        "ai_processing_time": sum(
            step.get("processing_time", 0) 
            for step in user_log["processing_steps"] 
            if "processing_time" in step
        ),
        "api_push_time": sum(
            step.get("push_time", 0) 
            for step in user_log["processing_steps"] 
            if "push_time" in step
        )
    }
    
    # Save the comprehensive log
    save_user_log(user_id, user_log, session_dir)
    current_session["users_processed"].append(user_id)
    
    # Store in Redis cache with API response for ALL cases (success, failure, errors)
    redis_cache = get_cache()
    if redis_cache.enabled:
        try:
            redis_cache.store_verification_result(
                user_id=str(user_id),
                verification_result=ai_result if 'ai_result' in locals() else user_log["final_result"],
                api_response=push_api_response if push_api_response else user_log["final_result"].get("api_response")
            )
            logger.info(f"💾 Stored user {user_id} to Redis cache with API response")
        except Exception as redis_error:
            logger.error(f"⚠️  Failed to store user {user_id} in Redis: {redis_error}")
    
    # Log to Google Sheets if enabled
    if google_sheet_logger:
        try:
            google_sheet_logger.log(user_log)
        except Exception as e:
            logger.error(f"Failed to log User {user_id} to Google Sheets: {e}")

    try:
        local_db.save_record(user_log)
    except Exception as e:
        logger.error(f"Failed to save to Local DB: {e}")

    
    return user_log

async def process_agent_batch(session, agent_id, session_dir):
    """Process a single batch for a specific agent"""
    logger.info(f"🔄 --- Starting Cycle for Agent {agent_id} ---")
    
    # Get Redis cache instance
    redis_cache = get_cache()
    
    # 1. Lock Batch
    locked_ids = await lock_batch(session, agent_id)
    
    if not locked_ids:
        logger.warning(f"⚠️ No users available to lock for Agent {agent_id} (Server may be down or no pending users)")
        return False
    
    # 2. Fetch User Details
    users_to_process = await fetch_user_details(session, agent_id)
    
    if not users_to_process:
        logger.warning(f"Locked batch for Agent {agent_id} but got no user details. Releasing batch...")
        await release_batch(session, agent_id)
        return False
    
    # 3. Check Redis Cache - Filter out already processed users
    if redis_cache.enabled:
        logger.info(f"🔍 Checking Redis cache for {len(users_to_process)} users...")
        user_ids_to_check = [str(u.get("user_id", u.get("id"))) for u in users_to_process]
        cache_status = redis_cache.bulk_check_users(user_ids_to_check)
        
        # Filter out cached users
        users_before = len(users_to_process)
        users_to_process = [
            u for u in users_to_process 
            if not cache_status.get(str(u.get("user_id", u.get("id"))), False)
        ]
        users_after = len(users_to_process)
        
        cached_count = users_before - users_after
        if cached_count > 0:
            logger.info(f"⚡ Skipped {cached_count} already verified users from cache")
            logger.info(f"📋 Processing {users_after} new users")
        
        # If all users were cached, release batch and return
        if not users_to_process:
            logger.info(f"✨ All {users_before} users already verified - nothing to process!")
            await release_batch(session, agent_id)
            return True
    
    # 4. Process Batch
    batch_start_time = time.time()
    batch_results = []
    
    # Process in chunks of 5 users at a time
    chunk_size = 5
    for i in range(0, len(users_to_process), chunk_size):
        chunk = users_to_process[i:i + chunk_size]
        tasks = [process_single_user(session, u, session_dir, agent_id) for u in chunk]
        chunk_results = await asyncio.gather(*tasks)
        batch_results.extend(chunk_results)
    
    batch_elapsed = time.time() - batch_start_time
    
    # Create batch summary with detailed information
    batch_summary = []
    for user_log in batch_results:
        if user_log:
            final_result = user_log["final_result"]
            batch_summary.append({
                "user_id": user_log["user_id"],
                "final_decision": final_result.get("final_decision"),
                "status_code": final_result.get("status_code"),
                "score": final_result.get("score", 0),
                "breakdown": final_result.get("breakdown", {}),
                "rejection_reasons": final_result.get("rejection_reasons", []),
                "extracted_data": final_result.get("extracted_data", {}),
                "processing_time": user_log["timing"].get("total_time"),
                "errors": len(user_log["errors"]),
                "has_errors": len(user_log["errors"]) > 0
            })
    
    # Save batch summary
    save_batch_summary(session_dir, {
        "batch_elapsed_time": batch_elapsed,
        "users": batch_summary
    })
    
    # Calculate and log batch statistics
    approved = sum(1 for u in batch_summary if u['final_decision'] == 'APPROVED')
    rejected = sum(1 for u in batch_summary if u['final_decision'] == 'REJECTED')
    review = sum(1 for u in batch_summary if u['final_decision'] == 'REVIEW')
    
    logger.info(f"--- Batch Complete for Agent {agent_id}. Processed {len(batch_results)} users in {batch_elapsed:.2f}s ---")
    logger.info(f"📊 Results: ✅ {approved} Approved | ❌ {rejected} Rejected | ⏳ {review} Review")
    
    # Log common rejection reasons if any
    all_rejection_reasons = []
    for u in batch_summary:
        all_rejection_reasons.extend(u.get('rejection_reasons', []))
    
    if all_rejection_reasons:
        from collections import Counter
        reason_counts = Counter(all_rejection_reasons)
        logger.info(f"🔍 Top Rejection Reasons:")
        for reason, count in reason_counts.most_common(5):
            logger.info(f"   • {reason}: {count} times")
    
    # 4. Release the batch for this agent
    await release_batch(session, agent_id)
    
    return True

async def run_pipeline():
    """Main pipeline that cycles through all agents continuously"""
    global google_sheet_logger
    
    async with aiohttp.ClientSession() as session:
        try:
            agent_sessions = {}
            
            # Initialize Redis Cache
            redis_cache = get_cache()
            if redis_cache.enabled:
                logger.info("✅ Redis cache enabled")
                cache_stats = redis_cache.get_cache_stats()
                logger.info(f"📊 Redis Cache Stats: {cache_stats['total_entries']} entries, TTL: {cache_stats['default_ttl_hours']}h")
            else:
                logger.info("ℹ️  Redis cache disabled (set REDIS_ENABLED=true to enable)")
            
            # Initialize Google Sheets Logger (optional)
            # Set GOOGLE_SHEETS_ENABLED=true and provide credentials to enable
            import os
            if os.getenv("GOOGLE_SHEETS_ENABLED", "false").lower() == "true":
                try:
                    creds_file = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "google_sheets_credentials.json")
                    sheet_name = os.getenv("GOOGLE_SHEETS_NAME", "KYC Verification Logs")
                    google_sheet_logger = GoogleSheetLogger(creds_file, sheet_name)
                    logger.info(f"✅ Google Sheets logging enabled: '{sheet_name}'")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Google Sheets: {e}")
                    logger.info("Continuing without Google Sheets logging...")
            else:
                logger.info("ℹ️ Google Sheets logging disabled (set GOOGLE_SHEETS_ENABLED=true to enable)")
            
            # Create session directories for each agent
            for agent_id in AGENT_IDS:
                session_dir = create_session_log_file(agent_id)
                agent_sessions[agent_id] = session_dir
                logger.info(f"📁 Agent {agent_id} logs will be saved to: {session_dir}")
            
            logger.info(f"🚀 Starting continuous processing loop for Agents: {AGENT_IDS}")
            
            while True:
                # Cycle through each agent
                for agent_id in AGENT_IDS:
                    session_dir = agent_sessions[agent_id]
                    
                    # Process one batch for this agent
                    processed = await process_agent_batch(session, agent_id, session_dir)
                    
                    if not processed:
                        logger.info(f"⏭️ No work available for Agent {agent_id}, moving to next agent...")
                    
                    # Delay between agents to avoid rate limiting
                    await asyncio.sleep(AGENT_CYCLE_DELAY)
                
                # After processing all agents, take a longer break before the next cycle
                logger.info(f"--- Completed cycle for all agents. Starting next cycle in {FULL_CYCLE_DELAY}s... ---")
                await asyncio.sleep(FULL_CYCLE_DELAY)
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested...")
            raise
        finally:
            # Save session summaries for all agents before exiting
            logger.info("💾 Saving final session summaries for all agents...")
            for agent_id, session_dir in agent_sessions.items():
                save_session_summary(session_dir)
                logger.info(f"✅ Agent {agent_id} logs saved to: {session_dir}")

if __name__ == "__main__":
    print(f"🚀 Starting Production Dispatcher for Agents: {AGENT_IDS}")
    print(f"📋 Will process agents in rotation continuously")
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")