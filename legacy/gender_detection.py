import asyncio
import logging
import os
import sys
import uuid
import cv2
import numpy as np
import pytesseract
import uvicorn
import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from ultralytics import YOLO

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GenderDetectionAPI")

class Config:
    # Adjust path if your model is elsewhere
    MODEL_PATH = "models/bestgender.pt" 
    DOWNLOAD_DIR = Path("downloads")
    OUTPUT_DIR = Path("outputs")
    
    # OCR Configuration
    TESSERACT_CONFIG = r'--oem 3 --psm 6'

config = Config()

# Ensure directories exist
config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Core Pipeline Logic ---

class GenderDetectionPipeline:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        logger.info(f"Loading YOLO model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            logger.critical(f"Model not found at {self.model_path}")
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model = YOLO(self.model_path)
        self._check_tesseract()
        logger.info("Gender Detection Pipeline initialized successfully.")

    def _check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            logger.critical("Tesseract not found. Please install Tesseract OCR.")
            raise RuntimeError("Tesseract not found")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Main logic: Detect -> Crop -> OCR -> Draw -> Return Result
        """
        logger.info(f"Processing image: {image_path}")
        
        # 1. Read Image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file.")

        # 2. Run YOLO Inference
        results = self.model(img)
        
        gender_result = "Unknown"
        ocr_text = ""
        output_filename = None

        # Check if detections exist
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Take the detection with highest confidence
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0])

            logger.info(f"Gender box detected with confidence: {confidence:.2f}")

            # 3. Crop Region
            # Add a small padding if needed, ensuring we don't go out of bounds
            h, w, _ = img.shape
            pad = 2
            y1_c = max(0, y1 - pad)
            y2_c = min(h, y2 + pad)
            x1_c = max(0, x1 - pad)
            x2_c = min(w, x2 + pad)
            
            crop = img[y1_c:y2_c, x1_c:x2_c]

            # 4. Preprocess for OCR (Grayscale + Threshold)
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Simple thresholding to make text pop
            _, thresh_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 5. Run OCR
            ocr_text = pytesseract.image_to_string(thresh_crop, lang='eng', config=config.TESSERACT_CONFIG)
            clean_text = ocr_text.upper().strip()
            
            logger.info(f"OCR Raw Text: {clean_text}")

            # 6. Determine Gender Logic
            if "MALE" in clean_text or "M " in clean_text or clean_text == "M":
                gender_result = "Male"
            elif "FEMALE" in clean_text or "F " in clean_text or clean_text == "F":
                gender_result = "Female"
            
            # 7. Draw Visuals on Original Image
            color = (255, 0, 0) # Blue
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{gender_result} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            logger.warning("No gender detection found by YOLO.")

        # 8. Save Output Image
        out_name = f"{uuid.uuid4()}_out.jpg"
        out_path = config.OUTPUT_DIR / out_name
        cv2.imwrite(str(out_path), img)

        return {
            "gender": gender_result,
            "ocr_text_read": ocr_text.strip(),
            "output_image_path": str(out_path)
        }

# --- FastAPI Setup ---

app = FastAPI(title="Gender Detection API", version="2.0")

pipeline: Optional[GenderDetectionPipeline] = None

class GenderRequest(BaseModel):
    image_url: HttpUrl

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        pipeline = GenderDetectionPipeline(model_path=config.MODEL_PATH)
    except Exception as e:
        logger.critical(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

async def download_image(session: aiohttp.ClientSession, url: str, filepath: Path) -> bool:
    try:
        async with session.get(str(url), timeout=30) as response:
            response.raise_for_status()
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False

@app.post("/aadhar-gender")
async def detect_gender(req: GenderRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Generate unique ID for this request
    req_id = str(uuid.uuid4())
    input_filename = f"{req_id}.jpg"
    input_path = config.DOWNLOAD_DIR / input_filename

    logger.info(f"Received request for URL: {req.image_url}")

    # Async Download
    async with aiohttp.ClientSession() as session:
        success = await download_image(session, str(req.image_url), input_path)
    
    if not success:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Failed to download image from URL"}
        )

    try:
        # Run Pipeline
        # We run the synchronous processing in a separate thread to not block the event loop
        # although for heavy CPU tasks usually separate processes are better, 
        # but for this scale, direct call or run_in_executor is fine.
        result = pipeline.process_image(str(input_path))

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "gender": result["gender"],
                    "ocr_debug": result["ocr_text_read"],
                    "input_image": str(input_path),
                    "output_image": result["output_image_path"]
                }
            }
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Internal processing error: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("gender_detection:app", host="0.0.0.0", port=8005, reload=True)