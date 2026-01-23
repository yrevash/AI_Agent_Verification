import cv2
import numpy as np
import requests
from qreader import QReader
from pyaadhaar.decode import AadhaarSecureQr
from pyzbar.pyzbar import decode as pyzbar_decode
import base64
from io import BytesIO
from PIL import Image

# Initialize AI Detector (Standard Model)
qreader_detector = QReader(model_size='s', min_confidence=0.5)

def download_image_from_cdn(url: str) -> str:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        temp_filename = "temp_aadhaar.jpg"
        with open(temp_filename, 'wb') as f:
            f.write(response.content)
        return temp_filename
    except Exception as e:
        raise ValueError(f"Failed to download image: {str(e)}")

def add_white_border(img, border_size=50):
    """Adds a white 'Quiet Zone' around the image. Crucial for screenshots."""
    return cv2.copyMakeBorder(
        img, 
        top=border_size, 
        bottom=border_size, 
        left=border_size, 
        right=border_size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )

def try_decode_strategy(img_array, strategy_name):
    """
    Helper to attempt decoding with different engines.
    """
    print(f"Attempting Strategy: {strategy_name}")
    
    # Strategy A: AI Detection (QReader)
    try:
        decoded_text_list = qreader_detector.detect_and_decode(image=img_array)
        if decoded_text_list and decoded_text_list[0]:
            return decoded_text_list[0]
    except Exception as e:
        print(f"QReader failed: {e}")

    # Strategy B: PyZbar (Math-based, good for clean screenshots)
    # PyZbar needs Grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
        
    barcodes = pyzbar_decode(gray)
    for barcode in barcodes:
        return barcode.data.decode("utf-8")
        
    return None

def extract_aadhaar_data(image_path: str):
    try:
        # 1. Load Original Image
        original_img = cv2.imread(image_path)
        if original_img is None:
            return {"status": "failed", "error": "Could not read image file."}

        qr_data = None
        
        # --- ATTEMPT 1: Raw Image (Best for good photos) ---
        qr_data = try_decode_strategy(original_img, "Raw")

        # --- ATTEMPT 2: Padding + Upscale (Best for Screenshots) ---
        if not qr_data:
            # Scale up by 2x (helps with small dots)
            upscaled = cv2.resize(original_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            # Add Border (Fixes tight crop issues)
            padded = add_white_border(upscaled)
            qr_data = try_decode_strategy(padded, "Upscale+Pad")

        # --- ATTEMPT 3: High Contrast Binary (Best for Glare/Lighting) ---
        if not qr_data:
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            # Adaptive Thresholding creates a strict Black/White image
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
            )
            # Add border to the binary image too
            binary_padded = add_white_border(binary)
            # Convert back to BGR for QReader compatibility
            binary_bgr = cv2.cvtColor(binary_padded, cv2.COLOR_GRAY2BGR)
            qr_data = try_decode_strategy(binary_bgr, "Binary+Threshold")

        # --- FINAL CHECK ---
        if not qr_data:
            # Save the debug image so you can see what failed
            cv2.imwrite("debug_failed_frame.jpg", original_img)
            return {
                "status": "failed", 
                "error": "QR code not detected in any mode. Check 'debug_failed_frame.jpg' on server."
            }

        # --- DECODING LOGIC (Same as before) ---
        try:
            if qr_data.isdigit():
                # Validate QR data length (Secure Aadhaar QRs are typically 2000+ characters)
                if len(qr_data) < 500:
                    return {
                        "status": "failed",
                        "error": f"QR data too short ({len(qr_data)} chars). Expected secure Aadhaar QR (2000+ chars).",
                        "qr_data_sample": qr_data
                    }
                
                try:
                    secure_qr = AadhaarSecureQr(int(qr_data))
                    decoded_data = secure_qr.decodeddata()
                    photo_data = secure_qr.image()
                    
                    buffered = BytesIO()
                    if isinstance(photo_data, Image.Image):
                        photo_data.save(buffered, format="JPEG")
                    else:
                        Image.open(BytesIO(photo_data)).save(buffered, format="JPEG")
                        
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    return {
                        "status": "success",
                        "strategy_used": "Secure QR",
                        "data": decoded_data,
                        "photo_base64": img_base64
                    }
                except IndexError as ie:
                    import traceback
                    return {
                        "status": "failed",
                        "error": f"QR data format error (IndexError): {str(ie)}. The QR may be corrupted or invalid.",
                        "qr_data_length": len(qr_data),
                        "qr_data_sample": qr_data[:100],
                        "full_traceback": traceback.format_exc()
                    }
                except ValueError as ve:
                    import traceback
                    return {
                        "status": "failed",
                        "error": f"QR data parsing error (ValueError): {str(ve)}. The QR may be malformed.",
                        "qr_data_length": len(qr_data),
                        "qr_data_sample": qr_data[:100],
                        "full_traceback": traceback.format_exc()
                    }
            else:
                 return {
                    "status": "failed", 
                    "error": "QR detected but data is not numeric (not a Secure Aadhaar QR).",
                    "raw_sample": qr_data[:20]
                }
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            return {
                "status": "failed", 
                "error": f"Parsing Error: {str(e)}", 
                "qr_data_length": len(qr_data) if qr_data else 0,
                "qr_data_sample": qr_data[:100] if qr_data else None,
                "full_traceback": error_traceback
            }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        return {
            "status": "failed",
            "error": f"Unexpected Error in extract_aadhaar_data: {str(e)}",
            "full_traceback": error_traceback,
            "image_path": image_path
        }