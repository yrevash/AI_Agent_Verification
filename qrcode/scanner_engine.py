import cv2
import numpy as np
import zxingcpp

class RobustScanner:
    def __init__(self):
        pass

    def scan(self, image_bytes):
        """
        Scans an image and returns the raw QR code value.
        """
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "failed", "error": "Invalid image file"}

        # --- STRATEGY 1: ZXing-CPP (Best for Rotated/Clean Codes) ---
        results = zxingcpp.read_barcodes(img)
        for result in results:
            if result.text:
                return {
                    "status": "success",
                    "qr_value": result.text
                }
        
        # --- STRATEGY 2: Preprocessing + Binarization ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
        )
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        results = zxingcpp.read_barcodes(binary_bgr)
        for result in results:
            if result.text:
                return {
                    "status": "success",
                    "qr_value": result.text
                }
        
        return {"status": "failed", "error": "QR Code not detected. Ensure image is clear."}

# Create a singleton instance
scanner = RobustScanner()