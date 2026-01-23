import cv2
import numpy as np
import os
import requests

class QRCropperEnhancer:
    def __init__(self):
        self.models_dir = "models"
        self._setup_models()
        
        # Initialize WeChat Detector (Best for finding the box)
        self.detector = cv2.wechat_qrcode_WeChatQRCode(
            os.path.join(self.models_dir, "detect_2021nov.prototxt"),
            os.path.join(self.models_dir, "detect_2021nov.caffemodel"),
            os.path.join(self.models_dir, "sr_2021nov.prototxt"),
            os.path.join(self.models_dir, "sr_2021nov.caffemodel")
        )

    def _setup_models(self):
        """Auto-downloads the required AI models if missing."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        # Use Hugging Face (most reliable)
        urls = {
            "detect_2021nov.prototxt": "https://huggingface.co/opencv/qrcode_wechatqrcode/resolve/main/detect_2021nov.prototxt",
            "detect_2021nov.caffemodel": "https://huggingface.co/opencv/qrcode_wechatqrcode/resolve/main/detect_2021nov.caffemodel",
            "sr_2021nov.prototxt": "https://huggingface.co/opencv/qrcode_wechatqrcode/resolve/main/sr_2021nov.prototxt",
            "sr_2021nov.caffemodel": "https://huggingface.co/opencv/qrcode_wechatqrcode/resolve/main/sr_2021nov.caffemodel",
        }

        for name, url in urls.items():
            path = os.path.join(self.models_dir, name)
            if not os.path.exists(path):
                print(f"Downloading {name}...")
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    with open(path, "wb") as f:
                        f.write(response.content)
                    print(f"✓ Downloaded {name}")
                except Exception as e:
                    print(f"✗ Failed to download {name}: {e}")

    def process_image(self, image_path, output_path="enhanced_crop.jpg"):
        """
        1. Finds QR
        2. Crops it with padding
        3. Saves result
        """
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not read image.")
            return

        print(f"Processing: {image_path}")

        # 1. Detect QR Bounding Box
        _, points = self.detector.detectAndDecode(img)

        if not points:
            print("No QR code detected to crop.")
            # Fallback: Try cropping the center 50% if detection fails
            h, w = img.shape[:2]
            crop_img = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            print("Warning: Using center crop fallback.")
        else:
            # 2. Extract the Box
            pt = points[0] 
            
            # Find min/max coordinates
            x_min = int(min(pt[:, 0]))
            x_max = int(max(pt[:, 0]))
            y_min = int(min(pt[:, 1]))
            y_max = int(max(pt[:, 1]))

            # Add Padding (Quiet Zone is essential for decoding)
            padding = 20
            h, w = img.shape[:2]
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            print(f"QR Found! Cropping to box: {x_min},{y_min} -> {x_max},{y_max}")
            crop_img = img[y_min:y_max, x_min:x_max]

        # 3. Save
        cv2.imwrite(output_path, crop_img)
        print(f"Success! Cropped image saved to: {output_path}")
        return output_path

# --- USAGE ---
if __name__ == "__main__":
    processor = QRCropperEnhancer()
    
    # Replace this with your actual image path
    target_image = "test.png" 
    
    if os.path.exists(target_image):
        processor.process_image(target_image, "final_enhanced_qr.jpg")
    else:
        print(f"File not found: {target_image}")