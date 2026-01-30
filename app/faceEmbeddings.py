"""
Face Verification System
Compares a selfie image with a face detected in an Aadhaar card image.
Uses YOLOv12 for face detection and DeepFace for face embedding comparison.
Stores verified face embeddings and checks for duplicates.
"""

from ultralytics import YOLO
from PIL import Image
import os
import io
import json
import numpy as np
import pickle
import uuid
import urllib.request
from typing import Union

# Force TensorFlow (used by DeepFace) to CPU to avoid CUDA_ERROR_INVALID_HANDLE
# when PyTorch (YOLO) already holds the GPU context
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

# For face embedding and comparison
from deepface import DeepFace

# Configuration
SIMILARITY_THRESHOLD = 0.50
DUPLICATE_THRESHOLD = 0.85  # Higher threshold for duplicate detection
REPO_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(REPO_DIR, "model")
MODEL_NAME = "yolov12l-face.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
MODEL_URL = "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt"
OUTPUT_DIR = os.path.join(REPO_DIR, "extracted_data")
SELFIE_EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "selfie_embeddings.pkl")
AADHAAR_EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "aadhaar_embeddings.pkl")

# Lazy-initialized face detector
_face_detector = None


def download_model():
    """Download the YOLO face detection model if not present."""
    if os.path.exists(MODEL_PATH):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model downloaded to {MODEL_PATH}")


def get_face_detector() -> YOLO:
    """Get the face detector, initializing lazily if needed."""
    global _face_detector
    if _face_detector is None:
        initialize_face_detector()
    return _face_detector


def initialize_face_detector():
    """Explicitly initialize the face detector. Call during server lifespan startup."""
    global _face_detector
    download_model()
    _face_detector = YOLO(MODEL_PATH)
    print(f"Face detector initialized: {MODEL_PATH}")


def _load_db(filepath: str) -> dict:
    """Load embeddings database from a pkl file."""
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return {}


def _save_db(db: dict, filepath: str) -> None:
    """Save embeddings database to a pkl file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(db, f)


def load_selfie_db() -> dict:
    return _load_db(SELFIE_EMBEDDINGS_FILE)

def load_aadhaar_db() -> dict:
    return _load_db(AADHAAR_EMBEDDINGS_FILE)

def save_selfie_embedding(user_id: str, embedding: np.ndarray) -> None:
    db = load_selfie_db()
    db[user_id] = embedding
    _save_db(db, SELFIE_EMBEDDINGS_FILE)

def save_aadhaar_embedding(user_id: str, embedding: np.ndarray) -> None:
    db = load_aadhaar_db()
    db[user_id] = embedding
    _save_db(db, AADHAAR_EMBEDDINGS_FILE)


def _check_duplicate_in_db(embedding: np.ndarray, db: dict) -> tuple[bool, str | None, float]:
    """
    Check if a face embedding already exists in a database.

    Returns:
        tuple: (is_duplicate, matching_user_id, similarity_score)
    """
    if not db:
        return False, None, 0.0

    max_similarity = 0.0
    matching_user_id = None

    for user_id, stored_embedding in db.items():
        similarity = cosine_similarity(embedding, stored_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            matching_user_id = user_id

    is_duplicate = max_similarity >= DUPLICATE_THRESHOLD
    return is_duplicate, matching_user_id if is_duplicate else None, float(max_similarity)


def check_all_duplicates(selfie_embedding: np.ndarray, aadhaar_embedding: np.ndarray) -> dict:
    """
    Check both selfie and Aadhaar embeddings against their respective databases.

    Returns:
        dict with duplicate info for both selfie and Aadhaar face
    """
    selfie_db = load_selfie_db()
    aadhaar_db = load_aadhaar_db()

    selfie_dup, selfie_dup_user, selfie_dup_sim = _check_duplicate_in_db(selfie_embedding, selfie_db)
    aadhaar_dup, aadhaar_dup_user, aadhaar_dup_sim = _check_duplicate_in_db(aadhaar_embedding, aadhaar_db)

    return {
        "selfie_duplicate": selfie_dup,
        "selfie_duplicate_user_id": selfie_dup_user,
        "selfie_duplicate_similarity": round(selfie_dup_sim, 4),
        "aadhaar_duplicate": aadhaar_dup,
        "aadhaar_duplicate_user_id": aadhaar_dup_user,
        "aadhaar_duplicate_similarity": round(aadhaar_dup_sim, 4),
        "is_duplicate": selfie_dup or aadhaar_dup,
    }


def detect_face(image_input: Union[str, io.BytesIO]) -> tuple[bool, list | None, np.ndarray | None]:
    """
    Detect face in an image using YOLOv12.
    Accepts a file path (str) or in-memory BytesIO object.

    Returns:
        tuple: (face_detected, face_box [x, y, w, h], cropped_face_array)
    """
    detector = get_face_detector()

    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        img = Image.open(image_input).convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")

    output = detector(img)

    res0 = output[0]
    boxes = []

    if hasattr(res0, "boxes"):
        try:
            xy = res0.boxes.xyxy.cpu().numpy()
            confs = res0.boxes.conf.cpu().numpy()
        except Exception:
            xy = res0.boxes.xyxy.numpy()
            confs = res0.boxes.conf.numpy()

        for b, c in zip(xy, confs):
            boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(c)))

    if not boxes:
        return False, None, None

    # Select highest confidence detection
    boxes.sort(key=lambda t: (t[4], (t[2] - t[0]) * (t[3] - t[1])), reverse=True)
    x1, y1, x2, y2, conf = boxes[0]

    # Add 10px padding on all sides (clamped to image bounds)
    padding = 10
    img_width, img_height = img.size
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(img_width, x2 + padding)
    y2_padded = min(img_height, y2 + padding)

    # Crop face with padding
    face_crop = img.crop((x1_padded, y1_padded, x2_padded, y2_padded))
    face_array = np.array(face_crop)

    face_box = [x1_padded, y1_padded, x2_padded - x1_padded, y2_padded - y1_padded]
    return True, face_box, face_array


def get_face_embedding(face_image: np.ndarray) -> np.ndarray:
    """
    Extract face embedding using DeepFace.
    Uses Facenet model for robust embeddings.
    """
    embedding = DeepFace.represent(
        face_image,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="skip"  # Already detected and cropped
    )
    return np.array(embedding[0]["embedding"])


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)


def verify_faces_memory(selfie_img: io.BytesIO, aadhaar_img: io.BytesIO, user_id: str = None) -> dict:
    """
    Verify if the face in selfie matches the face in Aadhaar card.
    In-memory version: accepts BytesIO objects, skips saving cropped images to disk.
    If verified, checks for duplicates and saves embedding.

    Args:
        selfie_img: BytesIO selfie image
        aadhaar_img: BytesIO Aadhaar card image
        user_id: Optional user ID (auto-generated if not provided)

    Returns:
        dict with verification result including similarity_score, verified, duplicate info
    """
    result = {
        "verified": False,
        "similarity_score": 0.0,
        "threshold": SIMILARITY_THRESHOLD,
        "selfie_face_detected": False,
        "aadhaar_face_detected": False,
        "is_duplicate": False,
        "duplicate_user_id": None,
        "duplicate_similarity": 0.0,
        "user_id": None,
        "embedding_saved": False,
        "error": None
    }

    # Detect face in selfie
    selfie_detected, selfie_box, selfie_face = detect_face(selfie_img)
    result["selfie_face_detected"] = selfie_detected

    if not selfie_detected:
        result["error"] = "No face detected in selfie image"
        return result

    # Detect face in Aadhaar card
    aadhaar_detected, aadhaar_box, aadhaar_face = detect_face(aadhaar_img)
    result["aadhaar_face_detected"] = aadhaar_detected

    if not aadhaar_detected:
        result["error"] = "No face detected in Aadhaar card image"
        return result

    # Get face embeddings
    try:
        selfie_embedding = get_face_embedding(selfie_face)
        aadhaar_embedding = get_face_embedding(aadhaar_face)
    except Exception as e:
        result["error"] = f"Failed to extract face embeddings: {str(e)}"
        return result

    # Check for duplicates FIRST — before similarity check
    # Checks selfie against selfie DB AND Aadhaar face against Aadhaar face DB
    dup_result = check_all_duplicates(selfie_embedding, aadhaar_embedding)
    result["is_duplicate"] = dup_result["is_duplicate"]
    result["selfie_duplicate"] = dup_result["selfie_duplicate"]
    result["selfie_duplicate_user_id"] = dup_result["selfie_duplicate_user_id"]
    result["aadhaar_duplicate"] = dup_result["aadhaar_duplicate"]
    result["aadhaar_duplicate_user_id"] = dup_result["aadhaar_duplicate_user_id"]

    if dup_result["is_duplicate"]:
        reasons = []
        if dup_result["selfie_duplicate"]:
            reasons.append(f"Selfie matches user: {dup_result['selfie_duplicate_user_id']}")
        if dup_result["aadhaar_duplicate"]:
            reasons.append(f"Aadhaar face matches user: {dup_result['aadhaar_duplicate_user_id']}")
        result["duplicate_user_id"] = dup_result["selfie_duplicate_user_id"] or dup_result["aadhaar_duplicate_user_id"]
        result["error"] = f"Duplicate detected: {'; '.join(reasons)}"
        result["embedding_saved"] = False
        return result

    # Calculate similarity between selfie and Aadhaar
    similarity = cosine_similarity(selfie_embedding, aadhaar_embedding)
    result["similarity_score"] = round(float(similarity), 4)

    # Verify based on threshold
    result["verified"] = bool(similarity >= SIMILARITY_THRESHOLD)

    # If verified, save embeddings to separate databases
    if result["verified"]:
        if user_id is None:
            user_id = str(uuid.uuid4())[:8]

        save_selfie_embedding(user_id, selfie_embedding)
        save_aadhaar_embedding(user_id, aadhaar_embedding)
        result["user_id"] = user_id
        result["embedding_saved"] = True

    return result


def verify_faces(selfie_path: str, aadhaar_path: str, user_id: str = None) -> dict:
    """
    Verify if the face in selfie matches the face in Aadhaar card.
    File-path version for standalone CLI usage.
    If verified, checks for duplicates and saves embedding.

    Args:
        selfie_path: Path to selfie image
        aadhaar_path: Path to Aadhaar card image
        user_id: Optional user ID (auto-generated if not provided)

    Returns:
        dict with verification result
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = {
        "verified": False,
        "similarity_score": 0.0,
        "threshold": SIMILARITY_THRESHOLD,
        "selfie_face_detected": False,
        "aadhaar_face_detected": False,
        "is_duplicate": False,
        "duplicate_user_id": None,
        "duplicate_similarity": 0.0,
        "user_id": None,
        "embedding_saved": False,
        "error": None
    }

    # Detect face in selfie
    selfie_detected, selfie_box, selfie_face = detect_face(selfie_path)
    result["selfie_face_detected"] = selfie_detected
    result["selfie_face_box"] = selfie_box

    if not selfie_detected:
        result["error"] = "No face detected in selfie image"
        return result

    # Save cropped selfie face
    selfie_crop_path = os.path.join(OUTPUT_DIR, "selfie_face_crop.jpg")
    Image.fromarray(selfie_face).save(selfie_crop_path)
    result["selfie_crop_path"] = selfie_crop_path

    # Detect face in Aadhaar card
    aadhaar_detected, aadhaar_box, aadhaar_face = detect_face(aadhaar_path)
    result["aadhaar_face_detected"] = aadhaar_detected
    result["aadhaar_face_box"] = aadhaar_box

    if not aadhaar_detected:
        result["error"] = "No face detected in Aadhaar card image"
        return result

    # Save cropped Aadhaar face
    aadhaar_crop_path = os.path.join(OUTPUT_DIR, "aadhaar_face_crop.jpg")
    Image.fromarray(aadhaar_face).save(aadhaar_crop_path)
    result["aadhaar_crop_path"] = aadhaar_crop_path

    
    
    # Get face embeddings
    try:
        selfie_embedding = get_face_embedding(selfie_face)
        aadhaar_embedding = get_face_embedding(aadhaar_face)
    except Exception as e:
        result["error"] = f"Failed to extract face embeddings: {str(e)}"
        return result

    # Check for duplicates FIRST — before similarity check
    # Checks selfie against selfie DB AND Aadhaar face against Aadhaar face DB
    dup_result = check_all_duplicates(selfie_embedding, aadhaar_embedding)
    result["is_duplicate"] = dup_result["is_duplicate"]
    result["duplicate_user_id"] = dup_result["selfie_duplicate_user_id"] or dup_result["aadhaar_duplicate_user_id"]
    result["duplicate_similarity"] = max(dup_result["selfie_duplicate_similarity"], dup_result["aadhaar_duplicate_similarity"])

    if dup_result["is_duplicate"]:
        reasons = []
        if dup_result["selfie_duplicate"]:
            reasons.append(f"Selfie matches user: {dup_result['selfie_duplicate_user_id']}")
        if dup_result["aadhaar_duplicate"]:
            reasons.append(f"Aadhaar face matches user: {dup_result['aadhaar_duplicate_user_id']}")
        result["error"] = f"Duplicate detected: {'; '.join(reasons)}"
        result["embedding_saved"] = False
        return result

    # Calculate similarity between selfie and Aadhaar
    similarity = cosine_similarity(selfie_embedding, aadhaar_embedding)
    result["similarity_score"] = round(float(similarity), 4)

    # Verify based on threshold
    result["verified"] = bool(similarity >= SIMILARITY_THRESHOLD)

    # If verified, save embeddings to separate databases
    if result["verified"]:
        if user_id is None:
            user_id = str(uuid.uuid4())[:8]

        save_selfie_embedding(user_id, selfie_embedding)
        save_aadhaar_embedding(user_id, aadhaar_embedding)
        result["user_id"] = user_id
        result["embedding_saved"] = True

    return result


def list_registered_users() -> dict:
    """List all registered user IDs from both selfie and Aadhaar databases."""
    selfie_db = load_selfie_db()
    aadhaar_db = load_aadhaar_db()
    return {
        "selfie_users": list(selfie_db.keys()),
        "aadhaar_users": list(aadhaar_db.keys()),
        "all_users": list(set(selfie_db.keys()) | set(aadhaar_db.keys()))
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python main.py <selfie_image_path> <aadhaar_image_path> [user_id]")
        print("Example: python main.py selfie.jpg aadhaar_card.jpg")
        print("Example with user_id: python main.py selfie.jpg aadhaar_card.jpg user123")
        print("\nTo list registered users: python main.py --list")
        sys.exit(1)

    if sys.argv[1] == "--list":
        users = list_registered_users()
        print(json.dumps({
            "selfie_users": users["selfie_users"],
            "aadhaar_users": users["aadhaar_users"],
            "all_users": users["all_users"],
            "count": len(users["all_users"])
        }, indent=2))
        sys.exit(0)

    selfie_path = sys.argv[1]
    aadhaar_path = sys.argv[2]
    user_id = sys.argv[3] if len(sys.argv) > 3 else None

    # Validate paths
    if not os.path.exists(selfie_path):
        print(json.dumps({"error": f"Selfie image not found: {selfie_path}"}))
        sys.exit(1)

    if not os.path.exists(aadhaar_path):
        print(json.dumps({"error": f"Aadhaar image not found: {aadhaar_path}"}))
        sys.exit(1)

    # Run verification
    result = verify_faces(selfie_path, aadhaar_path, user_id)

    # Simplified output - only essential fields
    output = {
        "verified": bool(result["verified"]),
        "similarity_score": float(result["similarity_score"]),
        "is_duplicate": bool(result["is_duplicate"]),
        "duplicate_user_id": result["duplicate_user_id"],
        "user_id": result["user_id"],
        "embedding_saved": bool(result["embedding_saved"])
    }

    # Add error if present
    if result.get("error"):
        output["error"] = result["error"]

    print(json.dumps(output, indent=2))
