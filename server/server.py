"""
Face Duplicate Check API — runs on the secondary server
Path: /workspace/AI_Agent_Verification/app/face_duplicate_api.py

Start with:
  pip install fastapi uvicorn numpy opencv-python deepface httpx python-multipart
  uvicorn face_duplicate_api:app --host 0.0.0.0 --port 8001
"""

import os
import base64
import pickle
import numpy as np
import cv2
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from deepface import DeepFace

app = FastAPI(title="Face Duplicate Check API")

EMBEDDINGS_DIR = "/workspace/AI_Agent_Verification/app/extracted_data"
SELFIE_PKL = os.path.join(EMBEDDINGS_DIR, "selfie_embeddings.pkl")
AADHAAR_PKL = os.path.join(EMBEDDINGS_DIR, "aadhaar_embeddings.pkl")

SIMILARITY_THRESHOLD = float(os.environ.get("FACE_SIMILARITY_THRESHOLD", "0.81"))
DEEPFACE_MODEL = os.environ.get("DEEPFACE_MODEL", "Facenet512")


# -- Request / Response Models ------------------------------------------------

class CheckImageRequest(BaseModel):
    """Base64 image input."""
    user_id: str | int
    image_base64: str


class CheckUrlRequest(BaseModel):
    """URL image input."""
    user_id: str | int
    image_url: str


class CheckResponse(BaseModel):
    duplicate: bool
    matched_user_id: str | None = None
    matched_source: str | None = None
    similarity: float | None = None
    message: str


class DeleteResponse(BaseModel):
    user_id: str
    deleted_from: list[str]
    message: str


# -- Helpers -------------------------------------------------------------------

def load_pkl(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, dict) else {}


def save_pkl(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def decode_base64_to_image(image_base64: str) -> np.ndarray:
    """Decode a base64 string to an OpenCV BGR image."""
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img


def download_image_from_url(url: str) -> np.ndarray:
    """Download an image from a URL and return it as an OpenCV BGR image."""
    try:
        print(f"Downloading image from URL: {url}")
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP {e.response.status_code} when fetching image URL")
    except httpx.RequestError as e:
        raise ValueError(f"Network error fetching image URL: {e}")

    np_arr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image downloaded from URL")
    print(f"Image downloaded and decoded: {img.shape}")
    return img


def extract_embedding(img: np.ndarray) -> np.ndarray | None:
    """Extract a face embedding using DeepFace."""
    try:
        print(f"Extracting embedding: image shape={img.shape}, model={DEEPFACE_MODEL}")
        results = DeepFace.represent(
            img_path=img,
            model_name=DEEPFACE_MODEL,
            enforce_detection=False,
            detector_backend="mtcnn",
        )
        if results and len(results) > 0:
            embedding = np.array(results[0]["embedding"], dtype=np.float32)
            print(f"Embedding extracted: {embedding.shape[0]}-dim")
            return embedding
        print("No embedding results returned")
        return None
    except Exception as e:
        print(f"DeepFace error: {e}")
        return None


def find_duplicate(
    embedding: np.ndarray,
    user_id: str,
    stored: dict,
    source_name: str,
) -> dict | None:
    """Scan stored embeddings and return the best cosine-similarity match (if any)."""
    best_match = None
    best_sim = 0.0

    for stored_uid, stored_emb in stored.items():
        if str(stored_uid) == user_id:
            continue

        stored_vec = np.array(stored_emb, dtype=np.float32).flatten()
        if stored_vec.shape != embedding.shape:
            continue

        sim = cosine_similarity(embedding, stored_vec)
        if sim >= SIMILARITY_THRESHOLD and sim > best_sim:
            best_sim = sim
            best_match = {
                "matched_user_id": str(stored_uid),
                "matched_source": source_name,
                "similarity": round(sim, 4),
            }

    return best_match


def _run_duplicate_check(user_id: str, img: np.ndarray) -> CheckResponse:
    """Shared logic: extract embedding, compare, store if new."""

    embedding = extract_embedding(img)
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in image")

    selfie_store = load_pkl(SELFIE_PKL)
    aadhaar_store = load_pkl(AADHAAR_PKL)

    match = find_duplicate(embedding, user_id, selfie_store, "selfie")
    if not match:
        match = find_duplicate(embedding, user_id, aadhaar_store, "aadhaar")

    if match:
        return CheckResponse(
            duplicate=True,
            matched_user_id=match["matched_user_id"],
            matched_source=match["matched_source"],
            similarity=match["similarity"],
            message=(
                f"Face matches user {match['matched_user_id']} "
                f"in {match['matched_source']} embeddings"
            ),
        )

    selfie_store[user_id] = embedding.tolist()
    save_pkl(SELFIE_PKL, selfie_store)

    return CheckResponse(
        duplicate=False,
        message="No duplicate found. Embedding saved.",
    )


# -- Endpoints -----------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": DEEPFACE_MODEL,
        "selfie_embeddings_count": len(load_pkl(SELFIE_PKL)),
        "aadhaar_embeddings_count": len(load_pkl(AADHAAR_PKL)),
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }


@app.post("/check-duplicate-image", response_model=CheckResponse)
def check_duplicate_image(req: CheckImageRequest):
    """
    Check for duplicate face using a **base64-encoded** image.

    Request body:
        user_id      – unique user identifier
        image_base64 – base64-encoded image (with or without data-URI prefix)
    """
    user_id = str(req.user_id)
    print(f"[check-duplicate-image] user_id={user_id}, base64 len={len(req.image_base64)}")

    try:
        img = decode_base64_to_image(req.image_base64)
        print(f"Image decoded: {img.shape}")
    except ValueError as e:
        print(f"Image decode failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    return _run_duplicate_check(user_id, img)


@app.post("/check-duplicate-url", response_model=CheckResponse)
def check_duplicate_url(req: CheckUrlRequest):
    """
    Check for duplicate face using an **image URL**.

    Request body:
        user_id   – unique user identifier
        image_url – publicly accessible URL of the face image
    """
    user_id = str(req.user_id)
    print(f"[check-duplicate-url] user_id={user_id}, url={req.image_url}")

    try:
        img = download_image_from_url(req.image_url)
        print(f"Image ready: {img.shape}")
    except ValueError as e:
        print(f"Image download failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    return _run_duplicate_check(user_id, img)


@app.post("/check-duplicate-image-file", response_model=CheckResponse)
async def check_duplicate_image_file(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Check for duplicate face using a **direct file upload**.

    Form fields:
        user_id – unique user identifier
        file    – image file (JPEG, PNG, WebP, BMP, etc.)

    Example curl:
        curl -X POST http://localhost:8001/check-duplicate-image-file \\
             -F "user_id=123" \\
             -F "file=@/path/to/photo.jpg"
    """
    print(
        f"[check-duplicate-image-file] user_id={user_id}, "
        f"filename={file.filename}, content_type={file.content_type}"
    )

    # Validate content type (optional but recommended)
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {', '.join(allowed_types)}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode uploaded image. Ensure the file is a valid image.",
        )

    print(f"Image decoded from file upload: {img.shape}")
    return _run_duplicate_check(user_id, img)


@app.delete("/embeddings/{user_id}", response_model=DeleteResponse)
def delete_embeddings(user_id: str):
    """Delete all stored embeddings (selfie + aadhaar) for the given user_id."""
    print(f"Delete request for user_id={user_id}")

    deleted_from: list[str] = []

    selfie_store = load_pkl(SELFIE_PKL)
    if user_id in selfie_store:
        del selfie_store[user_id]
        save_pkl(SELFIE_PKL, selfie_store)
        deleted_from.append("selfie")
        print(f"Removed {user_id} from selfie embeddings")

    aadhaar_store = load_pkl(AADHAAR_PKL)
    if user_id in aadhaar_store:
        del aadhaar_store[user_id]
        save_pkl(AADHAAR_PKL, aadhaar_store)
        deleted_from.append("aadhaar")
        print(f"Removed {user_id} from aadhaar embeddings")

    if not deleted_from:
        raise HTTPException(
            status_code=404,
            detail=f"No embeddings found for user_id={user_id}",
        )

    return DeleteResponse(
        user_id=user_id,
        deleted_from=deleted_from,
        message=f"Embeddings deleted from: {', '.join(deleted_from)}",
    )