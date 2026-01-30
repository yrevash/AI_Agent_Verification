# CHANGES.md

## Face Embeddings + Aadhaar Front/Back Cross-Check Integration

### Overview

Three categories of changes to the verification pipeline:

1. **Replaced Qwen face similarity with YOLOv12+DeepFace embedding-based verification** — includes duplicate detection via stored embeddings
2. **Split Aadhaar extraction into separate front/back calls** — enables cross-checking Aadhaar numbers and VID numbers between front and back
3. **Removed all gender verification** — face embeddings inherently reject mismatched identities (different-gender faces won't produce matching embeddings), making separate gender checks redundant
4. **Updated scoring** — three components only: Aadhaar (30pts), DOB (30pts), face similarity (40pts)

Qwen handles Aadhaar document OCR only (no PAN). YOLOv12 + DeepFace handles all face/identity verification.

No API endpoint changes. Only internal logic modifications.

---

### Files Modified

#### `app/faceEmbeddings.py`

| Change | Before | After |
|--------|--------|-------|
| Model initialization | Module-level `download_model()` + `face_detector = YOLO(...)` on import | Lazy init via `get_face_detector()` + explicit `initialize_face_detector()` for lifespan |
| `detect_face()` | Accepts `str` (file path) only | Accepts `Union[str, io.BytesIO]` — seeks to 0 for BytesIO |
| `verify_faces_memory()` | Did not exist | New function: accepts BytesIO, skips disk saves, returns similarity + embedding for duplicate check |
| `verify_faces()` | Used module-level `face_detector` | Uses `get_face_detector()` (still saves crops to disk for CLI usage) |

#### `batch.py`

| Change | Before | After |
|--------|--------|-------|
| Aadhaar extraction | Single `extract_aadhaar_data()` with both images | Separate `extract_aadhaar_front()` and `extract_aadhaar_back()` |
| Cross-check | None | `cross_check_aadhaar()` validates Aadhaar number + VID match between front/back |
| Face similarity | Qwen-based similarity prompt (`similarity_percentage: 0-100`) | YOLOv12 face detection + DeepFace Facenet embedding cosine similarity |
| Duplicate detection | None | Via `verify_faces_memory()` — checks selfie embedding against stored database |
| Gender verification | Qwen selfie gender detection + GenderPipeline | Removed entirely — embeddings handle identity mismatch |
| Lifespan init | Qwen + GenderPipeline + Scorer | Qwen + Scorer + Face Detector (no GenderPipeline) |
| Health endpoint | `qwen_agent`, `gender_pipeline` | `qwen_agent`, `face_verifier` |
| `extracted_data` response | aadhaar, name, dob, address, gender | + `vid_number`, `face_similarity`, `face_verified`, `is_duplicate_face`, `aadhaar_cross_check_passed` |
| Post-scoring overrides | Gender mismatch → REVIEW, low Qwen similarity → REVIEW | Face not verified → REJECT, no face detected → REVIEW |

#### `scoring.py`

| Change | Before | After |
|--------|--------|-------|
| Weights | Aadhaar 35, DOB 35, Gender 15, Face Gender 15 | Aadhaar 30, DOB 30, Face Similarity 40 |
| `calculate_score()` signature | `face_data, entity_data, expected_gender, expected_dob, qwen_face_result, face_gender_match` | `entity_data, expected_dob, face_verification_result, cross_check_failures` |
| Gender scoring | Two sections (input vs OCR + selfie vs Aadhaar) | Removed entirely |
| Face similarity scoring | None | 40pts if verified, partial (40%) if similarity >= 0.4 but below threshold, 0 otherwise |
| Cross-check failures | None | Aadhaar number mismatch and VID mismatch as critical failures |
| Duplicate face | None | Critical failure (auto-reject) |

#### `CLAUDE.md`

Updated architecture docs, pipeline flow, key files table, scoring weights, verification checks. Removed gender pipeline references.

---

### Pipeline Flow

```
Step 1a: Extract Aadhaar FRONT via Qwen (aadharnumber, name, dob, gender, vid, is_masked)
Step 1b: Extract Aadhaar BACK via Qwen (aadharnumber, address, pincode, vid, is_masked)
Step 1c: Cross-check front/back numbers (Aadhaar + VID) → rejection if mismatch
Step 1d: Masked Aadhaar check → rejection if masked
Step 2:  Face embedding verification (YOLOv12 + DeepFace)
         - Cosine similarity threshold: 0.55
         - Duplicate detection threshold: 0.85
         - Inherently catches gender mismatches via embedding comparison
Step 3:  Score calculation (3 components + critical failures)
Step 4:  Build response (enriched extracted_data)
```

### Scoring Weights (Before → After)

| Component | Before | After |
|-----------|--------|-------|
| Aadhaar validity | 35 | 30 |
| DOB/Age | 35 | 30 |
| Gender match | 15 | Removed |
| Face gender match | 15 | Removed |
| Face similarity | - | 40 |
| **Total** | **100** | **100** |

### Critical Failures (Auto-Reject)

- Masked Aadhaar (existing)
- Underage / DOB unreadable (existing)
- Birth year mismatch (existing)
- **Aadhaar number mismatch front/back** (new)
- **VID number mismatch front/back** (new)
- **Duplicate face detected** (new)
