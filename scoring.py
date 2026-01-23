import re
from datetime import datetime

class VerificationScorer:
    def __init__(self):
        # Weights (Total 100) - Updated Configuration
        self.weights = {
            "aadhar": 35,           # Valid Aadhaar Number
            "dob": 35,              # Valid Age/DOB
            "gender": 15,           # Gender Match (Input vs Aadhaar)
            "face_gender": 15       # Face Gender Match (Selfie vs Aadhaar using Qwen)
        }
    
    def _normalize_dob(self, dob_string):
        """Standardizes date formats (dd-mm-yyyy, yyyy-mm-dd) to yyyy-mm-dd"""
        if not dob_string: return None
        formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d.%m.%Y', '%Y/%m/%d']
        for fmt in formats:
            try:
                return datetime.strptime(str(dob_string).strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None
    
    def _extract_year(self, dob_string):
        """Extract year from DOB string - handles both full dates and year-only formats"""
        if not dob_string:
            return None
        
        dob_str = str(dob_string).strip()
        
        # Try to parse as full date first
        normalized = self._normalize_dob(dob_str)
        if normalized:
            return normalized.split('-')[0]  # Extract year from yyyy-mm-dd
        
        # If not a full date, look for a 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', dob_str)
        if year_match:
            return year_match.group(0)
        
        return None

    def calculate_score(self, face_data, entity_data, expected_gender, expected_dob=None, qwen_face_result=None, face_gender_match=None):
        score = 0
        breakdown = {}
        rejection_reasons = []
        critical_failure = False  # If True, status is REJECTED regardless of score
        qwen_override = False  # Flag to track if Qwen provided override decision


        # --- 1. Aadhaar Validity & Masking (0 - 40 points) ---
        aadhaar_num = entity_data.get("aadharnumber", "").replace(" ", "")
        aadhaar_original = entity_data.get("aadharnumber", "")
        
        # Check for Masking or Invalid Aadhaar
        is_masked = "X" in aadhaar_original.upper()
        is_invalid = not (len(aadhaar_num) == 12 and aadhaar_num.isdigit())
        
        if is_masked or is_invalid:
            # Masked or Invalid Aadhaar -> Critical Failure (Auto REJECT)
            breakdown["aadhar_score"] = 0
            critical_failure = True
            
            if is_masked:
                rejection_reasons.append(f"Masked Aadhaar Number Detected ({aadhaar_original})")
            else:
                rejection_reasons.append("Invalid/Unreadable Aadhaar Number")
        else:
            # Valid 12-digit Aadhaar
            score += self.weights["aadhar"]
            breakdown["aadhar_score"] = self.weights["aadhar"]

        # --- 2. DOB & Age Check (0 - 40 points) ---
        extracted_dob = entity_data.get("dob", "")
        age_status = entity_data.get("age_status") # Calculated in entity.py
        
        dob_score = 0
        # Rule A: Must be 18+
        if age_status == "age_approved":
            # Rule B: If Input DOB provided, it MUST match Aadhaar DOB (YEAR-BASED MATCHING)
            if expected_dob and extracted_dob:
                input_year = self._extract_year(expected_dob)
                extracted_year = self._extract_year(extracted_dob)
                
                if input_year and extracted_year:
                    if input_year == extracted_year:
                        # Year matches - PASS
                        dob_score = self.weights["dob"]
                    else:
                        # Year Mismatch -> Critical Failure
                        dob_score = 0
                        critical_failure = True
                        rejection_reasons.append(f"Birth Year Mismatch (Input Year: {input_year} vs Aadhaar Year: {extracted_year})")
                else:
                    # Could not extract year from one or both - If Age approved, give benefit of doubt
                    dob_score = self.weights["dob"]
            else:
                # No expected_dob provided or no extracted_dob - If Age approved, pass
                dob_score = self.weights["dob"]
        else:
            dob_score = 0
            critical_failure = True
            rejection_reasons.append("User is Under 18 or DOB Unreadable")
            
        score += dob_score
        breakdown["dob_score"] = dob_score

        # --- 3. Gender Check (0 - 20 points) ---
        # Normalize Helper
        def clean_gender(g):
            if not g: return "unknown"
            g = g.lower()
            if "female" in g: return "female"
            if "male" in g: return "male"
            return "other"

        extracted_gen = entity_data.get("gender", "Other")
        input_gen = expected_gender if expected_gender else ""
        
        norm_extracted = clean_gender(extracted_gen)
        norm_input = clean_gender(input_gen)

        gender_score = 0
        
        # Case A: OCR detected Male/Female
        if norm_extracted in ["male", "female"]:
            if norm_input and norm_input == norm_extracted:
                gender_score = self.weights["gender"]
            elif norm_input:
                # Mismatch -> Critical Failure
                gender_score = 0
                critical_failure = True
                rejection_reasons.append(f"Gender Mismatch (Input: {norm_input} vs OCR: {norm_extracted})")
        
        # Case B: OCR is 'Other' or 'Not Detected' -> Gender should come from entity_data (via gender_pipeline in main.py)
        else:
            # If OCR failed, the main.py should have already tried gender_pipeline
            # and updated entity_data['gender'] with the fallback result
            # So if we're here and gender is still 'Other', it means all methods failed
            if norm_input:
                # We have expected gender but couldn't verify it
                critical_failure = True
                rejection_reasons.append("Gender could not be verified from document (OCR failed)")
            else:
                # No expected gender provided, can't verify
                # Don't fail, but give 0 points
                pass

        score += gender_score
        breakdown["gender_score"] = gender_score

        # --- 4. Face Gender Match (0 - 15 points) ---
        # Qwen-based selfie gender verification
        face_gender_score = 0
        if face_gender_match is not None:
            if face_gender_match:
                # Selfie gender matches Aadhaar gender
                face_gender_score = self.weights["face_gender"]
            else:
                # Selfie gender doesn't match - give 0 points but don't reject
                face_gender_score = 0
                # Note: We don't set critical_failure here, just reduce score
        
        score += face_gender_score
        breakdown["face_gender_score"] = face_gender_score

        # --- FINAL DECISION ---
        if critical_failure:
            status = "REJECTED"
        elif score > 60:
            status = "APPROVED"
        elif score >= 40:
            status = "REVIEW"
        else:
            status = "REJECTED"

        return {
            "total_score": round(score, 2),
            "status": status,
            "breakdown": breakdown,
            "rejection_reasons": rejection_reasons
        }