import re
from datetime import datetime

class VerificationScorer:
    def __init__(self):
        # Weights (Total 100)
        # No gender checks — face embeddings inherently reject mismatched identities
        self.weights = {
            "aadhar": 30,           # Valid Aadhaar Number
            "dob": 30,              # Valid Age/DOB
            "face_similarity": 40   # Face Embedding Similarity (YOLOv12 + DeepFace)
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

    def calculate_score(self, entity_data, expected_dob=None,
                        face_verification_result=None, cross_check_failures=None):
        score = 0
        breakdown = {}
        rejection_reasons = []
        critical_failure = False  # If True, status is REJECTED regardless of score

        # --- 1. Aadhaar Validity & Masking (0 - 30 points) ---
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

        # --- 2. DOB & Age Check (0 - 30 points) ---
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

        # --- 3. Face Similarity (0 - 40 points) ---
        face_sim_score = 0
        if face_verification_result is not None:
            verified = face_verification_result.get("verified", False)
            is_duplicate = face_verification_result.get("is_duplicate", False)
            similarity = face_verification_result.get("similarity_score", 0.0)

            if is_duplicate:
                # Duplicate face -> Critical Failure (Auto REJECT)
                face_sim_score = 0
                critical_failure = True
                dup_user = face_verification_result.get("duplicate_user_id", "unknown")
                rejection_reasons.append(f"Duplicate Face Detected (matches user: {dup_user})")
            elif verified:
                # Face verified — full points
                face_sim_score = self.weights["face_similarity"]
            else:
                # Face not verified (below threshold) — partial points based on similarity
                if similarity >= 0.4:
                    face_sim_score = int(self.weights["face_similarity"] * 0.4)
                else:
                    face_sim_score = 0

        score += face_sim_score
        breakdown["face_similarity_score"] = face_sim_score

        # --- 4. Cross-Check Failures (Aadhaar front/back mismatch) ---
        if cross_check_failures:
            for failure in cross_check_failures:
                critical_failure = True
                rejection_reasons.append(failure)
            breakdown["cross_check_failed"] = True
        else:
            breakdown["cross_check_failed"] = False

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

    @staticmethod
    def _normalize_name(name_string):
        """Lowercase, strip punctuation, collapse whitespace -> set of tokens."""
        if not name_string:
            return set()
        cleaned = re.sub(r'[^a-z0-9\s]', ' ', str(name_string).lower())
        return set(t for t in cleaned.split() if t)

    def names_match(self, name_a, name_b, threshold=0.5):
        """
        Token-overlap name comparison tolerant of middle names / ordering / spacing.
        Returns True/False if both names are present, or None if either is missing
        (in which case the caller should skip the name check).
        """
        tokens_a = self._normalize_name(name_a)
        tokens_b = self._normalize_name(name_b)
        if not tokens_a or not tokens_b:
            return None
        smaller = min(len(tokens_a), len(tokens_b))
        overlap = len(tokens_a & tokens_b) / smaller if smaller else 0.0
        return overlap >= threshold

    def calculate_passport_score(self, entity_data, expected_dob=None,
                                 expected_name=None, face_verification_result=None):
        """
        Scoring for the PASSPORT (non-India) pipeline.

        Critical failures (auto-REJECT): DOB year mismatch, name mismatch.
        Face is NEVER a critical failure here — a face mismatch/duplicate routes to
        REVIEW (handled by the caller), it does not reject.
        Weights (Total 100): passport 20, dob 20, name 20, face 40.
        """
        weights = {"passport": 20, "dob": 20, "name": 20, "face_similarity": 40}
        score = 0
        breakdown = {}
        rejection_reasons = []
        critical_failure = False

        # --- 1. Passport number validity (0 - 20 points, non-critical) ---
        passport_num = str(entity_data.get("passport_number", "")).replace(" ", "")
        # Typical passport numbers are 6-9 alphanumeric characters
        passport_valid = bool(re.fullmatch(r'[A-Za-z0-9]{6,12}', passport_num))
        if passport_valid:
            score += weights["passport"]
            breakdown["passport_score"] = weights["passport"]
        else:
            breakdown["passport_score"] = 0
            rejection_reasons.append("Invalid/Unreadable Passport Number")

        # --- 2. DOB year match (0 - 20 points, CRITICAL on mismatch) ---
        extracted_dob = entity_data.get("dob", "")
        dob_score = 0
        if expected_dob and extracted_dob:
            input_year = self._extract_year(expected_dob)
            extracted_year = self._extract_year(extracted_dob)
            if input_year and extracted_year:
                if input_year == extracted_year:
                    dob_score = weights["dob"]
                else:
                    dob_score = 0
                    critical_failure = True
                    rejection_reasons.append(
                        f"Birth Year Mismatch (Input Year: {input_year} vs Passport Year: {extracted_year})"
                    )
            else:
                # Could not parse a year from one/both — benefit of the doubt
                dob_score = weights["dob"]
        else:
            # No expected or extracted DOB to compare — benefit of the doubt
            dob_score = weights["dob"]
        score += dob_score
        breakdown["dob_score"] = dob_score

        # --- 3. Name match (0 - 20 points, CRITICAL on mismatch) ---
        extracted_name = entity_data.get("name", "")
        name_score = 0
        name_result = self.names_match(expected_name, extracted_name)
        if name_result is None:
            # One side missing — cannot compare, give benefit of the doubt
            name_score = weights["name"]
            breakdown["name_checked"] = False
        elif name_result:
            name_score = weights["name"]
            breakdown["name_checked"] = True
        else:
            name_score = 0
            critical_failure = True
            breakdown["name_checked"] = True
            rejection_reasons.append(
                f"Name Mismatch (Input: {expected_name} vs Passport: {extracted_name})"
            )
        score += name_score
        breakdown["name_score"] = name_score

        # --- 4. Face similarity (0 - 40 points, NEVER critical) ---
        face_sim_score = 0
        if face_verification_result is not None:
            verified = face_verification_result.get("verified", False)
            is_duplicate = face_verification_result.get("is_duplicate", False)
            similarity = face_verification_result.get("similarity_score", 0.0)

            if verified:
                face_sim_score = weights["face_similarity"]
            elif similarity >= 0.4:
                face_sim_score = int(weights["face_similarity"] * 0.4)
            else:
                face_sim_score = 0
            breakdown["face_duplicate"] = bool(is_duplicate)
        score += face_sim_score
        breakdown["face_similarity_score"] = face_sim_score

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
