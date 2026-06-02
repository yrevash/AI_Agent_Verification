"""
Offline test of the passport extraction path.

Uses the REAL OllamaQwenAgent.validate_passport / extract_passport methods from
batch.py, pointed at the local Ollama OpenAI-compatible endpoint with a vision
model (stand-in for production's vLLM Qwen3-VL-8B). Reads the reference image
passport.jpeg and prints the structured extraction result.
"""
import io
import json
import sys

from batch import OllamaQwenAgent

OLLAMA_OPENAI_URL = "http://localhost:11434/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5vl:3b"
IMAGE_PATH = "passport.jpeg"


def main():
    agent = OllamaQwenAgent(ollama_url=OLLAMA_OPENAI_URL, model=MODEL)

    with open(IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    # use a fresh BytesIO per call (methods seek/convert internally)
    print("=== validate_passport ===")
    v = agent.validate_passport(io.BytesIO(img_bytes))
    print(json.dumps(v, indent=2))

    print("\n=== extract_passport ===")
    data = agent.extract_passport(io.BytesIO(img_bytes), None)
    print(json.dumps(data, indent=2))

    # quick sanity check on the fields the passport pipeline relies on
    print("\n=== field check ===")
    for key in ("passport_number", "name", "dob", "gender", "nationality", "country", "date_of_expiry"):
        val = data.get(key) if isinstance(data, dict) else None
        status = "OK" if val else "MISSING"
        print(f"  {key:16} [{status}] {val!r}")


if __name__ == "__main__":
    main()
