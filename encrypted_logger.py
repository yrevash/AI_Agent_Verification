"""
Encrypted Per-User Logging using Fernet (AES-128-CBC + HMAC-SHA256).
Each verification result is stored as an encrypted .enc file per user.
"""

import json
import os
from pathlib import Path
from cryptography.fernet import Fernet


class EncryptedUserLogger:
    """Logs verification results as Fernet-encrypted JSON files, one per user."""

    def __init__(self, logs_dir: str = "data/encrypted_logs", key_path: str = "data/encryption.key"):
        self.logs_dir = Path(logs_dir)
        self.key_path = Path(key_path)
        self._fernet = self._init_fernet()
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _init_fernet(self) -> Fernet:
        """Load or generate the Fernet key.

        Priority:
        1. ENCRYPTION_KEY environment variable
        2. Key file on disk
        3. Auto-generate and persist to key file
        """
        env_key = os.environ.get("ENCRYPTION_KEY")
        if env_key:
            return Fernet(env_key.encode())

        if self.key_path.exists():
            key = self.key_path.read_bytes().strip()
            return Fernet(key)

        # Auto-generate
        key = Fernet.generate_key()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_bytes(key)
        print(f"Generated new encryption key at {self.key_path}")
        return Fernet(key)

    def log_user(self, user_id: str, record: dict) -> Path:
        """Encrypt and write a user's verification record to disk.

        Returns the path to the written .enc file.
        """
        safe_id = str(user_id).replace("/", "_").replace("\\", "_")
        out_path = self.logs_dir / f"{safe_id}.enc"
        plaintext = json.dumps(record, default=str).encode("utf-8")
        out_path.write_bytes(self._fernet.encrypt(plaintext))
        return out_path

    def read_user_log(self, user_id: str) -> dict:
        """Decrypt and return a user's verification record.

        Raises FileNotFoundError if the log does not exist.
        """
        safe_id = str(user_id).replace("/", "_").replace("\\", "_")
        enc_path = self.logs_dir / f"{safe_id}.enc"
        if not enc_path.exists():
            raise FileNotFoundError(f"No encrypted log for user {user_id}")
        plaintext = self._fernet.decrypt(enc_path.read_bytes())
        return json.loads(plaintext.decode("utf-8"))

    def list_logged_users(self) -> list[str]:
        """Return a list of user IDs that have encrypted logs."""
        return [p.stem for p in sorted(self.logs_dir.glob("*.enc"))]
