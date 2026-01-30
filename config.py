"""
Environment Configuration for Production Deployment
Loads configuration from environment variables with sensible defaults
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class RedisConfig(BaseModel):
    """Redis configuration"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    ttl_hours: int = Field(default=24, env="REDIS_TTL_HOURS")
    enabled: bool = Field(default=True, env="REDIS_ENABLED")


class AppConfig(BaseModel):
    """Application configuration"""
    # Server settings
    host: str = Field(default="0.0.0.0", env="APP_HOST")
    port: int = Field(default=8101, env="APP_PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # GPU settings
    use_gpu: bool = Field(default=True, env="USE_GPU")
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = Field(default="logs", env="LOG_DIR")
    
    # File handling
    temp_dir: str = Field(default="temp", env="TEMP_DIR")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    
    # Model paths
    model1_path: str = Field(default="models/best4.pt", env="MODEL1_PATH")
    model2_path: str = Field(default="models/best.pt", env="MODEL2_PATH")
    
    # Feature flags
    enable_qwen_fallback: bool = Field(default=True, env="ENABLE_QWEN_FALLBACK")
    enable_debug_images: bool = Field(default=False, env="ENABLE_DEBUG_IMAGES")
    
    # Database
    aadhaar_db_path: str = Field(default="data/aadhaar_records.pkl", env="AADHAAR_DB_PATH")

    # Encrypted logging
    encrypted_logs_dir: str = Field(default="data/encrypted_logs", env="ENCRYPTED_LOGS_DIR")
    encryption_key_path: str = Field(default="data/encryption.key", env="ENCRYPTION_KEY_PATH")
    
    # Performance tuning
    cleanup_interval: int = Field(default=120, env="CLEANUP_INTERVAL")  # seconds
    temp_file_ttl: int = Field(default=300, env="TEMP_FILE_TTL")  # seconds
    http_timeout: int = Field(default=30, env="HTTP_TIMEOUT")  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.app = self._load_app_config()
        self.redis = self._load_redis_config()
        
    def _load_app_config(self) -> AppConfig:
        """Load application configuration from environment"""
        return AppConfig(
            host=os.getenv("APP_HOST", "0.0.0.0"),
            port=int(os.getenv("APP_PORT", "8101")),
            workers=int(os.getenv("WORKERS", "1")),
            use_gpu=os.getenv("USE_GPU", "true").lower() == "true",
            cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_dir=os.getenv("LOG_DIR", "logs"),
            temp_dir=os.getenv("TEMP_DIR", "temp"),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", "10485760")),
            model1_path=os.getenv("MODEL1_PATH", "models/best4.pt"),
            model2_path=os.getenv("MODEL2_PATH", "models/best.pt"),
            enable_qwen_fallback=os.getenv("ENABLE_QWEN_FALLBACK", "true").lower() == "true",
            enable_debug_images=os.getenv("ENABLE_DEBUG_IMAGES", "false").lower() == "true",
            aadhaar_db_path=os.getenv("AADHAAR_DB_PATH", "data/aadhaar_records.pkl"),
            encrypted_logs_dir=os.getenv("ENCRYPTED_LOGS_DIR", "data/encrypted_logs"),
            encryption_key_path=os.getenv("ENCRYPTION_KEY_PATH", "data/encryption.key"),
            cleanup_interval=int(os.getenv("CLEANUP_INTERVAL", "120")),
            temp_file_ttl=int(os.getenv("TEMP_FILE_TTL", "300")),
            http_timeout=int(os.getenv("HTTP_TIMEOUT", "30"))
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment"""
        return RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ttl_hours=int(os.getenv("REDIS_TTL_HOURS", "24")),
            enabled=os.getenv("REDIS_ENABLED", "true").lower() == "true"
        )
    
    def create_directories(self):
        """Create necessary directories"""
        Path(self.app.log_dir).mkdir(exist_ok=True, parents=True)
        Path(self.app.temp_dir).mkdir(exist_ok=True, parents=True)
        Path(self.app.aadhaar_db_path).parent.mkdir(exist_ok=True, parents=True)
        
        if self.app.enable_debug_images:
            Path("debug_output").mkdir(exist_ok=True, parents=True)
    
    def validate(self):
        """Validate configuration"""
        # Check model files exist
        if not Path(self.app.model1_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.app.model1_path}")
        if not Path(self.app.model2_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.app.model2_path}")
    
    def print_config(self):
        """Print current configuration (for debugging)"""
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print(f"Server: {self.app.host}:{self.app.port}")
        print(f"Workers: {self.app.workers}")
        print(f"GPU Enabled: {self.app.use_gpu}")
        print(f"Log Level: {self.app.log_level}")
        print(f"Redis: {self.redis.host}:{self.redis.port} (enabled={self.redis.enabled})")
        print(f"Model1: {self.app.model1_path}")
        print(f"Model2: {self.app.model2_path}")
        print(f"Qwen Fallback: {self.app.enable_qwen_fallback}")
        print("=" * 80)


# Singleton instance
_config = None

def get_config() -> Config:
    """Get singleton configuration instance"""
    global _config
    if _config is None:
        _config = Config()
        _config.create_directories()
    return _config
