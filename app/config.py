"""config.py — Centralized application settings via Pydantic."""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    model_path: str = ""
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    rate_limit: str = "10/minute"
    max_image_size_mb: int = 20
    log_level: str = "INFO"

    class Config:
        env_file = str(Path(__file__).parent / ".env")
        env_file_encoding = "utf-8"

settings = Settings()
