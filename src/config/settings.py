"""
=============================================================================
Configuration Settings Module
=============================================================================

Pydantic-based settings management with environment variable support.
All configuration is loaded from .env file or environment variables.

CACHE OPTIMIZATION NOTE:
------------------------
The MANUAL_PATH setting determines where the 25k-token operations manual
is loaded from. This content becomes the shared prefix for all agents.
=============================================================================
"""

from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ for libraries like 'prompty' that read from env vars
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables or .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # -------------------------------------------------------------------------
    # vLLM Remote Endpoint
    # -------------------------------------------------------------------------
    # The remote vLLM server handles LMCache/prefix caching server-side.
    # We cannot modify server configuration, only optimize our prompts.
    vllm_base_url: str = "http://89.169.108.198:30080/v1"
    vllm_api_key: str = ""
    vllm_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # -------------------------------------------------------------------------
    # Langfuse Observability (MANDATORY)
    # -------------------------------------------------------------------------
    # Langfuse is required from day one for all LLM calls.
    # This enables TTFT tracking for cache efficiency analysis.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"
    
    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Path to the operations manual (25k tokens, ~50 pages)
    # This becomes the SHARED PREFIX for all agents
    manual_path: str = "data/operations_manual.txt"
    
    @property
    def manual_full_path(self) -> Path:
        """Get the full path to the operations manual."""
        return Path(self.manual_path)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
