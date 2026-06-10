from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Configurations
    ENV: Literal["development", "production", "testing"] = "development"
    PROJECT_NAME: str = "Sales Assistant AI Platform"
    API_V1_STR: str = "/api/v1"
    
    # Dynamic LLM Provider Selection
    LLM_PROVIDER: Literal["groq", "gemini", "huggingface"] = "groq"
    
    # Targeted Model Deployments
    GROQ_MODEL: str = "llama3-70b-8192"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    HUGGINGFACE_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Core LLM Hyperparameters
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1000

    # API Tokens (Kept optional so missing keys for inactive providers won't crash boot)
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = None

    # Database Infrastructure Wire
    DATABASE_URL: str = "sqlite+aiosqlite:///./sales_agent.db"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Package-level singleton instance
_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings