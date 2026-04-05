from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_URL: str
    OPENAI_API_KEY: str
    REDIS_URL: str | None = None
    EMBEDDING_CACHE_TTL_SECONDS: int = 86400
    AZURE_TRANSLATOR_KEY: str | None = None
    AZURE_TRANSLATOR_ENDPOINT: str | None = None
    AZURE_TRANSLATOR_REGION: str | None = None
    AZURE_DOCUMENT_INTELLIGENCE_KEY: str | None = None
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: str | None = None
    AZURE_DOCUMENT_INTELLIGENCE_API_VERSION: str = "2024-11-30"
    AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID: str = "prebuilt-read"
    AZURE_DOCUMENT_INTELLIGENCE_TIMEOUT_SECONDS: int = 60
    AZURE_DOCUMENT_INTELLIGENCE_POLL_INTERVAL_MS: int = 1200
    FRONTEND_URL: str = "http://localhost:3000"
    APP_ENV: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
