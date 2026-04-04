from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_URL: str
    OPENAI_API_KEY: str
    AZURE_TRANSLATOR_KEY: str | None = None
    AZURE_TRANSLATOR_ENDPOINT: str | None = None
    AZURE_TRANSLATOR_REGION: str | None = None
    FRONTEND_URL: str = "http://localhost:3000"
    APP_ENV: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
