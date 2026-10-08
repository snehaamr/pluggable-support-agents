from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./support.db"
    model_provider: str = "deterministic"
    model_name: str = "gpt-4o-mini"
    model_base_url: str = "https://api.openai.com/v1"
    model_api_key: str = ""
    model_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
