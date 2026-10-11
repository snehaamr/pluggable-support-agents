from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./support.db"
    model_provider: str = "deterministic"
    model_name: str = "gpt-4o-mini"
    model_base_url: str = "https://api.openai.com/v1"
    model_api_key: str = ""
    model_timeout_seconds: float = 60.0
    embedding_model: str = "text-embedding-3-small"
    order_agent_url: str = "http://127.0.0.1:8001"
    refund_agent_url: str = "http://127.0.0.1:8002"
    agent_transport: str = "http"
    datadog_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("datadog_api_key", "DD_API_KEY", "DATADOG_API_KEY"),
    )
    datadog_site: str = Field(
        default="datadoghq.com",
        validation_alias=AliasChoices("datadog_site", "DD_SITE", "DATADOG_SITE"),
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)
