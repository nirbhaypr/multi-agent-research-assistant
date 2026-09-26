from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    redis_url: str = "redis://localhost:6379/0"
    redis_prefix: str = "research:v1"
    research_mode: Literal["live", "demo"] = "live"
    openai_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    api_token: SecretStr | None = None
    max_pending_runs: int = Field(default=100, ge=1, le=10000)
    retention_seconds: int = Field(default=604800, ge=3600)
    lease_seconds: int = Field(default=15, ge=5, le=120)

    def require_providers(self, mode=None):
        if (mode or self.research_mode) == "live" and (
            not self.openai_api_key or not self.tavily_api_key
        ):
            raise ValueError("Live mode requires OPENAI_API_KEY and TAVILY_API_KEY")
