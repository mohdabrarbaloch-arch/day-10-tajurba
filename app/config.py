"""Settings, read from the environment and nowhere else.

No secret has a usable default in production. `SECRET_KEY` falls back to a
development value only when `ENVIRONMENT` is not "production", and the app
refuses to start in production without a real one.
"""

from __future__ import annotations

import secrets
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Tajurba"
    environment: str = "development"
    debug: bool = False

    secret_key: str = ""
    access_token_expire_minutes: int = 60 * 12
    algorithm: str = "HS256"

    database_url: str = "sqlite:///./tajurba.db"

    # Upload limits
    max_upload_bytes: int = 5 * 1024 * 1024
    max_rows: int = 20_000

    # Rate limiting
    login_rate_limit: str = "10/minute"
    train_rate_limit: str = "20/hour"

    cors_origins: str = "*"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def resolved_secret(self) -> str:
        if self.secret_key:
            return self.secret_key
        if self.is_production:
            raise RuntimeError(
                "SECRET_KEY must be set when ENVIRONMENT=production. "
                "Generate one with: python -c 'import secrets;print(secrets.token_urlsafe(48))'"
            )
        return "dev-only-insecure-key-do-not-use-in-production"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def generate_secret() -> str:
    """Helper for the setup docs."""
    return secrets.token_urlsafe(48)
