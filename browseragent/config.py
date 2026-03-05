"""Configuration loaded from .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str = ""
    auth_username: str = ""
    auth_password: str = ""
    model_name: str = "gpt-4o"
    headless: bool = True

    # Record-Then-Replay settings
    paths_dir: str = "~/.browseragent/paths"
    replay_enabled: bool = True
    optimize_paths: bool = True

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "Settings":
        """Load settings from .env file and environment variables."""
        load_dotenv(dotenv_path or ".env")
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            auth_username=os.getenv("AUTH_USERNAME", ""),
            auth_password=os.getenv("AUTH_PASSWORD", ""),
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            headless=os.getenv("HEADLESS", "true").lower() in ("true", "1", "yes"),
            paths_dir=os.getenv("PATHS_DIR", "~/.browseragent/paths"),
            replay_enabled=os.getenv("REPLAY_ENABLED", "true").lower()
            in ("true", "1", "yes"),
            optimize_paths=os.getenv("OPTIMIZE_PATHS", "true").lower()
            in ("true", "1", "yes"),
        )

    def validate(self) -> None:
        """Raise if required settings are missing."""
        if not self.openai_api_key:
            raise SystemExit(
                "OPENAI_API_KEY is required. Set it in .env or as an env var."
            )

