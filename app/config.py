"""Settings from .env / environment (API_ROUTING_PLAN Stage R1). The key is checked at startup, not at import."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(Exception):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = field(default="", repr=False)          # never printed
    router_model: str = "openai/gpt-oss-120b"                  # Q1
    router_temperature: float = 0.0                            # classification must be stable
    router_reasoning_effort: str = "low"                       # gpt-oss: keep the routing call fast
    history_turns: int = 2                                     # A-D11
    router_fallback_route: str = "out_of_scope"                # Q10: safe default when router output is invalid twice
    max_message_chars: int = 2000                              # Q6
    host: str = "127.0.0.1"                                    # Q7
    port: int = 8000
    llm_max_attempts: int = 4                                  # Q8
    llm_base_delay: float = 1.0
    llm_max_delay: float = 20.0                                # never wait longer than this for one retry
    llm_timeout: float = 30.0

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        env = os.environ if env is None else env
        def get(name, default, cast=str):
            raw = env.get(name)
            if raw is None or str(raw).strip() == "":
                return default
            try:
                return cast(raw)
            except ValueError as e:
                raise ConfigError(f"{name}={raw!r} is not a valid {cast.__name__}") from e
        d = cls()
        return cls(
            groq_api_key=str(env.get("GROQ_API_KEY", "")).strip().strip("\"'"),
            router_model=get("ROUTER_MODEL", d.router_model),
            router_reasoning_effort=get("ROUTER_REASONING_EFFORT", d.router_reasoning_effort),
            history_turns=get("HISTORY_TURNS", d.history_turns, int),
            max_message_chars=get("MAX_MESSAGE_CHARS", d.max_message_chars, int),
            host=get("APP_HOST", d.host), port=get("APP_PORT", d.port, int),
            llm_max_attempts=get("LLM_MAX_ATTEMPTS", d.llm_max_attempts, int),
            llm_base_delay=get("LLM_BASE_DELAY", d.llm_base_delay, float),
            llm_max_delay=get("LLM_MAX_DELAY", d.llm_max_delay, float),
            llm_timeout=get("LLM_TIMEOUT", d.llm_timeout, float),
        )

    def require_key(self) -> "Settings":
        if not self.groq_api_key:
            raise ConfigError("GROQ_API_KEY is not set. Put it in .env (see .env.example) or the environment, "
                              "then restart. Get a free key at https://console.groq.com/keys")
        return self


def get_settings() -> Settings:
    return Settings.from_env()
