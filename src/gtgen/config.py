"""Settings loaded from the environment (and a .env file, if present)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _get(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


@dataclass(frozen=True)
class Settings:
    provider: str
    openai_api_key: str
    openai_base_url: str
    azure_api_key: str
    azure_endpoint: str
    azure_api_version: str
    labeller_model: str
    judge_model: str
    closed_set_model: str
    labeller_reasoning_effort: str
    judge_reasoning_effort: str
    summ_service_src: str
    max_concurrency: int
    max_label_attempts: int
    request_timeout: float

    def validate_for_labelling(self) -> None:
        """Fail early, before any call data is read, if the run cannot work."""
        problems = []
        if self.provider not in ("openai", "azure"):
            problems.append(f"LLM_PROVIDER must be 'openai' or 'azure', got {self.provider!r}")
        if self.provider == "openai" and not self.openai_api_key:
            problems.append("OPENAI_API_KEY is not set")
        if self.provider == "azure" and not (self.azure_api_key and self.azure_endpoint):
            problems.append("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set")
        for name in ("labeller_model", "judge_model", "closed_set_model"):
            if not getattr(self, name):
                problems.append(f"{name.upper()} is not set")
        if self.labeller_model and self.labeller_model == self.judge_model:
            problems.append(
                "JUDGE_MODEL must differ from LABELLER_MODEL, so one model's blind spots "
                "are not checked by the same model"
            )
        if problems:
            raise SystemExit("Configuration error:\n  - " + "\n  - ".join(problems))


def load_settings(env_file: str | None = None) -> Settings:
    load_dotenv(env_file or Path.cwd() / ".env", override=False)
    return Settings(
        provider=_get("LLM_PROVIDER", "openai").lower(),
        openai_api_key=_get("OPENAI_API_KEY"),
        openai_base_url=_get("OPENAI_BASE_URL"),
        azure_api_key=_get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=_get("AZURE_OPENAI_ENDPOINT"),
        azure_api_version=_get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        labeller_model=_get("LABELLER_MODEL"),
        judge_model=_get("JUDGE_MODEL"),
        closed_set_model=_get("CLOSED_SET_MODEL"),
        labeller_reasoning_effort=_get("LABELLER_REASONING_EFFORT"),
        judge_reasoning_effort=_get("JUDGE_REASONING_EFFORT"),
        summ_service_src=_get("SUMM_SERVICE_SRC"),
        max_concurrency=int(_get("MAX_CONCURRENCY", "4")),
        max_label_attempts=int(_get("MAX_LABEL_ATTEMPTS", "2")),
        request_timeout=float(_get("REQUEST_TIMEOUT_SECONDS", "300")),
    )
