"""GPT client: one call returns JSON that matches a schema (OpenAI structured outputs)."""

import json
from dataclasses import dataclass, field

from openai import AzureOpenAI, OpenAI

from .config import Settings


@dataclass
class LLMResult:
    data: dict
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: str = ""
    refusal: str | None = None


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, result: LLMResult) -> None:
        self.calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        m = self.by_model.setdefault(result.model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        m["calls"] += 1
        m["input_tokens"] += result.input_tokens
        m["output_tokens"] += result.output_tokens


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        if settings.provider == "azure":
            self.client = AzureOpenAI(
                api_key=settings.azure_api_key,
                azure_endpoint=settings.azure_endpoint,
                api_version=settings.azure_api_version,
                timeout=settings.request_timeout,
                max_retries=5,
            )
        else:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
                timeout=settings.request_timeout,
                max_retries=5,
            )

    def json_call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        reasoning_effort: str = "",
        max_output_tokens: int = 16000,
    ) -> LLMResult:
        kwargs = {}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
            max_completion_tokens=max_output_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        result = LLMResult(
            data={},
            model=response.model or model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=choice.message.content or "",
            refusal=getattr(choice.message, "refusal", None),
        )
        if result.refusal:
            raise LLMError(f"model refused: {result.refusal}")
        if choice.finish_reason == "length":
            raise LLMError("output hit the token limit before the JSON was complete")
        try:
            result.data = json.loads(result.raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"output is not valid JSON: {e}") from e
        return result
