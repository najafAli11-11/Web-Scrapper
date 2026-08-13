"""Provider-agnostic LLM client (AGENTS.md: config-driven, never hardcoded).

The extractor depends on the `LLMClient` protocol only. `LiteLLMClient` is
the default implementation, backed by litellm so any supported provider
(NVIDIA NIM, OpenAI, Anthropic, Groq, Ollama, OpenAI-compatible endpoints, ...)
is a config change, not a code change.

Resilience: schema-constrained via tool-use when the provider supports it,
with an automatic strict-JSON fallback for providers/models that don't.
"""

from __future__ import annotations

import json
from typing import Optional, Protocol, Sequence, Union

from agents.llm.env import get_env

Messages = Sequence[dict]


class LLMClient(Protocol):
    def complete_structured(
        self,
        *,
        messages: Messages,
        tool_name: str,
        tool_schema: dict,
        temperature: float,
        max_tokens: int,
    ) -> Optional[dict]:
        """Return parsed structured arguments or None if unusable."""
        ...


class LiteLLMClient:
    def __init__(self, agent_cfg: dict):
        self.cfg = agent_cfg["llm"]

    def _model(self) -> str:
        provider = self.cfg["provider"]
        model = self.cfg["model"]
        if provider in ("openai", "openai-compatible", "custom"):
            return f"openai/{model}"
        return f"{provider}/{model}"

    def _api_key(self) -> Optional[str]:
        env = self.cfg.get("api_key_env")
        return get_env(env) if env else None

    def _request_kwargs(self) -> dict:
        kwargs: dict = {
            "model": self._model(),
            "temperature": float(self.cfg["temperature"]),
            "max_tokens": int(self.cfg["max_tokens"]),
            "api_key": self._api_key(),
        }
        base = self.cfg.get("api_base")
        if base:
            kwargs["api_base"] = base
        return kwargs

    @staticmethod
    def _tool(tool_name: str, tool_schema: dict) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Structured extraction result",
                    "parameters": tool_schema,
                },
            }
        ]

    def complete_structured(
        self,
        *,
        messages: Messages,
        tool_name: str,
        tool_schema: dict,
        temperature: float,
        max_tokens: int,
    ) -> Optional[dict]:
        import litellm

        kwargs = self._request_kwargs()
        try:
            response = litellm.completion(
                **kwargs,
                messages=list(messages),
                tools=self._tool(tool_name, tool_schema),
                tool_choice=self.cfg.get("tool_choice", "required"),
            )
        except Exception:
            # Provider/model doesn't support tool-calling -> strict-JSON mode.
            return self._json_mode(messages, tool_schema, temperature, max_tokens, **kwargs)

        try:
            message = response.choices[0].message
            for call in message.tool_calls or []:
                if call.function and call.function.arguments:
                    return json.loads(call.function.arguments)
        except Exception:
            return None
        return None

    def _json_mode(self, messages, tool_schema, temperature, max_tokens, **kwargs) -> Optional[dict]:
        import litellm

        instruction = {
            "role": "system",
            "content": (
                "Respond with a single JSON object ONLY — no prose, no markdown. "
                "The object must conform exactly to this JSON Schema:\n"
                f"{json.dumps(tool_schema)}"
            ),
        }
        try:
            response = litellm.completion(
                **kwargs,
                messages=list(messages) + [instruction],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            return json.loads(text) if text else None
        except Exception:
            return None
