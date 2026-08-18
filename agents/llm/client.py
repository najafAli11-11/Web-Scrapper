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
import re
from typing import Optional, Protocol, Sequence, Union

from agents.llm.env import get_env

Messages = Sequence[dict]


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract a JSON object from text that may contain markdown formatting.

    Handles: raw JSON, ```json fenced blocks, and ``` fenced blocks.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    for pattern in (r"```json\s*\n(.*?)\n\s*```", r"```\s*\n(.*?)\n\s*```"):
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                continue
    return None


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
    def __init__(self, agent_cfg: dict, logger=None):
        self.cfg = agent_cfg["llm"]
        self.last_raw_text: Optional[str] = None
        self._logger = logger

    def _model(self) -> str:
        provider = self.cfg["provider"]
        model = self.cfg["model"]
        if provider in ("openai", "openai-compatible", "custom"):
            return f"openai/{model}"
        return f"{provider}/{model}"

    def _api_key(self) -> Optional[str]:
        env = self.cfg.get("api_key_env")
        return get_env(env) if env else None

    def _log(self, event_type: str, **kwargs) -> None:
        if self._logger is not None:
            self._logger.log_event(event_type, **kwargs)

    def set_logger(self, logger) -> None:
        self._logger = logger

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
        except Exception as exc:
            self._log(
                "llm_call",
                outcome="tool_use_failed",
                reason=f"tool-use call failed, falling back to json_mode: {exc}",
                details={"model": self._model(), "fallback": "json_mode", "error": str(exc)[:300]},
            )
            return self._json_mode(messages, tool_schema, temperature, max_tokens)

        try:
            message = response.choices[0].message
            self.last_raw_text = message.content
            for call in message.tool_calls or []:
                if call.function and call.function.arguments:
                    parsed = json.loads(call.function.arguments)
                    self._log("llm_call", outcome="success", reason="tool_use",
                              details={"model": self._model(), "mode": "tool_use"})
                    return parsed
            if message.content:
                parsed = _extract_json_from_text(message.content)
                if parsed is not None:
                    self._log("llm_call", outcome="success", reason="json_from_content",
                              details={"model": self._model(), "mode": "json_from_content"})
                    return parsed
            self._log(
                "llm_call",
                outcome="tool_use_parse_failed",
                reason="tool_calls empty and content not parseable, falling back to json_mode",
                details={"model": self._model(), "has_tool_calls": bool(message.tool_calls)},
            )
        except Exception as exc:
            self._log(
                "llm_call",
                outcome="tool_use_parse_failed",
                reason=f"failed to parse tool-use response: {exc}",
                details={"model": self._model(), "error": str(exc)[:300]},
            )
        return self._json_mode(messages, tool_schema, temperature, max_tokens)

    def _json_mode(self, messages, tool_schema, temperature, max_tokens) -> Optional[dict]:
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
                model=self._model(),
                api_key=self._api_key(),
                messages=list(messages) + [instruction],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            self.last_raw_text = text
            parsed = json.loads(text) if text else None
            if parsed is not None:
                self._log("llm_call", outcome="success", reason="json_mode",
                          details={"model": self._model(), "mode": "json_mode"})
            return parsed
        except Exception as exc:
            self._log(
                "llm_call",
                outcome="json_mode_failed",
                reason=f"json_mode fallback failed: {exc}",
                details={"model": self._model(), "error": str(exc)[:300]},
            )
            return None
