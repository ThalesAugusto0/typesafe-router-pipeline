"""
title: TypeSafe Router Pipeline
author: open-webui
date: 2026-09-19
version: 0.1.0
license: MIT
description: Routes each prompt to a cheap or an expensive backend LLM based on a TypeSafe AI (system_one) classification of the prompt's complexity.
requirements: typesafe-sdk
environment_variables: TYPESAFE_API_KEY, TYPESAFE_BASE_URL, TYPESAFE_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, ORCAROUTER_API_KEY, ORCAROUTER_BASE_URL, CHEAP_PROVIDER, CHEAP_MODEL, EXPENSIVE_PROVIDER, EXPENSIVE_MODEL
"""

import json
import os
import re
from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel
from typesafe_sdk import (
    Choice,
    RetryPolicy,
    TypeSafeAPIConnectionError as APIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError as APITimeoutError,
    TypeSafeAuthenticationError as AuthenticationError,
    TypeSafeBadRequestError as BadRequestError,
    TypeSafeClient,
    TypeSafeInternalServerError as InternalServerError,
    TypeSafeRateLimitError as RateLimitError,
)


class Pipeline:
    class Valves(BaseModel):
        # TypeSafe AI (classification) settings.
        # Leave TYPESAFE_BASE_URL empty to use the SDK default (https://api.typesafe.ai);
        # system_one() calls POST {base_url}/v1/systemone under the hood, so if this
        # server sits behind an egress proxy/firewall, allowlist that host + path.
        TYPESAFE_API_KEY: str = ""
        TYPESAFE_BASE_URL: str = ""
        TYPESAFE_MODEL: str = ""
        TYPESAFE_TIMEOUT_SECONDS: float = 3.0
        TYPESAFE_MAX_RETRIES: int = 1

        # How the routing decision is framed for the classifier
        ROUTING_INSTRUCTIONS: str = (
            "Decide whether answering this request well requires a powerful, "
            "expensive LLM or whether a cheap, fast LLM is enough."
        )
        CHEAP_CRITERION: str = (
            "Simple, short, factual, or conversational request that a small/fast model handles well."
        )
        EXPENSIVE_CRITERION: str = (
            "Complex reasoning, long-form writing, multi-step analysis, hard or large "
            "code (architecture, non-trivial debugging, big refactors), or high-stakes "
            "request that benefits from a stronger model."
        )
        # Second axis: is the prompt about code or plain text?
        KIND_INSTRUCTIONS: str = "Decide whether the message is about programming/code or not."
        CODE_CRITERION: str = (
            "Contains source code, a stack trace, a code block, or asks to write, review, "
            "debug, explain or refactor code, SQL, shell commands or config files."
        )
        TEXT_CRITERION: str = "Plain-language request with no code involved."
        # Tier used when the classification call fails for any reason
        FALLBACK_TIER: str = "cheap"

        # Backend providers that actually serve the request (both are OpenAI-compatible
        # /chat/completions APIs and both offer models across the price spectrum).
        OPENROUTER_API_KEY: str = ""
        OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
        ORCAROUTER_API_KEY: str = ""
        ORCAROUTER_BASE_URL: str = "https://api.orcarouter.ai/v1"

        # Each model valve is an ordered, comma-separated list of "provider:model" entries.
        # provider is "openrouter" or "orcarouter". The first entry is tried first; on any
        # failure (429, 5xx, timeout...) the next entry is tried. Providers can be mixed freely.
        CHEAP_MODEL: str = (
            "openrouter:deepseek/deepseek-v4-flash-0731:free, "
            "orcarouter:deepseek/deepseek-v4-flash-free"
        )
        EXPENSIVE_MODEL: str = "openrouter:z-ai/glm-5.3, openrouter:deepseek/deepseek-v4-pro"
        # Used when the prompt is classified as code.
        CODE_CHEAP_MODEL: str = (
            "openrouter:deepseek/deepseek-v4-flash-0731:free, "
            "orcarouter:deepseek/deepseek-v4-flash-free"
        )
        CODE_EXPENSIVE_MODEL: str = "openrouter:moonshotai/kimi-k3, openrouter:deepseek/deepseek-v4-pro"
        BACKEND_TIMEOUT_SECONDS: float = 120.0

        # Classify using the last N chat messages (not just the latest), so a short
        # follow-up like "and with tests?" inherits the complexity/kind of the conversation.
        CLASSIFY_CONTEXT_MESSAGES: int = 4
        CLASSIFY_MAX_CHARS_PER_MESSAGE: int = 2000
        # Append "which model answered" to each reply.
        SHOW_MODEL_FOOTER: bool = True

    def __init__(self):
        # self.id left unset on purpose, see other examples in this repo.
        self.name = "TypeSafe Router"

        self.valves = self.Valves(
            **{
                "TYPESAFE_API_KEY": os.getenv("TYPESAFE_API_KEY", ""),
                "TYPESAFE_BASE_URL": os.getenv("TYPESAFE_BASE_URL", ""),
                "TYPESAFE_MODEL": os.getenv("TYPESAFE_MODEL", ""),
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
                "OPENROUTER_BASE_URL": os.getenv(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                ),
                "ORCAROUTER_API_KEY": os.getenv("ORCAROUTER_API_KEY", ""),
                "ORCAROUTER_BASE_URL": os.getenv(
                    "ORCAROUTER_BASE_URL", "https://api.orcarouter.ai/v1"
                ),
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    FOOTER_RE = re.compile(r"\n\n> 🔀 [^\n]*\s*$")

    @staticmethod
    def _text(content) -> str:
        """Message content is a string, or a list of parts for multimodal messages."""
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""

    def _classifier_state(self, messages: List[dict], user_message: str) -> str:
        """Last N turns as a transcript; a lone message is sent as-is."""
        v = self.valves
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        turns = turns[-max(v.CLASSIFY_CONTEXT_MESSAGES, 1):]
        if len(turns) <= 1:
            return user_message
        limit = v.CLASSIFY_MAX_CHARS_PER_MESSAGE
        lines = [
            f"{m['role']}: {self.FOOTER_RE.sub('', self._text(m.get('content')))[:limit]}"
            for m in turns[:-1]
        ]
        return (
            "Conversation so far:\n" + "\n".join(lines)
            + "\n\nLatest user message (classify this one, using the conversation "
            + f"as context):\n{user_message[:limit]}"
        )

    def classify(self, messages: List[dict], user_message: str) -> tuple[str, str]:
        """Ask TypeSafe AI for (tier, kind): cheap/expensive and code/text.

        Never raises: any SDK/network error falls back to `Valves.FALLBACK_TIER`
        so a TypeSafe outage never blocks the chat flow.
        """
        state = self._classifier_state(messages, user_message)
        try:
            client = TypeSafeClient(
                api_key=self.valves.TYPESAFE_API_KEY or None,
                # The SDK appends /v1/systemone itself; tolerate a URL pasted with it.
                base_url=self.valves.TYPESAFE_BASE_URL.rstrip(
                    "/").removesuffix("/v1/systemone") or None,
                model=self.valves.TYPESAFE_MODEL or None,
                retry=RetryPolicy(
                    max_retries=self.valves.TYPESAFE_MAX_RETRIES,
                    timeout=self.valves.TYPESAFE_TIMEOUT_SECONDS,
                ),
            )
            with client:
                response = client.system_one(
                    state,
                    {
                        "tier": Choice(
                            instructions=self.valves.ROUTING_INSTRUCTIONS,
                            criteria={
                                "cheap": self.valves.CHEAP_CRITERION,
                                "expensive": self.valves.EXPENSIVE_CRITERION,
                            },
                        ),
                        "kind": Choice(
                            instructions=self.valves.KIND_INSTRUCTIONS,
                            criteria={
                                "code": self.valves.CODE_CRITERION,
                                "text": self.valves.TEXT_CRITERION,
                            },
                        ),
                    },
                )
            tier = response.choices["tier"].choice
            kind = response.choices["kind"].choice
            print(f"[TypeSafeRouter] classified prompt as '{tier}'/'{kind}'")
            return tier, kind
        except (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
            BadRequestError,
            InternalServerError,
            TypeSafeAPIError,
        ) as e:
            print(
                f"[TypeSafeRouter] TypeSafe API error ({type(e).__name__}): {e} "
                f"-> falling back to '{self.valves.FALLBACK_TIER}'"
            )
            return self.valves.FALLBACK_TIER, self._guess_kind(user_message)
        except Exception as e:
            print(
                f"[TypeSafeRouter] unexpected classification error: {e} "
                f"-> falling back to '{self.valves.FALLBACK_TIER}'"
            )
            return self.valves.FALLBACK_TIER, self._guess_kind(user_message)

    @staticmethod
    def _guess_kind(user_message: str) -> str:
        # ponytail: classifier is down, so only a fenced block counts as code
        return "code" if "```" in user_message else "text"

    def _provider_config(self, provider: str) -> tuple[str, str]:
        """Resolve a provider name ("openrouter"/"orcarouter") to (base_url, api_key)."""
        v = self.valves
        if provider == "openrouter":
            return v.OPENROUTER_BASE_URL, v.OPENROUTER_API_KEY
        if provider == "orcarouter":
            return v.ORCAROUTER_BASE_URL, v.ORCAROUTER_API_KEY
        raise ValueError(
            f"unknown provider '{provider}' (use openrouter or orcarouter)")

    @staticmethod
    def _parse_targets(spec: str) -> List[tuple[str, str]]:
        """"openrouter:a/b:free, orcarouter:c/d" -> [("openrouter", "a/b:free"), ("orcarouter", "c/d")]"""
        targets = []
        for entry in spec.split(","):
            if entry.strip():
                provider, _, model = entry.strip().partition(":")
                targets.append((provider, model))
        return targets

    @staticmethod
    def _stream_with_footer(lines: Iterator, footer: str) -> Iterator:
        """Pass the SSE stream through, injecting the footer as a last content chunk."""
        for line in lines:
            text = line.decode() if isinstance(line, bytes) else line
            if footer and text.strip() == "data: [DONE]":
                chunk = {"choices": [{"index": 0, "delta": {
                    "content": footer}, "finish_reason": None}]}
                yield "data: " + json.dumps(chunk)
            yield line

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        tier, kind = self.classify(messages, user_message)
        v = self.valves
        if tier == "expensive":
            spec = v.CODE_EXPENSIVE_MODEL if kind == "code" else v.EXPENSIVE_MODEL
        else:
            spec = v.CODE_CHEAP_MODEL if kind == "code" else v.CHEAP_MODEL

        stream = bool(body.get("stream", False))
        payload = {**body}
        for key in ("user", "chat_id", "title"):
            payload.pop(key, None)
        # Don't feed our own "answered by" footer back to the model as context.
        payload["messages"] = [
            {**m, "content": self.FOOTER_RE.sub("", m["content"])}
            if m.get("role") == "assistant" and isinstance(m.get("content"), str)
            else m
            for m in body.get("messages", [])
        ]

        errors = []
        for provider, model in self._parse_targets(spec):
            try:
                base_url, api_key = self._provider_config(provider)
                r = requests.post(
                    url=f"{base_url.rstrip('/')}/chat/completions",
                    json={**payload, "model": model},
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    stream=stream,
                    timeout=v.BACKEND_TIMEOUT_SECONDS,
                )
                r.raise_for_status()
                print(f"[TypeSafeRouter] {tier}/{kind} -> {provider}:{model}")
                footer = f"\n\n> 🔀 `{model}` · {tier}/{kind}" if v.SHOW_MODEL_FOOTER else ""
                if stream:
                    return self._stream_with_footer(r.iter_lines(), footer)
                data = r.json()
                if footer and data.get("choices"):
                    msg = data["choices"][0]["message"]
                    msg["content"] = (msg.get("content") or "") + footer
                return data
            except Exception as e:
                print(
                    f"[TypeSafeRouter] {tier}/{kind} {provider}:{model} failed: {e}")
                errors.append(f"{provider}:{model}: {e}")
        return "Error: all backends failed - " + " | ".join(errors)
