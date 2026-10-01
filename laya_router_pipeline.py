"""
title: Laya Router Pipeline
author: open-webui
date: 2026-10-01
version: 0.1.0
license: MIT
description: Routes each prompt to a cheap or an expensive backend LLM based on a Laya (System 1 decision engine) classification of the prompt's complexity. Laya is open source and runs locally on the CPU.
environment_variables: LAYA_MODE, LAYA_CHECKPOINT, LAYA_DEVICE, LAYA_BASE_URL, LAYA_API_KEY, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, ORCAROUTER_API_KEY, ORCAROUTER_BASE_URL, CHEAP_PROVIDER, CHEAP_MODEL, EXPENSIVE_PROVIDER, EXPENSIVE_MODEL
"""

import inspect
import json
import os
import re
import threading
from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel

# transformers imports TensorFlow when it is installed next to torch, which can
# deadlock inside the Pipelines worker. Laya's docs ask for this; it has to be set
# before `laya` (and therefore `transformers`) is first imported.
os.environ.setdefault("USE_TF", "0")

# Short checkpoint name -> (hugging face repo id, subfolder). All three ship inside
# the same repo; convaiinnovations/laya-multilingual and friends mirror the subfolders.
CHECKPOINTS = {
    "english": ("convaiinnovations/laya", None),
    "multilingual": ("convaiinnovations/laya", "multilingual"),
    "typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}


class Pipeline:
    class Valves(BaseModel):
        # Laya (classification) settings.
        # "server" (default) talks to a `laya-serve` sidecar over HTTP, which keeps torch
        # out of the Pipelines container. "local" runs the model in-process and needs
        # `pip install laya` (torch + transformers) in the Pipelines container first; it
        # is not in the frontmatter `requirements`, which would install it on every boot.
        LAYA_MODE: str = "server"
        # Empty means Laya's Router picks the checkpoint per request from the detected
        # script/language. Otherwise one of CHECKPOINTS above, or any "repo/id" (with an
        # optional "repo/id#subfolder") to pin a single checkpoint.
        LAYA_CHECKPOINT: str = ""
        LAYA_DEVICE: str = "cpu"
        # torch.set_num_threads() for the classifier; 0 leaves torch's own default.
        # On a busy box, capping this keeps the router from starving everything else.
        LAYA_NUM_THREADS: int = 0
        # Build the checkpoint during on_startup instead of on the first chat message.
        # The first build costs ~7-10s on CPU, plus the download on a cold cache.
        LAYA_PRELOAD: bool = True
        # How many checkpoints the Router may keep resident. Auto-routing switches between
        # english and multilingual, so 1 rebuilds a checkpoint (seconds) on every language
        # switch; 2 keeps both warm. Use 1 only together with a pinned LAYA_CHECKPOINT.
        LAYA_MAX_LOADED: int = 2
        # Force a language hint instead of letting Laya detect it ("pt", "de", ...).
        LAYA_LANG: str = ""
        # Truncate the classifier input; 0 uses the checkpoint's own context length.
        LAYA_MAX_LEN: int = 0
        # Only used when LAYA_MODE is "server": where `laya-serve` listens. The
        # pipeline posts to {base_url}/v1/systemone.
        LAYA_BASE_URL: str = "http://localhost:8000"
        LAYA_API_KEY: str = ""
        LAYA_TIMEOUT_SECONDS: float = 10.0

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
        # Laya only picks *which* of them answers; it never generates the reply itself.
        OPENROUTER_API_KEY: str = ""
        OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
        ORCAROUTER_API_KEY: str = ""
        ORCAROUTER_BASE_URL: str = "https://api.orcarouter.ai/v1"

        # Each model valve is an ordered, comma-separated list of "provider:model" entries.
        # provider is "openrouter" or "orcarouter". The first entry is tried first; on any
        # failure (429, 5xx, timeout...) the next entry is tried. Providers can be mixed freely.
        CHEAP_MODEL: str = (
            "orcarouter:deepseek/deepseek-v4-flash-free, openrouter:google/gemma-4-31b-it:free, "
            "orcarouter:z-ai/glm-5.3-flash-free, openrouter:openrouter/free"
        )
        EXPENSIVE_MODEL: str = "openrouter:z-ai/glm-5.3, openrouter:deepseek/deepseek-v4-pro"
        # Used when the prompt is classified as code.
        CODE_CHEAP_MODEL: str = (
            "orcarouter:deepseek/deepseek-v4-flash-free, openrouter:poolside/laguna-s-2.1:free, "
            "orcarouter:z-ai/glm-5.3-flash-free, openrouter:openrouter/free"
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
        self.name = "Laya Router"
        # ponytail: single slot, shared across chats; upgrade to a dict keyed by chat_id if concurrent tool rounds mix
        self._route = ("", "", "")

        self.valves = self.Valves(
            **{
                "LAYA_MODE": os.getenv("LAYA_MODE", "server"),
                "LAYA_CHECKPOINT": os.getenv("LAYA_CHECKPOINT", ""),
                "LAYA_DEVICE": os.getenv("LAYA_DEVICE", "cpu"),
                "LAYA_BASE_URL": os.getenv("LAYA_BASE_URL", "http://localhost:8000"),
                "LAYA_API_KEY": os.getenv("LAYA_API_KEY", ""),
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

        # The checkpoint is expensive to build, so it is loaded once and reused for
        # every request. pipe() can run on several worker threads, hence the lock.
        self._agent = None
        self._agent_key = None
        self._agent_lock = threading.Lock()

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        if self.valves.LAYA_MODE == "local" and self.valves.LAYA_PRELOAD:
            import asyncio

            try:
                # Off the event loop: a cold checkpoint build takes seconds, and the
                # very first one also downloads a few hundred MB from the Hub.
                await asyncio.to_thread(self._agent_for_valves)
            except Exception as e:
                # A failed preload is not fatal; classify() retries and falls back.
                print(f"[LayaRouter] preload failed: {e}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        self._agent = None
        self._agent_key = None

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

    def _questions(self) -> dict:
        """The two routing axes, as Laya typed questions.

        Laya is non-autoregressive, so both are answered in a single forward pass -
        asking for the second axis is close to free.
        """
        v = self.valves
        return {
            "tier": {
                "type": "choice",
                "instructions": v.ROUTING_INSTRUCTIONS,
                "criteria": {
                    "cheap": v.CHEAP_CRITERION,
                    "expensive": v.EXPENSIVE_CRITERION,
                },
            },
            "kind": {
                "type": "choice",
                "instructions": v.KIND_INSTRUCTIONS,
                "criteria": {
                    "code": v.CODE_CRITERION,
                    "text": v.TEXT_CRITERION,
                },
            },
        }

    def _agent_for_valves(self):
        """Build (or reuse) the local Laya agent described by the current valves."""
        v = self.valves
        key = (v.LAYA_CHECKPOINT, v.LAYA_DEVICE,
               v.LAYA_MAX_LOADED, v.LAYA_NUM_THREADS)
        with self._agent_lock:
            # Valves are editable at runtime, so a changed one has to rebuild.
            if self._agent is not None and self._agent_key == key:
                return self._agent

            import laya

            if v.LAYA_NUM_THREADS > 0:
                import torch

                torch.set_num_threads(v.LAYA_NUM_THREADS)

            spec = v.LAYA_CHECKPOINT.strip()
            if not spec:
                # Router auto-detects the script/language and loads the matching
                # checkpoint, which is what makes non-English prompts route well.
                agent = laya.Router(
                    device=v.LAYA_DEVICE,
                    max_loaded=max(v.LAYA_MAX_LOADED, 1),
                )
                if v.LAYA_PRELOAD:
                    # Not Router(preload=True): that builds all three checkpoints (~1.16B
                    # params) and lifts max_loaded to 3. Auto-routing only picks these two.
                    agent.preload(["english", "multilingual"][: agent.max_loaded])
                print(f"[LayaRouter] loaded Laya Router on {v.LAYA_DEVICE}")
            else:
                repo, subfolder = CHECKPOINTS.get(spec, (None, None))
                if repo is None:
                    repo, _, subfolder = spec.partition("#")
                    subfolder = subfolder or None
                kwargs = {"device": v.LAYA_DEVICE}
                if subfolder:
                    kwargs["subfolder"] = subfolder
                agent = laya.load(repo, **self._supported(laya.load, kwargs))
                print(
                    f"[LayaRouter] loaded Laya checkpoint {repo}"
                    f"{'/' + subfolder if subfolder else ''} on {v.LAYA_DEVICE}"
                )

            self._agent, self._agent_key = agent, key
            return agent

    @staticmethod
    def _supported(fn, kwargs: dict) -> dict:
        """Drop kwargs the installed Laya build does not accept.

        Router.predict takes lang_guess/max_len while a single-checkpoint agent may
        not, and the signatures move between releases; passing only what the callable
        declares keeps the pipeline working across versions instead of raising TypeError.
        """
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return kwargs
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: val for k, val in kwargs.items() if k in params}

    def _predict_local(self, state: str, questions: dict) -> dict:
        v = self.valves
        agent = self._agent_for_valves()
        kwargs = {}
        if v.LAYA_LANG.strip():
            kwargs["lang_guess"] = v.LAYA_LANG.strip()
        if v.LAYA_MAX_LEN > 0:
            kwargs["max_len"] = v.LAYA_MAX_LEN
        return agent.predict(state, questions, **self._supported(agent.predict, kwargs))

    def _predict_server(self, state: str, questions: dict) -> dict:
        """Same call against a `laya-serve` sidecar (POST /v1/systemone)."""
        v = self.valves
        payload = {"state": {"body": state}, "questions": questions}
        if v.LAYA_LANG.strip():
            payload["lang_guess"] = v.LAYA_LANG.strip()
        if v.LAYA_MAX_LEN > 0:
            payload["max_len"] = v.LAYA_MAX_LEN
        headers = {"Content-Type": "application/json"}
        if v.LAYA_API_KEY:
            headers["Authorization"] = f"Bearer {v.LAYA_API_KEY}"
        # Tolerate a URL pasted with the endpoint path already on it.
        base = v.LAYA_BASE_URL.rstrip("/").removesuffix("/v1/systemone")
        r = requests.post(
            url=f"{base}/v1/systemone",
            json=payload,
            headers=headers,
            timeout=v.LAYA_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _choice(result: dict, key: str, allowed: tuple) -> str:
        """Read one typed answer, rejecting anything outside the declared criteria."""
        choice = result["answers"][key]["choice"]
        if choice not in allowed:
            raise ValueError(f"Laya returned unexpected {key} '{choice}'")
        return choice

    def classify(self, messages: List[dict], user_message: str) -> tuple[str, str]:
        """Ask Laya for (tier, kind): cheap/expensive and code/text.

        Never raises: a missing package, an unbuildable checkpoint or an unreachable
        `laya-serve` all fall back to `Valves.FALLBACK_TIER`, so a broken classifier
        degrades routing quality without ever blocking the chat.
        """
        state = self._classifier_state(messages, user_message)
        try:
            if self.valves.LAYA_MODE == "server":
                result = self._predict_server(state, self._questions())
            else:
                result = self._predict_local(state, self._questions())
            tier = self._choice(result, "tier", ("cheap", "expensive"))
            kind = self._choice(result, "kind", ("code", "text"))
            routed = (result.get("routing") or {}).get(
                "model", self.valves.LAYA_MODE)
            print(
                f"[LayaRouter] classified prompt as '{tier}'/'{kind}' (laya: {routed})")
            return tier, kind
        except ImportError as e:
            print(
                f"[LayaRouter] laya is not installed ({e}) -> falling back to "
                f"'{self.valves.FALLBACK_TIER}'; `pip install laya` or set LAYA_MODE=server"
            )
            return self.valves.FALLBACK_TIER, self._guess_kind(user_message)
        except (requests.RequestException, KeyError, ValueError, OSError) as e:
            print(
                f"[LayaRouter] Laya classification failed ({type(e).__name__}): {e} "
                f"-> falling back to '{self.valves.FALLBACK_TIER}'"
            )
            return self.valves.FALLBACK_TIER, self._guess_kind(user_message)
        except Exception as e:
            print(
                f"[LayaRouter] unexpected classification error: {e} "
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
        """Pass the SSE data chunks through, injecting the footer as a last content chunk.

        The Pipelines server wraps any line not starting with "data:" as reply text, so
        SSE comments/keep-alives (": OPENROUTER PROCESSING") and blank lines are dropped.
        A round that ends in tool calls gets no footer: Open WebUI runs the tools and
        calls us again, and only the final round should carry it.
        """
        has_tool_calls = False
        for line in lines:
            text = line.decode() if isinstance(line, bytes) else line
            if not text.startswith("data:"):
                continue
            if text.strip() == "data: [DONE]":
                if footer and not has_tool_calls:
                    chunk = {"choices": [{"index": 0, "delta": {"content": footer}, "finish_reason": None}]}
                    yield "data: " + json.dumps(chunk)
            elif not has_tool_calls and '"tool_calls"' in text:
                try:
                    choices = json.loads(text[5:]).get("choices") or [{}]
                    has_tool_calls = bool((choices[0].get("delta") or {}).get("tool_calls"))
                except ValueError:
                    pass
            yield line

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # A tool round (web search, ...) re-calls us with the same user message; keep its
        # route instead of re-classifying, so one answer doesn't hop between models.
        body_messages = body.get("messages", [])
        if body_messages and body_messages[-1].get("role") == "tool" and self._route[0] == user_message:
            tier, kind = self._route[1:]
        else:
            tier, kind = self.classify(messages, user_message)
            self._route = (user_message, tier, kind)
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
                print(f"[LayaRouter] {tier}/{kind} -> {provider}:{model}")
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
                    f"[LayaRouter] {tier}/{kind} {provider}:{model} failed: {e}")
                errors.append(f"{provider}:{model}: {e}")
        return "Error: all backends failed - " + " | ".join(errors)
