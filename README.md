# TypeSafe Router Pipeline

> Two pipelines live in this repo, same router, different classifier:
> **`typesafe_router_pipeline.py`** (below) classifies with the hosted TypeSafe AI API, and
> **`laya_router_pipeline.py`** ([docs](./README-laya.md)) classifies with [Laya](https://github.com/NandhaKishorM/laya),
> an Apache-2.0 System 1 decision engine that runs locally on the CPU with no API key.

An [Open WebUI Pipelines](https://github.com/open-webui/pipelines) function that routes every incoming chat message to a **cheap** or an **expensive** backend LLM, picked automatically by classifying the prompt with [TypeSafe AI](https://typesafe.ai)'s `system_one` primitive.

Instead of hard-coding "always use GPT-4" or "always use the free model", this pipeline asks a fast classifier two questions about the request and routes accordingly:

1. **Tier** — does answering this well need a powerful, expensive model, or is a cheap/fast model enough?
2. **Kind** — is this about code (source, stack traces, debugging, refactors) or plain text?

Those two answers select one of four model pools, and the request is forwarded to the first backend that responds successfully.

## How it works

```
user message
     │
     ▼
classify(messages) ──► TypeSafe AI system_one()
     │                     returns: tier = cheap | expensive
     │                              kind = code  | text
     ▼
pick model list:
  expensive + code  -> CODE_EXPENSIVE_MODEL
  expensive + text  -> EXPENSIVE_MODEL
  cheap     + code  -> CODE_CHEAP_MODEL
  cheap     + text  -> CHEAP_MODEL
     │
     ▼
try each "provider:model" entry in order
  (openrouter or orcarouter) until one succeeds
     │
     ▼
stream/return the response, optionally
appended with a "🔀 answered by <model>" footer
```

### 1. Classification

`classify()` builds a short transcript from the last `CLASSIFY_CONTEXT_MESSAGES` turns (so a follow-up like *"and with tests?"* inherits the complexity/kind of the conversation) and sends it to TypeSafe AI's `system_one` endpoint with two `Choice` axes (`tier`, `kind`) defined by the `*_INSTRUCTIONS` / `*_CRITERION` valves.

If the TypeSafe API is unreachable, times out, rate-limits, or errors in any way, classification **never raises** — it falls back to `Valves.FALLBACK_TIER` (default `"cheap"`) and guesses `kind` locally by checking for a fenced code block (```` ``` ````). This means a TypeSafe outage degrades routing quality but never blocks the chat.

### 2. Model selection & failover

Each of the four model valves (`CHEAP_MODEL`, `EXPENSIVE_MODEL`, `CODE_CHEAP_MODEL`, `CODE_EXPENSIVE_MODEL`) holds a comma-separated, ordered list of `provider:model` entries, e.g.:

```
orcarouter:deepseek/deepseek-v4-flash-free, openrouter:google/gemma-4-31b-it:free, openrouter:openrouter/free
```

The pipeline tries each entry in order against the matching provider's OpenAI-compatible `/chat/completions` endpoint. On any failure (HTTP error, timeout, connection error) it logs the error and moves to the next entry. If every entry fails, the pipeline returns a single error string listing all the failures instead of raising.

Two backend providers are supported out of the box, both OpenAI-compatible and both offering models across the price spectrum:

- **OpenRouter** (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`)
- **OrcaRouter** (`ORCAROUTER_API_KEY`, `ORCAROUTER_BASE_URL`)

### 3. Streaming & footer

Responses are proxied through as-is, streaming or not (`body["stream"]`). When `SHOW_MODEL_FOOTER` is enabled, a Markdown footer such as:

> 🔀 `openrouter:deepseek/deepseek-v4-pro` · expensive/code

is appended to the reply so users can see which tier/model actually answered. The footer is stripped back out of assistant messages before they're re-sent as context on the next turn, so the router never confuses its own footer for conversation content.

## Configuration

All settings are exposed as Pipelines **Valves** and can also be seeded from environment variables at startup.

| Valve | Env var | Purpose |
|---|---|---|
| `TYPESAFE_API_KEY` | `TYPESAFE_API_KEY` | TypeSafe AI credential |
| `TYPESAFE_BASE_URL` | `TYPESAFE_BASE_URL` | Override TypeSafe API host (defaults to `https://api.typesafe.ai`) |
| `TYPESAFE_MODEL` | `TYPESAFE_MODEL` | Classifier model, if the SDK default should be overridden |
| `TYPESAFE_TIMEOUT_SECONDS` / `TYPESAFE_MAX_RETRIES` | — | Classification request resiliency |
| `ROUTING_INSTRUCTIONS`, `CHEAP_CRITERION`, `EXPENSIVE_CRITERION` | — | How the tier decision is framed to the classifier |
| `KIND_INSTRUCTIONS`, `CODE_CRITERION`, `TEXT_CRITERION` | — | How the code-vs-text decision is framed |
| `FALLBACK_TIER` | — | Tier used when classification fails (default `cheap`) |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | same | OpenRouter backend credentials |
| `ORCAROUTER_API_KEY` / `ORCAROUTER_BASE_URL` | same | OrcaRouter backend credentials |
| `CHEAP_MODEL`, `EXPENSIVE_MODEL`, `CODE_CHEAP_MODEL`, `CODE_EXPENSIVE_MODEL` | — | Ordered `provider:model` fallback lists per tier/kind |
| `BACKEND_TIMEOUT_SECONDS` | — | Timeout for the backend `/chat/completions` call |
| `CLASSIFY_CONTEXT_MESSAGES` / `CLASSIFY_MAX_CHARS_PER_MESSAGE` | — | How much conversation history feeds the classifier |
| `SHOW_MODEL_FOOTER` | — | Toggle the "answered by" footer |

## Requirements

- An [Open WebUI Pipelines](https://github.com/open-webui/pipelines) server to host this function.
- A [TypeSafe AI](https://typesafe.ai) API key for classification.
- At least one of an OpenRouter or OrcaRouter API key for the actual backend completions.

Python dependency: `typesafe-sdk` (declared in the pipeline's `requirements` frontmatter, installed automatically by the Pipelines server).

## Installation

1. Copy `typesafe_router_pipeline.py` into your Open WebUI Pipelines server (via the Admin UI's "Upload Pipeline" or by mounting it in the `pipelines` directory).
2. Set the required valves/environment variables listed above.
3. Select "TypeSafe Router" as a model in Open WebUI — every message sent to it is classified and routed automatically.

## License

MIT
