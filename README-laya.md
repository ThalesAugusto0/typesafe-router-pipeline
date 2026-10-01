# Laya Router Pipeline

The same router as [`typesafe_router_pipeline.py`](./README.md), with the remote classifier swapped for one that runs on your own CPU.

An [Open WebUI Pipelines](https://github.com/open-webui/pipelines) function that routes every incoming chat message to a **cheap** or an **expensive** backend LLM, picked automatically by classifying the prompt with [Laya](https://github.com/NandhaKishorM/laya) — an Apache-2.0 *System 1 decision engine* that runs locally.

## Laya is a classifier, not a chat model

Laya is **not** a generative LLM, so it does not replace the models that write the answers. It replaces exactly the part TypeSafe AI's `system_one` (Jev) was doing: turning a prompt into typed judgments that code can branch on. OpenRouter/OrcaRouter still serve the actual completion.

| | TypeSafe Router | Laya Router |
|---|---|---|
| Classifier | TypeSafe AI `system_one` (hosted) | Laya (local, in-process or sidecar) |
| Needs an API key to route | yes | no |
| Needs the network to route | yes | no |
| Model | Jev | ModernBERT-large 421M / mmBERT-base 322M |
| Device | — | CPU (or CUDA) |
| Licence | commercial API | Apache-2.0 |
| Backends that answer the chat | OpenRouter / OrcaRouter | unchanged |

Laya is non-autoregressive: it answers every question about the prompt in one forward pass (~33 ms on a T4, higher but workable on CPU), with outputs calibrated by strictly proper scoring rules rather than parsed out of generated text. It also detects the script/language of the prompt and picks a matching checkpoint, so a Portuguese prompt routes as well as an English one.

## How it works

```
user message
     │
     ▼
classify(messages) ──► Laya predict()  (local CPU, no network)
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

`classify()` builds a short transcript from the last `CLASSIFY_CONTEXT_MESSAGES` turns (so a follow-up like *"and with tests?"* inherits the complexity/kind of the conversation) and sends it to Laya as two `choice` questions (`tier`, `kind`) defined by the `*_INSTRUCTIONS` / `*_CRITERION` valves. Because Laya is non-autoregressive, the second axis costs almost nothing.

Answers are validated against the criteria they were asked with: anything outside `cheap`/`expensive` or `code`/`text` is treated as a failure rather than routed on blindly.

Classification **never raises**. A missing `laya` package, a checkpoint that will not build, an unreachable `laya-serve`, or a malformed answer all fall back to `Valves.FALLBACK_TIER` (default `"cheap"`) and guess `kind` locally by checking for a fenced code block (```` ``` ````). A broken classifier degrades routing quality but never blocks the chat.

### 2. Two deployment modes

`LAYA_MODE` picks where the model lives:

- **`server`** (default) — the pipeline POSTs to a [`laya-serve`](https://github.com/NandhaKishorM/laya) sidecar at `{LAYA_BASE_URL}/v1/systemone`. This keeps `torch` out of the Pipelines container and lets several pipelines (or an MCP server) share one warm model. A Compose service built from Laya's own Dockerfile:

  ```yaml
  laya:
    build: https://github.com/NandhaKishorM/laya.git#6d942c92081fbc139e736bbd9ac0023223c29b7f  # v0.3.22
    command: ["laya-serve"]
    environment:
      - LAYA_DEVICE=cpu
      - LAYA_MODELS=english,multilingual   # the two auto-routing picks; skips typed-decisions
      - LAYA_THREADS=4
      - LAYA_API_KEY=${LAYA_API_KEY}
    volumes:
      - laya-cache:/home/laya/.cache/huggingface
  ```

  Then set `LAYA_BASE_URL=http://laya:8000` and the same `LAYA_API_KEY` on the pipeline.
- **`local`** — the checkpoint is loaded in-process on the Pipelines server and reused for every request. Needs `pip install laya` in the Pipelines container (it pulls `torch` and `transformers`); it is deliberately not in the frontmatter `requirements`, which the Pipelines server would reinstall on every boot. The first build costs ~7–10 s on CPU plus a few hundred MB of download on a cold Hub cache; `LAYA_PRELOAD` (default on) builds the `english` and `multilingual` checkpoints during `on_startup`, off the event loop, instead of on the first chat message. Loading is guarded by a lock, so concurrent requests build it once.

`LAYA_CHECKPOINT` empty (the default) uses Laya's `Router`, which picks a checkpoint per request from the detected language. Set it to `english`, `multilingual`, `typed-decisions`, or any `repo/id` (optionally `repo/id#subfolder`) to pin one.

### 3. Model selection & failover

Identical to the TypeSafe variant. Each of the four model valves (`CHEAP_MODEL`, `EXPENSIVE_MODEL`, `CODE_CHEAP_MODEL`, `CODE_EXPENSIVE_MODEL`) holds a comma-separated, ordered list of `provider:model` entries, e.g.:

```
openrouter:deepseek/deepseek-v4-flash-0731:free, orcarouter:deepseek/deepseek-v4-flash-free
```

The pipeline tries each entry in order against the matching provider's OpenAI-compatible `/chat/completions` endpoint. On any failure (HTTP error, timeout, connection error) it logs the error and moves to the next entry. If every entry fails, the pipeline returns a single error string listing all the failures instead of raising.

- **OpenRouter** (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`)
- **OrcaRouter** (`ORCAROUTER_API_KEY`, `ORCAROUTER_BASE_URL`)

### 4. Streaming & footer

Responses are proxied through as-is, streaming or not (`body["stream"]`). When `SHOW_MODEL_FOOTER` is enabled, a Markdown footer such as:

> 🔀 `deepseek/deepseek-v4-pro` · expensive/code

is appended to the reply. The footer is stripped back out of assistant messages before they're re-sent as context on the next turn, so neither the router nor the backend ever sees its own footer as conversation content.

## Configuration

All settings are exposed as Pipelines **Valves** and can also be seeded from environment variables at startup.

| Valve | Env var | Purpose |
|---|---|---|
| `LAYA_MODE` | `LAYA_MODE` | `server` (`laya-serve` over HTTP, default) or `local` (in-process) |
| `LAYA_CHECKPOINT` | `LAYA_CHECKPOINT` | Pin a checkpoint (`english`, `multilingual`, `typed-decisions`, `repo/id[#subfolder]`); empty = auto-route per language |
| `LAYA_DEVICE` | `LAYA_DEVICE` | `cpu` (default) or `cuda` |
| `LAYA_NUM_THREADS` | — | `torch.set_num_threads()` for the classifier; `0` leaves torch's default |
| `LAYA_PRELOAD` | — | Build the checkpoint during startup instead of on the first message |
| `LAYA_MAX_LOADED` | — | How many checkpoints the `Router` may keep resident (default `2`: auto-routing switches between `english` and `multilingual`, so `1` rebuilds one per language switch) |
| `LAYA_LANG` | — | Force a language hint (`pt`, `de`, …) instead of auto-detection |
| `LAYA_MAX_LEN` | — | Truncate classifier input; `0` = the checkpoint's own context length |
| `LAYA_BASE_URL` / `LAYA_API_KEY` | same | `laya-serve` location and auth (`server` mode only) |
| `LAYA_TIMEOUT_SECONDS` | — | Timeout for the `server`-mode classification call |
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

- An [Open WebUI Pipelines](https://github.com/open-webui/pipelines) server to host this function, Python 3.10+.
- At least one of an OpenRouter or OrcaRouter API key for the actual backend completions.
- **No key of any kind for the routing** — Laya is Apache-2.0 and runs locally.

Python dependencies: only `requests` and `pydantic` in `server` mode. `local` mode also needs `laya`, which brings `torch` 2.14+, `transformers` 5.x and `huggingface_hub` 1.x along, so budget ~2 GB of image/disk and a few hundred MB per checkpoint.

## Installation

1. Copy `laya_router_pipeline.py` into your Open WebUI Pipelines server (via the Admin UI's "Upload Pipeline" or by mounting it in the `pipelines` directory).
2. Run `laya-serve` (see [Two deployment modes](#2-two-deployment-modes)) and set `LAYA_BASE_URL` / `LAYA_API_KEY` plus the backend valves listed above.
3. Select "Laya Router" as a model in Open WebUI — every message sent to it is classified and routed automatically.

The first `laya-serve` startup downloads the checkpoints from the Hugging Face Hub; `/health` answers once they are built. Classifications are logged as `[LayaRouter] classified prompt as 'expensive'/'code' (laya: english)`.

## Tests

```bash
pip install requests pydantic
python test_laya_router_pipeline.py
```

Runs in `server` mode with `requests.post` stubbed, so neither Laya nor the network is needed.

## License

MIT (this pipeline). Laya itself is Apache-2.0.
