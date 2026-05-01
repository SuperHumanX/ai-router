# @pmuppirala/ai-router

> Provider-agnostic AI router with streaming, tool-use, and automatic fallback.

This repo contains **two independent implementations** of the same routing concept:

| | TypeScript (`src/`) | Python (`gateway.py`) |
|---|---|---|
| **Package** | `@pmuppirala/ai-router` (npm) | Drop-in single file |
| **Used by** | Web frontends, Node.js services | CatalogValidator · portfolio_tracker · physician |
| **Providers** | Anthropic · OpenAI · Gemini · Local | Anthropic · OpenAI · Local SLM |
| **Routing** | weighted · round-robin · single · auto | weighted · round-robin · local-first · auto |
| **Local LLM** | Ollama / LM Studio / vLLM | Ollama + mlx_lm.server (auto-detected) |

---

## Python Gateway (`gateway.py`)

Single-file, zero-dependency LLM gateway for all Python projects. Copy into any project as `ai_router.py`.

### Three-tier routing

```
Tier 1  Local SLM          probed at call time — free, private, no API cost
          • mlx_lm.server   auto-detected when ":11435" or "mlx" in LOCAL_INTEL_URL
          • Ollama           all other URLs → /api/chat

Tier 2  Cloud (weighted)   automatic fallback when local is unreachable
          • Anthropic Claude  AI_ROUTER_WEIGHT_ANTHROPIC (default 0.7)
          • OpenAI GPT-4o     AI_ROUTER_WEIGHT_OPENAI    (default 0.3)

Tier 3  Error              raises RuntimeError so callers can surface it
```

No flags, no restarts. The gateway probes `LOCAL_INTEL_URL` on every call. If the SLM is up, it's used for free. If it's down, cloud kicks in transparently.

### Centralized key store

All API keys live in **one place** — `~/Projects/ai-router/.env`. Projects never store cloud keys:

```dotenv
# ~/Projects/ai-router/.env  — the single source of truth
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=AIza...     # future
GROK_API_KEY=xai-...       # future
```

Per-project `.env` only needs routing weights and the local SLM URL:

```dotenv
# CatalogValidator / portfolio_tracker / physician  — .env
LOCAL_INTEL_URL=http://100.66.27.15:11435   # Tailscale IP of Mac SLM (omit to skip Tier 1)
AI_ROUTER_MODE=weighted
AI_ROUTER_WEIGHT_ANTHROPIC=0.7
AI_ROUTER_WEIGHT_OPENAI=0.3
```

### Quick start

```python
from agents.ai_router import router, ChatMessage   # or catalog_router / finance_router / health_router

# Multi-turn chat
result = router.chat(
    messages=[ChatMessage(role="user", content="Summarise this week's market moves.")],
    system="You are a financial analyst.",
    model_hint="smart",    # "fast" | "smart" | "structured"
    max_tokens=800,
)
print(result.text)      # answer string
print(result.provider)  # "local" | "anthropic" | "openai"
print(result.model)     # e.g. "claude-sonnet-4-5"

# Single-turn convenience (CatalogValidator style)
cypher = router.complete(
    system="You are a KuzuDB Cypher expert. Return only raw Cypher.",
    user="Which suppliers stock more than 500 SKUs?",
    task="cypher",         # "cypher"|"structured" → fast model
                           # "summarize"|"analysis"|"general" → smart model
)
```

### Domain aliases

Each project imports the same singleton under a domain-specific name:

```python
from agents.ai_router import catalog_router   # CatalogValidator (retail)
from agents.ai_router import finance_router   # portfolio_tracker (finance)
from agents.ai_router import health_router    # physician (health)
```

### Deploying to a new project

```bash
cp ~/Projects/ai-router/gateway.py <project>/agents/ai_router.py
# Set LOCAL_INTEL_URL + AI_ROUTER_* weights in project .env
# API keys are picked up automatically from ~/Projects/ai-router/.env
```

### Env var reference (Python gateway)

| Variable | Default | Description |
|---|---|---|
| `LOCAL_INTEL_URL` | `http://localhost:11435` | Local SLM endpoint (empty = skip Tier 1) |
| `OLLAMA_URL` | — | Alias for `LOCAL_INTEL_URL` (legacy) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model name for Ollama requests |
| `AI_ROUTER_MODE` | `weighted` | `weighted` · `round-robin` · `anthropic` · `openai` · `local` · `auto` |
| `AI_ROUTER_WEIGHT_ANTHROPIC` | `0.7` | Relative weight for Anthropic |
| `AI_ROUTER_WEIGHT_OPENAI` | `0.3` | Relative weight for OpenAI |
| `AI_ROUTER_ANTHROPIC_MODEL` | `claude-haiku-4-5` | Fast/structured model |
| `AI_ROUTER_ANTHROPIC_SMART` | `claude-sonnet-4-5` | Smart model |
| `AI_ROUTER_OPENAI_MODEL` | `gpt-4o-mini` | Fast/structured model |
| `AI_ROUTER_OPENAI_SMART` | `gpt-4o` | Smart model |
| `AI_GATEWAY_CONFIG` | `~/Projects/ai-router/.env` | Override path for centralized key store |

---

## TypeScript Package (`src/`)

Supports **Anthropic Claude · OpenAI GPT · Google Gemini · Local LLMs** (Ollama, LM Studio, vLLM).

- 🔀 **4 routing modes**: single-provider, weighted random, round-robin, auto-fallback
- 🛠 **Tool-use / function-calling** unified across all providers
- ⚡ **Streaming** via `AsyncGenerator<StreamEvent>`
- 🔌 **Zero framework coupling** — no Supabase, no HTTP layer, no Next.js
- 🔑 **Env-var driven config** out of the box; inject a `configFetcher` for dynamic DB-backed config

---

## Install (TypeScript)

```bash
npm install @pmuppirala/ai-router
```

---

## Quick start — env vars (TypeScript)

```ts
// router.ts
import { createRouterFromEnv } from "@pmuppirala/ai-router";
export const router = createRouterFromEnv();
```

Set your keys and mode in `.env`:

```dotenv
# Provider keys (include only the providers you want active)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Local LLM (Ollama default — remove or set empty to disable)
AI_ROUTER_LOCAL_BASE_URL=http://localhost:11434/v1

# Routing
AI_ROUTER_MODE=weighted          # anthropic | openai | gemini | local | weighted | round-robin

# Weights (for "weighted" mode — any positive numbers, auto-normalised)
AI_ROUTER_WEIGHT_ANTHROPIC=2
AI_ROUTER_WEIGHT_OPENAI=2
AI_ROUTER_WEIGHT_GEMINI=1
AI_ROUTER_WEIGHT_LOCAL=1

# Model overrides
AI_ROUTER_ANTHROPIC_MODEL=claude-opus-4-5
AI_ROUTER_OPENAI_MODEL=gpt-4o
AI_ROUTER_GEMINI_MODEL=gemini-2.0-flash
AI_ROUTER_LOCAL_MODEL=llama3.2
```

---

## Usage

```ts
import { createRouterFromEnv } from "@pmuppirala/ai-router";

const router = createRouterFromEnv();

for await (const event of router.stream({
  systemPrompt: "You are a helpful assistant.",
  messages:     [{ role: "user", content: "What is the capital of France?" }],
  tools:        [],                          // no tools → pure chat
  toolExecutor: async () => ({              // required even when tools = []
    summary:   "",
    citations: [],
  }),
})) {
  switch (event.t) {
    case "provider": console.log("Using:", event.v); break;
    case "model":    console.log("Model:", event.v); break;
    case "text":     process.stdout.write(event.v);  break;
    case "cite":     console.log("Citations:", event.v); break;
    case "done":     console.log("\n[done]"); break;
    case "error":    console.error("Error:", event.v); break;
  }
}
```

---

## Tool use

```ts
const TOOLS = [{
  name:        "search_web",
  description: "Search the web for current information",
  input_schema: {
    type:       "object" as const,
    properties: { query: { type: "string", description: "Search query" } },
    required:   ["query"],
  },
}];

for await (const event of router.stream({
  systemPrompt: "You are a research assistant.",
  messages:     [{ role: "user", content: "Latest AI news?" }],
  tools:        TOOLS,
  maxRounds:    5,
  toolExecutor: async (name, input) => {
    if (name === "search_web") {
      const results = await mySearchFn(String(input.query));
      return {
        summary:   results.map((r) => r.snippet).join("\n"),
        citations: results,   // surfaced in the "cite" event
      };
    }
    return { summary: `Unknown tool: ${name}`, citations: [] };
  },
})) { /* handle events */ }
```

---

## Advanced: DB-backed dynamic config

```ts
import { AIRouter } from "@pmuppirala/ai-router";

const router = new AIRouter({
  anthropicApiKey: process.env.ANTHROPIC_API_KEY!,
  openaiApiKey:    process.env.OPENAI_API_KEY!,
  geminiApiKey:    process.env.GEMINI_API_KEY!,

  // Config fetched from your database — router caches for 10 s
  configFetcher: async () => {
    const row = await db.from("ai_router_config").select("*").single();
    return row.data;
  },

  // Round-robin persistence
  // onRoundRobin is passed per-call in stream() params
});

for await (const event of router.stream({
  // ...
  onRoundRobin: async (selected) => {
    await db.from("ai_router_config").update({ last_provider: selected });
    router.bustCache();
  },
})) { /* ... */ }
```

---

## StreamEvent reference

| `t`        | `v` type   | Description                                    |
|------------|------------|------------------------------------------------|
| `provider` | `Provider` | Which provider was selected                    |
| `model`    | `string`   | Model name                                     |
| `tool`     | `string`   | Tool/query being executed                      |
| `text`     | `string`   | Incremental text chunk (~30 chars)             |
| `cite`     | `unknown[]`| Citation objects from toolExecutor             |
| `done`     | —          | Stream finished successfully                   |
| `error`    | `string`   | Unrecoverable error (all providers failed)     |

---

## Env var reference

| Variable                     | Default                          | Description                        |
|------------------------------|----------------------------------|------------------------------------|
| `ANTHROPIC_API_KEY`          | —                                | Enables Anthropic provider         |
| `OPENAI_API_KEY`             | —                                | Enables OpenAI provider            |
| `GEMINI_API_KEY`             | —                                | Enables Gemini provider            |
| `AI_ROUTER_LOCAL_BASE_URL`   | `http://localhost:11434/v1`      | Local LLM endpoint (empty=disable) |
| `AI_ROUTER_MODE`             | `weighted`                       | Routing strategy                   |
| `AI_ROUTER_LAST_PROVIDER`    | `openai`                         | Round-robin cold-start seed        |
| `AI_ROUTER_WEIGHT_ANTHROPIC` | `1`                              | Relative weight                    |
| `AI_ROUTER_WEIGHT_OPENAI`    | `1`                              | Relative weight                    |
| `AI_ROUTER_WEIGHT_GEMINI`    | `1`                              | Relative weight                    |
| `AI_ROUTER_WEIGHT_LOCAL`     | `1`                              | Relative weight                    |
| `AI_ROUTER_ANTHROPIC_MODEL`  | `claude-opus-4-5`                | Model override                     |
| `AI_ROUTER_OPENAI_MODEL`     | `gpt-4o`                         | Model override                     |
| `AI_ROUTER_GEMINI_MODEL`     | `gemini-2.0-flash`               | Model override                     |
| `AI_ROUTER_LOCAL_MODEL`      | `llama3.2`                       | Model override                     |
| `AI_ROUTER_CACHE_TTL_MS`     | `10000`                          | Config cache TTL (0 = no cache)    |

---

## License

MIT © Prakash Muppirala
