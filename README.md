# @pmuppirala/ai-router

> Provider-agnostic AI router with streaming, tool-use, and automatic fallback.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how routing actually works across
clients, config precedence quirks, and the drift-history record from the
2026-09-23 consolidation. This README covers env vars and quick-start usage.

This repo contains **two independent implementations** of the same routing concept:

| | TypeScript (`src/`) | Python (`gateway.py`) |
|---|---|---|
| **Package** | `@pmuppirala/ai-router` (npm) | Drop-in single file |
| **Used by** | physician-web (Next.js) | CatalogValidator · portfolio_tracker · physician |
| **Providers** | Anthropic · OpenAI · Gemini · Local | OpenRouter · Gemini · Anthropic · OpenAI · Local SLM |
| **Routing** | weighted · round-robin · single-provider pin · auto | weighted · round-robin · single-provider pin · local-only · remote-forward |
| **Local LLM** | Ollama / LM Studio / vLLM | Ollama + mlx_lm.server (auto-detected) |

Both sides independently converged on the same shape (OpenRouter as a shared
BYOK fallback so one key covers all cloud providers) — see each section below
for how it's wired on that side.

---

## Python Gateway (`gateway.py`)

Single-file, zero-dependency LLM gateway for all Python projects. Copy into any project as `ai_router.py`.

### Routing tiers

```
Tier 0  Remote gateway     if REMOTE_GATEWAY_URL is set, every call is forwarded
                            there (e.g. server.py over Tailscale) instead of
                            calling providers directly — lets a remote host use
                            the router without holding any API keys itself.

Tier 1  Local SLM          probed at call time — free, private, no API cost.
                            Skipped when USE_LOCAL_SLM=false or model_hint="smart".
          • mlx_lm.server   auto-detected when ":11435" or "mlx" in LOCAL_INTEL_URL;
                             LOCAL_INTEL_DOMAIN, if set, hot-swaps the right LoRA
          • Ollama           all other URLs → /api/chat

Tier 2  Cloud (weighted)   automatic fallback when local is unreachable
          • OpenRouter        AI_ROUTER_WEIGHT_OPENROUTER (default 0.75) — primary;
                               one key, routes to Anthropic/OpenAI/Google upstream
          • Gemini            AI_ROUTER_WEIGHT_GEMINI     (default 0.15) — Vertex AI ADC, else OpenRouter BYOK
          • Anthropic Claude  AI_ROUTER_WEIGHT_ANTHROPIC  (default 0.05) — direct key, else OpenRouter BYOK
          • OpenAI GPT-4o     AI_ROUTER_WEIGHT_OPENAI     (default 0.05) — direct key, else OpenRouter BYOK

        Gemini/Anthropic/OpenAI each prefer their own direct path and only
        route through OpenRouter (OPENROUTER_API_KEY) when that path is missing
        or the direct call fails — so a single OPENROUTER_API_KEY in the master
        .env can serve as a catch-all without holding a per-provider key, while
        still giving each provider its own weighted slot when it has one.
        Caveat: the OpenRouter fallback path for Anthropic doesn't forward
        `tools` (no Anthropic-native tool schema over OpenRouter's OpenAI-compat
        API) — a call that needs tools should have a direct ANTHROPIC_API_KEY.

Tier 3  Error              raises RuntimeError so callers can surface it
```

No flags, no restarts. The gateway probes `LOCAL_INTEL_URL` on every call (unless skipped per above). If the SLM is up, it's used for free. If it's down, cloud kicks in transparently.

### Centralized key store

All API keys live in **one place** — `~/Projects/ai-router/.env`. Projects never store cloud keys:

```dotenv
# ~/Projects/ai-router/.env  — the single source of truth
OPENROUTER_API_KEY=sk-or-...   # primary cloud path + shared BYOK fallback for the rest
GEMINI_PROJECT=gen-lang-client-0271077908   # Vertex AI project (ADC, no key)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GROK_API_KEY=xai-...       # future
```

Per-project `.env` only needs routing weights and the local SLM URL:

```dotenv
# CatalogValidator / portfolio_tracker / physician  — .env
LOCAL_INTEL_URL=http://100.66.27.15:11435   # Tailscale IP of Mac SLM (omit to skip Tier 1)
LOCAL_INTEL_DOMAIN=retail                   # optional — hot-swaps the matching LoRA adapter
AI_ROUTER_MODE=weighted
# Defaults (openrouter=0.75, gemini=0.15, anthropic=0.05, openai=0.05) are usually fine as-is
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
print(result.provider)  # "local" | "openrouter" | "gemini" | "anthropic" | "openai"
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
cp ~/Projects/ai-router/gateway.py ~/Projects/ai-router/router_policy.py <project>/agents/
# router_policy.py is optional (only needed if AI_ROUTER_INTELLIGENT_ROUTING
# will be enabled — see ARCHITECTURE.md's "Intelligent routing" section) but
# copy it alongside gateway.py by default so it's there when needed. Copy
# both into whichever directory currently holds the project's ai_router.py —
# not always agents/ (CatalogValidator's is backend/mcp_domain/).
# Set LOCAL_INTEL_URL + AI_ROUTER_* weights in project .env
# API keys are picked up automatically from ~/Projects/ai-router/.env
```

### Env var reference (Python gateway)

| Variable | Default | Description |
|---|---|---|
| `REMOTE_GATEWAY_URL` | — | If set, forwards calls to this `server.py` instance instead of calling providers locally. `chat()` and `complete_with_meta()` (both `skip_local` values), `record_feedback()`, and `get_metrics_snapshot()` all forward — Phase 10 extended `server.py`'s surface (`/v1/complete_with_meta`, `/v1/record_feedback`, `GET /v1/metrics`, alongside the original `/v1/chat`+`/health`) to cover them. Pinned-provider calls (`chat(provider=...)`) forward too — the remote check runs before the pin is examined. |
| `LOCAL_INTEL_URL` | `http://localhost:11435` | Local SLM endpoint (empty = skip Tier 1) |
| `LOCAL_INTEL_DOMAIN` | — | LoRA domain to hot-swap (`retail`/`finance`/`health`) |
| `OLLAMA_URL` | — | Alias for `LOCAL_INTEL_URL` (legacy) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model name for Ollama requests |
| `USE_LOCAL_SLM` | `true` | `false` skips the local tier entirely |
| `AI_ROUTER_MODE` | `weighted` | `weighted` · `round-robin` · `openrouter` · `gemini` · `anthropic` · `openai` · `local` |
| `AI_ROUTER_WEIGHT_OPENROUTER` | `0.75` | Relative weight for OpenRouter |
| `AI_ROUTER_WEIGHT_GEMINI` | `0.15` | Relative weight for Gemini |
| `AI_ROUTER_WEIGHT_ANTHROPIC` | `0.05` | Relative weight for Anthropic |
| `AI_ROUTER_WEIGHT_OPENAI` | `0.05` | Relative weight for OpenAI |
| `AI_ROUTER_OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Fast/structured model |
| `AI_ROUTER_OPENROUTER_SMART` | `anthropic/claude-sonnet-4.5` | Smart model |
| `AI_ROUTER_ANTHROPIC_MODEL` | `claude-haiku-4-5` | Fast/structured model |
| `AI_ROUTER_ANTHROPIC_SMART` | `claude-sonnet-4-5` | Smart model |
| `AI_ROUTER_OPENAI_MODEL` | `gpt-4o-mini` | Fast/structured model |
| `AI_ROUTER_OPENAI_SMART` | `gpt-4o` | Smart model |
| `AI_ROUTER_GEMINI_MODEL` | `gemini-2.5-flash` | Fast/structured model |
| `AI_ROUTER_GEMINI_SMART` | `gemini-2.5-pro` | Smart model |
| `GEMINI_PROJECT` | `gen-lang-client-0271077908` | Vertex AI project (ADC, no key) |
| `GEMINI_LOCATION` | `us-central1` | Vertex AI region |
| `AI_GATEWAY_CONFIG` | `~/Projects/ai-router/.env` | Override path for centralized key store |
| `AI_ROUTER_INTELLIGENT_ROUTING` | `false` | Shadow-mode routing-decision logging (Phase 1 of the intelligent-routing plan — see ARCHITECTURE.md). Computes and logs a `RoutingDecision` per call via `router_policy.py`; does not yet change dispatch. |
| `AI_ROUTER_DOMAIN_MODE` | `override` | `override`\|`hint`\|`auto` — Phase 2. Only takes effect when `AI_ROUTER_INTELLIGENT_ROUTING=true`. `override` preserves today's exact behavior (`LOCAL_INTEL_DOMAIN` always wins); `hint`/`auto` let keyword-based domain classification actually influence the local call — see ARCHITECTURE.md. |
| `AI_ROUTER_ROUTE_GATING` | `false` | Phase 3. Only takes effect when `AI_ROUTER_INTELLIGENT_ROUTING=true`. Off (default): `chat()` dispatch is byte-identical to pre-Phase-3. On: `RoutingDecision.route` actually determines whether Tier 1 (local) is attempted, and can let a `model_hint="smart"` request reach local when the router decided it's genuinely viable — see ARCHITECTURE.md. |
| `AI_ROUTER_VERIFY_LOCAL` | `false` | Phase 4. Only takes effect when `AI_ROUTER_INTELLIGENT_ROUTING=true`. Off (default): a local response is returned as soon as `_call_local` returns non-`None` — same as always. On: a deterministic `ResponseVerifier` check must pass first, or the call escalates to cloud instead — see ARCHITECTURE.md for what it does and does not catch. |
| `AI_ROUTER_FRONTIER_POLICY` | `weighted` | Phase 5. Only takes effect when `AI_ROUTER_INTELLIGENT_ROUTING=true`. `weighted` (default) preserves today's exact primary cloud-provider selection. `capability` picks a specific `(provider, model)` pair via expected-utility scoring, scoped to the requested `model_hint`'s tier — see ARCHITECTURE.md for the current placeholder-data caveat. |
| `AI_ROUTER_EXPERIENCE_STORE` | `false` | Phase 6. Only takes effect when `AI_ROUTER_INTELLIGENT_ROUTING=true`. Off (default): nothing is recorded, zero behavior change. On: an append-only JSONL trace of each Tier 1/Tier 2 request is written to `data/experience_traces.jsonl` — see ARCHITECTURE.md for exact scope. |
| `AI_ROUTER_EXPERIENCE_REDACT` | `full` | Phase 6, redaction mode upgraded in Phase 7. Only matters when `AI_ROUTER_EXPERIENCE_STORE=true`. `full` (default; `true` still accepted as an alias): request/response text is redacted to a length+hash placeholder before being stored — not usable for training content. `partial`: structured PII (dollar amounts, SSNs, phone numbers, emails, long account numbers) is regex-scrubbed to `<REDACTED_...>` markers, rest of the text kept real — usable for training, but **cannot** catch free-text PII like names or addresses; a stated limitation, not a guarantee. `none` (`false` still accepted as an alias): raw content is persisted. `partial`/`none` are explicit per-deployment opt-ins, never a default — see ARCHITECTURE.md's Phase 7 section. |
| `AI_ROUTER_CLIENT_APP` | `unknown` | Phase 6. A label on each trace's `client_app` field — no effect on routing. |
| `AI_ROUTER_POLICY_ID` | unset | Phase 9. Only takes effect when `AI_ROUTER_INTELLIGENT_ROUTING=true`. Unset (default): `DeterministicRouterPolicy` records `router_version` as `"deterministic-v1"`, unchanged. Set: tags this instance's traces/metrics with the given string instead — lets two differently-configured instances (e.g. a canary with a lower threshold table) be told apart downstream, without subclassing anything. No traffic-splitting — see ARCHITECTURE.md's Phase 9 section for how a deployment actually uses this. |
| `AI_ROUTER_VERIFIER_ID` | unset | Phase 9. Only takes effect when `AI_ROUTER_VERIFY_LOCAL=true`. Same idea as `AI_ROUTER_POLICY_ID`, for `DeterministicVerifier`'s `verifier_version`. |

### Observability + feedback (Phase 8, Python gateway)

No new env vars — these ride on flags already in the table above. New
public methods on `AIGateway`, available to any deployment that wants to
wire them up (none does live yet — see ARCHITECTURE.md's Phase 8 section):

- **`get_metrics_snapshot() -> dict`** — live, in-process, content-free
  counters (request/route/verifier counts, per-domain benchmark deviation,
  latency percentiles). Needs only `AI_ROUTER_INTELLIGENT_ROUTING=true`, no
  new flag. `log_metrics_snapshot()` logs the same dict at INFO level.
- **`record_feedback(trace_id, label, note=None) -> bool`** — attach real
  feedback (`"correct"`/`"incorrect"`/`"unrated"`) to a past response.
  `chat()`'s returned `RouterResponse.trace_id` is populated only when
  `AI_ROUTER_EXPERIENCE_STORE=true` (feedback needs a matching trace record
  to correlate against).
- **`training/compute_metrics.py`** — offline script (same posture as
  `training/generate_candidates.py`): reads `data/experience_traces.jsonl` +
  `data/experience_feedback.jsonl`, computes per-domain rates including the
  real false-accept rate (feedback-backed, not a proxy) plus
  `feedback_coverage`. Run by hand, never wired into `chat()`. Phase 9
  added `by_router_version`/`by_verifier_version` breakdowns alongside
  `by_domain` — the canary-comparison tool for `AI_ROUTER_POLICY_ID`/
  `AI_ROUTER_VERIFIER_ID`-tagged cohorts.

---

## TypeScript Package (`src/`)

Supports **Anthropic Claude · OpenAI GPT · Google Gemini · Local LLMs** (Ollama, LM Studio, vLLM).

- 🔀 **4 routing modes**: single-provider, weighted random, round-robin, auto-fallback
- 🛠 **Tool-use / function-calling** unified across all providers
- ⚡ **Streaming** via `AsyncGenerator<StreamEvent>`
- 🔌 **Zero framework coupling** — no Supabase, no HTTP layer, no Next.js
- 🔑 **Env-var driven config** out of the box; inject a `configFetcher` for dynamic DB-backed config

**Type definitions only, no logic** (Phase 10): `RoutingDecision`,
`VerificationResult`, `ExperienceTrace`, `FeedbackRecord`, `DomainConfig`,
`ModelCapability` are exported, mirroring the Python gateway's Phase 1-9
data model shapes field-for-field (snake_case, matching the real JSON) —
useful for typing Node tooling that reads a Python-produced
`data/experience_traces.jsonl`/`data/experience_feedback.jsonl`. `AIRouter`
itself does not use or reference these — the actual routing/verification/
experience-store logic stays Python-only. See ARCHITECTURE.md's Phase 10
section.

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
