# @pmuppirala/ai-router

> Provider-agnostic AI router with streaming, tool-use, and automatic fallback.

Supports **Anthropic Claude · OpenAI GPT · Google Gemini · Local LLMs** (Ollama, LM Studio, vLLM).

- 🔀 **4 routing modes**: single-provider, weighted random, round-robin, auto-fallback
- 🛠 **Tool-use / function-calling** unified across all providers
- ⚡ **Streaming** via `AsyncGenerator<StreamEvent>`
- 🔌 **Zero framework coupling** — no Supabase, no HTTP layer, no Next.js
- 🔑 **Env-var driven config** out of the box; inject a `configFetcher` for dynamic DB-backed config

---

## Install

```bash
npm install @pmuppirala/ai-router
```

---

## Quick start — env vars

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

MIT © Praveen Muppirala
