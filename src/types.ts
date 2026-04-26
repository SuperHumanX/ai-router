// ── Public types for @pmuppirala/ai-router ───────────────────────────────────

/** All supported providers. */
export type Provider =
  | "anthropic"  // Anthropic Claude (native SDK)
  | "openai"     // OpenAI GPT (native SDK)
  | "gemini"     // Google Gemini (OpenAI-compatible endpoint)
  | "local";     // Ollama / LM Studio (OpenAI-compatible endpoint)

/** Routing strategy. */
export type RouterMode =
  | "anthropic"    // always Anthropic
  | "openai"       // always OpenAI
  | "gemini"       // always Gemini
  | "local"        // always local LLM
  | "weighted"     // probabilistic, per-provider weights
  | "round-robin"; // strict rotation across enabled providers

/**
 * Per-provider weights used in "weighted" mode.
 * Values are normalised automatically so they don't need to sum to 1.
 * Providers not in the map (or with weight 0) are skipped.
 */
export type ProviderWeights = Partial<Record<Provider, number>>;

/** Full router configuration. */
export interface RouterConfig {
  mode:            RouterMode;
  /** Used when mode = "weighted". Default: equal weight for all enabled providers. */
  weights:         ProviderWeights;
  /** For round-robin state persistence: tracks the last used provider. */
  last_provider:   Provider;
  /** Model names per provider. */
  anthropic_model: string;
  openai_model:    string;
  gemini_model:    string;
  local_model:     string;
}

/**
 * Structured events emitted by stream().
 *
 * t = "provider"  → which provider was selected (may appear twice if fallback)
 * t = "model"     → model name for the selected provider
 * t = "tool"      → tool call in progress (value = query or tool name)
 * t = "text"      → incremental text chunk
 * t = "cite"      → citation objects collected during tool calls
 * t = "done"      → stream finished successfully
 * t = "error"     → unrecoverable error (all providers failed)
 */
export type StreamEvent =
  | { t: "provider"; v: Provider }
  | { t: "model";    v: string }
  | { t: "tool";     v: string }
  | { t: "text";     v: string }
  | { t: "cite";     v: unknown[] }
  | { t: "done" }
  | { t: "error";    v: string };

/** Minimal chat message shape (role + string content). */
export interface CommonMessage {
  role:    "user" | "assistant";
  content: string;
}

/** Provider-agnostic tool definition (mirrors Anthropic's input_schema). */
export interface RouterTool {
  name:        string;
  description: string;
  input_schema: {
    type:       "object";
    properties: Record<string, { type: string; description: string; enum?: string[] }>;
    required:   string[];
  };
}

/** What a tool executor must return. */
export interface ToolResult {
  /** Text shown back to the model. */
  summary:   string;
  /** Structured objects surfaced in the "cite" event. */
  citations: unknown[];
}

/**
 * App-injected function to execute a named tool.
 * Receives the parsed input object from the model.
 */
export type ToolExecutor = (
  name:  string,
  input: Record<string, unknown>,
) => Promise<ToolResult>;

/**
 * Optional callback for persisting round-robin state.
 * The consuming app stores `selected` wherever it likes (DB, KV, env, etc.)
 * so the next router instance can pick up the correct next provider.
 */
export type RoundRobinPersist = (selected: Provider) => Promise<void> | void;

/** Parameters for a single streaming agent-loop call. */
export interface AgentLoopParams {
  systemPrompt:  string;
  messages:      CommonMessage[];
  tools:         RouterTool[];
  maxRounds?:    number;
  toolExecutor:  ToolExecutor;
  /** Called after round-robin selects — use to persist last_provider. */
  onRoundRobin?: RoundRobinPersist;
  /**
   * Override the model for this specific call, bypassing the config's
   * per-provider model setting. Useful for domain-specific models
   * (e.g. "meditron" for health queries, "finllama" for finance) or
   * one-off calls that need a different capability tier.
   *
   * The provider is still selected by the normal routing logic —
   * only the model name is swapped.
   */
  modelOverride?: string;
}

/** Options passed to the AIRouter constructor. */
export interface AIRouterOptions {
  /** Anthropic API key. Leave empty string to disable. */
  anthropicApiKey?: string;
  /** OpenAI API key. Leave empty string to disable. */
  openaiApiKey?:   string;
  /** Google Gemini API key. Leave empty string to disable. */
  geminiApiKey?:   string;
  /**
   * Base URL for a local OpenAI-compatible server (Ollama, LM Studio, etc.).
   * Default: http://localhost:11434/v1 (Ollama default).
   * Set to empty string to disable local provider.
   */
  localBaseUrl?:   string;

  /**
   * Async function that returns partial config overrides.
   * If omitted the router uses DEFAULT_CONFIG.
   * Implement this to load from a DB, env, remote config, etc.
   */
  configFetcher?: () => Promise<Partial<RouterConfig>>;

  /**
   * TTL (ms) for the local config cache. Default: 10 000 (10 s).
   * Set to 0 to always call configFetcher fresh.
   */
  configTtlMs?: number;
}
