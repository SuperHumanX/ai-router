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
  /** Optional domain sent to the local gateway so it hot-swaps the right LoRA
   *  adapter (e.g. "health"|"retail"|"finance"). Cloud providers ignore it. */
  local_domain?:   string;
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
   * OpenRouter API key (BYOK). When anthropicApiKey/openaiApiKey/geminiApiKey
   * are NOT supplied for a given provider, this key is used as a fallback so
   * that single key alone can power all three cloud providers via OpenRouter
   * instead of juggling separate direct keys. Anthropic requests routed this
   * way go through the OpenAI-compatible loop (OpenRouter's interface),
   * not the native Anthropic Messages API.
   */
  openrouterApiKey?: string;
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

// ── Intelligent-routing data model types (Phase 10) ──────────────────────────
//
// TYPE DEFINITIONS ONLY — mirroring the shape of the Python dataclasses in
// ai-router's router_policy.py (Phases 1-9), field-for-field, snake_case to
// match the real JSON these produce. NONE of the decision logic
// (DeterministicRouterPolicy, DeterministicVerifier, CapabilityFrontierRouter,
// the domain/model registries, JSONLExperienceStore, JSONLFeedbackStore, the
// redaction functions, etc.) is ported here or anywhere in this package —
// that logic is intentionally Python-only. These interfaces exist so
// TypeScript/Node tooling (e.g. a future dashboard reading
// data/experience_traces.jsonl or data/experience_feedback.jsonl) gets type
// safety over that JSON without a hand-maintained parallel implementation of
// the routing/verification/frontier-selection behavior itself. AIRouter (this
// package's actual router class) does not use or reference any of these.

/** Mirrors router_policy.py's RoutingDecision. */
export interface RoutingDecision {
  domain: string;
  subdomain: string | null;
  task: string;
  complexity: number;            // 0..1
  risk: number;                  // 0..1
  predicted_local_success: number; // 0..1
  selected_adapter: string | null;
  required_capabilities: string[];
  quality_threshold: number;
  route: "local" | "frontier";
  reason: string;
  router_version: string;
  /** DomainConfig.adapter_version for the resolved domain — the real
   *  version string (e.g. "v3"), NOT the adapter name (selected_adapter
   *  already covers that) — null when the domain has no adapter. */
  adapter_version: string | null;
}

/** Mirrors router_policy.py's VerificationResult. */
export interface VerificationResult {
  accepted: boolean;
  score: number;
  confidence: number;
  failure_reasons: string[];
  groundedness: number | null;
  completeness: number | null;
  policy_risk: string | null;
  verifier_version: string;
}

/** Mirrors router_policy.py's ExperienceTrace — one JSON line in
 *  data/experience_traces.jsonl when AI_ROUTER_EXPERIENCE_STORE is on.
 *  request/local_response/frontier_response's actual shape depends on the
 *  redact_mode that was active when the trace was written ("full" ->
 *  length+hash placeholder strings; "partial"/"none" -> real
 *  structure/text) — typed loosely (unknown/string) rather than as a
 *  discriminated union, since the mode isn't itself a trace field. */
export interface ExperienceTrace {
  trace_id: string;
  timestamp: string;             // ISO 8601
  client_app: string;
  request: unknown;
  context_metadata: {
    model_hint: string;
    domain_mode: string;
    force_frontier: boolean;
  };
  routing_decision: RoutingDecision;
  adapter_version: string | null;
  model_versions: { local: string | null; frontier: string | null };
  local_response: string | null;
  local_latency_ms: number | null;
  local_tokens: number | null;
  verification: VerificationResult | null;
  escalated: boolean;
  escalation_reason: string | null;
  frontier_provider: string | null;
  frontier_model: string | null;
  frontier_response: string | null;
  frontier_latency_ms: number | null;
  frontier_tokens: number | null;
  frontier_cost: number | null;
  tool_calls: unknown[];
  errors: string[];
  /** Always null on the trace itself — real feedback is a separate,
   *  append-only FeedbackRecord joined by trace_id, not written back here.
   *  See FeedbackRecord below. */
  user_feedback: null;
  final_source: "local" | "frontier";
  /** "capability-v1" | "weighted-v1" | null (null when the request never
   *  reached frontier selection at all — local succeeded). */
  frontier_router_version: string | null;
}

/** Mirrors router_policy.py's FeedbackRecord — one JSON line in
 *  data/experience_feedback.jsonl, correlated to an ExperienceTrace by
 *  trace_id (joined at read time, e.g. by training/compute_metrics.py —
 *  not stored on the trace itself). */
export interface FeedbackRecord {
  feedback_id: string;
  trace_id: string;
  timestamp: string;             // ISO 8601
  label: "correct" | "incorrect" | "unrated";
  note: string | null;
  client_app: string;
}

/** Mirrors router_policy.py's DomainConfig (the Phase 2 domain registry). */
export interface DomainConfig {
  name: string;
  adapter_name: string | null;
  adapter_version: string | null;
  supported_tasks: string[];
  local_model: string;
  tools: string[];
  retrieval_config: unknown | null;
  default_quality_threshold: number;
  risk_constraints: unknown | null;
  /** Real offline benchmark (e.g. execution accuracy), if measured. */
  benchmark_success_rate: number | null;
  keywords: string[];
}

/** Mirrors router_policy.py's ModelCapability (the Phase 5 model registry). */
export interface ModelCapability {
  provider: string;
  model: string;
  reasoning: number;
  coding: number;
  tool_use: number;
  structured_output: number;
  long_context: number;
  modalities: string[];
  cost_per_1k_tokens: number;
  avg_latency_ms: number;
  available: boolean;
}
