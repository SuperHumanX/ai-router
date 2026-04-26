/**
 * AIRouter — provider-agnostic multi-model router
 *
 * Supports: Anthropic Claude · OpenAI GPT · Google Gemini · Local (Ollama/LM Studio)
 *
 * Design principles:
 *  • Zero framework coupling — no Supabase, no HTTP, no Next.js
 *  • All config injected via configFetcher (app decides source: DB, env, file)
 *  • Tool execution fully injected — no hardcoded tools
 *  • Round-robin persistence delegated to onRoundRobin callback
 *  • Automatic fallback: primary fails → retry on next enabled provider
 *  • Gemini + Local reuse the OpenAI-compatible loop (fewer SDK deps)
 */

import Anthropic from "@anthropic-ai/sdk";
import OpenAI    from "openai";

import type {
  Provider,
  RouterConfig,
  ProviderWeights,
  StreamEvent,
  AgentLoopParams,
  AIRouterOptions,
} from "./types.js";

import { anthropicLoop }    from "./providers/anthropic.js";
import { openaiCompatLoop } from "./providers/openai-compat.js";

// ── Defaults ──────────────────────────────────────────────────────────────────

export const DEFAULT_CONFIG: RouterConfig = {
  mode:            "weighted",
  weights:         { anthropic: 0.5, openai: 0.5 },
  last_provider:   "openai",
  anthropic_model: "claude-opus-4-5",
  openai_model:    "gpt-4o",
  gemini_model:    "gemini-2.0-flash",
  local_model:     "llama3.2",
};

const DEFAULT_TTL_MS              = 10_000; // 10 s
const GEMINI_OPENAI_COMPAT_BASE   = "https://generativelanguage.googleapis.com/v1beta/openai/";
const OLLAMA_DEFAULT_BASE         = "http://localhost:11434/v1";

// ── AIRouter ──────────────────────────────────────────────────────────────────

export class AIRouter {
  // Anthropic uses its own SDK
  private anthropicClient: Anthropic | null = null;

  // One OpenAI client per OpenAI-compat endpoint
  private openaiClient: OpenAI  | null = null;
  private geminiClient: OpenAI  | null = null;
  private localClient:  OpenAI  | null = null;

  private configFetcher: (() => Promise<Partial<RouterConfig>>) | undefined;
  private ttlMs:         number;

  // Per-instance cache
  private _cachedConfig: RouterConfig | null = null;
  private _cacheExpiry:  number              = 0;

  constructor(options: AIRouterOptions) {
    // Only create clients for providers with credentials / endpoints
    if (options.anthropicApiKey) {
      this.anthropicClient = new Anthropic({ apiKey: options.anthropicApiKey });
    }
    if (options.openaiApiKey) {
      this.openaiClient = new OpenAI({ apiKey: options.openaiApiKey });
    }
    if (options.geminiApiKey) {
      this.geminiClient = new OpenAI({
        apiKey:  options.geminiApiKey,
        baseURL: GEMINI_OPENAI_COMPAT_BASE,
      });
    }
    // Local is enabled when localBaseUrl is set (even if empty key — Ollama needs none)
    const localBase = options.localBaseUrl ?? OLLAMA_DEFAULT_BASE;
    if (localBase) {
      this.localClient = new OpenAI({
        apiKey:  "ollama",   // placeholder — local servers typically ignore this
        baseURL: localBase,
      });
    }

    this.configFetcher = options.configFetcher;
    this.ttlMs         = options.configTtlMs ?? DEFAULT_TTL_MS;
  }

  // ── Config ─────────────────────────────────────────────────────────────────

  async fetchConfig(): Promise<RouterConfig> {
    const now = Date.now();
    if (this._cachedConfig && now < this._cacheExpiry) return this._cachedConfig;

    let overrides: Partial<RouterConfig> = {};
    if (this.configFetcher) {
      try {
        overrides = await this.configFetcher();
      } catch (err) {
        console.warn("[ai-router] configFetcher failed, using defaults:", err);
      }
    }

    this._cachedConfig = { ...DEFAULT_CONFIG, ...overrides };
    this._cacheExpiry  = now + this.ttlMs;
    return this._cachedConfig;
  }

  bustCache() {
    this._cachedConfig = null;
    this._cacheExpiry  = 0;
  }

  // ── Provider availability ──────────────────────────────────────────────────

  private enabledProviders(): Provider[] {
    const enabled: Provider[] = [];
    if (this.anthropicClient) enabled.push("anthropic");
    if (this.openaiClient)    enabled.push("openai");
    if (this.geminiClient)    enabled.push("gemini");
    if (this.localClient)     enabled.push("local");
    return enabled;
  }

  private isEnabled(p: Provider): boolean {
    switch (p) {
      case "anthropic": return !!this.anthropicClient;
      case "openai":    return !!this.openaiClient;
      case "gemini":    return !!this.geminiClient;
      case "local":     return !!this.localClient;
    }
  }

  // ── Provider selection ─────────────────────────────────────────────────────

  async selectProvider(
    onRoundRobin?: AgentLoopParams["onRoundRobin"],
  ): Promise<{ provider: Provider; config: RouterConfig }> {
    const config  = await this.fetchConfig();
    const enabled = this.enabledProviders();

    if (enabled.length === 0) {
      throw new Error("[ai-router] No providers configured. Supply at least one API key.");
    }

    let provider: Provider;

    switch (config.mode) {
      case "anthropic":
      case "openai":
      case "gemini":
      case "local": {
        // Explicit single-provider mode
        if (!this.isEnabled(config.mode)) {
          console.warn(`[ai-router] mode="${config.mode}" but that provider has no key — falling back to first enabled.`);
          provider = enabled[0];
        } else {
          provider = config.mode;
        }
        break;
      }

      case "round-robin": {
        // Cycle through enabled providers after last_provider
        const idx  = enabled.indexOf(config.last_provider);
        provider   = enabled[(idx + 1) % enabled.length];
        this.bustCache();
        if (onRoundRobin) {
          try { await onRoundRobin(provider); } catch { /* non-fatal */ }
        }
        break;
      }

      case "weighted":
      default: {
        provider = this._weightedSelect(config.weights, enabled);
        break;
      }
    }

    return { provider, config };
  }

  /** Weighted random selection; normalises weights automatically. */
  private _weightedSelect(weights: ProviderWeights, enabled: Provider[]): Provider {
    // Build list of (provider, weight) for enabled providers only
    const candidates = enabled.map((p) => ({ p, w: weights[p] ?? 1 })).filter((c) => c.w > 0);
    if (candidates.length === 0) return enabled[0];

    const total = candidates.reduce((s, c) => s + c.w, 0);
    let   roll  = Math.random() * total;
    for (const c of candidates) {
      roll -= c.w;
      if (roll <= 0) return c.p;
    }
    return candidates[candidates.length - 1].p;
  }

  // ── Streaming entrypoint ───────────────────────────────────────────────────

  /**
   * Run the full agentic loop, yielding StreamEvents.
   *
   * Normal event sequence:
   *   provider → model → (tool → …)* → text* → cite? → done
   *
   * On primary failure the router emits a new provider+model event and
   * transparently retries on the next available provider.
   */
  async *stream(params: AgentLoopParams): AsyncGenerator<StreamEvent> {
    const { onRoundRobin, modelOverride, ...loopParams } = params;
    const { provider, config } = await this.selectProvider(onRoundRobin);

    const enabled   = this.enabledProviders();
    const remaining = enabled.filter((p) => p !== provider);

    // modelOverride takes precedence over the per-provider config model
    const resolvedModel = modelOverride ?? this._modelFor(provider, config);

    yield { t: "provider", v: provider };
    yield { t: "model",    v: resolvedModel };

    try {
      yield* this._runLoop(provider, resolvedModel, loopParams);
      return;
    } catch (primaryErr) {
      console.error(`[ai-router] ${provider} failed:`, String(primaryErr));
    }

    // Try remaining providers in order until one succeeds.
    // modelOverride is preserved on fallback — same domain context, different provider.
    for (const fallback of remaining) {
      const fallbackModel = modelOverride ?? this._modelFor(fallback, config);
      console.warn(`[ai-router] → falling back to ${fallback} (${fallbackModel})`);
      yield { t: "provider", v: fallback };
      yield { t: "model",    v: fallbackModel };
      try {
        yield* this._runLoop(fallback, fallbackModel, loopParams);
        return;
      } catch (err) {
        console.error(`[ai-router] ${fallback} also failed:`, String(err));
      }
    }

    yield { t: "error", v: "All configured providers failed. Check API keys and connectivity." };
  }

  // ── Internal helpers ───────────────────────────────────────────────────────

  private _modelFor(provider: Provider, config: RouterConfig): string {
    switch (provider) {
      case "anthropic": return config.anthropic_model;
      case "openai":    return config.openai_model;
      case "gemini":    return config.gemini_model;
      case "local":     return config.local_model;
    }
  }

  private _runLoop(
    provider: Provider,
    model:    string,
    params:   Omit<AgentLoopParams, "onRoundRobin" | "modelOverride">,
  ): AsyncGenerator<StreamEvent> {
    switch (provider) {
      case "anthropic": return anthropicLoop   (this.anthropicClient!, model, params);
      case "openai":    return openaiCompatLoop(this.openaiClient!,    model, params);
      case "gemini":    return openaiCompatLoop(this.geminiClient!,    model, params);
      case "local":     return openaiCompatLoop(this.localClient!,     model, params);
    }
  }
}
