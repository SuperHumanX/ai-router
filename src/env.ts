/**
 * env.ts — Zero-code env-var driven configuration factory
 *
 * Usage (Node / Next.js / Vercel — any runtime that exposes process.env):
 *
 *   import { createRouterFromEnv } from "@pmuppirala/ai-router/env";
 *   export const router = createRouterFromEnv();
 *
 * Supported environment variables:
 *
 *   Provider keys:
 *     ANTHROPIC_API_KEY          Anthropic Claude
 *     OPENAI_API_KEY             OpenAI GPT
 *     GEMINI_API_KEY             Google Gemini
 *
 *   Local LLM (Ollama / LM Studio / vLLM):
 *     AI_ROUTER_LOCAL_BASE_URL   Default: http://localhost:11434/v1
 *                                Set to empty string to disable local provider.
 *
 *   Routing:
 *     AI_ROUTER_MODE             anthropic | openai | gemini | local |
 *                                weighted (default) | round-robin
 *     AI_ROUTER_LAST_PROVIDER    Last provider used (for round-robin cold-start)
 *
 *   Weights (for "weighted" mode — any positive numbers, auto-normalised):
 *     AI_ROUTER_WEIGHT_ANTHROPIC  Default: 1
 *     AI_ROUTER_WEIGHT_OPENAI     Default: 1
 *     AI_ROUTER_WEIGHT_GEMINI     Default: 1
 *     AI_ROUTER_WEIGHT_LOCAL      Default: 1
 *
 *   Model overrides:
 *     AI_ROUTER_ANTHROPIC_MODEL   Default: claude-opus-4-5
 *     AI_ROUTER_OPENAI_MODEL      Default: gpt-4o
 *     AI_ROUTER_GEMINI_MODEL      Default: gemini-2.0-flash
 *     AI_ROUTER_LOCAL_MODEL       Default: llama3.2
 *
 *   Cache:
 *     AI_ROUTER_CACHE_TTL_MS      Default: 10000 (10 s)
 *                                 Set to 0 to disable caching.
 */

import { AIRouter, DEFAULT_CONFIG } from "./router.js";
import type { RouterConfig, RouterMode, Provider, AIRouterOptions } from "./types.js";

/** Read a float env var, return undefined if missing/invalid. */
function envFloat(key: string): number | undefined {
  const v = process.env[key];
  if (!v) return undefined;
  const n = parseFloat(v);
  return isNaN(n) ? undefined : n;
}

/** Read an int env var, return undefined if missing/invalid. */
function envInt(key: string): number | undefined {
  const v = process.env[key];
  if (!v) return undefined;
  const n = parseInt(v, 10);
  return isNaN(n) ? undefined : n;
}

const VALID_MODES: RouterMode[] = ["anthropic", "openai", "gemini", "local", "weighted", "round-robin"];
const VALID_PROVIDERS: Provider[] = ["anthropic", "openai", "gemini", "local"];

/** Build a RouterConfig from environment variables. */
export function configFromEnv(): Partial<RouterConfig> {
  const modeRaw = process.env.AI_ROUTER_MODE;
  const mode    = modeRaw && VALID_MODES.includes(modeRaw as RouterMode)
    ? (modeRaw as RouterMode)
    : undefined;

  const lastRaw     = process.env.AI_ROUTER_LAST_PROVIDER;
  const last_provider = lastRaw && VALID_PROVIDERS.includes(lastRaw as Provider)
    ? (lastRaw as Provider)
    : undefined;

  const weights: Partial<RouterConfig["weights"]> = {};
  const wA = envFloat("AI_ROUTER_WEIGHT_ANTHROPIC"); if (wA !== undefined) weights.anthropic = wA;
  const wO = envFloat("AI_ROUTER_WEIGHT_OPENAI");    if (wO !== undefined) weights.openai    = wO;
  const wG = envFloat("AI_ROUTER_WEIGHT_GEMINI");    if (wG !== undefined) weights.gemini    = wG;
  const wL = envFloat("AI_ROUTER_WEIGHT_LOCAL");     if (wL !== undefined) weights.local     = wL;

  const config: Partial<RouterConfig> = {};
  if (mode)                                config.mode            = mode;
  if (last_provider)                       config.last_provider   = last_provider;
  if (Object.keys(weights).length > 0)     config.weights         = weights;
  if (process.env.AI_ROUTER_ANTHROPIC_MODEL) config.anthropic_model = process.env.AI_ROUTER_ANTHROPIC_MODEL;
  if (process.env.AI_ROUTER_OPENAI_MODEL)    config.openai_model    = process.env.AI_ROUTER_OPENAI_MODEL;
  if (process.env.AI_ROUTER_GEMINI_MODEL)    config.gemini_model    = process.env.AI_ROUTER_GEMINI_MODEL;
  if (process.env.AI_ROUTER_LOCAL_MODEL)     config.local_model     = process.env.AI_ROUTER_LOCAL_MODEL;

  return config;
}

/**
 * Create a fully configured AIRouter from environment variables.
 * Pass optional overrides to supplement or override env values.
 */
export function createRouterFromEnv(overrides?: Partial<AIRouterOptions>): AIRouter {
  const localBaseEnv = process.env.AI_ROUTER_LOCAL_BASE_URL;
  // undefined → use default (Ollama); "" → disabled
  const localBaseUrl = localBaseEnv !== undefined ? localBaseEnv : DEFAULT_CONFIG.local_model;

  return new AIRouter({
    anthropicApiKey: process.env.ANTHROPIC_API_KEY ?? "",
    openaiApiKey:    process.env.OPENAI_API_KEY    ?? "",
    geminiApiKey:    process.env.GEMINI_API_KEY    ?? "",
    localBaseUrl:    localBaseUrl as string,
    configFetcher:   async () => configFromEnv(),
    configTtlMs:     envInt("AI_ROUTER_CACHE_TTL_MS") ?? 10_000,
    ...overrides,
  });
}

/** Singleton router built from env — convenient for serverless/edge use. */
export const router = createRouterFromEnv();
