// Core router
export { AIRouter, DEFAULT_CONFIG } from "./router.js";

// Types
export type {
  Provider,
  RouterMode,
  RouterConfig,
  ProviderWeights,
  StreamEvent,
  CommonMessage,
  RouterTool,
  ToolResult,
  ToolExecutor,
  RoundRobinPersist,
  AgentLoopParams,
  AIRouterOptions,
} from "./types.js";

// Intelligent-routing data model types (Phase 10) — type definitions only,
// mirroring router_policy.py's Python dataclasses; see types.ts for the
// full explanation of what this is (and is explicitly not) for.
export type {
  RoutingDecision,
  VerificationResult,
  ExperienceTrace,
  FeedbackRecord,
  DomainConfig,
  ModelCapability,
} from "./types.js";

// Env-var factory (re-exported for convenience; also available at /env subpath)
export { createRouterFromEnv, configFromEnv, router } from "./env.js";
