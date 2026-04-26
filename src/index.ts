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

// Env-var factory (re-exported for convenience; also available at /env subpath)
export { createRouterFromEnv, configFromEnv, router } from "./env.js";
