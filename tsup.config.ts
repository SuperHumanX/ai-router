import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    env:   "src/env.ts",       // subpath export: @pmuppirala/ai-router/env
  },
  format:     ["esm", "cjs"],
  dts:        true,
  sourcemap:  true,
  clean:      true,
  splitting:  false,
  treeshake:  true,
  target:     "es2020",
  // Don't bundle peer dependencies — consumers bring their own
  external:   ["@anthropic-ai/sdk", "openai"],
});
