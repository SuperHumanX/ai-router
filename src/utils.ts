import type { StreamEvent } from "./types.js";

/** Yield text in ~30-char chunks (matches provider streaming granularity). */
export async function* chunkText(text: string): AsyncGenerator<StreamEvent> {
  for (let i = 0; i < text.length; i += 30) {
    yield { t: "text", v: text.slice(i, i + 30) };
  }
}
