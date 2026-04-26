/**
 * OpenAI-compatible loop — handles OpenAI, Gemini (via compat endpoint),
 * and any local OpenAI-compatible server (Ollama, LM Studio, vLLM, etc.).
 *
 * Callers pass a pre-configured OpenAI client instance; this module is
 * intentionally unaware of which provider is on the other end.
 */

import OpenAI from "openai";
import type { AgentLoopParams, StreamEvent } from "../types.js";
import { chunkText } from "../utils.js";

export async function* openaiCompatLoop(
  client: OpenAI,
  model:  string,
  params: Omit<AgentLoopParams, "onRoundRobin">,
): AsyncGenerator<StreamEvent> {
  const { systemPrompt, messages, tools, maxRounds = 5, toolExecutor } = params;
  const allCitations: unknown[] = [];

  const chatMessages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    ...messages.map((m): OpenAI.ChatCompletionMessageParam => ({
      role:    m.role,
      content: m.content,
    })),
  ];

  const chatTools: OpenAI.ChatCompletionTool[] = tools.map((t) => ({
    type:     "function" as const,
    function: {
      name:        t.name,
      description: t.description,
      parameters:  t.input_schema,
    },
  }));

  for (let round = 0; round < maxRounds; round++) {
    const response = await client.chat.completions.create({
      model,
      messages:    chatMessages,
      tools:       chatTools.length > 0 ? chatTools : undefined,
      tool_choice: chatTools.length > 0 ? "auto" : undefined,
      max_tokens:  2048,
    });

    const choice = response.choices[0];

    if (choice.finish_reason === "tool_calls" && choice.message.tool_calls?.length) {
      chatMessages.push(choice.message);

      for (const tc of choice.message.tool_calls) {
        const fn    = tc.function;
        let   input: Record<string, unknown> = {};
        try { input = JSON.parse(fn.arguments) as Record<string, unknown>; } catch { /* empty args */ }

        yield { t: "tool", v: String(input.query ?? fn.name) };

        const result = await toolExecutor(fn.name, input);
        allCitations.push(...result.citations);
        chatMessages.push({
          role:         "tool",
          tool_call_id: tc.id,
          content:      result.summary,
        });
      }
      continue;
    }

    yield* chunkText(choice.message.content ?? "");
    break;
  }

  if (allCitations.length > 0) yield { t: "cite", v: allCitations.slice(0, 12) };
  yield { t: "done" };
}
