import Anthropic from "@anthropic-ai/sdk";
import type { AgentLoopParams, StreamEvent } from "../types.js";
import { chunkText } from "../utils.js";

export async function* anthropicLoop(
  client: Anthropic,
  model:  string,
  params: Omit<AgentLoopParams, "onRoundRobin">,
): AsyncGenerator<StreamEvent> {
  const { systemPrompt, messages, tools, maxRounds = 5, toolExecutor } = params;
  const allCitations: unknown[] = [];

  const anthropicMessages: Anthropic.MessageParam[] = messages.map((m) => ({
    role:    m.role,
    content: m.content,
  }));

  const anthropicTools: Anthropic.Tool[] = tools.map((t) => ({
    name:         t.name,
    description:  t.description,
    input_schema: t.input_schema,
  }));

  for (let round = 0; round < maxRounds; round++) {
    const response = await client.messages.create({
      model,
      max_tokens: 2048,
      system:     systemPrompt,
      tools:      anthropicTools,
      messages:   anthropicMessages,
    });

    if (response.stop_reason === "tool_use") {
      const toolResults: Anthropic.ToolResultBlockParam[] = [];

      for (const block of response.content) {
        if (block.type !== "tool_use") continue;

        const input = block.input as Record<string, unknown>;
        yield { t: "tool", v: String(input.query ?? block.name) };

        const result = await toolExecutor(block.name, input);
        allCitations.push(...result.citations);
        toolResults.push({
          type:        "tool_result",
          tool_use_id: block.id,
          content:     result.summary,
        });
      }

      anthropicMessages.push({ role: "assistant", content: response.content });
      anthropicMessages.push({ role: "user",      content: toolResults });
      continue;
    }

    for (const block of response.content) {
      if (block.type === "text") yield* chunkText(block.text);
    }
    break;
  }

  if (allCitations.length > 0) yield { t: "cite", v: allCitations.slice(0, 12) };
  yield { t: "done" };
}
