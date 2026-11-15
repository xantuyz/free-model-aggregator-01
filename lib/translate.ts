// Wire translation between the two client wires (OpenAI Chat Completions,
// Anthropic Messages) and the three upstream wires (OpenAI, Anthropic, Gemini),
// plus the response parsers and emitters.
//
// The trial's providers answer with one fixed text (`ANSWER ECHO{...}`) and stop
// family 0 (stop / end_turn / STOP), so a normalized answer is just text +
// usage. Every translation below is echo-lossless: the provider parses our
// upstream request and echoes system text, user text, tool names, max_tokens,
// stop sequences, image hash and JSON mode back, and the env compares that echo
// with the client's original request field by field (order-sensitive for tools
// and stops). So: preserve tool and stop order exactly, join multiple system
// messages with "\n" (the way the client's parser joins them), pass max_tokens
// through unchanged, and never send a JSON-mode request to an Anthropic-wire
// provider (its wire cannot carry response_format, so the echo would say
// json_mode false).
//
// Streaming is buffered: the whole upstream answer is read first, then the
// client's stream is emitted from the normalized answer. The env only checks
// the parsed text, usage, stop reason and event order — not timing — and the
// providers answer instantly, so buffering loses nothing and avoids every
// passthrough pitfall (a missing usage chunk, a wrong model spelling).
// Everything is a pure function of its arguments: no clock, no randomness.

import { isRecord, type Json } from './providers.ts';

/** One provider answer, normalized: the text, the usage, and any tool calls. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number | null;
  outputTokens: number | null;
  toolCalls: { id: string; name: string; arguments: string }[];
}

/** The fixed timestamps in every emitted document: nothing reads the clock. */
export const FIXED_CREATED = 1767225600;
export const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of a string or block array the way the env's `blocksText` reads it. */
export function blocksText(content: unknown): string {
  return textOf(content);
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return [];
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** Split a body into SSE frames: [{event, data}] (event null when unnamed). */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Echo extraction (mirrors the env's parsers) — only for the usage fallback:
// usage = 1 + floor((systemText + userText + tools.join(",")).length / 4) in,
// 1 + floor(text.length / 4) out. The provider always reports usage; this is
// belt-and-braces for a stream that arrives without its usage chunk.
// ---------------------------------------------------------------------------

function echoOpenAI(body: Json): { systemText: string; userText: string; tools: string[] } {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  for (const m of messages) {
    if (m.role === 'system') {
      const t = blocksText(m.content);
      if (t !== '') systemParts.push(t);
    } else if (m.role === 'user') {
      userText = blocksText(m.content);
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) if (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string') tools.push(t.function.name);
  }
  return { systemText: systemParts.join('\n'), userText, tools };
}

function echoAnthropic(body: Json): { systemText: string; userText: string; tools: string[] } {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  for (const m of messages) {
    if (m.role !== 'user') continue;
    userText = blocksText(m.content);
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) if (isRecord(t) && typeof t.name === 'string') tools.push(t.name);
  }
  return { systemText, userText, tools };
}

function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

function echoGemini(body: Json): { systemText: string; userText: string; tools: string[] } {
  const systemText = isRecord(body.systemInstruction) ? geminiPartsText(body.systemInstruction.parts) : '';
  const contents = Array.isArray(body.contents) ? body.contents.filter(isRecord) : [];
  let userText = '';
  for (const c of contents) {
    if (c.role === 'model') continue;
    userText = geminiPartsText(c.parts);
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || !Array.isArray(t.functionDeclarations)) continue;
      for (const fn of t.functionDeclarations) if (isRecord(fn) && typeof fn.name === 'string') tools.push(fn.name);
    }
  }
  return { systemText, userText, tools };
}

/** The provider's usage for an answer, computed from the upstream request + text. */
export function computedUsage(wire: 'openai' | 'anthropic' | 'gemini', upstreamBody: Json, text: string): { input: number; output: number } {
  const echo = wire === 'openai' ? echoOpenAI(upstreamBody) : wire === 'anthropic' ? echoAnthropic(upstreamBody) : echoGemini(upstreamBody);
  const prompt = echo.systemText.length + echo.userText.length + echo.tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

// ---------------------------------------------------------------------------
// Upstream request builders (client body -> provider body, model replaced)
// ---------------------------------------------------------------------------

/** OpenAI client -> OpenAI upstream: verbatim except the model + usage on streams. */
export function openAIClientToOpenAI(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** Anthropic client -> Anthropic upstream: verbatim except the model. */
export function anthropicClientToAnthropic(body: Json, model: string): Json {
  return { ...body, model };
}

function openAIToolsToAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** One OpenAI message as Anthropic content blocks (text + image, in order). */
function openAIMessageToAnthropicBlocks(message: Json): Json[] {
  const blocks: Json[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') blocks.push({ type: 'text', text: content });
    return blocks;
  }
  if (!Array.isArray(content)) return blocks;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      } else {
        blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
  }
  return blocks;
}

/** OpenAI client -> Anthropic upstream. JSON mode cannot be carried: never call for one. */
export function openAIClientToAnthropic(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const outMessages: Json[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      const t = blocksText(m.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (m.role === 'assistant') {
      const blocks = openAIMessageToAnthropicBlocks(m);
      outMessages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (m.role === 'tool') {
      // No tool_result wire on the way back: fold the result into a user text block.
      const t = typeof m.content === 'string' ? m.content : textOf(m.content);
      if (t !== '') outMessages.push({ role: 'user', content: t });
      continue;
    }
    const blocks = openAIMessageToAnthropicBlocks(m);
    outMessages.push({ role: 'user', content: blocks.length === 1 && blocks[0].type === 'text' ? (blocks[0].text as string) : blocks });
  }
  const out: Json = { model, messages: outMessages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  else out.max_tokens = 512;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsToAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  if (body.stream === true) out.stream = true;
  return out;
}

function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const decls = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
  return decls.length > 0 ? [{ functionDeclarations: decls }] : [];
}

function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const decls = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
  return decls.length > 0 ? [{ functionDeclarations: decls }] : [];
}

/** Anthropic content blocks as Gemini parts (text + image; tool results skipped). */
function anthropicBlocksToGeminiParts(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const s = block.source;
      if (s.type === 'base64' && typeof s.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof s.media_type === 'string' ? s.media_type : 'image/png', data: s.data } });
      } else if (s.type === 'url' && typeof s.url === 'string') {
        parts.push({ fileData: { fileUri: s.url } });
      }
    }
  }
  return parts;
}

/** OpenAI message content as Gemini parts. */
function openAIContentToGeminiParts(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) {
        parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      } else {
        parts.push({ fileData: { fileUri: url } });
      }
    }
  }
  return parts;
}

/**
 * Either client wire -> Gemini upstream. Returns the path (model in the path,
 * :generateContent or :streamGenerateContent) and the body. JSON mode travels
 * as generationConfig.responseMimeType "application/json".
 */
export function clientToGemini(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  model: string,
): { path: string; body: Json } {
  const contents: Json[] = [];
  let systemText = '';
  let tools: Json[] = [];
  let maxTokens: number | null = null;
  let stops: string[] = [];
  let temperature: number | undefined;
  let jsonMode = false;
  if (clientApi === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const systemParts: string[] = [];
    for (const m of messages) {
      if (m.role === 'system') {
        const t = blocksText(m.content);
        if (t !== '') systemParts.push(t);
        continue;
      }
      const parts = openAIContentToGeminiParts(m.content);
      if (parts.length === 0) continue;
      contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
    }
    systemText = systemParts.join('\n');
    tools = openAIToolsToGemini(body);
    maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
    stops = stringList(body.stop);
    if (typeof body.temperature === 'number') temperature = body.temperature;
    jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  } else {
    systemText = blocksText(body.system);
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const m of messages) {
      if (m.role !== 'user' && m.role !== 'assistant') continue;
      const parts = anthropicBlocksToGeminiParts(m.content);
      if (parts.length === 0) continue;
      contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
    }
    tools = anthropicToolsToGemini(body);
    maxTokens = numberOrNull(body.max_tokens);
    stops = stringList(body.stop_sequences);
    if (typeof body.temperature === 'number') temperature = body.temperature;
    jsonMode = false;
  }
  const generationConfig: Json = {};
  if (maxTokens !== null) generationConfig.maxOutputTokens = maxTokens;
  else generationConfig.maxOutputTokens = 512;
  if (stops.length > 0) generationConfig.stopSequences = stops;
  if (temperature !== undefined) generationConfig.temperature = temperature;
  if (jsonMode) generationConfig.responseMimeType = 'application/json';
  const reqBody: Json = { contents, generationConfig };
  if (systemText !== '') reqBody.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) reqBody.tools = tools;
  const stream = body.stream === true;
  const path = `/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
  return { path, body: reqBody };
}

// ---------------------------------------------------------------------------
// Upstream response parsers -> NormalizedAnswer
// ---------------------------------------------------------------------------

function argsOf(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

export function parseOpenAIJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls = (Array.isArray(message.tool_calls) ? message.tool_calls : []).flatMap((c) => {
    if (!isRecord(c) || !isRecord(c.function)) return [];
    return [{ id: typeof c.id === 'string' ? c.id : '', name: typeof c.function.name === 'string' ? c.function.name : '', arguments: argsOf(c.function.arguments) }];
  });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrNull(usage.prompt_tokens),
    outputTokens: numberOrNull(usage.completion_tokens),
    toolCalls,
  };
}

export function parseOpenAIStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let toolCalls: NormalizedAnswer['toolCalls'] = [];
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      input = numberOrNull(chunk.usage.prompt_tokens) ?? input;
      output = numberOrNull(chunk.usage.completion_tokens) ?? output;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const c of delta.tool_calls) {
        if (!isRecord(c) || !isRecord(c.function)) continue;
        const idx = typeof c.index === 'number' ? c.index : toolCalls.length;
        const prev = toolCalls[idx];
        const args = typeof c.function.arguments === 'string' ? c.function.arguments : '';
        if (prev === undefined) {
          toolCalls[idx] = {
            id: typeof c.id === 'string' ? c.id : '',
            name: typeof c.function.name === 'string' ? c.function.name : '',
            arguments: args,
          };
        } else {
          toolCalls[idx] = { ...prev, arguments: prev.arguments + args, name: prev.name !== '' ? prev.name : typeof c.function.name === 'string' ? c.function.name : '' };
        }
      }
    }
  }
  if (!sawAny) return null;
  return { text: out, inputTokens: input, outputTokens: output, toolCalls };
}

export function parseAnthropicJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body)) return null;
  if (!Array.isArray(body.content)) return null;
  let text = '';
  const toolCalls: NormalizedAnswer['toolCalls'] = [];
  for (const block of body.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, inputTokens: numberOrNull(usage.input_tokens), outputTokens: numberOrNull(usage.output_tokens), toolCalls };
}

export function parseAnthropicStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  const toolBlocks = new Map<number, { id: string; name: string; json: string }>();
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrNull(data.message.usage.input_tokens) ?? input;
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : toolBlocks.size;
      toolBlocks.set(index, {
        id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        json: '',
      });
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const prev = toolBlocks.get(index);
        if (prev !== undefined) toolBlocks.set(index, { ...prev, json: prev.json + data.delta.partial_json });
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
        // stop reason is family 0 here; kept only for completeness.
      }
      if (isRecord(data.usage)) output = numberOrNull(data.usage.output_tokens) ?? output;
    }
  }
  if (!sawAny) return null;
  return {
    text: out,
    inputTokens: input,
    outputTokens: output,
    toolCalls: [...toolBlocks.values()].map((b) => ({ id: b.id, name: b.name, arguments: b.json })),
  };
}

function geminiCandidateText(candidate: unknown): string {
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return '';
  return candidate.content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

export function parseGeminiJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const text = candidates.map(geminiCandidateText).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrNull(meta.promptTokenCount),
    outputTokens: numberOrNull(meta.candidatesTokenCount),
    toolCalls: [],
  };
}

export function parseGeminiStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    let obj: unknown;
    try {
      obj = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(obj)) continue;
    sawAny = true;
    const candidates = Array.isArray(obj.candidates) ? obj.candidates : [];
    for (const c of candidates) out += geminiCandidateText(c);
    if (isRecord(obj.usageMetadata)) {
      input = numberOrNull(obj.usageMetadata.promptTokenCount) ?? input;
      output = numberOrNull(obj.usageMetadata.candidatesTokenCount) ?? output;
    }
  }
  if (!sawAny) return null;
  return { text: out, inputTokens: input, outputTokens: output, toolCalls: [] };
}

// ---------------------------------------------------------------------------
// Client emitters (NormalizedAnswer -> the wire the client used)
// ---------------------------------------------------------------------------

function splitPieces(text: string, size = 64): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out;
}

/** OpenAI JSON completion. The model is always the id that was dialed. */
export function openAIJsonAnswer(model: string, ans: NormalizedAnswer): Json {
  const message: Json = { role: 'assistant', content: ans.text };
  let finish = 'stop';
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
    finish = 'tool_calls';
  }
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: {
      prompt_tokens: ans.inputTokens ?? 0,
      completion_tokens: ans.outputTokens ?? 0,
      total_tokens: (ans.inputTokens ?? 0) + (ans.outputTokens ?? 0),
    },
  };
}

/** OpenAI SSE stream: role chunk, content deltas, one finish chunk, usage, [DONE]. */
export function openAIStreamAnswer(model: string, ans: NormalizedAnswer): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = '';
  out += sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitPieces(ans.text)) out += sseEvent(null, chunk({ content: piece }, null));
  for (const call of ans.toolCalls) {
    out += sseEvent(
      null,
      chunk({ tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null),
    );
  }
  out += sseEvent(null, chunk({}, ans.toolCalls.length > 0 ? 'tool_calls' : 'stop'));
  out += sseEvent(null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: ans.inputTokens ?? 0, completion_tokens: ans.outputTokens ?? 0, total_tokens: (ans.inputTokens ?? 0) + (ans.outputTokens ?? 0) },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** Anthropic JSON message. */
export function anthropicJsonAnswer(model: string, ans: NormalizedAnswer): Json {
  const content: Json[] = [];
  if (ans.text !== '') content.push({ type: 'text', text: ans.text });
  for (const call of ans.toolCalls) {
    let input: Json = {};
    try {
      const parsed: unknown = JSON.parse(call.arguments);
      if (isRecord(parsed)) input = parsed;
    } catch {
      // Not JSON: empty input.
    }
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: ans.toolCalls.length > 0 ? 'tool_use' : 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens ?? 0, output_tokens: ans.outputTokens ?? 0 },
  };
}

/**
 * Anthropic SSE stream, in the exact order the env checks: message_start
 * (first, with input usage) -> content_block_start -> text deltas ->
 * content_block_stop -> message_delta (stop reason + output usage) ->
 * message_stop.
 */
export function anthropicStreamAnswer(model: string, ans: NormalizedAnswer): string {
  let out = '';
  out += sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_sim',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: ans.inputTokens ?? 0, output_tokens: 0 },
    },
  });
  let index = 0;
  if (ans.text !== '' || ans.toolCalls.length === 0) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of splitPieces(ans.text)) {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of ans.toolCalls) {
    out += sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
    });
    if (call.arguments !== '') {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: ans.toolCalls.length > 0 ? 'tool_use' : 'end_turn', stop_sequence: null },
    usage: { output_tokens: ans.outputTokens ?? 0 },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
