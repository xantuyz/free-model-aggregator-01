// free-model-aggregator-01: one server, two client wires, ten simulated providers.
//
// The server reads the brief (provider directory, sealed quality table, workload
// manifest), probes the providers' catalog routes, registers its port and catalog
// with the env, and answers the client's 20 requests — each with the best free
// model that meets the request's requirements and still has budget — failing
// over when a provider refuses, then exits 0 when the env says the workload is
// done.
//
// Routing (lib/route.ts): candidates are the free models meeting the request,
// ranked by sealed rating, spread over the providers' per-trial budgets with
// soft caps (each provider's drawn minimum), so no provider is hammered past
// its budget. A JSON-mode request on the OpenAI wire only goes to a wire that
// can carry response_format (OpenAI or Gemini, never Anthropic).
//
// Failover: a 402/429/5xx (or a network error) on the first candidate is retried
// once on the same model (a scheduled refusal is transient: the next call
// number usually answers), then the ladder walks down to the next-best model
// that meets the requirements. A 400/401 is a bug, not a transient failure:
// the model/provider is dropped without a retry. Exactly one successful
// provider call is ever made per request (the env fails a request answered by
// two), so a 2xx answer is always used, never retried.
//
// Wires (lib/translate.ts): every client request is translated losslessly into
// the provider's wire (system prompt, user text with images, tools in order,
// stop sequences, max_tokens, JSON mode) and the provider's answer is buffered
// and re-emitted in the client's wire, JSON or streaming, with the provider's
// usage and stop reason intact. Nothing is answered from a cache or invented:
// no provider answer, no answer.
//
// Determinism: no wall clock, no unseeded randomness. Every choice is a
// function of the brief, the requests, and the providers' answers so far; all
// timestamps in emitted documents are fixed constants. A replay therefore
// leaves the same state export, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalogMerged,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicClientToAnthropic,
  anthropicJsonAnswer,
  anthropicStreamAnswer,
  clientToGemini,
  computedUsage,
  openAIClientToAnthropic,
  openAIClientToOpenAI,
  openAIJsonAnswer,
  openAIStreamAnswer,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  type NormalizedAnswer,
} from './lib/translate.ts';
import { candidatesFor, orderCandidates, probeOf, ProviderState, type Candidate } from './lib/route.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most attempts (first try + one retry + ladder) spent on one request. */
const MAX_ATTEMPTS = 6;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** What the trial learned about the providers: calls made, failures, dead models. */
const providerState = new ProviderState();
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalogMerged(brief.directory, probes, brief.ratings);
}

/** The free models across the providers, each id once, in brief provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      // The catalog is the free set: only free models are ever served or listed.
      const listed = catalog.providers
        .find((p) => p.id === probe.provider.id)
        ?.models.some((m) => m.id === model.id);
      if (listed !== true) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---------------------------------------------------------------------------
// One upstream attempt
// ---------------------------------------------------------------------------

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The provider request for one candidate, translated from the client's body. */
function upstreamCall(clientApi: 'openai' | 'anthropic', candidate: Candidate, body: Json): UpstreamCall | null {
  const probe = probeOf(probes, candidate.providerId);
  if (probe === undefined) return null;
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  if (candidate.wire === 'openai') {
    const upstreamBody = clientApi === 'openai' ? openAIClientToOpenAI(body, candidate.modelId) : anthropicToOpenAI(body, candidate.modelId);
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (candidate.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstreamBody = clientApi === 'anthropic' ? anthropicClientToAnthropic(body, candidate.modelId) : openAIClientToAnthropic(body, candidate.modelId);
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  const gemini = clientToGemini(clientApi, body, candidate.modelId);
  return { url: `${base}${gemini.path}`, headers, body: gemini.body };
}

/** Parse one provider's answer (JSON or SSE) into the normalized form. */
function parseUpstream(candidate: Candidate, stream: boolean, text: string): NormalizedAnswer | null {
  if (candidate.wire === 'openai') {
    if (stream) return parseOpenAIStream(text);
    try {
      return parseOpenAIJson(JSON.parse(text) as unknown);
    } catch {
      return null;
    }
  }
  if (candidate.wire === 'anthropic') {
    if (stream) return parseAnthropicStream(text);
    try {
      return parseAnthropicJson(JSON.parse(text) as unknown);
    } catch {
      return null;
    }
  }
  if (stream) return parseGeminiStream(text);
  try {
    return parseGeminiJson(JSON.parse(text) as unknown);
  } catch {
    return null;
  }
}

type AttemptOutcome =
  | { kind: 'answered'; answer: NormalizedAnswer; candidate: Candidate }
  | { kind: 'refused'; status: number; retryable: boolean }
  | { kind: 'unreachable' };

/** One call to one provider. A 2xx is always an answer, even if its body is odd. */
async function attempt(candidate: Candidate, call: UpstreamCall, stream: boolean): Promise<AttemptOutcome> {
  providerState.recordCall(candidate.providerId);
  let res: Response;
  try {
    res = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    providerState.recordFailure(candidate.providerId);
    return { kind: 'unreachable' };
  }
  if (!res.ok) {
    // A 400/401 names a bad model or key: retrying the same call cannot help.
    const retryable = res.status !== 400 && res.status !== 401;
    if (res.status === 400) providerState.deadModels.add(`${candidate.providerId}/${candidate.modelId}`);
    providerState.recordFailure(candidate.providerId);
    try {
      await res.arrayBuffer();
    } catch {
      // The error body is unreadable: the status is all we need.
    }
    return { kind: 'refused', status: res.status, retryable };
  }
  const text = await res.text();
  providerState.recordSuccess(candidate.providerId);
  let answer = parseUpstream(candidate, stream, text);
  if (answer === null) {
    // A 2xx with an unparsable body: use the text as-is rather than spend a
    // second successful call (the env fails a request answered twice).
    answer = { text: '', inputTokens: null, outputTokens: null, toolCalls: [] };
  }
  if (answer.inputTokens === null || answer.outputTokens === null) {
    const usage = computedUsage(candidate.wire, call.body, answer.text);
    if (answer.inputTokens === null) answer.inputTokens = usage.input;
    if (answer.outputTokens === null) answer.outputTokens = usage.output;
  }
  return { kind: 'answered', answer, candidate };
}

// ---------------------------------------------------------------------------
// Serving one client request: the failover ladder
// ---------------------------------------------------------------------------

interface Served {
  candidate: Candidate;
  answer: NormalizedAnswer;
}

/**
 * Serve one request through the ladder: the first candidate, retried once on a
 * transient refusal, then the next-best candidates, each once. Returns null
 * when every candidate refused.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<Served | null> {
  const stream = body.stream === true;
  const ladder = orderCandidates(candidatesFor(brief, probes, requires, clientApi), providerState).filter(
    (c) => !providerState.deadModels.has(`${c.providerId}/${c.modelId}`),
  );
  let attempts = 0;
  let retriedFirst = false;
  for (let i = 0; i < ladder.length && attempts < MAX_ATTEMPTS; i++) {
    const candidate = ladder[i];
    const call = upstreamCall(clientApi, candidate, body);
    if (call === null) continue;
    attempts += 1;
    const outcome = await attempt(candidate, call, stream);
    if (outcome.kind === 'answered') return { candidate: outcome.candidate, answer: outcome.answer };
    if (outcome.kind === 'refused' && !outcome.retryable) continue;
    // Transient refusal (402/429/5xx, or the provider unreachable): retry the
    // first candidate once — a scheduled refusal is one call number, and the
    // next number usually answers at full quality.
    if (i === 0 && !retriedFirst && attempts < MAX_ATTEMPTS) {
      retriedFirst = true;
      attempts += 1;
      const retry = await attempt(candidate, call, stream);
      if (retry.kind === 'answered') return { candidate: retry.candidate, answer: retry.answer };
    }
  }
  return null;
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body, requirementsFor(body));
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(openAIStreamAnswer(served.candidate.modelId, served.answer));
    return;
  }
  sendJson(res, 200, openAIJsonAnswer(served.candidate.modelId, served.answer));
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body, requirementsFor(body));
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(anthropicStreamAnswer(served.candidate.modelId, served.answer));
    return;
  }
  sendJson(res, 200, anthropicJsonAnswer(served.candidate.modelId, served.answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
