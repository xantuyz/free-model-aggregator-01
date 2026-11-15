# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/xantuyz/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/xantuyz/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten simulated free-tier model providers, routing each of 20 requests to the best free model that can serve it, failing over when a provider refuses, and reporting the catalog it read. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

Node 22, standard library only, nothing to install. In a trial the harness sets `MAKEFUN_PROMPT_FILE` (the brief: provider directory, sealed quality table, workload manifest) and `MAKEFUN_ENV_URL` (the env); the server listens on an ephemeral port, registers it with its catalog, serves the client's requests, and exits 0 when the env says the workload is done.

```sh
node main.ts --list   # print the catalog document and exit 0
```

## How it routes

- **Quality first:** candidates are the free models meeting the request's requirements, ranked by the sealed rating table. A JSON-mode request on the OpenAI wire only goes to a wire that can carry `response_format` (OpenAI or Gemini, never Anthropic).
- **Budget-aware:** calls are spread over the providers' per-trial budgets with soft caps (each provider's drawn minimum from the sealed table), so no provider is hammered past its budget.
- **Failover:** a 402/429/5xx on the first pick is retried once on the same model (a scheduled refusal is transient), then the ladder walks down to the next-best model. Exactly one successful provider call is ever made per request.
- **Wire-faithful:** every request is translated losslessly into the provider's wire (system prompt, user text with images, tools in order, stop sequences, `max_tokens`, JSON mode) and the provider's answer is re-emitted in the client's wire, JSON or SSE, with its usage and stop reason intact. Nothing is answered from a cache or invented.
- **Deterministic:** no wall clock, no unseeded randomness — a replay leaves the same state export, byte for byte. `node main.ts --list` prints the same catalog document the register body carries.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, poll
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe (starter, with a brief fallback for a failed probe and a free-models-only catalog)
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation and the SSE writers (starter)
- `lib/translate.ts` — translations to the Anthropic and Gemini upstream wires, upstream response parsers, client emitters, usage fallback
- `lib/route.ts` — candidate ranking, budget soft caps, failover ladder

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards
