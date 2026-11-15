// Routing: rank every free model that can serve a request, spread calls over the
// providers' per-trial budgets, and fail over when a provider refuses.
//
// The sealed provider table (kit program/generator + program/scorer, public by
// design) draws each provider's per-trial budget uniformly from [calls_min,
// calls_max] and spends 1-2 calls on scheduled refusals, so a provider's usable
// budget is roughly its minimum. The soft caps below are those minima: a provider
// is preferred while its calls stay under its cap, and only used past it when no
// fresher candidate can serve the request. The caps are a planning heuristic,
// never a hard limit: a provider past its cap may still answer (its real budget
// can be up to its max), and a refusal is always handled by the failover ladder.
//
// Candidate order inside one request: a provider that just failed twice in a row
// ("suspect", most likely out of budget or in outage) sorts after a healthy one,
// then a provider past its soft cap, then rating (highest first), then context
// length, then provider and model id for a deterministic tie-break. Nothing here
// reads the clock or unseeded randomness: the order is a function of the brief,
// the requests so far, and the providers' answers so far, so a replay ranks the
// same ladder.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProbedModel,
  type Requirement,
  type Wire,
} from './providers.ts';

/** One dialable (provider, model) pair with its sealed rating and capabilities. */
export interface Candidate {
  providerId: string;
  providerIndex: number;
  wire: Wire;
  modelId: string;
  rating: number;
  contextLength: number | null;
}

/** A provider's per-trial budget window, from the sealed table (calls_min/max). */
export const SOFT_CAPS: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** Per-provider state learned during the trial: calls made, consecutive failures. */
export class ProviderState {
  readonly calls = new Map<string, number>();
  readonly consecutiveFailures = new Map<string, number>();
  readonly deadModels = new Set<string>();

  callsOf(providerId: string): number {
    return this.calls.get(providerId) ?? 0;
  }

  recordCall(providerId: string): void {
    this.calls.set(providerId, this.callsOf(providerId) + 1);
  }

  recordSuccess(providerId: string): void {
    this.consecutiveFailures.set(providerId, 0);
  }

  recordFailure(providerId: string): void {
    this.consecutiveFailures.set(providerId, (this.consecutiveFailures.get(providerId) ?? 0) + 1);
  }

  isSuspect(providerId: string): boolean {
    return (this.consecutiveFailures.get(providerId) ?? 0) >= 2;
  }

  isOverCap(providerId: string): boolean {
    const cap = SOFT_CAPS[providerId];
    return cap !== undefined && this.callsOf(providerId) >= cap;
  }
}

/** All free models that can serve `requires`, from the probes and the ratings table. */
export function candidatesFor(
  brief: Brief,
  probes: readonly Probe[],
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
): Candidate[] {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      // A JSON-mode request on the OpenAI wire must reach a wire that can carry
      // response_format: the Anthropic wire echoes json_mode false, so the echo
      // check would fail. (On the Anthropic client wire there is no JSON-mode
      // flag to carry, so any wire serves it as long as the model is JSON-able.)
      if (requires.json === true && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
      const key = `${probe.provider.id}/${model.id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({
        providerId: probe.provider.id,
        providerIndex,
        wire: probe.provider.wire,
        modelId: model.id,
        rating: row?.rating_milli ?? 0,
        contextLength: row?.context_length ?? model.context_length ?? null,
      });
    }
  });
  return out;
}

/** The failover ladder for one request: best first, given what the trial learned. */
export function orderCandidates(cands: readonly Candidate[], state: ProviderState): Candidate[] {
  return [...cands].sort((a, b) => {
    const suspect = (state.isSuspect(a.providerId) ? 1 : 0) - (state.isSuspect(b.providerId) ? 1 : 0);
    if (suspect !== 0) return suspect;
    const cap = (state.isOverCap(a.providerId) ? 1 : 0) - (state.isOverCap(b.providerId) ? 1 : 0);
    if (cap !== 0) return cap;
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ctx = (b.contextLength ?? 0) - (a.contextLength ?? 0);
    if (ctx !== 0) return ctx;
    if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
    return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
  });
}

/** The probe of one provider, by id. */
export function probeOf(probes: readonly Probe[], providerId: string): Probe | undefined {
  return probes.find((p) => p.provider.id === providerId);
}

export type { ProbedModel };
