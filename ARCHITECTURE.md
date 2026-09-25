# ai-router — Architecture

How routing actually works across the clients that use this gateway, and how
the pieces stay (or drift) in sync. See [README.md](README.md) for the env
var reference and quick-start usage; this doc is about the *shape* of the
system and the decisions behind it.

## The picture

```
        physician              portfolio_tracker           CatalogValidator
     domain: health              domain: finance              domain: retail
            │                          │                            │
            └──────────────┬───────────┴──────────────┬─────────────┘
                            ▼                          
                       gateway.py  (one file, copied into each project
                                     as agents/ai_router.py, imported as
                                     health_router / finance_router / catalog_router)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
         local SLM                  cloud (weighted)
    tried first · domain LoRA     fallback · OpenRouter 75%
```

Three **independent Python processes** (physician, portfolio_tracker,
CatalogValidator) each hold their own *copy* of the same file — this is a
copy-paste distribution model, not a shared package/import. There is no
`pip install`, no symlink, no version pin. `gateway.py` at
`~/Projects/ai-router/` is the single source of truth; each project's
`agents/ai_router.py` (or `backend/mcp_domain/ai_router.py` for
CatalogValidator) is a snapshot taken at deploy time via:

```bash
cp ~/Projects/ai-router/gateway.py <project>/agents/ai_router.py
```

**This is the load-bearing fact to remember**: nothing automatically keeps
these in sync. See "Drift history" below for what happens when that's
forgotten.

physician-web (the Next.js app) is a fourth client but runs a *separate*,
independently-written TypeScript implementation (`src/router.ts`, installed
as `@pmuppirala/ai-router` via `github:SuperHumanX/ai-router`) — not a port
of `gateway.py`. The two converged on the same idea (OpenRouter as a shared
BYOK fallback) independently, without either side copying the other.

## Routing tiers

| Tier | What | Skip conditions |
|---|---|---|
| **0 — Remote** | If `REMOTE_GATEWAY_URL` is set, every call is forwarded to a `server.py` instance (e.g. over Tailscale) instead of touching providers locally. Lets a host run the router with zero API keys of its own. | Only active when the env var is set. Nothing here uses it today — it exists for a future remote/GCP client. |
| **1 — Local SLM** | Probes `LOCAL_INTEL_URL` (the shared LocalIntelligence gateway on port 11435) on every call. Free, private, no API cost. `LOCAL_INTEL_DOMAIN`, if set, is sent as an explicit `domain` field so the gateway hot-swaps the right LoRA adapter — without it, the client falls back to whatever `/v1/models` happens to list first, which is not guaranteed to be the right adapter for that client. | `USE_LOCAL_SLM=false`, or `model_hint="smart"` (smart-tier calls are quality-critical enough to skip local and go straight to cloud), or the probe fails/times out. |
| **2 — Cloud (weighted)** | `OpenRouter` is the primary path (default weight **0.75**) — one key, routes to Anthropic/OpenAI/Google models upstream, so no single direct-key account running dry takes cloud down. `Gemini` (0.15), `Anthropic` (0.05), `OpenAI` (0.05) each prefer their own direct key/path and fall back to OpenRouter BYOK themselves when that path is missing or fails. | `AI_ROUTER_MODE` pinned to a specific provider, or all cloud providers unconfigured. |
| **3 — Error** | Raises `RuntimeError` so the caller can surface it. | — |

The OpenRouter-fallback-within-each-provider detail matters: it's not just
"OpenRouter OR direct providers" — every provider slot (Gemini, Anthropic,
OpenAI) has its *own* two-hop fallback (direct key → OpenRouter with that
provider's model-ID prefix) independent of OpenRouter's own 75% weighted
slot. One caveat: the OpenRouter fallback path for Anthropic doesn't forward
`tools` (no Anthropic-native tool schema over OpenRouter's OpenAI-compat
API) — a call that needs tools should have a direct `ANTHROPIC_API_KEY`.

## Config precedence

Three layers, loaded in this order — **but the actual precedence is not
what the layering implies**:

1. Master key store (`~/Projects/ai-router/.env`) — loaded first, via
   `_load_master_env()`, which does `os.environ[k] = v` unconditionally for
   every non-empty key it finds.
2. Project `.env` (read via `python-dotenv`'s `dotenv_values()`, cwd-relative)
   — but only consulted as a *fallback* when `os.getenv(k)` already came back
   empty.
3. OS-level environment (shell, systemd, launchd) — same mechanism as #2,
   effectively merged with it since both just populate `os.environ`.

Because step 1 runs first and unconditionally overwrites, **the master
store wins over a project's own `.env` for any key both define** — which is
the opposite of what the module docstring says ("Project .env values and
OS-level env vars always take precedence"). In practice this rarely bites,
because the master store is scoped to keys (`OPENROUTER_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_PROJECT`, ...) and projects are scoped to
routing knobs (`AI_ROUTER_WEIGHT_*`, `LOCAL_INTEL_DOMAIN`, `USE_LOCAL_SLM`)
— but `LOCAL_INTEL_URL` is defined in *both* places, and the master store's
value (a Tailscale IP) silently wins over whatever a project sets locally.
Known, not yet fixed — flagging it here so the next person debugging "why
is my `.env` override not taking effect" finds this before re-discovering it.

## Per-client configuration

Each client sets three things in its own `.env`; everything else falls back
to `gateway.py`'s defaults.

| Client | `LOCAL_INTEL_DOMAIN` | `USE_LOCAL_SLM` | Cloud weight overrides |
|---|---|---|---|
| physician (health) | `health` | `true` | none — uses the 75/15/5/5 default |
| portfolio_tracker (finance) | `finance` | `true` | explicit 75/15/5/5 (same as default, set anyway) |
| CatalogValidator (retail) | *(unset — relies on `/v1/models[0]` happening to be `"retail"`)* | `true` (default) | `openrouter=0.85, anthropic=0.1` (skews harder toward OpenRouter than the default) |

The local gateway (LocalIntelligence, port 11435) exposes its domains
through `/v1/models` as `["retail", "finance", "health", "research"]` — in
that fixed order. Any client that doesn't set `LOCAL_INTEL_DOMAIN` silently
gets whichever domain is first in that list when it resolves its display
model name, which today happens to be `"retail"`. This is why
CatalogValidator's omission is harmless (it *wants* retail) and why it would
have been a real bug for physician/portfolio_tracker before they set the
domain explicitly.

Note: the `model` field on a `RouterResponse` from a **local** call is
cosmetic — it echoes `/v1/models[0]`, not the domain that actually served
the request (that's controlled entirely by the `domain` field in the
request payload). Don't use `result.model` to verify which LoRA answered a
local call; check the response content or the gateway's own request logs.

## Drift history (2026-09-23)

This is worth keeping as a record because it's exactly the failure mode to
watch for again.

The three project copies were **not merely stale** — two of them
(portfolio_tracker, CatalogValidator) had independently grown a materially
*better* architecture than canonical `gateway.py`: OpenRouter as a
first-class weighted provider (not just a same-provider fallback),
`REMOTE_GATEWAY_URL`, `USE_LOCAL_SLM`, `model_hint="smart"` skipping local,
and `LOCAL_INTEL_DOMAIN`. None of that had ever been copied back to
`~/Projects/ai-router/gateway.py`. Canonical, meanwhile, had a feature
neither project copy had picked up: Gemini's Vertex→OpenRouter BYOK
fallback, plus a corrected `GEMINI_PROJECT` default (the project copies
still had the wrong id, `mcp-research-2026`, months after canonical fixed
it).

physician's copy was further behind still — missing `LOCAL_INTEL_DOMAIN`
entirely, and its own `.env` had `USE_LOCAL_SLM=false` (local-first
disabled) plus stale `AI_ROUTER_WEIGHT_ANTHROPIC=0.7` /
`AI_ROUTER_WEIGHT_OPENAI=0.3` overrides left over from before the
OpenRouter-primary scheme existed — which, combined with the new
OpenRouter default weight, would have skewed routing to roughly 43%
openrouter / 40% anthropic / 17% openai instead of the intended 75/5/5.

**Resolution**: merged both directions into one canonical `gateway.py`
(kept portfolio_tracker/CatalogValidator's architecture as the base, ported
in canonical's Gemini fallback + corrected project id), re-copied to all
three Python clients, and fixed physician's `.env` (re-enabled
`USE_LOCAL_SLM`, dropped the stale weight overrides, added
`LOCAL_INTEL_DOMAIN=health`, fixed a leftover `OLLAMA_MODEL=retail-cypher-7b`
that was a copy-paste artifact from CatalogValidator). All three were then
smoke-tested live: local-first confirmed working for all three (each served
by its correct domain adapter), and the OpenRouter-primary cloud path
confirmed with a real end-to-end call.

**Why it happened**: there's no automation syncing `gateway.py` to its
copies — deployment is a manual `cp`, documented in the README but not
enforced anywhere. Any project that patches its local copy directly (as
portfolio_tracker/CatalogValidator did) permanently diverges from canonical
unless someone manually diffs and merges both directions, which is what
this pass did. If this repo starts drifting again, that manual-copy step is
the first thing to look at — either enforce it (a package, a symlink, a
`pip install -e`-style local dependency) or accept the drift as a known
cost of the copy-paste model and schedule periodic diff-and-reconcile
passes like this one.

## Intelligent routing (Phase 1 of 10, shadow mode — added 2026-09-24)

The gateway is evolving from a statically domain-directed router into a
dynamic, self-improving control plane: a router that predicts whether local
can handle a request, a verifier that checks whether it actually did, a
capability-aware frontier router for escalations, and an experience store
that turns those escalations into curated training candidates — without
ever training anything inside the request path. This is a 10-phase plan;
**only Phase 1 is implemented.**

Phase 1 adds one new sibling file, `router_policy.py` (deployed the same way
as `gateway.py` — copied into whichever directory holds a project's
`ai_router.py`), defining:

- `RequestContext` / `RoutingDecision` — the only types with real behavior.
- `VerificationResult`, `ExperienceTrace`, `DomainConfig`, `ModelCapability`
  — inert dataclasses, interfaces agreed on now so Phases 2/4/5/6 don't need
  a breaking type change when they add behavior.
- `RouterPolicy` (a `typing.Protocol`, not an ABC — deliberate, given the
  flat-file-copy deployment model) and `DeterministicRouterPolicy`, the only
  concrete policy shipped so far: rule-based placeholders (a fixed
  `model_hint` → quality-threshold table, a length-based complexity proxy, a
  keyword-based risk proxy, and a flat `predicted_local_success = 0.5`
  constant — there's no calibration data yet, that's Phase 7's job).

Gated behind `AI_ROUTER_INTELLIGENT_ROUTING` (default `false`). When
enabled, `gateway.py`'s `chat()` computes and logs a `RoutingDecision` on
every call — **but does not yet consult it.** The existing Tier 0-3 dispatch
is untouched; this is pure shadow-mode observability, laying groundwork for
Phase 2/3 to actually wire `decision.route` into dispatch. Because
`predicted_local_success` is a flat 0.5 and every default threshold is
≥ 0.70, `route` will read `"frontier"` for nearly every request today — that
is expected, not a bug.

`router_policy.py` is loaded dynamically via `Path(__file__).parent` (the
same idiom `_rr_path()` already used) rather than a plain `import
router_policy`, since `gateway.py` has no reliable package context across
its three deployments. A deployment that hasn't copied the new file, or
where the load fails for any reason, falls back to legacy behavior with a
logged warning — never silently.

First-ever tests for this repo also landed with Phase 1, in `tests/`, run
via `python3 -m unittest discover -s tests -v` (stdlib `unittest`, not
pytest — this repo has no venv/dependency story of its own, and
`unittest.TestCase` is pytest-collectible later if that changes).
`test_gateway_backward_compat.py` specifically freezes today's dispatch
behavior (byte-identical `chat()` output regardless of the flag) before
Phase 2+ starts actually changing it.

## Intelligent routing — Phase 2 of 10, domain/adapter registry (added 2026-09-24)

Phase 2 makes `DomainConfig` real and, for the first time, lets a computed
decision affect an actual request — double-gated behind two flags so every
existing deployment's behavior stays byte-identical unless both are
explicitly set.

**The registry is real data, not placeholders.** `DEFAULT_DOMAIN_REGISTRY`
in `router_policy.py` is seeded from this Mac's own local-adapter
benchmarks: retail (`retail_v3`, 60% execution accuracy vs. ~20-30%
baseline), finance (`finance`, 87% full-label match vs. 27% baseline),
health (`cardiology`, 55% full-label match vs. 42% baseline), and `general`
(no dedicated adapter — confirmed nothing trained). Two domains named in the
original spec were investigated and deliberately excluded: `manufacturing`
has zero infrastructure anywhere, and `research` in LocalIntelligence is a
*model pool* (a bigger base model for deep synthesis), not a routable
domain — unrelated to the separate MedResearch/Research Bridge service at
port 8011. The registry loader is fully config-driven regardless, so adding
a domain later is a config edit, not a code change.

**YAML is optional, never required.** `load_domain_registry()` always
starts from the hardcoded `DEFAULT_DOMAIN_REGISTRY` — the complete,
authoritative dataset, not a degraded fallback — and only layers overrides
from `config/domains.yaml` on top if PyYAML happens to be installed and that
file happens to exist. This sidesteps a real deployment problem: `config/`
was never part of the `cp gateway.py router_policy.py <target>/` deploy
step, so Phase 2 needed to work correctly without it.

**Classification is deterministic keyword matching** against each domain's
`keywords` list (`_classify_domain()`) — no LLM call, no added latency or
cost, consistent with Phase 1's approach. `DomainConfig` gained two new
fields to support this: `benchmark_success_rate` (feeds
`predicted_local_success` — real calibration data now, not the Phase 1 flat
0.5 placeholder, for any domain that has one) and `keywords`.

**`AI_ROUTER_DOMAIN_MODE`** (default `override`) controls whether any of
this can touch a real request:
- `override` — `LOCAL_INTEL_DOMAIN` always wins; classifier never runs
  against real dispatch. Byte-identical to pre-Phase-2 behavior.
- `hint` — classifier runs; a confident keyword match wins, otherwise falls
  back to the `LOCAL_INTEL_DOMAIN` hint (or `"general"`).
- `auto` — classifier's answer wins outright; the hint is ignored for
  domain selection.

`_call_local()` gained a `domain_override` parameter threaded from
`chat()`'s `RoutingDecision.domain`, but only populated when
`AI_ROUTER_DOMAIN_MODE != "override"` — nothing changes unless BOTH
`AI_ROUTER_INTELLIGENT_ROUTING=true` and `AI_ROUTER_DOMAIN_MODE=hint`/`auto`
are explicitly set. Verified live (stubbed `_call_local`) across all four
flag combinations: only the flags-on + hint/auto + a confidently-classified
message actually changes the domain sent to LocalIntelligence.

`quality_threshold` computation is still exactly Phase 1's `model_hint`
table — `DomainConfig.default_quality_threshold` is populated but not
consulted yet; reconciling the two is Phase 3's job. `subdomain`
(health's `cardiology` specialty) stays unpopulated — there's only one
specialty today and it's already LocalIntelligence's own default. `risk`
still uses Phase 1's flat keyword list, not `DomainConfig.risk_constraints`
— domain-aware risk scoring is a Phase 4/8 concern.

## Intelligent routing — Phase 3 of 10, route gating (added 2026-09-24)

Phase 3 does two things: fixes an unrelated pre-existing production bug
found while researching it, and makes `RoutingDecision.route` actually gate
dispatch for the first time (Phases 1-2 only ever logged or redirected
domain — never decided whether local was attempted at all).

**Unrelated bug fix, bundled in because it lives in the same file**:
`complete_with_meta()` (returns `(text, provider, model)`, with a
`skip_local=True` fast path for a 2-candidate cloud-only fallback) and its
`_is_quota_error()` helper are now part of canonical `gateway.py`.
CatalogValidator's `system.py` (unrelated to any of this router-intelligence
work, last touched 2026-05-26) has called this method for a long time —
verified via `git show HEAD` that it never actually existed in any committed
version of `ai_router.py`, only in a long-detached stale worktree from
before the OpenRouter/Gemini provider architecture existed. The resulting
`AttributeError` was silently caught by a broad `except`, degrading real
Cypher-gen/summarization calls to error JSON / generic fallback text. Fixed
canonically so it survives future syncs instead of silently vanishing again;
`complete()` is now a thin wrapper delegating to it. **This Mac's local copy
at `CatalogValidator/backend/mcp_domain/ai_router.py` has the fix — the
machine that actually executes `system.py`'s calls in production
(`retailmcp.local`, reached over HTTP/SSE, not local import) still needs the
same patch applied separately; not something this environment has access to.**

Because the fix lives in the same `gateway.py` as Phases 1-3's routing code,
deploying it to CatalogValidator's local copy necessarily also deployed that
code there for the first time — earlier phases were explicitly kept
undeployed. Verified harmless: none of `AI_ROUTER_INTELLIGENT_ROUTING` /
`AI_ROUTER_ROUTE_GATING` / `AI_ROUTER_DOMAIN_MODE` are set anywhere in
CatalogValidator's or the master `.env`, so all of it stays fully inert
there — flagging the discrepancy from the original "Phase 3 stays
canonical-only" plan rather than letting it pass silently.

**Route gating**: `force_frontier` (new `chat()` kwarg, default `False`)
forces `route="frontier"` regardless of predicted success/threshold.
`AI_ROUTER_ROUTE_GATING` (new flag, default `false`, double-gated with
`AI_ROUTER_INTELLIGENT_ROUTING`) controls whether `decision.route` actually
determines whether Tier 1 is attempted at all, and whether a
`model_hint="smart"` request can still reach local (`_call_local` gained
`bypass_smart_skip`, default `False` — preserves the original hard skip
exactly). Off (default): behavior is provably identical to pre-Phase-3.

**A finding, not a fix**: with today's default thresholds
(`fast=0.70`/`structured=0.82`/`smart=0.95`) and Phase 2's real benchmark
data (retail=0.60, finance=0.87, health=0.55), route gating produces **zero
behavioral change** for real traffic shapes — nothing but finance's
`"fast"` calls would ever clear its threshold. Verified end-to-end with a
local-only script mimicking CatalogValidator's exact `complete_with_meta`
call pattern: default thresholds → still routes frontier (safe, no
surprise); a manually lowered `"structured"` threshold → correctly routes
local for retail's real Cypher-gen task. Recalibrating the default
thresholds (or `DomainConfig.default_quality_threshold`, still unconsulted)
against real benchmark data is a deliberate open decision, not resolved here.

Testing note: an early version of the local-only verification script
accidentally made one real (tiny, harmless) network call to Vertex AI
Gemini via this Mac's ADC credentials, because it only stubbed the
OpenRouter HTTP path and missed that Gemini's direct path uses the
`google-genai` SDK directly. Fixed by disabling Gemini for that script.
Separately, three new tests initially had the same gap (`GEMINI_ENABLED`
unset, so weighted selection could occasionally pick gemini for a live
call) — both `GEMINI_ENABLED=false` and `AI_ROUTER_WEIGHT_GEMINI=0` are now
set in every test that doesn't pin a specific provider, closing this for
good rather than relying on one of the two guards alone.

**Update, 2026-09-24 (later)**: the `complete_with_meta` fix was deployed to
`retailmcp.local` and verified live against real production queries — real
Cypher generation + summarization, correct engine attribution (`gemini`/
`openrouter`), no more degraded fallback text. One of four test queries
surfaced an unrelated real bug (a generated Cypher date filter assumed
`order_date` contains the literal string `"May"`, which never matched the
actual stored format) — fixed separately, not a router issue. The weighted
multi-provider selection that put one query on `gemini` instead of
`openrouter` is pre-existing behavior (CatalogValidator's `backend/.env`
never set `AI_ROUTER_WEIGHT_GEMINI`, so it inherited the gateway's
hardcoded 0.15 default) — user confirmed the current 0.85/0.15/0.1/0.05
OpenRouter/Gemini/Anthropic/OpenAI split is fine as-is, no change made.

## Intelligent routing — Phase 4 of 10, response verifier (added 2026-09-24)

Until now, `_call_local()` returning any non-empty response was sufficient
— "local responded" was the only bar. The live CatalogValidator test above
demonstrated exactly the gap: a confident, well-formed, non-empty answer
that was simply wrong (the `order_date CONTAINS 'May'` bug). Phase 4
introduces a real quality gate between "local responded" and "local
response is used."

**Deliberately deterministic-only, domain-agnostic** (both asked and
confirmed): `DeterministicVerifier` checks for an empty response, a small
set of conservative refusal-phrase signatures, and error/exception
artifacts — nothing else. **Explicit, acknowledged limitation**: this would
**not** have caught the Encompass/May-2026 bug — that answer was non-empty,
not a refusal, and contained no error artifact. Catching a confidently
wrong-but-well-formed answer needs either domain-specific execution
verification (e.g. actually running the generated Cypher, closer to what
the retail dataset generator's own verification loop already does per
memory) or a judge model — both explicitly deferred, not solved here.
`confidence` is deliberately capped at `_DETERMINISTIC_CONFIDENCE = 0.5`
rather than 1.0 — passing these checks means "no red flags found," not
"this is correct," and the constant exists specifically so a future
judge/execution-verifier has room to signal a real confidence increase.
`groundedness`/`completeness` stay `None` — undetermined, not silently
defaulted.

Gated by `AI_ROUTER_VERIFY_LOCAL` (default `false`), independent of
`AI_ROUTER_ROUTE_GATING` — verification applies to whatever local response
comes back, regardless of how the decision to attempt local was made. Off
(default): `_response_verifier` stays `None`, Tier 1 dispatch is
byte-identical to pre-Phase-4. On: a rejected local response falls through
to Tier 2 (cloud) instead of returning — the real behavioral change, an
opt-in cost/latency tradeoff (a rejected local answer now costs an extra
cloud call). A verifier exception fails **open** (returns the local
response as-is, logged as a warning) — same "never block a response" rule
as the router policy's own try/except. `complete_with_meta()`'s normal path
inherits verification automatically via `chat()`; its `skip_local=True`
path stays exempt by design.

Verified with a local-only script (stubbed HTTP, Gemini explicitly disabled
per the lesson above): a refusal-phrase local response correctly escalates
to a stubbed cloud call; a normal local response is correctly returned
directly with no cloud call made.

**Live since 2026-09-24**: `AI_ROUTER_VERIFY_LOCAL` is on for
`portfolio_tracker` (finance) — the first real deployment of anything from
this whole plan. Investigation before enabling found `USE_LOCAL_SLM=true`
and `LOCAL_INTEL_DOMAIN=finance` were already live there, meaning Tier 1
local was already serving real Telegram-bot/dashboard traffic with zero
verification; this flag can only improve on that baseline. `gateway.py`/
`router_policy.py` were synced to both `portfolio_tracker/agents/` and
`physician/agents/` first (physician's copy is unused there — confirmed no
real call site imports `ai_router.py` at all in that project today, so
nothing to enable yet). `com.portfoliotracker.bot`/`.web` were restarted to
pick up the new `.env`; the bot's own log confirmed the exact expected flag
state live.

## Intelligent routing — Phase 5 of 10, frontier router (added 2026-09-24)

Today, Tier 2 cloud selection is pure weighted-random across *providers*
(`_select_cloud_provider()`) — no notion of picking a specific *model*
based on its actual capability/cost/latency profile. Phase 5 adds
`ModelCapability` (already defined inert since Phase 1) + `FrontierRouter` +
`CapabilityFrontierRouter`, implementing the spec's expected-utility formula
(predicted capability − cost penalty − latency penalty), while keeping
weighted selection as the unchanged default and fallback mechanism.

**Model scores are deliberate uniform tier placeholders** (asked and
confirmed): every fast-tier model (`gpt-4o-mini`, `claude-haiku-4-5`,
`gemini-2.5-flash`) scores identically (`0.60` across
reasoning/coding/tool_use/structured_output/long_context); every smart-tier
model (`gpt-4o`, `claude-sonnet-4-5`, `gemini-2.5-pro`) scores identically
(`0.85`). No claim that any provider is objectively better at anything —
no verified basis for that here. **Direct, acknowledged consequence**:
since capability ties within a tier, the utility formula currently
differentiates purely on cost/latency — "capability" mode today behaves as
"prefer the cheapest/fastest provider within the requested tier." Real
per-model differentiation needs real data, not solved here. Cost/latency
themselves are rough illustrative estimates (asked and confirmed to
include them anyway) based on general public pricing patterns, not
live-checked figures — e.g. Gemini Flash ≈ `$0.0003/1k tokens`, `~600ms`;
GPT-4o ≈ `$0.006/1k tokens`, `~1500ms`. Same config-driven registry pattern
as Phase 2's domains (`DEFAULT_MODEL_REGISTRY`, optional `config/models.yaml`
override, never raises on a bad file).

**Selection stays scoped to the requested `model_hint`'s tier** — a design
decision beyond what was literally asked, needed to avoid a `"fast"` call
silently escalating to smart-tier cost. `gateway.py` computes one candidate
`(provider, model)` per available provider (that provider's own configured
model for the requested hint) and passes the set to
`CapabilityFrontierRouter.select()`; `router_policy.py` never sees
`gateway.py`'s `self.models` config, and `gateway.py`'s routing logic never
contains a model-name literal ("do not hardcode specific model versions
into routing logic" — model names live only in config, on both sides).

Gated by `AI_ROUTER_FRONTIER_POLICY` (`"weighted"` default, unchanged;
`"capability"` opt-in), itself gated by `AI_ROUTER_INTELLIGENT_ROUTING`.
Only the **primary** pick changes — a caller-supplied `model_override`
bypasses capability selection entirely (same precedence it already has),
and a primary failure always falls back through the legacy
weighted/hint-based path on retry, never the capability pick again (per the
spec's "keep weighted routing... as fallback mode").

One real bug caught during testing, not in the design: the first
implementation's fallback-model selection used `capability_override`
whenever the loop was on the primary provider's first attempt, even when
`capability_override` was `None` (frontier policy off, or a caller-supplied
`model_override` should have applied) — silently dropping the caller's
explicit `model_override` on that first attempt. Fixed to check
`capability_override is not None` explicitly; caught by
`test_model_override_bypasses_capability_selection` before merge.

## Intelligent routing — Phase 6 of 10, experience store (added 2026-09-24)

Every routing/verification decision made by Phases 1-5 was computed and
then discarded — nothing persisted across requests. Phase 6 introduces an
append-only trace of what happened on each request, the foundation Phase
7's offline training-candidate generation will read from. **Changes zero
routing behavior** — pure observability, recording what already happens,
not altering it.

Real stakes this time: portfolio_tracker (and potentially physician) now
handle real financial/health questions through this code. **Redacted by
default** (asked and confirmed): `ExperienceTrace`'s content fields
(`request`, `local_response`, `frontier_response`) become length+hash
placeholders via `_redact_text()` — deterministic (repeat/duplicate
detection still possible) but never recoverable to the original text —
unless a deployment explicitly opts into a less-redacted mode (see the
Phase 7 section below for the 3-value `AI_ROUTER_EXPERIENCE_REDACT` mode
this became). In full-redact mode, tool-call *names* are kept (not
sensitive, matter for Phase 7); arguments are dropped. Error messages are
capped at 200 chars but not redacted — they describe the system's
behavior, not the user's data, and redacting them would hurt debuggability
far more than it protects privacy. Routing/verification/latency/cost
METADATA is always stored in full regardless of the redact setting —
domain, route, accept/reject, escalation reason, timing, token counts.

**Storage**: `JSONLExperienceStore` — one JSON line per trace, appended to
`data/experience_traces.jsonl` (same `Path(__file__).parent`-relative
convention as the round-robin state file), asked and confirmed over SQLite
for this phase: zero-dependency, trivially inspectable, easy to
batch-import into SQLite/Postgres later. `ExperienceStore` stays a
Protocol so a different backend is a drop-in swap. `dataclasses.asdict()`
serializes the nested `RoutingDecision`/`VerificationResult` automatically.
A write failure never raises — logged as a warning, the real response is
untouched, same guarantee as the router policy and verifier already have.

**Scoped to the Tier 1/Tier 2 local-vs-frontier loop only** — a stated
limitation, not a silent gap. Tier 0 (remote-forward, delegates entirely to
another machine's gateway) and explicit `provider=` pins (deliberate
one-off overrides, e.g. `stock_intelligence.py`'s `web_search` tool call)
are not traced — operationally distinct from the loop this system exists
to learn from. `complete_with_meta()`'s `skip_local=True` path bypasses
`chat()` entirely and isn't traced either, consistent with every prior
phase's scoping of that path. Mechanically: a small dict of trace fields
accumulates as Tier 1/2 execution proceeds (local latency, verification
result, frontier provider/model/latency, escalation reason), and
`_record_experience()` is called at each of that block's actual exit
points — not a restructure of the existing return flow, instrumentation
added alongside it.

Gated by `AI_ROUTER_EXPERIENCE_STORE` (default `false`, gated by
`AI_ROUTER_INTELLIGENT_ROUTING`). A new `AI_ROUTER_CLIENT_APP` env var
(default `"unknown"`) labels each trace's `client_app` field — no effect
on routing.

Verified with a local-only script (temp JSONL path, stubbed HTTP): a
good local response writes one `final_source="local"`,
`escalated=false` record; an empty local response correctly writes one
`final_source="frontier"`, `escalated=true`,
`escalation_reason="verification_rejected: empty response"` record after
escalating. Inspected the actual written JSON by eye — confirmed no raw
content anywhere in the file for either record.

## Intelligent routing — Phase 7 of 10, learning candidate generation (added 2026-09-24)

Turns Experience Store traces into training-candidate datasets — the
offline consumer of Phase 6's traces. **Explicitly outside the request
path**: `training/generate_candidates.py` is a standalone batch script,
never imported by `gateway.py`/`router_policy.py`, never called from
`chat()`, and not copied to any live deployment. It's a tool you run by
hand against a specific deployment's `data/experience_traces.jsonl` later,
once one exists.

**A Phase 6 upgrade came first, driven by a real tension**: Phase 6's
default redaction (hash-only) makes adapter-distillation candidates
impossible to build — a hash isn't trainable data. Asked how to resolve
this; the answer (asked and confirmed) was to design a middle-ground
redaction scheme rather than just accept the gap. `AI_ROUTER_EXPERIENCE_REDACT`
became a 3-value mode instead of a boolean:

- `"full"` (default; `"true"` is still accepted as an alias) — unchanged
  Phase 6 behavior, length+hash placeholders, tool-call arguments dropped.
- `"none"` (`"false"` still accepted as an alias) — unchanged, raw content.
- `"partial"` (new) — regex-based scrubbing of structured PII patterns
  (dollar amounts, SSN-shaped numbers, phone numbers, emails, long
  account-number-like digit runs) via `_redact_pii_patterns()`, leaving
  surrounding text intact and usable for training. Tool-call arguments are
  kept (PII-scrubbed) instead of dropped — scrubbed args still carry real
  training signal that full-redact mode throws away entirely.

**Stated limitation, not a silent gap**: partial-mode redaction is
regex-only (zero-dependency, matching every prior phase's constraint) and
**cannot** reliably catch free-text PII like names or addresses — it
reduces exposure, it does not guarantee anonymization. NER/ML-based
redaction was considered and explicitly ruled out as out of scope for a
stdlib-only repo. `"full"` stays the default; `"partial"` is an explicit
per-deployment opt-in, same as `"none"` already was.

Backward compatible: `build_experience_trace()`'s parameter changed from
`redact_content: bool` to `redact_mode: str = "full"` in place (safe —
Phase 6 was built this session and had never been deployed live with
content enabled anywhere, so nothing depended on the old shape).

**Three candidate datasets, one script run**:

- **`router`** — works regardless of redaction level (metadata only):
  domain, task, `model_hint`, complexity, risk, `predicted_local_success`,
  `quality_threshold`, route decided, and the ground-truth label
  `label_local_succeeded = (final_source == "local" and not escalated)`.
- **`distill`** — adapter/distillation candidates. Requires
  `escalated=True`, `final_source == "frontier"`, usable (non-fully-redacted)
  content, and the frontier response passing a quality gate — reuses Phase
  4's `DeterministicVerifier().evaluate(None, None, frontier_text)` directly
  against the frontier side (confirmed its `context`/`decision` params are
  unused in the deterministic implementation, safe to call this way).
  Deduplicated on `(domain, sha256(frontier_text))`. A `stats` dict
  explains every exclusion (`not_escalated`, `not_frontier`,
  `fully_redacted`, `failed_quality_gate`, `duplicate`, `accepted`) so a
  smaller-than-expected dataset is never a mystery.
- **`verifier`** — any trace where `verification is not None` (both
  accepted and rejected cases): request, local response, verifier's
  decision, frontier comparison when one exists, `user_feedback` (always
  `None` today — no feedback collection mechanism exists yet, a stated
  gap), final outcome.

**Dataset versioning** (the spec's explicit ask): each run writes to
`training/datasets/<candidate_type>/<version>/data.jsonl` + `manifest.json`
(version id = timestamp + uuid, generated-at, source trace file path,
source trace count, candidate count, router/verifier version strings
pulled from the traces, plus type-specific extras like `distill`'s
exclusion stats) — so a future training run can cite exactly which dataset
version it used.

**Summary report on every run**: candidate counts per type, and an
explicit warning when `distill` comes up empty specifically because source
traces were fully redacted (names the count, names the fix —
`AI_ROUTER_EXPERIENCE_REDACT=partial` or `none`) — never a silent or
misleading empty dataset.

Verified against a synthetic fixture (4 traces spanning all three redact
modes and all three candidate scenarios — local-accepted, escalated
successfully, escalated-but-frontier-also-fails, total-failure), built via
`router_policy.py`'s own `build_experience_trace()` for schema fidelity,
not hand-written JSON. Ran the script via CLI (not just unittest) and
inspected `data.jsonl`/`manifest.json` by eye: `router` produced 4/4
candidates (works at every redaction level, including the fully-redacted
trace); `distill` produced exactly 1/4 (the partial-redacted, escalated,
quality-gate-passing trace — correctly excluding the non-escalated trace,
the fully-redacted trace, and the trace whose frontier response was itself
a refusal), with PII correctly scrubbed to `<REDACTED_AMOUNT>` in both the
request and frontier response; `verifier` produced 3/4 (excluding only the
trace with no verification result), visibly showing all three redaction
modes side by side in the same file — hash placeholder, PII-scrubbed text,
and raw text. **No real trace data exists anywhere yet to run this
against** — `AI_ROUTER_EXPERIENCE_STORE` was built in Phase 6 but has never
been enabled live (portfolio_tracker only has `verify_local` on).

## Intelligent routing — Phase 8 of 10, observability + a real feedback loop (added 2026-09-25)

### Context

Phases 1-7 compute and persist routing/verification/escalation decisions,
but nothing surfaces them as metrics, and nothing lets a deployment tell
the system whether a locally-accepted answer was actually right. The
original spec calls **false-accept rate "the headline safety metric,"**
which is a real design problem: nothing in the system had ground truth.
Phase 4's `DeterministicVerifier` confidence is deliberately capped at 0.5
for exactly this reason (it can't tell well-formed-but-wrong from right —
the Encompass/May-2026 bug is the canonical example), and
`ExperienceTrace.user_feedback` had sat inert (always `None`) since Phase 1.

Asked how to handle this; the answer (asked and confirmed) was to **build a
real feedback-loop hook now**, not just report proxies for the missing
ground truth — bigger scope than the original 10-phase list implies (a
foundation more than a full feature — no live deployment has a UI wired to
call it yet, see Deployment below), but it's what makes a *real*, not
proxy, false-accept rate computable. Also asked and confirmed: **both**
metrics sources — live in-process counters (content-free, cheap, no new
flag) for real-time visibility in every deployment immediately, plus an
offline aggregator over Experience Store + feedback trace files (same
`training/` pattern as Phase 7) for the deeper analysis, including the real
false-accept rate.

### 1. A real feedback loop

`FeedbackRecord` + `FeedbackStore` + `JSONLFeedbackStore` +
`build_feedback_record()` (all in `router_policy.py`), append-only and
**joined against trace records at read time** — not an in-place update to
the trace file, same rationale as Phase 6's append-only trace store (no
locking, no concurrent-writer hazard across the long-running bot/web
processes). Multiple feedback events on the same `trace_id` are all
preserved (a user changing their mind isn't data loss); the offline
aggregator resolves conflicts by taking the *last* record per `trace_id`.

`label` is a controlled vocabulary (`"correct"` / `"incorrect"` /
`"unrated"` — `_VALID_FEEDBACK_LABELS`); an invalid label makes
`build_feedback_record()` return `None`, logged as a warning, nothing
written. The free-text `note` field gets the **same 3-value redact mode**
already governing trace content (`full`/`partial`/`none`) — a feedback
note is exactly the kind of field a user could put PII into ("this was
wrong, my number is..."), so it gets one privacy policy, not two.

### 2. Threading `trace_id` back to the caller

`chat()` now generates a `_trace_id = uuid.uuid4().hex` unconditionally
near its top (stable across all of a call's possible exit points) and
passes it into `build_experience_trace()` (which gained a
`trace_id: Optional[str] = None` param — omitted, it auto-generates
internally exactly as every prior phase's tests already expect).
`RouterResponse` gained one new field, `trace_id: Optional[str] = None`
(confirmed backward compatible — all 9 existing construction sites in
`gateway.py` use keyword args) — set on the returned response **only when
`AI_ROUTER_EXPERIENCE_STORE` is on**; otherwise left `None`, since a
populated `trace_id` with no matching trace record would be a dangling
reference nobody could look up.

### 3. `AIGateway.record_feedback(trace_id, label, note=None) -> bool`

The public method a deployment calls later to attach feedback to a
previously-returned response. No-op (returns `False`, never raises) if
`AI_ROUTER_EXPERIENCE_STORE` is off or `label` is invalid — same fail-open
guarantee as every other subsystem. `self._feedback_store` is constructed
alongside `self._experience_store`, gated by the **same flag**
(`AI_ROUTER_EXPERIENCE_STORE`) — no new env var, since a feedback record
with no matching trace record to correlate against is meaningless.

**Stated plainly**: no live deployment has a UI wired to call this today.
physician-web's separate thumbs-up/down loop is a different TypeScript
app/router (out of scope until Phase 10); wiring portfolio_tracker's
Telegram bot to call this would need its own follow-up work (persisting
`trace_id` alongside a message so a later 👎 reaction could call it) — a
real, useful, separate integration task, not part of this phase. Phase 8
builds the capability; wiring a live UI to it is future work.

### 4. Live in-process metrics — `AIGateway.get_metrics_snapshot()`

A small bounded `_MetricsCollector` (`gateway.py`), constructed
unconditionally whenever `AI_ROUTER_INTELLIGENT_ROUTING` is on — **no new
flag**: pure content-free counting (domain names, route/accept/reject
counts, latency numbers, never message text), so it rides for free on the
same shadow-computation flag Phase 1 already established as
safe-by-default. Folded into the existing `_record_experience()` method as
an unconditional first step (before its `AI_ROUTER_EXPERIENCE_STORE`-gated
part) — every exit point in `chat()` that already called it now feeds the
counters too, no new call sites needed. `threading.Lock`-guarded — a real
concern given portfolio_tracker's Flask dashboard + Telegram bot can share
one process.

Reports: `total_requests`, route split (local/frontier), verifier
accept/reject counts, per-provider frontier usage, error count, p50/p95/p99
latency (local and frontier, via `statistics.quantiles` over a
`deque(maxlen=500)` rolling window — bounded memory, not unbounded
history), and **per-domain**: total, local-accepted count, escalated count,
and `benchmark_deviation` — observed local-accepted-without-escalation rate
for that domain minus its `DomainConfig.benchmark_success_rate` (Phase 2's
registry, reused via `load_domain_registry()`). Large negative deviation is
a real, always-on canary — explicitly labeled a proxy, not a correctness
proof, same caveat Phase 4 already established for verifier confidence.

**Does NOT report real false-accept rate** — that needs feedback data,
which arrives asynchronously and can't be joined against bounded in-memory
counters without an unbounded `trace_id → outcome` map. That's
`training/compute_metrics.py`'s job. Exposed as a plain method
(`get_metrics_snapshot() -> dict`), plus a `log_metrics_snapshot()`
convenience wrapping it in a `logger.info` call — not a Prometheus endpoint
or background thread, since this repo owns no web framework
(CatalogValidator is Streamlit, portfolio_tracker is Flask+Telegram,
physician-web is TS) and starting an HTTP server inside a shared library
would be an unwanted new moving part. A caller that wants a `/metrics`
route wires this into whatever framework it already has.

### 5. Offline aggregator: `training/compute_metrics.py`

New sibling to `training/generate_candidates.py`, same posture (standalone
batch script, never imported by `gateway.py`/`router_policy.py`, never
called from `chat()`, not copied to any deployment). Imports `load_traces()`
from `generate_candidates.py` directly (reused, not duplicated); adds a
matching `load_feedback()` with the same defensive skip-on-malformed-line
behavior. Joins by `trace_id` (last record per id wins, by file order).

Computes, per domain: `verifier_accept_rate`/`verifier_reject_rate`/
`escalation_rate`/`local_route_share` (same definitions as the live
snapshot, from durable trace history instead of a bounded window),
`benchmark_deviation` (same proxy), and **`false_accept_rate`** — the real
metric the spec asks for. Numerator: `final_source=="local"` traces whose
*latest* joined feedback record has `label=="incorrect"`. Denominator:
`final_source=="local"` traces that have **any** feedback at all — not "all
local-accepted traces," since a rate over unlabeled data would be
fabricated, not measured. Reported alongside **`feedback_coverage`**
(labeled / all local-accepted), so the number is never presented without
the sample size behind it. **Zero coverage reports `false_accept_rate:
null`, not `0.0`** — "not enough data" and "measured zero false accepts"
are categorically different claims. Overall (not per-domain): latency
percentiles, provider distribution, error rate.

Writes `training/reports/metrics/<version>/report.json` (version =
timestamp + uuid, same convention as Phase 7's `write_dataset()`), with
`generated_at`/source file paths/counts alongside the metrics. Prints a
human-readable summary, with an explicit note whenever a domain's
`false_accept_rate` is `null` because coverage is zero — naming the count,
naming the fix (`record_feedback()` calls from a live UI).

Verified against a synthetic 2-trace/2-feedback fixture (built via real
`chat()` calls through the actual gateway, stubbed HTTP, `redact_mode=
"partial"`): confirmed `RouterResponse.trace_id` correctly threaded and
matched the trace file, feedback notes correctly PII-scrubbed (an embedded
fake SSN became `<REDACTED_SSN>`), `get_metrics_snapshot()` showed correct
route/verifier/domain bucketing and non-null latency percentiles, and the
CLI's `report.json` showed `false_accept_rate=1.0`/`feedback_coverage=1.0`
for the one local-accepted trace that received `"incorrect"` feedback —
exactly matching the feedback calls made, no more, no less. **No real
trace or feedback data exists anywhere live yet** — `AI_ROUTER_EXPERIENCE_
STORE` has never been enabled live, and `record_feedback()` has no live
caller (see above), so this script's output against any real deployment's
files today would legitimately be `false_accept_rate: null` everywhere
until a feedback UI is wired up.

### Backward compatibility

| Risk | Mitigation |
|---|---|
| `AI_ROUTER_INTELLIGENT_ROUTING` off (default everywhere) | `_metrics` stays `None`, `get_metrics_snapshot()` returns `{"enabled": False}`, the counter-update step in `_record_experience()` never runs (guarded by the same `decision is not None` check already gating the rest of that function) — zero overhead, zero behavior change. |
| `AI_ROUTER_EXPERIENCE_STORE` off (everywhere today, including portfolio_tracker) | `_feedback_store` stays `None`, `record_feedback()` always returns `False` immediately, no file ever created. `RouterResponse.trace_id` stays `None`. |
| New `RouterResponse.trace_id` field | Backward compatible — all 9 existing construction sites use keyword args; a new trailing `Optional[str] = None` field breaks nothing. |
| Orphaned feedback (`trace_id` that doesn't match any real trace) | Recorded anyway — harmless, just never matches anything in `compute_metrics.py`'s join and is silently excluded from every rate. |
| Counters under concurrent access | `threading.Lock` around all mutation/reads. |
| `false_accept_rate` misread as authoritative with near-zero coverage | `feedback_coverage` always reported alongside it; zero-coverage domains report `null`, not `0.0`. |

## Intelligent routing — Phase 9 of 10, versioning + shadow/canary groundwork (added 2026-09-25)

### Context

The spec's Phase 9 line: "Every trace records policy/adapter/verifier/
model versions; shadow/canary groundwork." Investigating what was already
recorded turned up two confirmed bugs, fixed regardless of anything else:

1. **`ExperienceTrace.adapter_version`/`model_versions["local"]` recorded
   the wrong value** — `decision.selected_adapter` (the adapter *name*,
   e.g. `"retail_v3"`) instead of the real version string
   (`DomainConfig.adapter_version`, e.g. `"v3"`). Every trace ever written
   had this wrong. `RoutingDecision` gained its own `adapter_version`
   field (populated in `decide()` from the already-resolved
   `domain_config.adapter_version`), and `build_experience_trace()` now
   reads that instead.
2. **`compute_metrics.py`'s `benchmark_deviation` re-derived its baseline
   from today's LIVE domain registry** at analysis time instead of from
   each trace's own frozen `predicted_local_success`. Harmless while the
   registry hasn't changed, but wrong the moment a domain gets
   rebenchmarked after some traces were already written — exactly the
   point-in-time-correctness problem "versioning" exists to prevent.

Asked how far "shadow/canary groundwork" should go (deliberately
open-ended in the spec); the answer (asked and confirmed): **instance-level
version tagging + env vars + grouped metrics** — not live traffic-splitting.
A deployment canaries by running two separately-configured `AIGateway`
processes and comparing their independently-collected trace files
afterward via `training/compute_metrics.py`. There is no dual-dispatch or
percentage-based bucketing code anywhere in `ai-router`, and none was added.

### 1. `policy_id` / `verifier_id` — canary tagging

`DeterministicRouterPolicy`/`DeterministicVerifier` gained
`policy_id`/`verifier_id` constructor params (default `None` → fall back to
`ROUTER_VERSION`/`VERIFIER_VERSION`, byte-identical to before). New env
vars `AI_ROUTER_POLICY_ID`/`AI_ROUTER_VERIFIER_ID` (both optional, unset
everywhere today) let a deployment tag an instance without subclassing
anything — two instances with different threshold tables or verifier
tuning, each tagged, produce traces whose `router_version`/
`verifier_version` fields identify which cohort they came from.

### 2. `frontier_router_version` — which selection path actually ran

New module constants `FRONTIER_ROUTER_VERSION = "capability-v1"` and
`WEIGHTED_FRONTIER_VERSION = "weighted-v1"`. `ExperienceTrace` gained a
matching `frontier_router_version: Optional[str] = None` field —
`gateway.py`'s `chat()` derives it from the *actual outcome* of the
existing `capability_override` computation (not just which mode is
configured — a capability-mode instance that falls through to weighted on
a given request correctly records `"weighted-v1"` for that request).
`None` when a request never reaches frontier selection at all (local
succeeded) — verified by eye in the smoke test below.

### 3. `compute_metrics.py`: the staleness fix + version-cohort breakdowns

Replaced the per-domain-only calculation with a generic `_group_metrics()`
helper used three ways: `compute_domain_metrics()` (existing, signature
simplified from `(traces, feedback_by_trace, registry)` to `(traces,
feedback_by_trace)` — the registry lookup is gone, replaced by each
group's own traces' averaged `predicted_local_success`, now exposed as a
new `avg_predicted_local_success` field alongside `benchmark_deviation`),
`compute_router_version_metrics()` and `compute_verifier_version_metrics()`
(both new — the direct analytical payoff of `policy_id`/`verifier_id`
tagging: point this script at two cohorts' trace files, or one merged
file, and compare `false_accept_rate`/`verifier_accept_rate` between them).
`compute_verifier_version_metrics()` naturally excludes traces where no
verifier ran (nothing to group them under).

### Backward compatibility

| Risk | Mitigation |
|---|---|
| `AI_ROUTER_POLICY_ID`/`AI_ROUTER_VERIFIER_ID` unset (everywhere today) | Both resolve to `None`; constructors fall back to `ROUTER_VERSION`/`VERIFIER_VERSION` — every trace's version fields are byte-identical to pre-Phase-9. |
| New `RoutingDecision.adapter_version`/`ExperienceTrace.frontier_router_version` fields | Both are only ever constructed through their one builder (`decide()`/`build_experience_trace()`), updated together — no external direct-construction call site to break. |
| `compute_domain_metrics()` signature change | Canonical-only script, no live caller — same in-place-fix precedent as Phase 7's `redact_content`→`redact_mode` rename; existing tests updated, not shimmed. |
| Historical trace files predating Phase 9 | None exist live (`AI_ROUTER_EXPERIENCE_STORE` has never been enabled live) — moot today, noted so it's not a silent surprise later. |

### Verification

147/147 tests passing (127 prior + 20 new), stable across 3 runs. Local-
only smoke test: two differently-tagged `AIGateway` instances (`policy_id=
"baseline"`/`"canary-lower-threshold"`, `verifier_id=
"baseline-verifier"`/`"canary-verifier"`, one on the retail domain, one on
finance) wrote to a shared trace file — inspected by eye: `adapter_version`
correctly showed `"v3"`/`"2026-08-22"` (not `"retail_v3"`/`"finance"`, the
old buggy values), `router_version`/`verifier_version` correctly identified
each cohort, `frontier_router_version` was `None` for the two local-success
traces and `"weighted-v1"` for the one that escalated. Ran
`training/compute_metrics.py`'s CLI against the shared file and inspected
`report.json`: `by_router_version`/`by_verifier_version` cleanly separated
the two cohorts with independent rates, and `avg_predicted_local_success`
(0.6 for retail, 0.87 for finance) matched the registry's current
benchmarks exactly, confirming the staleness-fix calculation produces
identical results to the old one today — it only diverges once the
registry actually changes, which is the whole point.

## Intelligent routing — Phase 10 of 10, central service + client integration (added 2026-09-25)

### Context

The final phase. Spec line: "router_policy.py → router/ package; canonical
Python package or central service; TS side shares policy defs or calls the
service; compatibility facade preserved." Investigation found a central
service already existed in embryonic form — `server.py`, stdlib
`http.server`, wraps `chat()` over HTTP for the pre-existing "Tier 0 remote
gateway" pattern — but it only exposed `/v1/chat`+`/health`, not the Phase
4-9 surface. Also found physician-web genuinely depends on this repo's TS
package (`@pmuppirala/ai-router` via GitHub), but that package (`src/
router.ts`) is the pre-intelligent-routing implementation — none of
Phases 1-9 exist in TypeScript anywhere, and physician-web has its own
separately-evolved DB-backed router-config system, unrelated to this.

Asked how far to go on both; answers (asked and confirmed): **extend
`server.py` into a real central service and migrate the 3 live clients to
call it over HTTP** (not a `router/` package split — a direct choice
between the two); **port just the TS type definitions**, not the decision
logic; and for client integration, **sync files to all 3 AND turn on
`AI_ROUTER_EXPERIENCE_STORE` live** for portfolio_tracker/CatalogValidator.

**A new risk class, stated plainly**: this is the first phase that changes
*where* a live client's LLM calls physically execute — local Python call →
network hop to a new standing service — rather than adding an opt-in local
code path. A bug or crash in the central service now affects every client
pointed at it simultaneously, where today a crash is isolated to one
process. Built and tested as new, additive, canonical-only capability
first; the live cutover of portfolio_tracker/CatalogValidator is the last,
most carefully staged step (see "Client integration" below).

### 1. Extended remote-forwarding surface

New `server.py` endpoints: `POST /v1/complete_with_meta`, `POST /v1/
record_feedback`, `GET /v1/metrics` (alongside the existing `POST /v1/chat`
+ `GET /health`). `gateway.py` gained a `_get_json()` sibling to the
existing `_post_json()` for the one read-only call. Of the four public
methods, three needed a genuinely new forward branch and one didn't:

- `chat()` already forwards via Tier 0 (`if self._remote_url: return self.
  _call_remote(...)`, checked before anything else, including provider
  pins — confirmed a pinned call like `stock_intelligence.py`'s
  `provider="anthropic"` remotes correctly with zero changes needed).
- `complete_with_meta(skip_local=False)` already remotes for free — it
  delegates to `self.chat()`, which already has the check above.
- `complete_with_meta(skip_local=True)` was the one real gap — it bypasses
  `chat()` entirely and dispatched cloud calls with local keys regardless
  of remote mode. Gained its own `if self._remote_url:` branch forwarding
  to the new endpoint.
- `record_feedback()`/`get_metrics_snapshot()` gained matching branches —
  when a client is fully remoted, ALL state (traces, feedback, metrics)
  correctly lives on the central service, not fragmented across processes.

No new auth — same threat model `server.py` already had ("Tailscale-only,
so being permissive here is safe"); this phase doesn't change who can
reach it, only what a reachable client can call.

### 2. TS type definitions only

`src/types.ts`/`src/index.ts` gained `RoutingDecision`, `VerificationResult`,
`ExperienceTrace`, `FeedbackRecord`, `DomainConfig`, `ModelCapability` —
interfaces mirroring the Python dataclasses' shapes field-for-field
(snake_case, matching the real JSON), published for optional future use
(e.g. Node tooling reading `data/experience_traces.jsonl` with type
safety). **No logic ported** — `AIRouter` (the TS class) doesn't reference
any of them; `DeterministicRouterPolicy`/`DeterministicVerifier`/the
stores/redaction stay Python-only. physician-web's own repo (`lib/ai-
router.ts`) was not touched — a separate project outside this session;
pulling these in is its own maintainer's call.

### 3. Persistent service

New `~/Library/LaunchAgents/com.airouter.server.plist` running `server.py`
with `KeepAlive`+`RunAtLoad`, same `launchctl kickstart -k` restart
convention as every other service in this ecosystem.

### 4. Client integration — staged

**Stage A** (sync): backed up and replaced each of the 3 clients'
`ai_router.py`/`router_policy.py` with the canonical Phase 1-9 files,
compiled + ran the backward-compat suite in each project's own venv.
Brought CatalogValidator (was Phase 1-3) and portfolio_tracker/physician
(were Phase 1-5) all up to full parity. Zero behavior change by itself.

**Stage B** (observability live): `AI_ROUTER_EXPERIENCE_STORE=true` in
portfolio_tracker's and CatalogValidator's own `.env` (backed up first)
and in `~/Projects/ai-router/.env` (the central service's config) — so
observability works whether a client dispatches locally or remotely.
Redaction mode untouched (stays the `"full"` default).

**Stage C** (service up): started the central service, confirmed `/health`
and a real `/v1/chat` call work directly against it.

**Stage D** (pilot): physician set `REMOTE_GATEWAY_URL` — zero live
callers today (confirmed dead code back in the mid-session investigation),
the lowest-risk possible validation target.

**Stage E** (live cutover): portfolio_tracker + CatalogValidator's *local*
processes only — not the Mac Mini (`retailmcp.local`), which stays out of
scope, same limitation as every prior phase touching it. Only
CatalogValidator's `system.py` call sites (`complete_with_meta`) are
actually affected — the other five call sites bypass `chat()`/`complete()`
entirely (confirmed back in Phase 3) and keep using local keys regardless.

### Backward compatibility

| Risk | Mitigation |
|---|---|
| `REMOTE_GATEWAY_URL` unset (every client, until Stage D/E) | Every new forward branch is gated by `if self._remote_url:` — unset means none of it runs, byte-identical to before. |
| Central service as a new single point of failure | Stated plainly, not eliminated — mitigated by staging (physician pilots with zero real risk first), `launchd`'s `KeepAlive`, and instant one-env-var rollback. A real, new risk class accepted by explicit choice. |
| New TS exports break existing consumers | Additive only — `npm run typecheck`/`build` verified clean, nothing existing renamed or reshaped. |

### Verification

164/164 tests passing (147 prior + 17 new: 9 gateway-side remote-forwarding
tests mocking `_post_json`/`_get_json`, 8 in new `test_server.py` — a real
`server.py` subprocess hit with real HTTP requests, no mocking). A real
credential leak was caught and fixed during this work: the test's
subprocess environment inherited the parent process's env (or, after an
initial fix attempt, still leaked via `gateway.py`'s `_env()` helper
falling back to `dotenv_values()`'s frame-based file lookup — an empty-
string override doesn't suppress it, only a non-empty fake value does).
Fixed by using an explicit env allowlist with non-empty fake credentials
for every provider — confirmed via a `ResourceWarning`-as-error rerun that
no real network calls occur. `npm run typecheck && npm run build` clean;
new type exports confirmed present in `dist/index.d.ts`.

### Client integration results (2026-09-25)

**Stage A** — all 3 clients synced (backed up first): CatalogValidator
(was Phase 1-3) and portfolio_tracker/physician (were Phase 1-5) all
brought to full Phase 1-9 parity. Verified via the ai-router backward-
compat suite run against each freshly-synced copy, in each project's own
venv: CatalogValidator (Python 3.12 + 3.13, both present), portfolio_tracker
(Python 3.14), physician (Python 3.11) — 127/127 in every case (the 2
canonical-only training-tooling test files excluded, since neither
`training/` nor `server.py` is copied to any client — same scoping every
prior phase established).

**Stage B** — `AI_ROUTER_EXPERIENCE_STORE=true` added to portfolio_tracker's
and CatalogValidator's own `.env` (backed up first; CatalogValidator also
got `AI_ROUTER_INTELLIGENT_ROUTING=true` for the first time — previously
off entirely). **Deliberately NOT added to `~/Projects/ai-router/.env`** —
that master file loads into every consuming project's own process, and
setting a behavior flag there (rather than just API keys) would have
silently forced it on everywhere, exactly the precedence-inversion bug
this same file already carries a comment about from a prior incident with
routing weights. The central service's own copy of these flags instead
lives in its launchd job's `EnvironmentVariables` (see Stage C), scoped to
that one process.

**Stage C** — `~/Library/LaunchAgents/com.airouter.server.plist` created
and loaded (`launchctl bootstrap`), `KeepAlive`+`RunAtLoad`, same
`launchctl kickstart -k gui/$(id -u)/com.airouter.server` restart
convention as every other service in this ecosystem. Verified: `/health`
and `/v1/metrics` respond correctly on `127.0.0.1:7861`.

**portfolio_tracker restart**: `launchctl kickstart -k` on both
`com.portfoliotracker.bot`/`.web`. Both came back up cleanly (new PIDs
confirmed); the bot's own log shows the expected line — `experience_
store=yes feedback=yes metrics=yes policy_id=deterministic-v1 verifier_
id=deterministic-v1` — no errors introduced (the one warning present,
`OPENAI_API_KEY not set`, is pre-existing and unrelated).

**Stage D (physician pilot)** — `REMOTE_GATEWAY_URL=http://127.0.0.1:7861`
added to physician's `.env` (backed up first; physician has zero live
callers today, confirmed dead code — the lowest-risk possible validation
target). Ran a real `complete()` call through physician's own deployed
files + its own `.env`: correctly detected `REMOTE_GATEWAY_URL`, forwarded
to the central service, got a real response back. The central service's
own `/v1/metrics` and `data/experience_traces.jsonl` both confirm the
round trip (`total_requests: 1`, `route_counts.frontier: 1`,
`provider_counts.openrouter: 1`, one real trace with the expected shape).

**Stage E (portfolio_tracker + CatalogValidator live cutover to
`REMOTE_GATEWAY_URL`) — NOT done.** Per the plan, this is the step with
real production stakes and stays explicitly deferred pending a separate
go-ahead. **CatalogValidator's own process restart (to pick up Stages A/B)
is also still pending** — its 5 live processes (3 Streamlit apps, a
supplier app, `telegram_bot.py`) are plain backgrounded shell scripts, not
launchd jobs, and restarting them was left to the user rather than guessed
at.

## Open items (not yet actioned)

- CatalogValidator has no `LOCAL_INTEL_DOMAIN` set — works today only
  because `"retail"` happens to be first in `/v1/models`. Worth setting
  explicitly so it doesn't silently break if LocalIntelligence's domain
  ordering ever changes.
- `DEFAULT_THRESHOLDS`/`DomainConfig.default_quality_threshold` are not
  calibrated against Phase 2's real benchmark data — route gating is
  currently a no-op for real domain traffic as a result. Deliberate, pending
  a decision on how to recalibrate.
- Phase 4's deterministic verifier cannot catch semantically-wrong-but-
  well-formed answers (see above) — domain-specific execution verification
  or a judge model would be needed, neither built yet.
- Phase 5's model capability/cost/latency data are illustrative placeholders,
  not verified — real numbers would meaningfully improve "capability" mode's
  usefulness beyond "prefer cheapest/fastest in tier."
- Phase 6's `experience_traces.jsonl` has no rotation/retention policy —
  append-only, unbounded growth over time. Fine for now, worth revisiting
  before this runs live for an extended period.
- Phase 6 doesn't trace Tier 0/pinned-provider calls or
  `complete_with_meta(skip_local=True)` — real signal from those paths
  (e.g. `stock_intelligence.py`'s tool-use escalations) isn't captured yet.
- The config-precedence inversion (master store beats project `.env`,
  contradicting the docstring) is real but low-impact today. Fix if it ever
  causes a real incident, otherwise leave as documented behavior.
  `AI_ROUTER_INTELLIGENT_ROUTING` is read through the same `_env()` closure
  and inherits this — relevant once per-project rollout starts (e.g.
  enabling it for one project before the others), since setting it in the
  master `.env` would silently override every project's own setting, same
  as `LOCAL_INTEL_URL` already does.
- No automated sync mechanism between canonical `gateway.py` and its three
  project copies — see "Drift history" above. `router_policy.py` will need
  the same manual re-copy discipline once Phase 1 is deployed to physician/
  portfolio_tracker/CatalogValidator.
- Phase 1's `predicted_local_success`/`complexity`/`risk` heuristics are
  intentionally crude placeholders (see module docstring in
  `router_policy.py`) — not yet backed by any real signal.
