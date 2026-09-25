"""
router_policy.py — Intelligent routing data models (Phase 1-9)
══════════════════════════════════════════════════════════════
Sibling file to gateway.py, deployed the same way (copied alongside it into
each project). Deliberately NOT a package yet — gateway.py is a flat file
with no reliable package context across its three deployments, so this stays
flat too until the Phase 10 migration to a real `router/` package.

Zero *required* third-party imports, by design — matches gateway.py's
dependency-free philosophy. PyYAML is optional (see "Domain registry" below)
and guarded exactly like gateway.py's httpx/google-genai imports.

Phase 1 scope
─────────────
`RoutingDecision` + `RouterPolicy` + `DeterministicRouterPolicy` have real
behavior. `VerificationResult`, `ExperienceTrace`, and `ModelCapability` are
defined here as inert dataclasses — interfaces agreed on now so later phases
(4/6/5 respectively) don't need a breaking type change when they add
behavior.

Phase 2 scope
─────────────
`DomainConfig` becomes real: a config-driven registry (`DEFAULT_DOMAIN_REGISTRY`,
optionally overridden by `config/domains.yaml`) replaces `LOCAL_INTEL_DOMAIN`
as a hard mandate. Domain selection now supports three modes via
`RequestContext.domain_mode` ("override" default = today's exact behavior,
"hint"/"auto" = deterministic keyword classification against the registry).
See gateway.py's `chat()` for how `AI_ROUTER_DOMAIN_MODE` threads through.

`subdomain` (e.g. health's `cardiology` specialty) stays unpopulated — only
one specialty exists today and it's already LocalIntelligence's own default.
`risk` still uses the flat keyword list below, not `DomainConfig.risk_constraints`
— domain-aware risk scoring is a Phase 4/8 concern.

Phase 3 scope
─────────────
`context.force_frontier` now actually forces `route="frontier"` regardless
of `predicted_local_success`/threshold. `quality_threshold` computation is
STILL the Phase 1 `model_hint`-keyed table, unreconciled with
`DomainConfig.default_quality_threshold` — with current defaults and real
Phase 2 benchmark data, this means `route` will read "frontier" for most
real domain traffic today (e.g. retail's 0.60 benchmark never clears the
0.82 "structured" threshold). That's a calibration question for whoever
configures thresholds per deployment, not something this module decides.
Whether `route` actually gates dispatch (vs. just being logged) is
gateway.py's `chat()`'s decision, via `AI_ROUTER_ROUTE_GATING`.

Phase 4 scope
─────────────
`VerificationResult` becomes real via `ResponseVerifier` +
`DeterministicVerifier`: deterministic, domain-agnostic checks only (empty
response, refusal phrases, error artifacts). Explicit, deliberate
limitation: this does NOT catch a well-formed, non-empty, non-refusal
answer that is simply semantically wrong (e.g. a plausible but incorrect
generated Cypher query) — that needs domain-specific execution verification
or a judge model, neither of which exists yet. `confidence` is deliberately
capped below 1.0 (`_DETERMINISTIC_CONFIDENCE`) to avoid overstating what a
red-flag-absence check actually proves. `groundedness`/`completeness` stay
`None` — undetermined, not silently defaulted. Whether verification result
actually gates whether a local response is returned (vs. escalating to
cloud) is gateway.py's `chat()`'s decision, via `AI_ROUTER_VERIFY_LOCAL`.

Phase 5 scope
─────────────
`ModelCapability` becomes real via a config-driven registry
(`DEFAULT_MODEL_REGISTRY`, optionally overridden by `config/models.yaml` —
same contract as the Phase 2 domain registry) + `FrontierRouter` +
`CapabilityFrontierRouter`. Capability scores are DELIBERATE UNIFORM TIER
PLACEHOLDERS (0.60 fast-tier / 0.85 smart-tier, identical across providers)
— not claiming any provider is objectively better at anything, no verified
basis for that here. Direct consequence: within a tier, every candidate's
capability score ties, so the utility formula currently differentiates
purely on cost/latency (themselves rough illustrative estimates, not
live-checked pricing) — i.e. "capability" mode today behaves as "prefer the
cheapest/fastest provider within the requested hint's tier." Real
per-model differentiation is a future data update, not solved here.
Selection is deliberately scoped to the requested `model_hint`'s tier
(gateway.py passes in only same-tier candidates) so enabling this can't
silently escalate a "fast" call to a smart-tier model's cost. Whether this
replaces weighted selection as primary is gateway.py's `chat()`'s decision,
via `AI_ROUTER_FRONTIER_POLICY` — `"weighted"` (default, unchanged) vs
`"capability"`.

Phase 6 scope
─────────────
`ExperienceTrace` becomes real via `ExperienceStore` + `JSONLExperienceStore`
+ `build_experience_trace()`. Changes ZERO routing behavior — pure
observability, recording what Phases 1-5 already decide, not altering it.
Content is redacted by default (`redact_mode="full"`, see Phase 7 below for
the other modes) — request/response text becomes a length+hash placeholder,
never the original text, unless a deployment explicitly opts into a less
redacted mode. Deliberately scoped to the Tier 1/Tier 2 local-vs-frontier
loop only: Tier 0 (remote-forward) and explicit `provider=` pins are NOT
traced (operationally distinct from the loop this system is built to learn
from), and neither is `complete_with_meta()`'s `skip_local=True` path
(bypasses `chat()` entirely, consistent with every prior phase's scoping of
that path). Whether tracing happens at all, and which redaction mode
applies, are gateway.py's `chat()`'s decisions, via
`AI_ROUTER_EXPERIENCE_STORE` and `AI_ROUTER_EXPERIENCE_REDACT`.

Phase 7 scope
─────────────
Learning-candidate generation is explicitly OFFLINE — `training/generate_
candidates.py`, a standalone script that reads a trace file and is never
called from `chat()` or any request path. `AI_ROUTER_EXPERIENCE_REDACT`
becomes a 3-value mode (`"full"`/`"partial"`/`"none"`, backward compatible
with the old `"true"`/`"false"` values) — `"partial"` (new) scrubs
structured PII patterns (`_redact_pii_patterns()`) via regex while keeping
the rest of the text usable for training, as a middle ground between
`"full"`'s hash-only placeholder (useless for distillation candidates) and
`"none"`'s raw content. Regex-only, by design (zero-dependency) — this
CANNOT reliably catch free-text PII like names or addresses; "partial"
reduces exposure, it does not guarantee anonymization. `"full"` stays the
default.

Phase 8 scope
─────────────
Observability + a real feedback loop. `ExperienceTrace.trace_id` is now
threaded back to the caller (gateway.py's `chat()` sets it on the returned
`RouterResponse`, only when the Experience Store is on) so a deployment can
later correlate a specific past response with real user feedback —
`FeedbackRecord` + `FeedbackStore` + `JSONLFeedbackStore` +
`build_feedback_record()` are new, append-only, joined against trace
records at read time (not an in-place update — same rationale as Phase 6's
append-only trace store: no locking, no concurrent-writer hazard). A
feedback `note` gets the same 3-value redact mode already governing trace
content — one privacy policy, not two. This is what makes a REAL
false-accept rate computable (via `training/compute_metrics.py`, offline)
instead of only a proxy — Phase 4's verifier confidence stays capped at 0.5
regardless; this doesn't change what the verifier itself can detect, only
what can be measured about it after the fact, and only for the subset of
traces that actually receive feedback (see `feedback_coverage`). No live
deployment calls `record_feedback()` yet — this phase builds the capability,
not a wired-up UI. Live in-process metrics (`get_metrics_snapshot()`) are a
separate, always-available (when `AI_ROUTER_INTELLIGENT_ROUTING` is on)
mechanism that needs neither the Experience Store nor feedback — see
gateway.py's `_MetricsCollector` and ARCHITECTURE.md's Phase 8 section.

Phase 9 scope
─────────────
Versioning + shadow/canary groundwork. Two confirmed bugs fixed:
`ExperienceTrace.adapter_version`/`model_versions["local"]` used to record
`decision.selected_adapter` (the adapter NAME, e.g. `"retail_v3"`) instead
of the real version string — `RoutingDecision` now has its own
`adapter_version` field (from `DomainConfig.adapter_version`), and
`build_experience_trace()` uses that instead. `DeterministicRouterPolicy`/
`DeterministicVerifier` gained `policy_id`/`verifier_id` constructor params
(default to `ROUTER_VERSION`/`VERIFIER_VERSION` — unset, byte-identical to
before) so two differently-CONFIGURED instances of the same code (e.g. a
canary with a lower threshold table) can be told apart in every trace and
metric downstream — set via `AI_ROUTER_POLICY_ID`/`AI_ROUTER_VERIFIER_ID`
in gateway.py, no subclassing needed. `ExperienceTrace.frontier_router_version`
is new — `FRONTIER_ROUTER_VERSION`/`WEIGHTED_FRONTIER_VERSION` naming
whichever selection path actually decided a given request's primary pick
(`None` when the request never reached frontier selection at all).
**This is groundwork, not a canary system** — no traffic-splitting code
lives here or in gateway.py; a deployment canaries by running two
separately-configured `AIGateway` processes and comparing their
independently-collected trace files via `training/compute_metrics.py`,
which now breaks down metrics by `router_version`/`verifier_version`
cohort (and fixes its own staleness bug: `benchmark_deviation` used to
re-derive its baseline from today's live domain registry at analysis time
instead of each trace's own frozen `predicted_local_success` — wrong the
moment a domain gets rebenchmarked after some traces were already
recorded).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("ai_gateway")   # same logger name as gateway.py, so log lines interleave sensibly

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _yaml = None   # type: ignore[assignment]
    _YAML_OK = False

# ── Constants ────────────────────────────────────────────────────────────────

ROUTER_VERSION = "deterministic-v1"
REQUIRES_TOOLS = "tools"   # named constant — Phase 4's ModelCapability.tool_use
                            # will need to cross-reference this exact string

DEFAULT_THRESHOLDS: dict[str, float] = {
    "fast":       0.70,
    "structured": 0.82,
    "smart":      0.95,
}

# Crude keyword-based risk heuristic — a placeholder pending a real classifier.
_RISK_KEYWORDS = ("diagnose", "prescribe", "trade", "buy", "sell")

VERIFIER_VERSION = "deterministic-v1"

# Phase 9: names which frontier selection path actually decided a given
# request's primary provider/model pick — "capability-v1" when
# CapabilityFrontierRouter.select() actually returned a pick, "weighted-v1"
# for the always-present legacy path (_select_cloud_provider()). Recorded
# per-trace in ExperienceTrace.frontier_router_version — see gateway.py's
# chat() for how it's derived from the real per-request outcome, not just
# which mode is configured.
FRONTIER_ROUTER_VERSION = "capability-v1"
WEIGHTED_FRONTIER_VERSION = "weighted-v1"

# Deliberately capped below 1.0 — a passing deterministic check means "no red
# flags found," not "this is correct." See module docstring's Phase 4 section.
_DETERMINISTIC_CONFIDENCE = 0.5

# Conservative on purpose — substring matches chosen to be unambiguous
# refusal/error signatures, not generic short-answer patterns (a correct
# answer like "0 orders" must never trip these).
_REFUSAL_PHRASES = (
    "i cannot", "i can't", "i don't have access", "i'm not able to",
    "i am not able to", "i'm sorry, but", "as an ai",
)
_ERROR_ARTIFACTS = (
    "traceback (most recent call last)", "nonetype", "keyerror",
    "attributeerror", "unable to submit request",
)


# ── Phase 1 data models ────────────────────────────────────────────────────

@dataclass
class RequestContext:
    """Input to RouterPolicy.decide(). `messages` holds ChatMessage-like
    objects (each with .role/.content) — typed loosely as list[Any] rather
    than importing gateway.py's ChatMessage, to avoid a circular dependency
    between these two sibling files (gateway.py dynamically loads this file,
    not the other way around)."""
    messages:        list[Any]
    system:          str
    model_hint:      str                     # "fast" | "structured" | "smart"
    app_domain_hint: Optional[str] = None     # from LOCAL_INTEL_DOMAIN — a hint, not a mandate
    tools:           Optional[list] = None
    max_tokens:      int = 800
    force_frontier:  bool = False             # reserved for Phase 3; unused in Phase 1
    domain_mode:     str = "override"         # "override" | "hint" | "auto" — see AI_ROUTER_DOMAIN_MODE.
                                               # Default preserves Phase 1's exact behavior.


@dataclass
class RoutingDecision:
    domain: str
    subdomain: Optional[str]              # None in Phase 1 — no subdomain classifier yet
    task: str                             # "unclassified" in Phase 1 — no task classifier yet
    complexity: float                     # 0..1
    risk: float                           # 0..1
    predicted_local_success: float        # 0..1 — flat placeholder in Phase 1
    selected_adapter: Optional[str]
    required_capabilities: list[str]
    quality_threshold: float
    route: str                            # "local" | "frontier"
    reason: str                           # numeric/specific, not generic
    router_version: str
    # Phase 9: the resolved domain's real adapter VERSION (DomainConfig.
    # adapter_version, e.g. "v3"/"2026-08-22") — NOT the adapter NAME
    # (selected_adapter already covers that). None when the domain has no
    # adapter (e.g. "general"). Defaulted so existing fixtures/callers that
    # don't pass it keep working.
    adapter_version: Optional[str] = None


# ── Phase 2/4/5/6 data models — defined now, inert until their phase lands ──

@dataclass
class VerificationResult:
    accepted: bool
    score: float
    confidence: float
    failure_reasons: list[str]
    groundedness: Optional[float] = None
    completeness: Optional[float] = None
    policy_risk: Optional[str] = None
    verifier_version: str = "unset"


@dataclass
class ExperienceTrace:
    trace_id: str
    timestamp: str
    client_app: str
    request: dict
    context_metadata: dict
    routing_decision: RoutingDecision
    adapter_version: Optional[str]
    model_versions: dict
    local_response: Optional[str]
    local_latency_ms: Optional[float]
    local_tokens: Optional[int]
    verification: Optional[VerificationResult]
    escalated: bool
    escalation_reason: Optional[str]
    frontier_provider: Optional[str]
    frontier_model: Optional[str]
    frontier_response: Optional[str]
    frontier_latency_ms: Optional[float]
    frontier_tokens: Optional[int]
    frontier_cost: Optional[float]
    tool_calls: list
    errors: list[str]
    user_feedback: Optional[dict]
    final_source: str   # "local" | "frontier"
    # Phase 9: which frontier selection path actually decided this request's
    # primary pick — FRONTIER_ROUTER_VERSION / WEIGHTED_FRONTIER_VERSION, or
    # None when the request never reached frontier selection at all (local
    # succeeded). Defaulted — genuinely optional, not silently omitted.
    frontier_router_version: Optional[str] = None


@dataclass
class DomainConfig:
    name: str
    adapter_name: Optional[str]
    adapter_version: Optional[str]
    supported_tasks: list[str]
    local_model: str
    tools: list[str] = field(default_factory=list)
    retrieval_config: Optional[dict] = None
    default_quality_threshold: float = 0.75
    risk_constraints: Optional[dict] = None
    # Phase 2 additions — both default to a no-op value so any Phase 1 code
    # constructing a DomainConfig without them keeps working unchanged.
    benchmark_success_rate: Optional[float] = None   # real offline benchmark, if measured
    keywords: list[str] = field(default_factory=list)  # for deterministic domain classification


@dataclass
class ModelCapability:
    provider: str
    model: str
    reasoning: float
    coding: float
    tool_use: float
    structured_output: float
    long_context: float
    modalities: list[str] = field(default_factory=lambda: ["text"])
    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    available: bool = True


# ── Domain registry (Phase 2) ───────────────────────────────────────────────
# Real, measured data — not placeholders. Benchmark numbers are from this
# Mac's own local-adapter benchmarks (see ARCHITECTURE.md's Phase 2 section
# for dates/methodology). This dict IS the authoritative default, not a
# degraded fallback — config/domains.yaml (optional, see load_domain_registry
# below) only layers overrides on top of it. `manufacturing` has zero
# infrastructure anywhere and isn't included; `research` in LocalIntelligence
# is a model *pool*, not a routable domain, and also isn't included here.

DEFAULT_DOMAIN_REGISTRY: dict[str, DomainConfig] = {
    "retail": DomainConfig(
        name="retail", adapter_name="retail_v3", adapter_version="v3",
        supported_tasks=["cypher_query", "inventory_lookup", "supplier_analysis", "catalog_search"],
        local_model="qwen3-8b-4bit", tools=[],
        retrieval_config={"store": "kuzudb", "path": "CatalogValidator/backend/data/kuzu_db"},
        default_quality_threshold=0.75, risk_constraints=None,
        benchmark_success_rate=0.60,   # execution accuracy vs held-out use-cases, 2026-08-21
        keywords=["inventory", "sku", "supplier", "catalog", "stock", "product", "order", "oem", "moq"],
    ),
    "finance": DomainConfig(
        name="finance", adapter_name="finance", adapter_version="2026-08-22",
        supported_tasks=["intent_classification", "portfolio_query", "position_lookup"],
        local_model="qwen3-8b-4bit", tools=[], retrieval_config=None,
        default_quality_threshold=0.80, risk_constraints={"requires_disclaimer": True},
        benchmark_success_rate=0.87,   # exact full-label match, n=60, 2026-09-15
        keywords=["portfolio", "stock", "ticker", "dividend", "ira", "holding", "position", "earnings", "allocation", "budget"],
    ),
    "health": DomainConfig(
        name="health", adapter_name="cardiology", adapter_version="2026-09-15",
        supported_tasks=["command_routing"],
        local_model="qwen3-8b-4bit", tools=[],
        retrieval_config={"store": "qdrant_cloud", "url_env": "QDRANT_URL"},
        default_quality_threshold=0.85, risk_constraints={"requires_disclaimer": True, "high_risk": True},
        benchmark_success_rate=0.55,   # exact full-label match, n=33 held-out, 2026-09-15
        keywords=["patient", "cardiology", "ecg", "chads", "diagnosis", "prescribe", "appointment", "soap", "calendar"],
    ),
    "general": DomainConfig(
        name="general", adapter_name=None, adapter_version=None,
        supported_tasks=["chat"], local_model="qwen3-8b-4bit", tools=[],
        retrieval_config=None, default_quality_threshold=0.70, risk_constraints=None,
        benchmark_success_rate=None,   # no dedicated adapter — flat 0.5 placeholder still applies
        keywords=[],
    ),
}


def load_domain_registry(path: Optional[Path] = None) -> dict[str, DomainConfig]:
    """Starts from DEFAULT_DOMAIN_REGISTRY (the complete, real dataset) and,
    only if PyYAML is available and a config/domains.yaml file exists next to
    this module, layers overrides/extensions on top. A missing or malformed
    YAML file must never break routing — falls back to the hardcoded registry
    on any failure."""
    registry = dict(DEFAULT_DOMAIN_REGISTRY)
    yaml_path = path or (Path(__file__).parent / "config" / "domains.yaml")
    if not _YAML_OK or not yaml_path.exists():
        return registry
    try:
        with open(yaml_path) as f:
            data = _yaml.safe_load(f) or {}
        for name, entry in (data.get("domains") or {}).items():
            registry[name] = DomainConfig(name=name, **entry)
        return registry
    except Exception:
        return registry


def _classify_domain(text: str, registry: dict[str, DomainConfig]) -> Optional[str]:
    """Simple keyword-count matcher — no ML, no dependency. Returns the
    domain with the most keyword hits, or None if nothing matched at all."""
    text = text.lower()
    scores = {name: sum(1 for kw in cfg.keywords if kw in text) for name, cfg in registry.items()}
    best = max(scores, key=scores.get, default=None)
    return best if best and scores[best] > 0 else None


# ── RouterPolicy interface ─────────────────────────────────────────────────

@runtime_checkable
class RouterPolicy(Protocol):
    """Structural interface, not an ABC — deliberately, since the deployment
    model is flat-file-copy with no shared package. A future learned-model
    policy (or a policy defined in a differently-drifted copy of this file)
    satisfies this without importing the exact class that defines it.

    Note: @runtime_checkable only checks that a `decide` attribute exists,
    not its signature or behavior — isinstance(x, RouterPolicy) is not proof
    `.decide()` is call-safe. Callers must still guard the actual call."""

    def decide(self, context: RequestContext) -> RoutingDecision: ...


class DeterministicRouterPolicy:
    """Phase 1-2's only concrete RouterPolicy: deterministic rules, no LLM
    classifier. See module docstring for what's still a placeholder vs real."""

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
        registry: Optional[dict[str, DomainConfig]] = None,
        policy_id: Optional[str] = None,
    ) -> None:
        # Constructor param (not a bare literal) keeps a data-driven seam
        # open even though Phase 1 wires the hardcoded default — Phase 2's
        # DomainConfig.default_quality_threshold is the real data-driven
        # source later.
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.registry = registry if registry is not None else load_domain_registry()
        # Phase 9: lets two differently-CONFIGURED instances of this same
        # CODE (e.g. a canary with a lower threshold table) be told apart in
        # every trace/metric downstream. Unset (default) -> identical to
        # pre-Phase-9 behavior.
        self.policy_id = policy_id or ROUTER_VERSION

    def decide(self, context: RequestContext) -> RoutingDecision:
        total_chars = sum(len(getattr(m, "content", "") or "") for m in context.messages)
        complexity = min(1.0, total_chars / 2000)

        combined_text = " ".join(
            (getattr(m, "content", "") or "") for m in context.messages
        ).lower()
        risk = 0.5 if any(kw in combined_text for kw in _RISK_KEYWORDS) else 0.0

        # ── Domain resolution (Phase 2) ─────────────────────────────────────
        if context.domain_mode == "override":
            domain = context.app_domain_hint or "general"
        else:
            classified = _classify_domain(combined_text, self.registry)
            if context.domain_mode == "hint":
                domain = classified or context.app_domain_hint or "general"
            else:   # "auto"
                domain = classified or "general"

        domain_config = self.registry.get(domain)
        selected_adapter = domain_config.adapter_name if domain_config else domain
        if domain_config and domain_config.benchmark_success_rate is not None:
            predicted_local_success = domain_config.benchmark_success_rate
            success_source = "benchmark"
        else:
            predicted_local_success = 0.5   # flat placeholder — see module docstring
            success_source = "placeholder"

        threshold = self.thresholds.get(context.model_hint, self.thresholds["structured"])

        required_capabilities = [REQUIRES_TOOLS] if context.tools else []

        # ── Route decision (Phase 3) ────────────────────────────────────────
        if context.force_frontier:
            route = "frontier"
            reason = f"force_frontier=True (explicit override) for model_hint={context.model_hint}"
        else:
            route = "local" if predicted_local_success >= threshold else "frontier"
            reason = (
                f"domain={domain} predicted_local_success={predicted_local_success:.2f} ({success_source}) "
                f"{'>=' if route == 'local' else '<'} threshold={threshold:.2f} "
                f"for model_hint={context.model_hint}"
            )

        return RoutingDecision(
            domain=domain,
            subdomain=None,
            task="unclassified",
            complexity=complexity,
            risk=risk,
            predicted_local_success=predicted_local_success,
            selected_adapter=selected_adapter,
            required_capabilities=required_capabilities,
            quality_threshold=threshold,
            route=route,
            reason=reason,
            router_version=self.policy_id,
            adapter_version=domain_config.adapter_version if domain_config else None,
        )


# ── ResponseVerifier interface (Phase 4) ────────────────────────────────────

@runtime_checkable
class ResponseVerifier(Protocol):
    """Structural interface, same rationale as RouterPolicy above (Protocol,
    not ABC — flat-file-copy deployment, no shared package to inherit from).

    Note: @runtime_checkable only checks that an `evaluate` attribute
    exists, not its signature or behavior. Callers must still guard the
    actual call — gateway.py's chat() does, with a fail-open except."""

    def evaluate(self, context: RequestContext, decision: RoutingDecision, local_text: str) -> VerificationResult: ...


class DeterministicVerifier:
    """Phase 4's only concrete ResponseVerifier: domain-agnostic deterministic
    checks, no LLM judge, no execution verification. See module docstring's
    Phase 4 section for what this does NOT catch (well-formed but
    semantically wrong answers)."""

    def __init__(self, min_score: float = 0.5, verifier_id: Optional[str] = None) -> None:
        # Constructor param, not a bare literal — same data-driven-seam
        # convention as DeterministicRouterPolicy.thresholds. score is
        # binary (0.0/1.0) in this MVP, so 0.5 just means "any triggered
        # failure reason rejects" — kept as a param so a future scored
        # verifier can reuse this class's acceptance logic.
        self.min_score = min_score
        # Phase 9: same canary-tagging rationale as DeterministicRouterPolicy.
        # policy_id — lets two differently-configured verifier instances be
        # told apart downstream. Unset (default) -> identical to pre-Phase-9.
        self.verifier_id = verifier_id or VERIFIER_VERSION

    def evaluate(self, context: RequestContext, decision: RoutingDecision, local_text: str) -> VerificationResult:
        reasons: list[str] = []
        text = (local_text or "").strip()

        if not text:
            reasons.append("empty response")

        lower = text.lower()
        if any(p in lower for p in _REFUSAL_PHRASES):
            reasons.append("refusal phrase detected")
        if any(p in lower for p in _ERROR_ARTIFACTS):
            reasons.append("error/exception artifact detected")

        score = 0.0 if reasons else 1.0
        accepted = score >= self.min_score

        return VerificationResult(
            accepted=accepted,
            score=score,
            confidence=_DETERMINISTIC_CONFIDENCE,
            failure_reasons=reasons,
            groundedness=None,      # undetermined — deterministic checks can't assess this
            completeness=None,      # undetermined — deterministic checks can't assess this
            policy_risk=None,
            verifier_version=self.verifier_id,
        )


# ── Model registry (Phase 5) ────────────────────────────────────────────────
# Uniform tier placeholders, not differentiated-by-reputation estimates — see
# module docstring's Phase 5 section for why, and for the direct consequence
# (capability score ties within a tier; cost/latency do the differentiating
# today). Cost/latency are rough illustrative estimates, not live-checked
# pricing. This dict IS the authoritative default, same contract as
# DEFAULT_DOMAIN_REGISTRY — config/models.yaml only layers overrides on top.

_FAST_TIER = dict(reasoning=0.60, coding=0.60, tool_use=0.60, structured_output=0.60, long_context=0.60)
_SMART_TIER = dict(reasoning=0.85, coding=0.85, tool_use=0.85, structured_output=0.85, long_context=0.85)

DEFAULT_MODEL_REGISTRY: dict[tuple[str, str], ModelCapability] = {
    ("openrouter", "openai/gpt-4o-mini"): ModelCapability(
        provider="openrouter", model="openai/gpt-4o-mini", **_FAST_TIER,
        cost_per_1k_tokens=0.0005, avg_latency_ms=750,
    ),
    ("openrouter", "anthropic/claude-sonnet-4.5"): ModelCapability(
        provider="openrouter", model="anthropic/claude-sonnet-4.5", **_SMART_TIER,
        cost_per_1k_tokens=0.007, avg_latency_ms=1800,
    ),
    ("anthropic", "claude-haiku-4-5"): ModelCapability(
        provider="anthropic", model="claude-haiku-4-5", **_FAST_TIER,
        cost_per_1k_tokens=0.0005, avg_latency_ms=700,
    ),
    ("anthropic", "claude-sonnet-4-5"): ModelCapability(
        provider="anthropic", model="claude-sonnet-4-5", **_SMART_TIER,
        cost_per_1k_tokens=0.007, avg_latency_ms=1800,
    ),
    ("openai", "gpt-4o-mini"): ModelCapability(
        provider="openai", model="gpt-4o-mini", **_FAST_TIER,
        cost_per_1k_tokens=0.0005, avg_latency_ms=800,
    ),
    ("openai", "gpt-4o"): ModelCapability(
        provider="openai", model="gpt-4o", **_SMART_TIER,
        cost_per_1k_tokens=0.006, avg_latency_ms=1500,
    ),
    ("gemini", "gemini-2.5-flash"): ModelCapability(
        provider="gemini", model="gemini-2.5-flash", **_FAST_TIER,
        cost_per_1k_tokens=0.0003, avg_latency_ms=600,
    ),
    ("gemini", "gemini-2.5-pro"): ModelCapability(
        provider="gemini", model="gemini-2.5-pro", **_SMART_TIER,
        cost_per_1k_tokens=0.005, avg_latency_ms=2000,
    ),
}


def load_model_registry(path: Optional[Path] = None) -> dict[tuple[str, str], ModelCapability]:
    """Same contract as load_domain_registry: DEFAULT_MODEL_REGISTRY is the
    complete, authoritative default; config/models.yaml (a list of entries,
    since keys are (provider, model) pairs, not names) is an optional
    override layer. Never raises — a bad file falls back to the hardcoded
    registry."""
    registry = dict(DEFAULT_MODEL_REGISTRY)
    yaml_path = path or (Path(__file__).parent / "config" / "models.yaml")
    if not _YAML_OK or not yaml_path.exists():
        return registry
    try:
        with open(yaml_path) as f:
            data = _yaml.safe_load(f) or {}
        for entry in (data.get("models") or []):
            registry[(entry["provider"], entry["model"])] = ModelCapability(**entry)
        return registry
    except Exception:
        return registry


# ── FrontierRouter interface (Phase 5) ──────────────────────────────────────

@runtime_checkable
class FrontierRouter(Protocol):
    """Structural interface, same rationale as RouterPolicy/ResponseVerifier
    above. candidates are (provider, model) pairs the CALLER already
    considers acceptable (gateway.py filters to the requested model_hint's
    tier before calling this) — keeps this module decoupled from gateway.py's
    self.models config, and keeps gateway.py free of model-name literals in
    its routing logic."""

    def select(
        self, context: RequestContext, decision: RoutingDecision,
        candidates: list[tuple[str, str]],
    ) -> Optional[tuple[str, str]]: ...


class CapabilityFrontierRouter:
    """Phase 5's only concrete FrontierRouter: expected-utility scoring over
    a registry-informed candidate set. See module docstring for why, given
    today's uniform tier placeholders, this currently behaves as "prefer
    cheapest/fastest within the requested tier" rather than true
    capability-based differentiation."""

    def __init__(
        self,
        registry: Optional[dict[tuple[str, str], ModelCapability]] = None,
        cost_weight: float = 0.1,
        latency_weight: float = 0.05,
    ) -> None:
        self.registry = registry if registry is not None else load_model_registry()
        # Constructor params, not bare literals — same data-driven-seam
        # convention as DeterministicRouterPolicy.thresholds.
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight

    def select(
        self, context: RequestContext, decision: RoutingDecision,
        candidates: list[tuple[str, str]],
    ) -> Optional[tuple[str, str]]:
        scored = [(pair, self.registry.get(pair)) for pair in candidates]
        scored = [(pair, cap) for pair, cap in scored if cap is not None and cap.available]
        if not scored:
            return None   # gateway.py falls back to weighted selection

        def utility(cap: ModelCapability) -> float:
            capability_score = (
                cap.tool_use if REQUIRES_TOOLS in decision.required_capabilities
                else (cap.reasoning + cap.coding + cap.structured_output) / 3
            )
            cost_penalty = self.cost_weight * cap.cost_per_1k_tokens
            latency_penalty = self.latency_weight * (cap.avg_latency_ms / 1000.0)
            return capability_score - cost_penalty - latency_penalty

        best_pair, _ = max(scored, key=lambda item: utility(item[1]))
        return best_pair


# ── Experience Store (Phase 6, redaction upgraded in Phase 7) ───────────────

def _redact_text(text: Optional[str]) -> Optional[str]:
    """"full" mode. Deterministic (repeat/duplicate detection stays possible)
    but never recoverable to the original content."""
    if text is None:
        return None
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"<redacted len={len(text)} sha256={digest}>"


# Ordered most-specific first — a broad catch-all last, so formatted patterns
# (SSN, phone) get their own clear label before an unformatted long digit run
# falls through to the generic account-number bucket. Regex-only, by design
# (zero-dependency) — this CANNOT reliably catch free-text PII like names or
# addresses. "partial" mode reduces exposure; it does not guarantee
# anonymization. See module docstring's Phase 7 section.
_PII_PATTERNS = [
    (re.compile(r'\$\s?[\d,]+(?:\.\d{2})?'),           '<REDACTED_AMOUNT>'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),              '<REDACTED_SSN>'),
    (re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'),    '<REDACTED_PHONE>'),
    (re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),           '<REDACTED_EMAIL>'),
    (re.compile(r'\b\d{9,17}\b'),                       '<REDACTED_ACCOUNT_NUMBER>'),
]


def _redact_pii_patterns(text: Optional[str]) -> Optional[str]:
    """"partial" mode. Scrubs structured PII patterns, leaves the rest of the
    text intact and usable for training. Over-redaction (a legitimate large
    number caught by the account-number pattern) is an accepted tradeoff —
    the safer failure mode for PII scrubbing than under-redaction."""
    if text is None:
        return None
    for pattern, placeholder in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def build_experience_trace(
    *,
    context: RequestContext,
    decision: RoutingDecision,
    trace_id: Optional[str] = None,
    client_app: str = "unknown",
    local_response: Optional[str] = None,
    local_latency_ms: Optional[float] = None,
    local_tokens: Optional[int] = None,
    verification: Optional[VerificationResult] = None,
    escalated: bool = False,
    escalation_reason: Optional[str] = None,
    frontier_provider: Optional[str] = None,
    frontier_model: Optional[str] = None,
    frontier_response: Optional[str] = None,
    frontier_latency_ms: Optional[float] = None,
    frontier_tokens: Optional[int] = None,
    frontier_cost: Optional[float] = None,
    tool_calls: Optional[list] = None,
    errors: Optional[list[str]] = None,
    final_source: str = "local",
    redact_mode: str = "full",   # "full" | "partial" | "none"
    frontier_router_version: Optional[str] = None,
) -> ExperienceTrace:
    """Constructs an ExperienceTrace from the raw pieces gateway.py collects
    during Tier 1/Tier 2 dispatch, applying redaction internally per
    redact_mode — keeps ExperienceStore a dumb serializer; a future second
    store implementation doesn't need to reimplement redaction. See module
    docstring's Phase 6/7 sections for scope."""
    tool_calls = tool_calls or []
    errors = errors or []

    combined_text = " ".join((getattr(m, "content", "") or "") for m in context.messages)

    if redact_mode == "full":
        request: dict = {
            "message_count": len(context.messages),
            "total_chars": len(combined_text),
            "content_hash": hashlib.sha256(combined_text.encode("utf-8", errors="replace")).hexdigest()[:12],
        }
        local_response_out = _redact_text(local_response)
        frontier_response_out = _redact_text(frontier_response)
        # Tool names are kept (not sensitive, matter for Phase 7); arguments are dropped.
        tool_calls_out = [{"name": tc.get("name")} if isinstance(tc, dict) else tc for tc in tool_calls]
    elif redact_mode == "partial":
        request = {
            "messages": [
                {"role": getattr(m, "role", "user"), "content": _redact_pii_patterns(getattr(m, "content", "") or "")}
                for m in context.messages
            ],
            "system": context.system,
        }
        local_response_out = _redact_pii_patterns(local_response)
        frontier_response_out = _redact_pii_patterns(frontier_response)
        # Kept (scrubbed), not dropped — scrubbed args still carry real
        # training signal that "full" mode's drop-entirely approach loses.
        tool_calls_out = [
            {**tc, "arguments": _redact_pii_patterns(json.dumps(tc.get("arguments")))} if isinstance(tc, dict) and "arguments" in tc else tc
            for tc in tool_calls
        ]
    else:   # "none"
        request = {
            "messages": [{"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")} for m in context.messages],
            "system": context.system,
        }
        local_response_out = local_response
        frontier_response_out = frontier_response
        tool_calls_out = tool_calls

    return ExperienceTrace(
        trace_id=trace_id or uuid.uuid4().hex,
        timestamp=datetime.now(timezone.utc).isoformat(),
        client_app=client_app,
        request=request,
        context_metadata={
            "model_hint": context.model_hint,
            "domain_mode": context.domain_mode,
            "force_frontier": context.force_frontier,
        },
        routing_decision=decision,
        # Phase 9 bug fix: these used to record decision.selected_adapter
        # (the adapter NAME, e.g. "retail_v3") here — wrong for a field
        # called *_version. decision.adapter_version is the real version
        # string (DomainConfig.adapter_version, e.g. "v3"/"2026-08-22").
        adapter_version=decision.adapter_version,
        model_versions={"local": decision.adapter_version, "frontier": frontier_model},
        local_response=local_response_out,
        local_latency_ms=local_latency_ms,
        local_tokens=local_tokens,
        verification=verification,
        escalated=escalated,
        escalation_reason=escalation_reason,
        frontier_provider=frontier_provider,
        frontier_model=frontier_model,
        frontier_response=frontier_response_out,
        frontier_latency_ms=frontier_latency_ms,
        frontier_tokens=frontier_tokens,
        frontier_cost=frontier_cost,
        tool_calls=tool_calls_out,
        # Capped, not redacted — an error is about the system's behavior, not
        # the user's data; redacting it would hurt debuggability far more
        # than it protects privacy.
        errors=[e[:200] for e in errors],
        # Always None on the trace itself — Phase 8's feedback mechanism
        # (FeedbackRecord/JSONLFeedbackStore) is a separate, append-only
        # store joined against traces by trace_id at read time, not written
        # back into the trace. See router_policy.py's Phase 8 section.
        user_feedback=None,
        final_source=final_source,
        frontier_router_version=frontier_router_version,
    )


@runtime_checkable
class ExperienceStore(Protocol):
    """Structural interface, same rationale as every prior phase's Protocol
    — flat-file-copy deployment, no shared package to inherit from."""

    def record(self, trace: ExperienceTrace) -> None: ...


class JSONLExperienceStore:
    """Phase 6's only concrete ExperienceStore: append-only JSONL, one line
    per trace. Never raises — a store-write failure must never block or
    alter a real response, same guarantee as the router policy and verifier."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (Path(__file__).parent / "data" / "experience_traces.jsonl")

    def record(self, trace: ExperienceTrace) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(trace), default=str) + "\n")
        except Exception as e:
            logger.warning("experience store write failed: %s", e)


# ── Feedback loop (Phase 8) ─────────────────────────────────────────────────
# Append-only, joined against ExperienceTrace records at read time (offline,
# in training/compute_metrics.py) — not an in-place update to the trace
# file. Same rationale as Phase 6's append-only trace store: no locking, no
# concurrent-writer hazard across long-running processes. Multiple feedback
# records on the same trace_id are all preserved (a user changing their mind
# isn't data loss); the reader resolves conflicts by taking the *last*
# record per trace_id (most recent write wins).

_VALID_FEEDBACK_LABELS = {"correct", "incorrect", "unrated"}   # named constant, not a bare literal


@dataclass
class FeedbackRecord:
    feedback_id: str
    trace_id: str            # correlates back to a specific ExperienceTrace.trace_id
    timestamp: str
    label: str                # one of _VALID_FEEDBACK_LABELS
    note: Optional[str] = None
    client_app: str = "unknown"


def build_feedback_record(
    *,
    trace_id: str,
    label: str,
    note: Optional[str] = None,
    client_app: str = "unknown",
    redact_mode: str = "full",   # "full" | "partial" | "none" — same 3 modes as trace content
) -> Optional[FeedbackRecord]:
    """Validates `label` against _VALID_FEEDBACK_LABELS (returns None on an
    invalid label — the caller logs a warning and writes nothing; this
    function never raises). Applies the same redact_mode already governing
    ExperienceTrace content to the free-text `note` field — a feedback note
    is exactly the kind of field a user could put PII into ("this was wrong,
    my number is..."), so it gets the same privacy treatment as everything
    else, not a second policy."""
    if label not in _VALID_FEEDBACK_LABELS:
        return None

    if redact_mode == "full":
        note_out = _redact_text(note)
    elif redact_mode == "partial":
        note_out = _redact_pii_patterns(note)
    else:   # "none"
        note_out = note

    return FeedbackRecord(
        feedback_id=uuid.uuid4().hex,
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        label=label,
        note=note_out,
        client_app=client_app,
    )


@runtime_checkable
class FeedbackStore(Protocol):
    """Structural interface, same rationale as every prior phase's Protocol
    — flat-file-copy deployment, no shared package to inherit from."""

    def record(self, feedback: FeedbackRecord) -> None: ...


class JSONLFeedbackStore:
    """Phase 8's only concrete FeedbackStore: append-only JSONL, one line per
    feedback event, sibling to JSONLExperienceStore. Never raises — a
    store-write failure must never block or alter a real response, same
    guarantee as every other subsystem in this file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (Path(__file__).parent / "data" / "experience_feedback.jsonl")

    def record(self, feedback: FeedbackRecord) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(feedback), default=str) + "\n")
        except Exception as e:
            logger.warning("feedback store write failed: %s", e)
