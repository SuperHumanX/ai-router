"""
gateway.py — Unified LLM Gateway (Python)
══════════════════════════════════════════
Single file, zero project-specific imports.  Drop into any Python project.

Tier 0  Remote gateway        when REMOTE_GATEWAY_URL is set, ALL chat() calls
                                are forwarded to that HTTP server (e.g. over
                                Tailscale) instead of calling providers directly —
                                lets a remote host (e.g. GCP) use the router
                                without holding any API keys itself.

Tier 1  Local LLM              tried first when LOCAL_INTEL_URL / OLLAMA_URL is
                                set, USE_LOCAL_SLM != false, and model_hint != "smart"
                                (smart calls skip local and go straight to cloud —
                                local is for fast/structured work, not quality-critical
                                calls)
          • mlx_lm.server      auto-detected by port :11435 or "mlx" in URL
            → OpenAI-compat    /v1/chat/completions
                                LOCAL_INTEL_DOMAIN, if set, is sent as `domain` so
                                the gateway hot-swaps the right LoRA adapter
                                (retail/finance/health) without relying on
                                model-string parsing or /v1/models ordering.
          • Ollama              all other URLs
            → /api/chat

  Tier 2  Cloud providers      weighted or round-robin when local is unavailable.
                                Each provider prefers its own direct path and
                                falls back to OpenRouter BYOK (OPENROUTER_API_KEY)
                                when that path is missing or the call fails —
                                enabled if EITHER backend is usable.
            • OpenRouter        if OPENROUTER_API_KEY is set (primary default, 75%) —
                                 one key, routes to Anthropic/OpenAI/Google models
                                 upstream; a provider-diversity hedge against any
                                 single direct-key account running dry (e.g. Anthropic
                                 credit exhaustion doesn't take this down — OpenRouter
                                 has its own pool covering the same Claude models).
            • Gemini            Vertex AI (ADC) direct (secondary, 15%), else
                                 OpenRouter BYOK fallback if Vertex fails (billing
                                 not enabled, quota, ADC missing, transient error)
            • Anthropic         ANTHROPIC_API_KEY direct (tertiary, 5%), else
                                 OpenRouter BYOK fallback on missing key or failure
                                 (OpenRouter fallback doesn't forward `tools` — no
                                 Anthropic-native tool schema on that path, so a
                                 pin requiring tools needs the direct key)
            • OpenAI            OPENAI_API_KEY direct (tertiary, 5%), else
                                 OpenRouter BYOK fallback on missing key or failure

  Tier 3  None                 raises / returns None so callers can surface the error

Routing modes  (AI_ROUTER_MODE env var)
───────────────────────────────────────
  weighted       probabilistic selection by weight (default)
  round-robin    strict alternation; state in data/.ai_router_rr.json
  openrouter     always OpenRouter (if available)
  gemini         always Gemini    (if available)
  anthropic      always Anthropic (if available)
  openai         always OpenAI    (if available)
  local          local LLM only, no cloud fallback

Centralized key store  ~/Projects/ai-router/.env
──────────────────────────────────────────────────
All API keys live here — one place, not repeated per project.
Override location: AI_GATEWAY_CONFIG=/path/to/keys.env

  OPENROUTER_API_KEY  = sk-or-...    (primary cloud path — embeds Anthropic/OpenAI/Google,
                                       and is the shared BYOK fallback for the other three)
  GEMINI_PROJECT       = gen-lang-client-0271077908   (Vertex AI project — no key, uses ADC)
  ANTHROPIC_API_KEY   = sk-ant-...   (direct — same models OpenRouter also serves)
  OPENAI_API_KEY      = sk-proj-...  (direct)
  GROK_API_KEY        = xai-...      (future)

Per-project .env  (weights + local LLM only)
────────────────────────────────────────────
  LOCAL_INTEL_URL          http://localhost:11435   MLX via Tailscale (auto-detected)
  LOCAL_INTEL_DOMAIN                                LoRA domain to hot-swap (retail|finance|health)
  OLLAMA_URL               http://localhost:11434   Ollama fallback
  OLLAMA_MODEL             qwen2.5:7b
  USE_LOCAL_SLM            true                     false = skip local tier entirely
  REMOTE_GATEWAY_URL                                Tailscale URL of a machine running server.py;
                                                     when set, forwards every call there instead

  AI_ROUTER_MODE               weighted              weighted|round-robin|openrouter|gemini|anthropic|openai|local
  AI_ROUTER_WEIGHT_OPENROUTER  0.75
  AI_ROUTER_WEIGHT_GEMINI      0.15
  AI_ROUTER_WEIGHT_ANTHROPIC   0.05
  AI_ROUTER_WEIGHT_OPENAI      0.05

  AI_ROUTER_OPENROUTER_MODEL  openai/gpt-4o-mini          fast / structured
  AI_ROUTER_OPENROUTER_SMART  anthropic/claude-sonnet-4.5 smart
  AI_ROUTER_ANTHROPIC_MODEL   claude-haiku-4-5      fast / structured
  AI_ROUTER_ANTHROPIC_SMART   claude-sonnet-4-5     smart
  AI_ROUTER_OPENAI_MODEL      gpt-4o-mini           fast / structured
  AI_ROUTER_OPENAI_SMART      gpt-4o                smart
  AI_ROUTER_GEMINI_MODEL      gemini-2.5-flash      fast / structured
  AI_ROUTER_GEMINI_SMART      gemini-2.5-pro        smart

Local LLM auto-detection (no flags needed)
──────────────────────────────────────────
The gateway probes LOCAL_INTEL_URL at call time.
  • If reachable → served locally (free, private)
  • If unreachable → transparent cloud fallback
No --mlx or --local flags required anywhere.

Public API
──────────
  from gateway import router, ChatMessage, RouterResponse

  # Multi-turn (portfolio_tracker / physician style)
  result: RouterResponse = router.chat(
      messages=[ChatMessage(role="user", content="...")],
      system="You are a helpful assistant.",
      model_hint="structured",   # default: "structured"/"fast" → haiku | "smart" → sonnet
      max_tokens=800,
  )
  print(result.text, result.provider, result.model)

  # Single-turn convenience (CatalogValidator style)
  text: str | None = router.complete(
      system="You are a KuzuDB Cypher expert.",
      user="How many orders from supplier X?",
      task="cypher",       # "cypher"|"structured" → fast model
                           # "summarize"|"analysis"|"general" → smart model
  )

Domain aliases (backward-compat)
─────────────────────────────────
  catalog_router  = router   # CatalogValidator
  finance_router  = router   # portfolio_tracker
  health_router   = router   # physician
"""

from __future__ import annotations

import json
import logging
import os
import random
import statistics
import threading
import time
import sys
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import httpx as _httpx
    _HTTPX_OK = True
except ImportError:
    _httpx    = None   # type: ignore[assignment]
    _HTTPX_OK = False

try:
    from google import genai as _genai
    _GENAI_OK = True
except ImportError:
    _genai    = None   # type: ignore[assignment]
    _GENAI_OK = False

logger = logging.getLogger("ai_gateway")

# ── Master key store ──────────────────────────────────────────────────────────
# API keys live in ONE place: ~/Projects/ai-router/.env
# Projects only need routing weights and LOCAL_INTEL_URL in their own .env.
# Override location via AI_GATEWAY_CONFIG env var if needed.

_MASTER_ENV_PATH = Path.home() / "Projects" / "ai-router" / ".env"


def _load_master_env() -> None:
    """
    Load the centralized API-key store into the environment (non-overriding).
    Project .env values and OS-level env vars always take precedence.
    """
    custom = os.environ.get("AI_GATEWAY_CONFIG", "").strip()
    master = Path(custom) if custom else _MASTER_ENV_PATH
    if not master.exists():
        logger.debug("No master gateway config found at %s", master)
        return
    # Load master keys: override=True so empty shell-inherited vars don't block them.
    # Project .env routing weights are loaded by the project's own startup code and
    # are not affected here (they use different variable names).
    try:
        from dotenv import dotenv_values
        for k, v in dotenv_values(str(master)).items():
            if v:   # only set non-empty values from the master store
                os.environ[k] = v
        logger.debug("Loaded master gateway config from %s", master)
    except ImportError:
        # Manual parse if python-dotenv isn't installed
        for line in master.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                os.environ[k] = v
        logger.debug("Loaded master gateway config (manual parse) from %s", master)


# ── Router policy (Phase 1, optional) ───────────────────────────────────────
# router_policy.py is a sibling file, deployed the same way as this one (see
# ARCHITECTURE.md). It's loaded dynamically via Path(__file__).parent rather
# than a plain `import router_policy`, because this file is deployed as a
# flat copy with no reliable package context — at least one live consumer
# already loads *this* file itself via importlib.util.spec_from_file_location,
# bypassing sys.path entirely, and Path(__file__).parent resolves correctly
# under every observed loading style (normal import, spec_from_file_location,
# Streamlit's st.cache_resource). A deployment that hasn't copied
# router_policy.py yet, or fails to load it for any reason, falls back to
# legacy behavior — this must never be silent, since failures here would
# otherwise be invisible across three differently-venv'd deployments.

def _load_router_policy():
    rp_path = Path(__file__).parent / "router_policy.py"
    if not rp_path.exists():
        logger.debug("router_policy.py not found alongside %s — intelligent routing unavailable", __file__)
        return None
    try:
        import importlib.util
        mod_key = "_ai_router_policy"
        if mod_key in sys.modules:
            return sys.modules[mod_key]
        spec = importlib.util.spec_from_file_location(mod_key, rp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_key] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning("router_policy load failed: %s — intelligent routing disabled", e)
        sys.modules.pop("_ai_router_policy", None)
        return None


# ── Model defaults ────────────────────────────────────────────────────────────

_ANT_FAST    = "claude-haiku-4-5"
_ANT_SMART   = "claude-sonnet-4-5"
_OAI_FAST    = "gpt-4o-mini"
_OAI_SMART   = "gpt-4o"
_GEM_FAST    = "gemini-2.5-flash"    # fast / structured — cheap, very capable
_GEM_SMART   = "gemini-2.5-pro"     # smart — highest quality
_LOCAL_MODEL = "qwen2.5:7b"
_LOCAL_URL   = "http://localhost:11435"

_LOCAL_TIMEOUT = 60.0      # seconds — covers cold-start on Tailscale


# ── Public types ──────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role:    str   # "user" | "assistant" | "system"
    content: str


@dataclass
class RouterResponse:
    text:     str
    provider: str            # "local" | "openrouter" | "gemini" | "anthropic" | "openai"
    model:    str
    tokens:   Optional[int] = None
    # Phase 8: set only when AI_ROUTER_EXPERIENCE_STORE is on — correlates
    # this specific response with its ExperienceTrace, so a caller can later
    # call AIGateway.record_feedback(trace_id, ...). None otherwise — an
    # unset trace_id would be a dangling reference nobody could look up.
    trace_id: Optional[str] = None


# ── Round-robin state ─────────────────────────────────────────────────────────

def _rr_path() -> Path:
    """Resolve round-robin state file relative to the script that imported us."""
    return Path(__file__).parent / "data" / ".ai_gateway_rr.json"


def _rr_load() -> dict:
    try:
        p = _rr_path()
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {"last_provider": "openai"}


def _rr_save(state: dict) -> None:
    try:
        p = _rr_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state))
    except Exception as e:
        logger.debug("round-robin state save failed: %s", e)


# ── Low-level HTTP helpers ────────────────────────────────────────────────────

def _post_json(url: str, payload: dict, headers: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from e


def _get_json(url: str, timeout: int = 30) -> dict:
    """Phase 10: sibling to _post_json for read-only remote calls (e.g.
    GET /v1/metrics) — same stdlib urllib.request, zero new dependencies."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from e


# ── Provider call implementations ─────────────────────────────────────────────

def _call_anthropic(
    messages:   list[ChatMessage],
    system:     str,
    model:      str,
    api_key:    str,
    max_tokens: int,
    tools:      Optional[list] = None,
) -> RouterResponse:
    payload = {
        "model":      model,
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   [{"role": m.role, "content": m.content}
                       for m in messages if m.role != "system"],
    }
    if tools:
        payload["tools"] = tools
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }
    data   = _post_json("https://api.anthropic.com/v1/messages", payload, headers)
    # Collect all text blocks — tool_use responses interleave tool_use + text blocks
    # so data["content"][0]["text"] crashes when index 0 is a tool_use block.
    texts  = [b["text"] for b in (data.get("content") or []) if b.get("type") == "text"]
    text   = "\n\n".join(texts) or "No response."
    tokens = data.get("usage", {}).get("output_tokens")
    return RouterResponse(text=text, provider="anthropic", model=model, tokens=tokens)


def _call_anthropic_openrouter(
    messages:   list[ChatMessage],
    system:     str,
    model:      str,
    api_key:    str,
    max_tokens: int,
) -> RouterResponse:
    """Anthropic via OpenRouter BYOK — fallback when ANTHROPIC_API_KEY isn't set
    (or the direct call fails). Same OpenAI-compat shape as _call_openrouter,
    pointed at OpenRouter with the "anthropic/<model>" model-ID prefix.
    Note: tools aren't forwarded here — Anthropic's native tool schema doesn't
    map onto OpenRouter's OpenAI-compat shape without translation, same
    limitation as the Gemini OpenRouter fallback below."""
    all_msgs = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content}
        for m in messages if m.role != "system"
    ]
    payload = {"model": f"anthropic/{model}", "messages": all_msgs, "max_tokens": max_tokens}
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data   = _post_json("https://openrouter.ai/api/v1/chat/completions", payload, headers)
    text   = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens")
    return RouterResponse(text=text, provider="anthropic", model=model, tokens=tokens)


def _call_openrouter(
    messages:   list[ChatMessage],
    system:     str,
    model:      str,
    api_key:    str,
    max_tokens: int,
) -> RouterResponse:
    """OpenRouter speaks the OpenAI chat-completions shape at its own base URL.
    model is OpenRouter's provider-prefixed id, e.g. "anthropic/claude-sonnet-4.5"
    or "openai/gpt-4o-mini" — one key serves all of them."""
    all_msgs = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content}
        for m in messages if m.role != "system"
    ]
    payload = {"model": model, "messages": all_msgs, "max_tokens": max_tokens}
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data   = _post_json("https://openrouter.ai/api/v1/chat/completions", payload, headers)
    text   = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens")
    return RouterResponse(text=text, provider="openrouter", model=model, tokens=tokens)


def _call_openai(
    messages:   list[ChatMessage],
    system:     str,
    model:      str,
    api_key:    str,
    max_tokens: int,
) -> RouterResponse:
    all_msgs = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content}
        for m in messages if m.role != "system"
    ]
    payload = {"model": model, "messages": all_msgs, "max_tokens": max_tokens}
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data   = _post_json("https://api.openai.com/v1/chat/completions", payload, headers)
    text   = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens")
    return RouterResponse(text=text, provider="openai", model=model, tokens=tokens)


def _call_openai_openrouter(
    messages:   list[ChatMessage],
    system:     str,
    model:      str,
    api_key:    str,
    max_tokens: int,
) -> RouterResponse:
    """OpenAI via OpenRouter BYOK — fallback when OPENAI_API_KEY isn't set (or
    the direct call fails). Same shape as _call_openai, pointed at OpenRouter
    with the "openai/<model>" model-ID prefix it expects."""
    all_msgs = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content}
        for m in messages if m.role != "system"
    ]
    payload = {"model": f"openai/{model}", "messages": all_msgs, "max_tokens": max_tokens}
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data   = _post_json("https://openrouter.ai/api/v1/chat/completions", payload, headers)
    text   = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens")
    return RouterResponse(text=text, provider="openai", model=model, tokens=tokens)


def _call_gemini(
    messages:   list[ChatMessage],
    system:     str,
    model:      str,
    project:    str,
    location:   str,
    max_tokens: int,
) -> RouterResponse:
    """Call Gemini via Vertex AI using Application Default Credentials (ADC).

    Gemini 2.5 models have thinking enabled by default. The SDK's response.text
    shortcut can return None when thought parts are present, so we iterate
    candidates and skip thought blocks explicitly.
    """
    if not _GENAI_OK:
        raise RuntimeError("google-genai library not installed (pip install google-cloud-aiplatform)")
    client = _genai.Client(vertexai=True, project=project, location=location)
    contents = [{"role": m.role, "parts": [{"text": m.content}]}
                for m in messages if m.role != "system"]
    from google.genai import types as _gtypes
    config = _gtypes.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=0.1,
        # Disable thinking for structured/fast calls — reduces latency & cost.
        # Smart callers can override by passing a higher thinking_budget via config.
        thinking_config=_gtypes.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(model=model, contents=contents, config=config)

    # Extract text robustly: iterate parts, skip thought blocks
    text = None
    try:
        text = response.text   # works when there are no thought parts
    except Exception:
        pass
    if not text:
        try:
            for candidate in (response.candidates or []):
                for part in (candidate.content.parts or []):
                    if getattr(part, "thought", False):
                        continue   # skip thinking block
                    part_text = getattr(part, "text", None)
                    if part_text:
                        text = part_text
                        break
                if text:
                    break
        except Exception:
            pass
    text = text or "No response."

    tokens = getattr(getattr(response, "usage_metadata", None), "candidates_token_count", None)
    return RouterResponse(text=text, provider="gemini", model=model, tokens=tokens)


def _call_gemini_openrouter(
    messages:   list[ChatMessage],
    system:     str,
    model:      str,
    api_key:    str,
    max_tokens: int,
) -> RouterResponse:
    """Gemini via OpenRouter BYOK — fallback when Vertex AI (ADC) fails or
    isn't billing-enabled. Same OpenAI-compat shape as _call_openai, pointed
    at OpenRouter with the "google/<model>" model-ID prefix it expects."""
    all_msgs = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content}
        for m in messages if m.role != "system"
    ]
    payload = {"model": f"google/{model}", "messages": all_msgs, "max_tokens": max_tokens}
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data   = _post_json("https://openrouter.ai/api/v1/chat/completions", payload, headers)
    text   = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens")
    return RouterResponse(text=text, provider="gemini", model=model, tokens=tokens)


# ── Live metrics (Phase 8) ──────────────────────────────────────────────────
# Content-free, in-process counters — domain names, route/accept/reject
# counts, latency numbers, never message text. Constructed unconditionally
# whenever AI_ROUTER_INTELLIGENT_ROUTING is on (no separate flag — rides on
# the same shadow-computation Phase 1 already established as safe-by-default).
# Bounded memory: latency percentiles come from a fixed-size rolling window,
# not unbounded history. threading.Lock guards mutation/reads since a
# deployment may share one AIGateway across threads (e.g. a Flask dashboard
# + a bot process's own threads).

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentiles(samples: "deque") -> dict:
    if not samples:
        return {"p50": None, "p95": None, "p99": None, "count": 0}
    data = sorted(samples)
    if len(data) < 2:
        v = data[0]
        return {"p50": v, "p95": v, "p99": v, "count": len(data)}
    # statistics.quantiles(n=100) needs >=2 points; clamp index for small n.
    q = statistics.quantiles(data, n=100, method="inclusive")
    def _at(pct: int) -> float:
        return q[min(max(pct - 1, 0), len(q) - 1)]
    return {"p50": _at(50), "p95": _at(95), "p99": _at(99), "count": len(data)}


class _MetricsCollector:
    """See module docstring's Phase 8 section and ARCHITECTURE.md for the
    live-snapshot vs. offline-aggregator split — this class intentionally
    does NOT attempt a real false-accept rate (needs feedback data, which
    arrives asynchronously and can't be joined against bounded counters
    without an unbounded trace_id->outcome map). That's
    training/compute_metrics.py's job."""

    def __init__(self, max_latency_samples: int = 500) -> None:
        self._lock = threading.Lock()
        self.started_at = _now_iso()
        self.total_requests = 0
        self.route_counts = {"local": 0, "frontier": 0}
        self.verifier_counts = {"accepted": 0, "rejected": 0, "not_run": 0}
        self.provider_counts: dict = {}
        self.error_count = 0
        self.by_domain: dict = {}
        self._local_latencies: "deque" = deque(maxlen=max_latency_samples)
        self._frontier_latencies: "deque" = deque(maxlen=max_latency_samples)

    def record(
        self, *,
        domain: Optional[str], final_source: str, escalated: bool,
        verification, frontier_provider: Optional[str],
        local_latency_ms: Optional[float], frontier_latency_ms: Optional[float],
        had_error: bool,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.route_counts[final_source] = self.route_counts.get(final_source, 0) + 1
            if verification is not None:
                self.verifier_counts["accepted" if verification.accepted else "rejected"] += 1
            elif final_source == "local":
                self.verifier_counts["not_run"] += 1
            if frontier_provider:
                self.provider_counts[frontier_provider] = self.provider_counts.get(frontier_provider, 0) + 1
            if had_error:
                self.error_count += 1
            if local_latency_ms is not None:
                self._local_latencies.append(local_latency_ms)
            if frontier_latency_ms is not None:
                self._frontier_latencies.append(frontier_latency_ms)
            if domain:
                bucket = self.by_domain.setdefault(domain, {"total": 0, "local_accepted": 0, "escalated": 0})
                bucket["total"] += 1
                if final_source == "local" and not escalated:
                    bucket["local_accepted"] += 1
                if escalated:
                    bucket["escalated"] += 1

    def snapshot(self, domain_registry: Optional[dict] = None) -> dict:
        with self._lock:
            by_domain_out = {}
            for name, bucket in self.by_domain.items():
                deviation = None
                if domain_registry is not None:
                    cfg = domain_registry.get(name)
                    if cfg is not None and cfg.benchmark_success_rate is not None and bucket["total"] > 0:
                        observed = bucket["local_accepted"] / bucket["total"]
                        deviation = observed - cfg.benchmark_success_rate
                by_domain_out[name] = {**bucket, "benchmark_deviation": deviation}
            return {
                "enabled": True,
                "started_at": self.started_at,
                "snapshot_at": _now_iso(),
                "total_requests": self.total_requests,
                "route_counts": dict(self.route_counts),
                "verifier_counts": dict(self.verifier_counts),
                "provider_counts": dict(self.provider_counts),
                "error_count": self.error_count,
                "by_domain": by_domain_out,
                "local_latency_ms": _percentiles(self._local_latencies),
                "frontier_latency_ms": _percentiles(self._frontier_latencies),
            }


# ── Gateway ───────────────────────────────────────────────────────────────────

class AIGateway:
    """
    Provider-agnostic LLM gateway.
    All config read from environment variables — no constructor args needed.
    Thread-safe singleton; instantiate once at module level.
    """

    def __init__(self) -> None:
        # Load env in priority order (later layers override earlier ones):
        #   1. Master key store  ~/Projects/ai-router/.env  (all API keys live here)
        #   2. Project .env      weights, model overrides, LOCAL_INTEL_URL
        #   3. OS environment    always wins (allows CI/systemd overrides)
        _load_master_env()

        def _env(k: str, default: str = "") -> str:
            v = os.getenv(k, "").strip().strip('"').strip("'")
            if not v:
                try:
                    from dotenv import dotenv_values
                    v = (dotenv_values().get(k) or "").strip().strip('"').strip("'")
                except ImportError:
                    pass
            return v or default

        # ── Remote gateway (Tailscale proxy) ─────────────────────────────────
        # When REMOTE_GATEWAY_URL is set, ALL chat() calls are forwarded to that
        # HTTP server instead of calling providers directly. This lets GCP
        # (or any remote host) use the AI router without holding API keys — keys
        # stay on the machine that runs ~/Projects/ai-router/server.py.
        # Example: REMOTE_GATEWAY_URL=http://100.x.x.x:7861  (Tailscale IP)
        self._remote_url = _env("REMOTE_GATEWAY_URL").rstrip("/")

        # ── Local LLM ─────────────────────────────────────────────────────────
        # Prefer LOCAL_INTEL_URL (shared convention), fall back to OLLAMA_URL
        self._local_url   = _env("LOCAL_INTEL_URL") or _env("OLLAMA_URL") or _LOCAL_URL
        self._local_model = _env("OLLAMA_MODEL", _LOCAL_MODEL)
        # Domain selects the LoRA adapter the gateway hot-swaps (retail/finance/
        # health). Sent explicitly so routing no longer depends on model-string
        # parsing or /v1/models ordering. Unset -> gateway keeps legacy behaviour.
        self._local_domain = _env("LOCAL_INTEL_DOMAIN") or None
        self._is_mlx      = ":11435" in self._local_url or "mlx" in self._local_url.lower()
        self._http        = _httpx.Client(timeout=_LOCAL_TIMEOUT) if _HTTPX_OK else None
        self._mlx_model_id: Optional[str] = None   # resolved lazily
        # USE_LOCAL_SLM=false → skip local entirely (e.g. crawler cron where the
        # local model is an intent classifier, not suitable for extraction).
        # Default true — try local first, fall through to cloud on failure.
        self._use_local_slm = _env("USE_LOCAL_SLM", "true").lower() != "false"

        # ── Gemini / Vertex AI (ADC — no key needed), OpenRouter BYOK fallback ──
        # Vertex is tried first when available (draws down GCP credit pools);
        # OpenRouter is the fallback when Vertex fails for any reason (billing
        # not enabled on the project, quota, transient error, ADC missing).
        # "gemini" is enabled as a provider slot if EITHER path is usable —
        # _call_cloud() decides per-call which one actually serves the request.
        self._gemini_project    = _env("GEMINI_PROJECT", "gen-lang-client-0271077908")
        self._gemini_location   = _env("GEMINI_LOCATION", "us-central1")
        self._vertex_available  = _GENAI_OK and _env("GEMINI_ENABLED", "true").lower() != "false"

        # ── Cloud keys ────────────────────────────────────────────────────────
        # OpenRouter is the primary cloud fallback — one key, embeds
        # Anthropic/OpenAI/Google upstream, so a single direct-key account
        # running dry (e.g. Anthropic credit exhaustion) doesn't take cloud
        # calls down. It's also the shared BYOK backstop for Gemini/Anthropic/
        # OpenAI's own direct-key slots below when those fail or aren't set.
        self._openrouter_key = _env("OPENROUTER_API_KEY")
        self._anthropic_key  = _env("ANTHROPIC_API_KEY")
        self._openai_key     = _env("OPENAI_API_KEY")

        self._gemini_enabled = self._vertex_available or bool(self._openrouter_key)

        # ── Routing config ────────────────────────────────────────────────────
        self.mode = _env("AI_ROUTER_MODE", "weighted").lower()
        self.weights: dict[str, float] = {
            "openrouter": float(_env("AI_ROUTER_WEIGHT_OPENROUTER", "0.75")),
            "gemini":     float(_env("AI_ROUTER_WEIGHT_GEMINI",     "0.15")),
            "anthropic":  float(_env("AI_ROUTER_WEIGHT_ANTHROPIC",  "0.05")),
            "openai":     float(_env("AI_ROUTER_WEIGHT_OPENAI",     "0.05")),
        }

        # ── Models per provider ───────────────────────────────────────────────
        self.models = {
            "openrouter": {
                "fast":       _env("AI_ROUTER_OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                "smart":      _env("AI_ROUTER_OPENROUTER_SMART", "anthropic/claude-sonnet-4.5"),
                "structured": _env("AI_ROUTER_OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            },
            "anthropic": {
                "fast":       _env("AI_ROUTER_ANTHROPIC_MODEL", _ANT_FAST),
                "smart":      _env("AI_ROUTER_ANTHROPIC_SMART", _ANT_SMART),
                "structured": _env("AI_ROUTER_ANTHROPIC_MODEL", _ANT_FAST),
            },
            "openai": {
                "fast":       _env("AI_ROUTER_OPENAI_MODEL", _OAI_FAST),
                "smart":      _env("AI_ROUTER_OPENAI_SMART", _OAI_SMART),
                "structured": _env("AI_ROUTER_OPENAI_MODEL", _OAI_FAST),
            },
            "gemini": {
                "fast":       _env("AI_ROUTER_GEMINI_MODEL", _GEM_FAST),
                "smart":      _env("AI_ROUTER_GEMINI_SMART", _GEM_SMART),
                "structured": _env("AI_ROUTER_GEMINI_MODEL", _GEM_FAST),
            },
        }

        # ── Intelligent routing (Phase 1, off by default) ──────────────────────
        # Shadow-mode only: when enabled, a RoutingDecision is computed and
        # logged for every chat() call, but it is NOT yet consulted to change
        # which tier/provider actually serves the request — see
        # ARCHITECTURE.md and router_policy.py's module docstring. That wiring
        # is a later phase. This flag is read via the same _env() closure as
        # everything else, so it inherits the master-vs-project config
        # precedence quirk documented in ARCHITECTURE.md — relevant once
        # per-project rollout starts.
        self._intelligent_routing = _env("AI_ROUTER_INTELLIGENT_ROUTING", "false").lower() == "true"
        # Phase 2: "override" (default) preserves today's exact behavior —
        # LOCAL_INTEL_DOMAIN always wins, the domain registry/classifier
        # never influences a real request. "hint"/"auto" opt into dynamic
        # domain selection — see router_policy.py's module docstring and
        # ARCHITECTURE.md's Phase 2 section. Same config-precedence caveat
        # as AI_ROUTER_INTELLIGENT_ROUTING above.
        self._domain_mode = _env("AI_ROUTER_DOMAIN_MODE", "override").lower()
        # Phase 3: off (default) preserves today's exact dispatch — Tier 1 is
        # always attempted (subject to _call_local's own model_hint=="smart"
        # skip), same as always. On: decision.route actually gates whether
        # Tier 1 is attempted at all, and can bypass the smart-skip when the
        # router decided "local" is genuinely viable — see chat() below and
        # ARCHITECTURE.md's Phase 3 section. Same config-precedence caveat.
        self._route_gating = _env("AI_ROUTER_ROUTE_GATING", "false").lower() == "true"
        # Phase 4: off (default) means local responses are returned as soon
        # as _call_local returns non-None — same as always. On: a real
        # ResponseVerifier check must pass first, or the call falls through
        # to Tier 2 instead — see chat() below and ARCHITECTURE.md's Phase 4
        # section. Independent of AI_ROUTER_ROUTE_GATING — verification is
        # orthogonal to how the decision to attempt local was made. Same
        # config-precedence caveat as the flags above.
        self._verify_local = _env("AI_ROUTER_VERIFY_LOCAL", "false").lower() == "true"
        # Phase 5: "weighted" (default) preserves today's exact primary cloud
        # selection (_select_cloud_provider()). "capability" only affects the
        # PRIMARY pick, scoped to the requested model_hint's tier — fallback
        # attempts always stay on the weighted/hint-based path, per the
        # spec's "keep weighted routing as fallback mode." See chat() below
        # and ARCHITECTURE.md's Phase 5 section. Same config-precedence caveat.
        self._frontier_policy = _env("AI_ROUTER_FRONTIER_POLICY", "weighted").lower()
        # Phase 6: off (default) means nothing is recorded anywhere — zero
        # behavior change, pure observability when on. Records only the
        # Tier 1/Tier 2 local-vs-frontier loop — see chat() below and
        # ARCHITECTURE.md's Phase 6 section for the exact scope.
        # AI_ROUTER_CLIENT_APP is purely a label on each trace, no effect on
        # routing. Same config-precedence caveat as the flags above.
        self._experience_enabled = _env("AI_ROUTER_EXPERIENCE_STORE", "false").lower() == "true"
        # Phase 7: AI_ROUTER_EXPERIENCE_REDACT is a 3-value mode, backward
        # compatible with Phase 6's original true/false — "true" (or unset)
        # -> "full" (unchanged hash-only default), "false" -> "none"
        # (unchanged raw), new: "partial" (regex PII-pattern scrubbing, see
        # router_policy.py's module docstring). Unrecognized values fall
        # back to "full", the safe default.
        _redact_raw = _env("AI_ROUTER_EXPERIENCE_REDACT", "true").lower()
        if _redact_raw in ("true", "full"):
            self._experience_redact_mode = "full"
        elif _redact_raw in ("false", "none"):
            self._experience_redact_mode = "none"
        elif _redact_raw == "partial":
            self._experience_redact_mode = "partial"
        else:
            self._experience_redact_mode = "full"
        self._client_app = _env("AI_ROUTER_CLIENT_APP", "unknown")
        # Phase 9: canary-tagging identifiers — unset (default, everywhere
        # today) means both DeterministicRouterPolicy/DeterministicVerifier
        # fall back to their module-level ROUTER_VERSION/VERIFIER_VERSION
        # constants, byte-identical to pre-Phase-9. Set, they let this
        # instance's traces/metrics be told apart from a baseline running
        # elsewhere — see ARCHITECTURE.md's Phase 9 section for how a
        # deployment actually uses this to canary (two separate processes,
        # compared offline via training/compute_metrics.py — no in-process
        # traffic-splitting here).
        self._policy_id = _env("AI_ROUTER_POLICY_ID", "") or None
        self._verifier_id = _env("AI_ROUTER_VERIFIER_ID", "") or None
        self._router_policy = None
        self._response_verifier = None
        self._frontier_router = None
        self._experience_store = None
        # Phase 8: feedback store rides on the SAME flag as the experience
        # store (AI_ROUTER_EXPERIENCE_STORE, no new env var) — a feedback
        # record with no matching trace record to correlate against is
        # meaningless, so its lifecycle is tied to whichever flag turns
        # trace recording on. Metrics ride on AI_ROUTER_INTELLIGENT_ROUTING
        # alone (below) — content-free counting, safe to always collect once
        # routing decisions exist at all. See ARCHITECTURE.md's Phase 8
        # section.
        self._feedback_store = None
        self._metrics = None
        if self._intelligent_routing:
            _rp_module = _load_router_policy()
            if _rp_module is not None:
                self._router_policy = _rp_module.DeterministicRouterPolicy(policy_id=self._policy_id)
                if self._verify_local:
                    self._response_verifier = _rp_module.DeterministicVerifier(verifier_id=self._verifier_id)
                if self._frontier_policy == "capability":
                    self._frontier_router = _rp_module.CapabilityFrontierRouter()
                if self._experience_enabled:
                    self._experience_store = _rp_module.JSONLExperienceStore()
                    self._feedback_store = _rp_module.JSONLFeedbackStore()
                self._metrics = _MetricsCollector()

        if self._remote_url:
            logger.info("AIGateway ready — mode=remote  url=%s (all calls forwarded; no local keys needed)", self._remote_url)
        else:
            logger.info(
                "AIGateway ready — mode=%s local=%s use_local=%s openrouter=%s gemini=%s anthropic=%s openai=%s intelligent_routing=%s domain_mode=%s route_gating=%s verify_local=%s frontier_policy=%s experience_store=%s feedback=%s metrics=%s policy_id=%s verifier_id=%s",
                self.mode, self._local_url,
                "yes" if self._use_local_slm   else "no",
                "yes" if self._openrouter_key   else "no",
                f"yes({self._gemini_project})"  if self._gemini_enabled else "no",
                "yes" if self._anthropic_key    else "no",
                "yes" if self._openai_key       else "no",
                "yes" if self._router_policy    else "no",
                self._domain_mode,
                "yes" if self._route_gating else "no",
                "yes" if self._response_verifier else "no",
                self._frontier_policy,
                "yes" if self._experience_store else "no",
                "yes" if self._feedback_store else "no",
                "yes" if self._metrics else "no",
                self._router_policy.policy_id if self._router_policy else "n/a",
                self._response_verifier.verifier_id if self._response_verifier else "n/a",
            )

    # ── Local LLM ─────────────────────────────────────────────────────────────

    def _resolve_mlx_model(self) -> str:
        if self._mlx_model_id:
            return self._mlx_model_id
        if not _HTTPX_OK or self._http is None:
            return self._local_model
        try:
            resp = self._http.get(f"{self._local_url}/v1/models", timeout=3.0)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    self._mlx_model_id = models[0]["id"]
                    return self._mlx_model_id
        except Exception:
            pass
        return self._local_model

    def _call_local(
        self,
        messages:          list[ChatMessage],
        system:            str,
        max_tokens:        int,
        model_hint:        str = "structured",
        domain_override:   Optional[str] = None,
        bypass_smart_skip: bool = False,
    ) -> Optional[RouterResponse]:
        # Phase 3: bypass_smart_skip is only True when AI_ROUTER_ROUTE_GATING
        # is on AND a real RoutingDecision said "local" is genuinely viable
        # for this request despite model_hint=="smart" — see chat() below.
        # Default False preserves this hard skip exactly as before Phase 3.
        if model_hint == "smart" and not bypass_smart_skip:
            logger.debug("model_hint=smart — skipping local LLM, routing to cloud")
            return None
        if not _HTTPX_OK or self._http is None:
            logger.debug("httpx not available — skipping local LLM")
            return None
        # Phase 2: domain_override (from a RoutingDecision, only set when
        # AI_ROUTER_DOMAIN_MODE is "hint"/"auto") takes precedence over the
        # static LOCAL_INTEL_DOMAIN env var. None (the default — always the
        # case when intelligent routing or domain_mode="override" apply)
        # means this line is byte-identical to before Phase 2.
        effective_domain = domain_override if domain_override is not None else self._local_domain
        msgs = [{"role": "system", "content": system}] + [
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        ]
        try:
            if self._is_mlx:
                model   = self._resolve_mlx_model()
                payload = {
                    "model": model, "messages": msgs,
                    "stream": False, "temperature": 0.1, "max_tokens": max_tokens,
                }
                if effective_domain:
                    payload["domain"] = effective_domain
                resp = self._http.post(
                    f"{self._local_url}/v1/chat/completions",
                    json=payload, timeout=_LOCAL_TIMEOUT,
                )
                resp.raise_for_status()
                text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                payload = {
                    "model": self._local_model, "messages": msgs,
                    "stream": False, "options": {"temperature": 0.1, "num_predict": max_tokens},
                }
                resp = self._http.post(
                    f"{self._local_url}/api/chat",
                    json=payload, timeout=_LOCAL_TIMEOUT,
                )
                resp.raise_for_status()
                text = resp.json().get("message", {}).get("content", "")

            if text:
                model_id = self._mlx_model_id or self._local_model
                logger.debug("local LLM served: model=%s tokens≈%d", model_id, len(text) // 4)
                return RouterResponse(text=text, provider="local", model=model_id)
        except Exception as e:
            _timeout_types = (_httpx.TimeoutException,) if _HTTPX_OK else ()
            if _timeout_types and isinstance(e, _timeout_types):
                logger.warning("local LLM timeout after %.0fs", _LOCAL_TIMEOUT)
            else:
                logger.warning("local LLM error: %s", e)
        return None

    # ── Remote gateway (Tailscale proxy) ──────────────────────────────────────

    def _call_remote(
        self,
        messages:       list[ChatMessage],
        system:         str,
        model_hint:     str,
        max_tokens:     int,
        model_override: Optional[str],
        tools:          Optional[list],
        provider:       Optional[str],
    ) -> RouterResponse:
        """Forward a chat request to the remote gateway server (e.g. over Tailscale)."""
        payload: dict = {
            "messages":   [{"role": m.role, "content": m.content} for m in messages],
            "system":     system,
            "model_hint": model_hint,
            "max_tokens": max_tokens,
        }
        if model_override:
            payload["model_override"] = model_override
        if tools:
            payload["tools"] = tools
        if provider:
            payload["provider"] = provider

        headers = {"Content-Type": "application/json"}
        try:
            data = _post_json(f"{self._remote_url}/v1/chat", payload, headers, timeout=120)
        except RuntimeError as e:
            raise RuntimeError(f"Remote gateway error: {e}") from e

        if "error" in data:
            raise RuntimeError(f"Remote gateway returned error: {data['error']}")

        return RouterResponse(
            text=data.get("text", ""),
            provider=f"remote:{data.get('provider', '?')}",
            model=data.get("model", "?"),
            tokens=data.get("tokens"),
        )

    # ── Compatibility shims (old CatalogAIRouter / FinanceRouter API) ────────────

    @property
    def openai_key(self) -> str:
        return self._openai_key

    @property
    def anthropic_key(self) -> str:
        return self._anthropic_key

    @property
    def openrouter_key(self) -> str:
        return self._openrouter_key

    def _enabled(self) -> list[str]:
        return self._cloud_providers()

    # ── Cloud provider selection ───────────────────────────────────────────────

    def _cloud_providers(self) -> list[str]:
        # Order matters for fallback: OpenRouter first (broadest coverage, own
        # credit pool), then Gemini (ADC, no key, or OpenRouter BYOK), then
        # direct Anthropic/OpenAI.
        available = []
        if self._openrouter_key:
            available.append("openrouter")
        if self._gemini_enabled:
            available.append("gemini")
        if self._anthropic_key:
            available.append("anthropic")
        if self._openai_key:
            available.append("openai")
        return available

    def _select_cloud_provider(self) -> Optional[str]:
        providers = self._cloud_providers()
        if not providers:
            return None

        mode = self.mode

        if mode in ("openrouter", "gemini", "anthropic", "openai"):
            return mode if mode in providers else (providers[0] if providers else None)

        if mode == "round-robin":
            state    = _rr_load()
            last     = state.get("last_provider", "openai")
            idx      = providers.index(last) if last in providers else -1
            selected = providers[(idx + 1) % len(providers)]
            _rr_save({"last_provider": selected})
            return selected

        # weighted (default)
        candidates = [
            (p, self.weights.get(p, 1.0))
            for p in providers
            if self.weights.get(p, 1.0) > 0
        ]
        if not candidates:
            return providers[0] if providers else None
        total = sum(w for _, w in candidates)
        roll  = random.random() * total
        for p, w in candidates:
            roll -= w
            if roll <= 0:
                return p
        return candidates[-1][0]

    def _call_cloud(
        self,
        provider:   str,
        messages:   list[ChatMessage],
        system:     str,
        model_hint: str,
        max_tokens: int,
        model_override: Optional[str],
        tools:      Optional[list] = None,
    ) -> RouterResponse:
        model = model_override or self.models[provider][model_hint]
        if provider == "openrouter":
            return _call_openrouter(messages, system, model, self._openrouter_key, max_tokens)
        if provider == "gemini":
            # Vertex first (leverages the GCP credit pool); OpenRouter BYOK is
            # the fallback, tried only on failure — not a weighted choice, since
            # from the caller's perspective this is still "the gemini provider."
            if self._vertex_available:
                try:
                    return _call_gemini(messages, system, model,
                                        self._gemini_project, self._gemini_location, max_tokens)
                except Exception as e:
                    if not self._openrouter_key:
                        raise
                    logger.warning("Vertex AI call failed (%s) — falling back to OpenRouter BYOK", e)
            if not self._openrouter_key:
                raise RuntimeError("AIGateway: gemini provider has no working backend (Vertex failed/unavailable, no OPENROUTER_API_KEY).")
            return _call_gemini_openrouter(messages, system, model, self._openrouter_key, max_tokens)
        if provider == "anthropic":
            # Direct key first (native Messages API, supports tools); OpenRouter
            # BYOK is the fallback — tried on failure — same shape as gemini above.
            if self._anthropic_key:
                try:
                    return _call_anthropic(messages, system, model, self._anthropic_key, max_tokens, tools=tools)
                except Exception as e:
                    if not self._openrouter_key:
                        raise
                    logger.warning("Anthropic direct call failed (%s) — falling back to OpenRouter BYOK", e)
            if not self._openrouter_key:
                raise RuntimeError("AIGateway: anthropic provider has no working backend (no ANTHROPIC_API_KEY, no OPENROUTER_API_KEY).")
            return _call_anthropic_openrouter(messages, system, model, self._openrouter_key, max_tokens)
        if provider == "openai":
            if self._openai_key:
                try:
                    return _call_openai(messages, system, model, self._openai_key, max_tokens)
                except Exception as e:
                    if not self._openrouter_key:
                        raise
                    logger.warning("OpenAI direct call failed (%s) — falling back to OpenRouter BYOK", e)
            if not self._openrouter_key:
                raise RuntimeError("AIGateway: openai provider has no working backend (no OPENAI_API_KEY, no OPENROUTER_API_KEY).")
            return _call_openai_openrouter(messages, system, model, self._openrouter_key, max_tokens)
        raise ValueError(f"Unknown provider: {provider}")

    # ── Experience Store (Phase 6) + live metrics (Phase 8) ──────────────────

    def _record_experience(
        self, ctx, decision, *,
        trace_id=None,
        local_response=None, local_latency_ms=None, verification=None,
        escalated=False, escalation_reason=None,
        frontier_provider=None, frontier_model=None, frontier_response=None,
        frontier_latency_ms=None, frontier_tokens=None,
        errors=None, final_source="local",
        frontier_router_version=None,
    ) -> None:
        """Updates in-process metrics unconditionally (Phase 8 — content-free,
        needs only AI_ROUTER_INTELLIGENT_ROUTING), then writes a full
        ExperienceTrace only if AI_ROUTER_EXPERIENCE_STORE is on. Never
        raises either way — a tracing/metrics bug must never block or alter
        a real response, same guarantee as the router policy and verifier
        already have."""
        if decision is not None and self._metrics is not None:
            try:
                self._metrics.record(
                    domain=decision.domain, final_source=final_source, escalated=escalated,
                    verification=verification, frontier_provider=frontier_provider,
                    local_latency_ms=local_latency_ms, frontier_latency_ms=frontier_latency_ms,
                    had_error=bool(errors),
                )
            except Exception as e:
                logger.warning("metrics update failed: %s", e)

        if self._experience_store is None or ctx is None or decision is None:
            return
        try:
            _rp = sys.modules.get("_ai_router_policy")
            if _rp is None:
                return
            trace = _rp.build_experience_trace(
                context=ctx, decision=decision, trace_id=trace_id, client_app=self._client_app,
                local_response=local_response, local_latency_ms=local_latency_ms,
                verification=verification, escalated=escalated, escalation_reason=escalation_reason,
                frontier_provider=frontier_provider, frontier_model=frontier_model,
                frontier_response=frontier_response, frontier_latency_ms=frontier_latency_ms,
                frontier_tokens=frontier_tokens, errors=errors or [],
                final_source=final_source, redact_mode=self._experience_redact_mode,
                frontier_router_version=frontier_router_version,
            )
            self._experience_store.record(trace)
        except Exception as e:
            logger.warning("experience trace build/record failed: %s", e)

    def record_feedback(self, trace_id: str, label: str, note: Optional[str] = None) -> bool:
        """Attach feedback to a previously-returned RouterResponse.trace_id
        (see chat()). No-op (returns False) if AI_ROUTER_EXPERIENCE_STORE is
        off, or if `label` isn't one of "correct"/"incorrect"/"unrated" —
        never raises, matches every other subsystem's fail-open guarantee.
        No live deployment calls this yet — see ARCHITECTURE.md's Phase 8
        section."""
        # Phase 10: when fully remoted, feedback belongs with the trace it
        # correlates to — which lives on the central service, not here.
        if self._remote_url:
            try:
                data = _post_json(
                    f"{self._remote_url}/v1/record_feedback",
                    {"trace_id": trace_id, "label": label, "note": note},
                    {"Content-Type": "application/json"}, timeout=30,
                )
                return bool(data.get("ok"))
            except Exception as e:
                logger.warning("remote record_feedback failed: %s", e)
                return False
        if self._feedback_store is None:
            return False
        try:
            _rp = sys.modules.get("_ai_router_policy")
            if _rp is None:
                return False
            record = _rp.build_feedback_record(
                trace_id=trace_id, label=label, note=note,
                client_app=self._client_app, redact_mode=self._experience_redact_mode,
            )
            if record is None:
                logger.warning("record_feedback: invalid label %r, ignored", label)
                return False
            self._feedback_store.record(record)
            return True
        except Exception as e:
            logger.warning("record_feedback failed: %s", e)
            return False

    def get_metrics_snapshot(self) -> dict:
        """Live, in-process, content-free metrics — see _MetricsCollector
        above and ARCHITECTURE.md's Phase 8 section for what this does and
        does not report (no real false-accept rate; that needs feedback
        data joined against durable trace history — see
        training/compute_metrics.py)."""
        # Phase 10: when fully remoted, the metrics that matter are the
        # central service's own (this process's local counters would just
        # be empty, since it never dispatches locally in that mode).
        if self._remote_url:
            try:
                return _get_json(f"{self._remote_url}/v1/metrics", timeout=30)
            except Exception as e:
                logger.warning("remote get_metrics_snapshot failed: %s", e)
                return {"enabled": False}
        if self._metrics is None:
            return {"enabled": False}
        registry = self._router_policy.registry if self._router_policy is not None else None
        return self._metrics.snapshot(domain_registry=registry)

    def log_metrics_snapshot(self) -> None:
        """Convenience for a caller that just wants periodic log visibility
        without building its own poller/metrics route."""
        logger.info("metrics_snapshot %s", json.dumps(self.get_metrics_snapshot(), default=str))

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages:       list[ChatMessage],
        system:         str           = "You are a helpful assistant.",
        model_hint:     str           = "structured",  # "structured"/"fast" → haiku | "smart" → sonnet
        max_tokens:     int           = 800,
        model_override: Optional[str] = None,
        retries:        int           = 1,
        tools:          Optional[list] = None,         # e.g. [{"type":"web_search_20250305","name":"web_search"}]
        provider:       Optional[str] = None,          # pin to "openrouter"/"anthropic"/"openai"/"gemini" for this call only
        force_frontier: bool          = False,         # Phase 3: explicit override, only consulted when AI_ROUTER_ROUTE_GATING is on
    ) -> RouterResponse:
        """
        Route a chat request.  Returns RouterResponse with .text, .provider, .model.

        Tier 0 — remote gateway (if REMOTE_GATEWAY_URL is set — bypasses everything below).
        Tier 1 — local LLM (skipped when USE_LOCAL_SLM=false, mode is pinned to
                 a cloud provider, model_hint="smart" (unless AI_ROUTER_ROUTE_GATING
                 says otherwise), or a RoutingDecision says route="frontier").
        Tier 2 — cloud providers, weighted or round-robin.

        Pass provider="anthropic" (or "openai"/"gemini"/"openrouter") to pin a single
        call to that provider without changing the global AI_ROUTER_MODE — useful
        when a tool (e.g. web_search) is only supported by one provider.
        """
        # Intelligent routing (Phase 1 shadow logging + Phase 2 domain
        # selection + Phase 3 route gating). decision.domain only affects a
        # real request when AI_ROUTER_DOMAIN_MODE is "hint"/"auto"; decision.route
        # only affects real dispatch when AI_ROUTER_ROUTE_GATING is on — see
        # the Tier 1 dispatch below. A policy exception can never block a
        # real chat() call.
        decision = None
        ctx = None   # Phase 4: must be safely referenceable later even if this block doesn't run
        # Phase 8: generated unconditionally (cost: one uuid call) so it's
        # stable across this call's several possible exit points below. Only
        # actually attached to the returned RouterResponse when the
        # Experience Store is on — see the _trace_id assignments below.
        _trace_id = uuid.uuid4().hex
        if self._router_policy is not None:
            try:
                _rp = sys.modules["_ai_router_policy"]
                ctx = _rp.RequestContext(
                    messages=messages, system=system, model_hint=model_hint,
                    app_domain_hint=self._local_domain,
                    tools=tools, max_tokens=max_tokens,
                    domain_mode=self._domain_mode,
                    force_frontier=force_frontier,
                )
                decision = self._router_policy.decide(ctx)
                logger.info(
                    "routing_decision domain=%s task=%s route=%s reason=%s",
                    decision.domain, decision.task, decision.route, decision.reason,
                )
            except Exception as e:
                logger.warning("router policy failed, continuing with legacy routing: %s", e)
                decision = None

        # Tier 0: remote gateway — forwards to ~/Projects/ai-router/server.py over Tailscale.
        # API keys stay on the gateway machine; this host needs no credentials.
        if self._remote_url:
            return self._call_remote(messages, system, model_hint, max_tokens, model_override, tools, provider)

        # Per-call provider pin — bypasses local tier and routing selection
        if provider:
            if provider not in self._cloud_providers():
                key_var = {"openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider, f"{provider.upper()}_API_KEY")
                raise RuntimeError(
                    f"AIGateway: pinned provider '{provider}' is not configured. "
                    f"Add {key_var} to ~/Projects/ai-router/.env (master key store) "
                    f"or to the project .env file."
                )
            return self._call_cloud(provider, messages, system, model_hint, max_tokens, model_override, tools=tools)

        # Phase 3: whether Tier 1 is even attempted, and whether it can bypass
        # the model_hint=="smart" skip, follow a real RoutingDecision only
        # when AI_ROUTER_ROUTE_GATING is on. Off (default): attempt_local is
        # always True and bypass_smart_skip is always False — byte-identical
        # to pre-Phase-3 behavior, including the hardcoded smart-skip.
        use_route_gate = self._route_gating and decision is not None
        attempt_local = (decision.route == "local") if use_route_gate else True

        # Phase 6: accumulated as Tier 1/2 execution proceeds, recorded at
        # each actual exit point below. Tracing is scoped to this Tier 1/2
        # loop only — Tier 0/pinned-provider calls above are not traced, and
        # neither is complete_with_meta()'s skip_local path (bypasses chat()
        # entirely) — see ARCHITECTURE.md's Phase 6 section.
        _local_response_text = None
        _local_latency_ms = None
        _verification = None
        _escalated = False
        _escalation_reason = None

        # Tier 1: local — skipped when USE_LOCAL_SLM=false, mode is pinned to
        # cloud, model_hint=smart (unless route-gated otherwise), or a
        # RoutingDecision says route="frontier" (only when route-gated)
        if self._use_local_slm and self.mode not in ("openrouter", "gemini", "anthropic", "openai") and attempt_local:
            # Phase 2: only override the domain sent to LocalIntelligence when
            # domain_mode is explicitly "hint"/"auto" — "override" (the
            # default) keeps this None, so _call_local falls back to
            # self._local_domain exactly as before Phase 2.
            domain_override = None
            if decision is not None and self._domain_mode != "override":
                domain_override = decision.domain
            _local_call_start = time.time()
            local_result = self._call_local(
                messages, system, max_tokens, model_hint,
                domain_override=domain_override,
                bypass_smart_skip=use_route_gate,
            )
            _local_latency_ms = (time.time() - _local_call_start) * 1000
            if local_result:
                _local_response_text = local_result.text
                # Phase 8: attach trace_id now — every branch below returns
                # this same local_result object. Only set when there's a
                # matching trace record to correlate it to.
                if self._experience_store is not None:
                    local_result.trace_id = _trace_id
                # Phase 4: verification is orthogonal to route gating — only
                # gated by AI_ROUTER_VERIFY_LOCAL. Off (default): falls
                # straight to the else branch below, byte-identical to
                # pre-Phase-4 behavior.
                if self._response_verifier is not None and decision is not None and ctx is not None:
                    try:
                        verification = self._response_verifier.evaluate(ctx, decision, local_result.text)
                        _verification = verification
                        logger.info(
                            "verification accepted=%s score=%.2f reasons=%s",
                            verification.accepted, verification.score, verification.failure_reasons,
                        )
                        if verification.accepted:
                            self._record_experience(
                                ctx, decision, trace_id=_trace_id, local_response=_local_response_text,
                                local_latency_ms=_local_latency_ms, verification=_verification,
                                final_source="local",
                            )
                            return local_result
                        logger.info("local response rejected by verifier — escalating to cloud")
                        _escalated = True
                        _escalation_reason = f"verification_rejected: {', '.join(verification.failure_reasons)}"
                        # falls through to Tier 2 below — no early return
                    except Exception as e:
                        # Fail OPEN — a verifier bug must never force every
                        # local call to escalate to paid cloud (same rule as
                        # the router policy try/except above).
                        logger.warning("verifier failed, accepting local response as-is: %s", e)
                        self._record_experience(
                            ctx, decision, trace_id=_trace_id, local_response=_local_response_text,
                            local_latency_ms=_local_latency_ms, final_source="local",
                            errors=[f"verifier_exception: {e}"],
                        )
                        return local_result
                else:
                    self._record_experience(
                        ctx, decision, trace_id=_trace_id, local_response=_local_response_text,
                        local_latency_ms=_local_latency_ms, final_source="local",
                    )
                    return local_result
            else:
                _escalated = True
                _escalation_reason = "local_unavailable_or_failed"

        if self.mode == "local":
            raise RuntimeError("AIGateway: local LLM unavailable and mode=local; no cloud fallback.")

        # Tier 2: cloud
        providers  = self._cloud_providers()
        if not providers:
            raise RuntimeError(
                "AIGateway: no cloud providers configured "
                "(set OPENROUTER_API_KEY, GEMINI_PROJECT, ANTHROPIC_API_KEY, or OPENAI_API_KEY)."
            )

        # Phase 5: capability-based selection only picks the PRIMARY provider
        # + a specific model — scoped to the requested model_hint's tier
        # (one candidate per available provider, each using that provider's
        # own configured model for this hint). Only runs when the caller
        # hasn't already pinned an explicit model_override — same precedence
        # explicit overrides already have over hint-based lookup. Fallback
        # attempts (below) always use the legacy weighted/hint-based path.
        capability_override = None
        if (self._frontier_policy == "capability" and self._frontier_router is not None
                and decision is not None and ctx is not None and model_override is None):
            candidates = [(p, self.models[p][model_hint]) for p in providers]
            picked = self._frontier_router.select(ctx, decision, candidates)
            if picked is not None:
                primary, capability_override = picked
                logger.info("frontier_capability_pick provider=%s model=%s", primary, capability_override)
            else:
                primary = self._select_cloud_provider()
        else:
            primary = self._select_cloud_provider()

        # Phase 9: names which selection path actually decided THIS
        # request's primary pick — derived from the real outcome above
        # (capability_override is only non-None when select() genuinely
        # returned a pick), not just which mode is configured. Only
        # meaningful once we've reached Tier 2 at all, so it's computed
        # here rather than earlier — a trace where local succeeded never
        # sees this and correctly keeps frontier_router_version=None.
        _rp_versions = sys.modules.get("_ai_router_policy")
        _frontier_router_version = None
        if _rp_versions is not None:
            _frontier_router_version = (
                _rp_versions.FRONTIER_ROUTER_VERSION if capability_override is not None
                else _rp_versions.WEIGHTED_FRONTIER_VERSION
            )

        order   = [primary] + [p for p in providers if p != primary]
        last_err: Exception = RuntimeError("All providers failed.")
        _cloud_errors: list = []

        for attempt, prov in enumerate(order[: retries + 1]):
            try:
                if attempt > 0:
                    logger.info("falling back to %s", prov)
                override = (
                    capability_override
                    if (attempt == 0 and prov == primary and capability_override is not None)
                    else model_override
                )
                _cloud_call_start = time.time()
                result = self._call_cloud(prov, messages, system, model_hint, max_tokens, override, tools=tools)
                _cloud_latency_ms = (time.time() - _cloud_call_start) * 1000
                logger.debug("cloud %s/%s served", result.provider, result.model)
                self._record_experience(
                    ctx, decision, trace_id=_trace_id, local_response=_local_response_text,
                    local_latency_ms=_local_latency_ms, verification=_verification,
                    escalated=_escalated, escalation_reason=_escalation_reason,
                    frontier_provider=result.provider, frontier_model=result.model,
                    frontier_response=result.text, frontier_latency_ms=_cloud_latency_ms,
                    frontier_tokens=result.tokens, errors=_cloud_errors,
                    final_source="frontier", frontier_router_version=_frontier_router_version,
                )
                if self._experience_store is not None:
                    result.trace_id = _trace_id
                return result
            except Exception as e:
                logger.warning("%s failed: %s", prov, e)
                last_err = e
                _cloud_errors.append(f"{prov}: {e}")
                time.sleep(0.5)

        self._record_experience(
            ctx, decision, trace_id=_trace_id, local_response=_local_response_text,
            local_latency_ms=_local_latency_ms, verification=_verification,
            escalated=True, escalation_reason=_escalation_reason or "all_cloud_providers_failed",
            errors=_cloud_errors + [f"final: {last_err}"], final_source="frontier",
            frontier_router_version=_frontier_router_version,
        )
        raise RuntimeError(f"AIGateway: all providers failed. Last: {last_err}") from last_err

    @staticmethod
    def _is_quota_error(e: Exception) -> bool:
        s = str(e).lower()
        return "429" in s or "quota" in s or "rate_limit" in s or "insufficient_quota" in s

    def complete(
        self,
        system: str,
        user:   str,
        task:   str = "general",   # "cypher"|"structured" → fast model; else smart model
        max_tokens: int = 800,
    ) -> Optional[str]:
        """
        Single-turn convenience method.  Returns text string or None on total failure.

        task="cypher" / "structured"  → model_hint="structured" (fast/cheap models)
        task="summarize" / "analysis" / "general" → model_hint="smart"
        """
        result = self.complete_with_meta(system=system, user=user, task=task, max_tokens=max_tokens)
        return result[0] if result else None

    def complete_with_meta(
        self,
        system:      str,
        user:        str,
        task:        str = "general",
        max_tokens:  int = 800,
        skip_local:  bool = False,
    ) -> Optional[tuple]:
        """
        Like complete() but returns (text, provider, model) so callers can
        surface the actual provider in engine labels / analytics.

        skip_local=True forces cloud-only — use this when a local-first Stage 1
        already ran and escalated, so Stage 2 should not retry the same model.

        Returns None on total failure.
        """
        hint = "structured" if task in ("cypher", "structured") else "smart"
        msgs = [ChatMessage(role="user", content=user)]
        try:
            # Phase 10: skip_local=False already goes through self.chat() below,
            # which already forwards to the remote gateway when self._remote_url
            # is set (Tier 0, checked first thing inside chat()) — nothing to do
            # for that path. skip_local=True is the one path that bypasses
            # chat() entirely and would otherwise dispatch locally regardless of
            # remote mode — it gets its own explicit forward here.
            if skip_local and self._remote_url:
                data = _post_json(
                    f"{self._remote_url}/v1/complete_with_meta",
                    {"system": system, "user": user, "task": task, "max_tokens": max_tokens, "skip_local": True},
                    {"Content-Type": "application/json"}, timeout=120,
                )
                if data is None or data.get("error"):
                    return None
                return (data["text"], data["provider"], data["model"])
            if skip_local:
                # Bypass Tier 1 (local MLX / Ollama) and go straight to cloud.
                # Try providers in priority order so one failure doesn't kill the call.
                providers = self._cloud_providers()
                if not providers:
                    raise RuntimeError("No cloud providers configured.")
                primary = self._select_cloud_provider()
                order = [primary] + [p for p in providers if p != primary]
                last_err: Exception = RuntimeError("All cloud providers failed.")
                result = None
                for _prov in order[:2]:   # try primary + one fallback
                    try:
                        result = self._call_cloud(_prov, msgs, system, hint, max_tokens, None)
                        break
                    except Exception as _ce:
                        if self._is_quota_error(_ce):
                            logger.debug("%s quota/rate-limit in complete_with_meta, trying next", _prov)
                        else:
                            logger.warning("%s failed in complete_with_meta: %s", _prov, _ce)
                        last_err = _ce
                if result is None:
                    raise last_err
            else:
                result = self.chat(
                    messages=msgs,
                    system=system,
                    model_hint=hint,
                    max_tokens=max_tokens,
                )
            return (result.text, result.provider, result.model)
        except Exception as e:
            logger.error("complete_with_meta() failed: %s", e)
            return None


# ── Module-level singleton + domain aliases ────────────────────────────────────

_gateway_instance: Optional[AIGateway] = None
_gateway_lock = threading.Lock()


def _get_gateway() -> AIGateway:
    global _gateway_instance
    if _gateway_instance is None:
        with _gateway_lock:
            if _gateway_instance is None:
                _gateway_instance = AIGateway()
    return _gateway_instance


# Primary singleton — used by all projects
router: AIGateway = _get_gateway()

# Domain aliases for readability in each project's codebase
catalog_router: AIGateway = router   # CatalogValidator (retail)
finance_router: AIGateway = router   # portfolio_tracker (finance)
health_router:  AIGateway = router   # physician (health)
