"""
gateway.py — Unified LLM Gateway (Python)
══════════════════════════════════════════
Single file, zero project-specific imports.  Drop into any Python project.

Three-tier routing
──────────────────
  Tier 1  Local LLM            always tried first when LOCAL_INTEL_URL / OLLAMA_URL is set
            • mlx_lm.server    auto-detected by port :11435 or "mlx" in URL
              → OpenAI-compat  /v1/chat/completions
            • Ollama            all other URLs
              → /api/chat

  Tier 2  Cloud providers      weighted or round-robin when local is unavailable
            • Anthropic         if ANTHROPIC_API_KEY is set
            • OpenAI            if OPENAI_API_KEY is set

  Tier 3  None                 raises / returns None so callers can surface the error

Routing modes  (AI_ROUTER_MODE env var)
───────────────────────────────────────
  weighted       probabilistic selection by weight (default)
  round-robin    strict alternation; state in data/.ai_router_rr.json
  anthropic      always Anthropic (if available)
  openai         always OpenAI    (if available)
  local          local LLM only, no cloud fallback
  auto           ordered: local → anthropic → openai (ignores weights)

Centralized key store  ~/Projects/ai-router/.env
──────────────────────────────────────────────────
All API keys live here — one place, not repeated per project.
Override location: AI_GATEWAY_CONFIG=/path/to/keys.env

  ANTHROPIC_API_KEY   = sk-ant-...
  OPENAI_API_KEY      = sk-proj-...
  GEMINI_API_KEY      = AIza...      (future)
  GROK_API_KEY        = xai-...      (future)

Per-project .env  (weights + local LLM only)
────────────────────────────────────────────
  LOCAL_INTEL_URL          http://localhost:11435   MLX via Tailscale (auto-detected)
  OLLAMA_URL               http://localhost:11434   Ollama fallback
  OLLAMA_MODEL             qwen2.5:7b

  AI_ROUTER_MODE           weighted                 weighted|round-robin|anthropic|openai|local
  AI_ROUTER_WEIGHT_ANTHROPIC  0.7
  AI_ROUTER_WEIGHT_OPENAI     0.3

  AI_ROUTER_ANTHROPIC_MODEL   claude-haiku-4-5      fast / structured
  AI_ROUTER_ANTHROPIC_SMART   claude-sonnet-4-5     smart
  AI_ROUTER_OPENAI_MODEL      gpt-4o-mini           fast / structured
  AI_ROUTER_OPENAI_SMART      gpt-4o                smart

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
      model_hint="fast",   # "fast" | "smart" | "structured"
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
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import httpx as _httpx
    _HTTPX_OK = True
except ImportError:
    _httpx    = None   # type: ignore[assignment]
    _HTTPX_OK = False

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


# ── Model defaults ────────────────────────────────────────────────────────────

_ANT_FAST    = "claude-haiku-4-5"
_ANT_SMART   = "claude-sonnet-4-5"
_OAI_FAST    = "gpt-4o-mini"
_OAI_SMART   = "gpt-4o"
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
    provider: str            # "local" | "anthropic" | "openai"
    model:    str
    tokens:   Optional[int] = None


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


# ── Provider call implementations ─────────────────────────────────────────────

def _call_anthropic(
    messages:   list[ChatMessage],
    system:     str,
    model:      str,
    api_key:    str,
    max_tokens: int,
) -> RouterResponse:
    payload = {
        "model":      model,
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   [{"role": m.role, "content": m.content}
                       for m in messages if m.role != "system"],
    }
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }
    data   = _post_json("https://api.anthropic.com/v1/messages", payload, headers)
    text   = data["content"][0]["text"]
    tokens = data.get("usage", {}).get("output_tokens")
    return RouterResponse(text=text, provider="anthropic", model=model, tokens=tokens)


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

        # ── Local LLM ─────────────────────────────────────────────────────────
        # Prefer LOCAL_INTEL_URL (shared convention), fall back to OLLAMA_URL
        self._local_url   = _env("LOCAL_INTEL_URL") or _env("OLLAMA_URL") or _LOCAL_URL
        self._local_model = _env("OLLAMA_MODEL", _LOCAL_MODEL)
        self._is_mlx      = ":11435" in self._local_url or "mlx" in self._local_url.lower()
        self._http        = _httpx.Client(timeout=_LOCAL_TIMEOUT) if _HTTPX_OK else None
        self._mlx_model_id: Optional[str] = None   # resolved lazily

        # ── Cloud keys ────────────────────────────────────────────────────────
        self._anthropic_key = _env("ANTHROPIC_API_KEY")
        self._openai_key    = _env("OPENAI_API_KEY")

        # ── Routing config ────────────────────────────────────────────────────
        self.mode = _env("AI_ROUTER_MODE", "weighted").lower()
        self.weights: dict[str, float] = {
            "anthropic": float(_env("AI_ROUTER_WEIGHT_ANTHROPIC", "0.7")),
            "openai":    float(_env("AI_ROUTER_WEIGHT_OPENAI",    "0.3")),
        }

        # ── Models per provider ───────────────────────────────────────────────
        self.models = {
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
        }

        logger.info(
            "AIGateway ready — mode=%s local=%s anthropic=%s openai=%s",
            self.mode, self._local_url,
            "yes" if self._anthropic_key else "no",
            "yes" if self._openai_key    else "no",
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
        messages:   list[ChatMessage],
        system:     str,
        max_tokens: int,
    ) -> Optional[RouterResponse]:
        if not _HTTPX_OK or self._http is None:
            logger.debug("httpx not available — skipping local LLM")
            return None
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

    # ── Compatibility shims (old CatalogAIRouter / FinanceRouter API) ────────────

    @property
    def openai_key(self) -> str:
        return self._openai_key

    @property
    def anthropic_key(self) -> str:
        return self._anthropic_key

    def _enabled(self) -> list[str]:
        return self._cloud_providers()

    # ── Cloud provider selection ───────────────────────────────────────────────

    def _cloud_providers(self) -> list[str]:
        available = []
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

        if mode in ("anthropic", "openai"):
            return mode if mode in providers else (providers[0] if providers else None)

        if mode == "round-robin":
            state    = _rr_load()
            last     = state.get("last_provider", "openai")
            idx      = providers.index(last) if last in providers else -1
            selected = providers[(idx + 1) % len(providers)]
            _rr_save({"last_provider": selected})
            return selected

        # weighted (default) and auto
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
    ) -> RouterResponse:
        model = model_override or self.models[provider][model_hint]
        if provider == "anthropic":
            return _call_anthropic(messages, system, model, self._anthropic_key, max_tokens)
        if provider == "openai":
            return _call_openai(messages, system, model, self._openai_key, max_tokens)
        raise ValueError(f"Unknown provider: {provider}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages:       list[ChatMessage],
        system:         str           = "You are a helpful assistant.",
        model_hint:     str           = "fast",       # "fast" | "smart" | "structured"
        max_tokens:     int           = 800,
        model_override: Optional[str] = None,
        retries:        int           = 1,
    ) -> RouterResponse:
        """
        Route a chat request.  Returns RouterResponse with .text, .provider, .model.

        Tier 1 — local LLM (skipped when mode="anthropic" or mode="openai").
        Tier 2 — cloud providers, weighted or round-robin.
        """
        # Tier 1: local (unless pinned to a specific cloud provider)
        if self.mode not in ("anthropic", "openai"):
            local_result = self._call_local(messages, system, max_tokens)
            if local_result:
                return local_result

        if self.mode == "local":
            raise RuntimeError("AIGateway: local LLM unavailable and mode=local; no cloud fallback.")

        # Tier 2: cloud
        providers  = self._cloud_providers()
        if not providers:
            raise RuntimeError("AIGateway: no cloud providers configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY).")

        primary = self._select_cloud_provider()
        order   = [primary] + [p for p in providers if p != primary]
        last_err: Exception = RuntimeError("All providers failed.")

        for attempt, provider in enumerate(order[: retries + 1]):
            try:
                if attempt > 0:
                    logger.info("falling back to %s", provider)
                result = self._call_cloud(provider, messages, system, model_hint, max_tokens, model_override)
                logger.debug("cloud %s/%s served", result.provider, result.model)
                return result
            except Exception as e:
                logger.warning("%s failed: %s", provider, e)
                last_err = e
                time.sleep(0.5)

        raise RuntimeError(f"AIGateway: all providers failed. Last: {last_err}") from last_err

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
        hint = "structured" if task in ("cypher", "structured") else "smart"
        try:
            result = self.chat(
                messages=[ChatMessage(role="user", content=user)],
                system=system,
                model_hint=hint,
                max_tokens=max_tokens,
            )
            return result.text
        except Exception as e:
            logger.error("complete() failed: %s", e)
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
