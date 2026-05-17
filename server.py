#!/usr/bin/env python3
"""
server.py — AI Gateway HTTP Server
====================================
Exposes the AI router as a lightweight HTTP API so remote machines (e.g. GCP)
can forward LLM calls here over Tailscale without holding any API keys themselves.

Usage (run on the machine that holds the API keys):
    python3 server.py                  # listens 0.0.0.0:7861
    python3 server.py --port 7862
    python3 server.py --host 100.x.x.x --port 7861   # Tailscale IP only

Clients set REMOTE_GATEWAY_URL=http://<tailscale-ip>:7861 in their .env.

Endpoints
---------
  POST /v1/chat
    Body:   {messages, system, model_hint, max_tokens, tools?, provider?, model_override?}
    Return: {text, provider, model, tokens?}

  GET  /health
    Return: {"ok": true, "providers": ["anthropic", "openai"]}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gateway_server")

# Load this directory's .env so API keys are available
_here = Path(__file__).parent
_env_file = _here / ".env"
if _env_file.exists():
    try:
        from dotenv import dotenv_values
        for k, v in dotenv_values(str(_env_file)).items():
            if v and k not in os.environ:
                os.environ[k] = v
        log.info("Loaded %s", _env_file)
    except ImportError:
        # Manual parse fallback
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
        log.info("Loaded %s (manual parse)", _env_file)

# Import the gateway (same directory)
sys.path.insert(0, str(_here))
from gateway import router, ChatMessage  # noqa: E402


class GatewayHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default per-request noise
        pass

    def _clean_path(self) -> str:
        """Return path without query string or trailing slash."""
        return self.path.split("?")[0].rstrip("/")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, msg: str, status: int = 500):
        self._send_json({"error": msg}, status)

    def do_GET(self):
        path = self._clean_path()
        log.info("GET %s", path)
        if path == "/health" or path.endswith("/health"):
            providers = router._cloud_providers()
            self._send_json({"ok": True, "providers": providers})
        else:
            self._send_error_json(f"Not found (got path: {self.path!r})", 404)

    def do_POST(self):
        path = self._clean_path()
        log.info("POST %s  (raw: %r)", path, self.path)
        # Accept /v1/chat (canonical) or / (sent by older clients that omit the path).
        # Server is Tailscale-only so being permissive here is safe.
        if path not in ("/v1/chat", "") and not path.endswith("/v1/chat"):
            log.warning("Unrecognised POST path: %r", self.path)
            self._send_error_json(f"Not found (got path: {self.path!r})", 404)
            return
        try:
            body = self._read_json()
        except Exception as e:
            self._send_error_json(f"Invalid JSON: {e}", 400)
            return

        try:
            messages = [ChatMessage(**m) for m in body["messages"]]
            result = router.chat(
                messages=messages,
                system=body.get("system", "You are a helpful assistant."),
                model_hint=body.get("model_hint", "structured"),
                max_tokens=int(body.get("max_tokens", 800)),
                model_override=body.get("model_override") or None,
                tools=body.get("tools") or None,
                provider=body.get("provider") or None,
            )
            log.info("→ %s/%s  tokens=%s", result.provider, result.model, result.tokens or "?")
            self._send_json({
                "text":     result.text,
                "provider": result.provider,
                "model":    result.model,
                "tokens":   result.tokens,
            })
        except Exception as e:
            log.warning("chat error: %s", e)
            self._send_error_json(str(e), 500)


def main():
    parser = argparse.ArgumentParser(description="AI Gateway HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7861, help="Port (default: 7861)")
    args = parser.parse_args()

    providers = router._cloud_providers()
    if not providers:
        log.error("No cloud providers configured — set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    log.info("AI Gateway server starting — providers: %s", providers)
    log.info("Listening on http://%s:%d", args.host, args.port)
    log.info("Clients: set REMOTE_GATEWAY_URL=http://<this-host>:%d", args.port)

    server = HTTPServer((args.host, args.port), GatewayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
