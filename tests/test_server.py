"""End-to-end tests for server.py (Phase 10) — starts the REAL server as a
subprocess and hits it with real HTTP requests, unlike
test_gateway_backward_compat.py's remote-forwarding tests, which mock
gateway.py's client-side _post_json/_get_json in isolation. This is the
complementary half: does server.py's own request parsing, dispatch, and
response serialization actually work over the wire.

No real LLM provider calls are made — ANTHROPIC_API_KEY is a fake string,
so /v1/chat and /v1/complete_with_meta are expected to fail at the provider
call (proving routing + error serialization work end-to-end); /health,
/v1/metrics, and /v1/record_feedback need no provider at all and are
expected to succeed for real.

Run via:
    python3 -m unittest discover -s tests -v
from the ai-router repo root.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _request(method: str, url: str, body: dict = None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, json.loads(e.read())


class TestServerEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.port = _free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        # Explicit allowlist, NOT dict(os.environ) — this session has a
        # confirmed history of a real, billed provider call leaking through
        # when a test env inherited the parent process's environment
        # wholesale (a real OPENROUTER_API_KEY/ANTHROPIC_API_KEY etc. can be
        # present in the shell). Only pass through what the interpreter
        # itself needs, plus explicit FAKE, NON-EMPTY values for every
        # provider credential this repo recognizes.
        #
        # Critical gotcha, confirmed the hard way (a real, billed OpenRouter
        # call leaked through twice while developing this test): gateway.py's
        # _env() helper does `v = os.getenv(k, "").strip(); if not v: v =
        # dotenv_values().get(k)` — dotenv_values() with NO path argument
        # locates ".env" via python-dotenv's find_dotenv(), which searches
        # from the CALLING FRAME's file location (gateway.py's own
        # directory) via stack introspection, NOT the process's cwd and NOT
        # AI_GATEWAY_CONFIG (that only governs _load_master_env()'s own,
        # separate load step). So setting a credential to "" does NOT
        # disable it — an empty string is falsy, which is exactly what
        # triggers the dotenv_values() fallback, which then finds the REAL
        # key in the real ~/Projects/ai-router/.env regardless of cwd or
        # AI_GATEWAY_CONFIG. The only reliable way to keep a credential off
        # is a non-empty FAKE value, so os.getenv() alone short-circuits
        # _env() before the dotenv fallback ever runs.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "AI_GATEWAY_CONFIG":         "/nonexistent/ai-router-test.env",   # belt-and-suspenders for _load_master_env()
            "ANTHROPIC_API_KEY":         "sk-ant-test-fake",     # non-empty so _cloud_providers() is non-empty; server won't sys.exit(1)
            "OPENAI_API_KEY":            "sk-openai-test-disabled",
            "OPENROUTER_API_KEY":        "sk-or-test-disabled",
            "GEMINI_PROJECT":            "test-disabled",
            "GEMINI_ENABLED":            "false",
            "AI_ROUTER_WEIGHT_GEMINI":   "0",
            "USE_LOCAL_SLM":             "false",
            # Deliberately absent, not "" — REMOTE_GATEWAY_URL isn't in the
            # real master .env (confirmed), so there's no dotenv-fallback
            # leak risk for this one key, and this test wants server.py's
            # own router dispatching locally, not forwarding to "remote."
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_EXPERIENCE_STORE":    "true",
        }
        cls.proc = subprocess.Popen(
            [sys.executable, str(_REPO_ROOT / "server.py"), "--host", "127.0.0.1", "--port", str(cls.port)],
            cwd=cls.tmpdir.name, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        cls._wait_for_health(timeout=10.0)

    @classmethod
    def _wait_for_health(cls, timeout: float) -> None:
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                out = cls.proc.stdout.read() if cls.proc.stdout else ""
                raise RuntimeError(f"server.py exited early (code {cls.proc.returncode}):\n{out}")
            try:
                status, _ = _request("GET", f"{cls.base_url}/health", timeout=1.0)
                if status == 200:
                    return
            except Exception as e:
                last_err = e
            time.sleep(0.2)
        raise RuntimeError(f"server.py never became healthy: {last_err}")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
            cls.proc.wait(timeout=5)
        if cls.proc.stdout:
            cls.proc.stdout.close()
        cls.tmpdir.cleanup()

    def test_health_returns_providers(self):
        status, data = _request("GET", f"{self.base_url}/health")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertIn("anthropic", data["providers"])

    def test_unknown_path_returns_404(self):
        status, data = _request("GET", f"{self.base_url}/nonexistent")
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    def test_invalid_json_body_returns_400(self):
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat", data=b"not json", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5.0)
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            with e:
                self.assertEqual(e.code, 400)

    def test_metrics_endpoint_works_with_no_llm_calls(self):
        status, data = _request("GET", f"{self.base_url}/v1/metrics")
        self.assertEqual(status, 200)
        self.assertTrue(data["enabled"])
        self.assertEqual(data["total_requests"], 0)

    def test_record_feedback_endpoint_works_with_no_llm_calls(self):
        status, data = _request("POST", f"{self.base_url}/v1/record_feedback",
                                 {"trace_id": "some-trace-id", "label": "correct"})
        self.assertEqual(status, 200)
        self.assertIn("ok", data)   # True (feedback store is on) — the value itself isn't the point here

    def test_record_feedback_invalid_label_returns_ok_false(self):
        status, data = _request("POST", f"{self.base_url}/v1/record_feedback",
                                 {"trace_id": "some-trace-id", "label": "not-a-real-label"})
        self.assertEqual(status, 200)
        self.assertFalse(data["ok"])

    def test_chat_endpoint_reaches_dispatch_and_returns_well_formed_error(self):
        # Fake API key -> the real Anthropic call fails -> proves request
        # parsing + routing + error serialization work end-to-end, even
        # though no real completion succeeds here (that's gateway.py's
        # own, separately-covered responsibility).
        status, data = _request("POST", f"{self.base_url}/v1/chat", {
            "messages": [{"role": "user", "content": "hi"}],
            "provider": "anthropic",
        })
        self.assertEqual(status, 500)
        self.assertIn("error", data)

    def test_complete_with_meta_endpoint_reaches_dispatch(self):
        status, data = _request("POST", f"{self.base_url}/v1/complete_with_meta", {
            "system": "s", "user": "u", "task": "general", "skip_local": True,
        })
        # complete_with_meta() catches its own failures and returns None ->
        # the server maps that to {"error": ...} with status 200, not 500.
        self.assertEqual(status, 200)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
