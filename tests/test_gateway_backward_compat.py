"""Backward-compatibility regression tests for gateway.py, written alongside
Phase 1 (router_policy.py) specifically because Phase 2+ will soon start
touching this exact dispatch logic — this freezes today's behavior first.

Run via:
    python3 -m unittest discover -s tests -v
from the ai-router repo root.

Uses AI_GATEWAY_CONFIG pointed at a nonexistent path in every test so the
real ~/Projects/ai-router/.env (with live secrets) never leaks into test
behavior — tests are fully hermetic.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NONEXISTENT_ENV = "/nonexistent/ai-router-test.env"


def _load_gateway_module():
    path = os.path.join(_REPO_ROOT, "gateway.py")
    spec = importlib.util.spec_from_file_location("_test_gateway", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_gateway"] = module
    spec.loader.exec_module(module)
    return module


class TestBackwardCompatibility(unittest.TestCase):

    def setUp(self):
        self.gw = _load_gateway_module()

    def test_intelligent_routing_off_by_default(self):
        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AI_ROUTER_INTELLIGENT_ROUTING", None)
            g = self.gw.AIGateway()
        self.assertFalse(g._intelligent_routing)
        self.assertIsNone(g._router_policy)

    def test_cloud_providers_and_weights_unaffected_by_flag(self):
        base = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "OPENROUTER_API_KEY": "sk-or-test",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, {**base, "AI_ROUTER_INTELLIGENT_ROUTING": "false"}, clear=False):
            g_off = self.gw.AIGateway()
        with patch.dict(os.environ, {**base, "AI_ROUTER_INTELLIGENT_ROUTING": "true"}, clear=False):
            g_on = self.gw.AIGateway()
        self.assertEqual(g_off._cloud_providers(), g_on._cloud_providers())
        self.assertEqual(g_off.weights, g_on.weights)
        self.assertEqual(g_off.models, g_on.models)

    def test_local_tier_still_tried_first_regardless_of_flag(self):
        sentinel = self.gw.RouterResponse(text="local-served", provider="local", model="x")
        for flag in ("false", "true"):
            with self.subTest(flag=flag):
                env = {
                    "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                    "AI_ROUTER_INTELLIGENT_ROUTING": flag,
                    "USE_LOCAL_SLM": "true",
                    "LOCAL_INTEL_URL": "http://localhost:11435",
                }
                with patch.dict(os.environ, env, clear=False):
                    g = self.gw.AIGateway()
                g._call_local = lambda *a, **k: sentinel
                result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
                self.assertIs(result, sentinel)

    def test_policy_exception_does_not_block_tool_calling(self):
        # Constraint #3 (do not break tool calling): a raising RouterPolicy
        # must never prevent a real chat() call — including one with tools —
        # from reaching the provider.
        captured = {}

        def fake_post_json(url, payload, headers, timeout=60):
            captured["payload"] = payload
            return {"content": [{"type": "text", "text": "ok"}], "usage": {"output_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "USE_LOCAL_SLM": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()

        self.assertIsNotNone(g._router_policy, "router_policy.py should have loaded successfully")
        g._router_policy.decide = lambda ctx: (_ for _ in ()).throw(RuntimeError("boom"))
        self.gw._post_json = fake_post_json

        tools = [{"name": "search_web", "description": "x", "input_schema": {"type": "object", "properties": {}}}]
        result = g.chat(
            messages=[self.gw.ChatMessage(role="user", content="hi")],
            provider="anthropic",
            tools=tools,
        )
        self.assertEqual(result.text, "ok")
        self.assertEqual(captured["payload"].get("tools"), tools)

    def test_router_policy_loads_when_flag_enabled(self):
        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.assertIsNotNone(g._router_policy)
        self.assertEqual(type(g._router_policy).__name__, "DeterministicRouterPolicy")

    def test_router_response_unchanged_shape(self):
        # RouterResponse must not gain new required fields in Phase 1 —
        # existing keyword-argument call sites across gateway.py must still work.
        r = self.gw.RouterResponse(text="hi", provider="local", model="x")
        self.assertEqual(r.tokens, None)

    def test_domain_mode_override_does_not_change_local_call(self):
        # Phase 2: AI_ROUTER_DOMAIN_MODE defaults to "override" — _call_local
        # must receive domain_override=None (falls back to LOCAL_INTEL_DOMAIN
        # internally), byte-identical to pre-Phase-2 behavior, even with
        # intelligent routing on and text that would classify differently.
        captured = {}

        def fake_call_local(messages, system, max_tokens, model_hint="structured", domain_override=None, bypass_smart_skip=False):
            captured["domain_override"] = domain_override
            return self.gw.RouterResponse(text="ok", provider="local", model="x")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_DOMAIN_MODE": "override",
            "LOCAL_INTEL_DOMAIN": "health",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._call_local = fake_call_local
        g.chat(messages=[self.gw.ChatMessage(role="user", content="check inventory sku for this supplier")])
        self.assertIsNone(captured["domain_override"])

    def test_domain_mode_hint_reclassifies_and_reaches_local_call(self):
        # The one genuine new behavior Phase 2 adds: with AI_ROUTER_DOMAIN_MODE
        # set to "hint" and text that clearly classifies to a different domain
        # than the LOCAL_INTEL_DOMAIN hint, _call_local receives the
        # classified domain — verified end-to-end through chat(), not just
        # inside router_policy.py's own unit tests.
        captured = {}

        def fake_call_local(messages, system, max_tokens, model_hint="structured", domain_override=None, bypass_smart_skip=False):
            captured["domain_override"] = domain_override
            return self.gw.RouterResponse(text="ok", provider="local", model="x")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_DOMAIN_MODE": "hint",
            "LOCAL_INTEL_DOMAIN": "health",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._call_local = fake_call_local
        g.chat(messages=[self.gw.ChatMessage(role="user", content="what's my portfolio dividend this month")])
        self.assertEqual(captured["domain_override"], "finance")

    def test_route_gating_off_smart_hint_still_hard_skips_local(self):
        # Phase 3: with AI_ROUTER_ROUTE_GATING off (default), a model_hint="smart"
        # call must still hard-skip local exactly as before Phase 3 — even with
        # intelligent routing on. Uses the REAL _call_local (not mocked) so its
        # actual internal model_hint=="smart" hard-skip is what's exercised —
        # it returns None immediately, before any network attempt, so no local
        # server needs to be reachable for this test.
        def fake_post_json(url, payload, headers, timeout=60):
            return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_ROUTE_GATING": "false",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",   # disables the direct Vertex path (avoids a real network call)
            "AI_ROUTER_WEIGHT_GEMINI": "0",   # excludes gemini from weighted selection entirely (OpenRouter-BYOK fallback would otherwise still let it be picked non-deterministically)
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.gw._post_json = fake_post_json
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")], model_hint="smart")
        self.assertEqual(result.text, "cloud")

    def test_route_gating_on_local_route_bypasses_smart_skip(self):
        # Phase 3's one genuine new behavior: route_gating on, and a real
        # RoutingDecision says route="local" for a model_hint="smart" request
        # (using a custom low threshold) — _call_local IS attempted and its
        # bypass_smart_skip kwarg is True.
        captured = {}

        def fake_call_local(messages, system, max_tokens, model_hint="structured", domain_override=None, bypass_smart_skip=False):
            captured["bypass_smart_skip"] = bypass_smart_skip
            captured["called"] = True
            return self.gw.RouterResponse(text="local", provider="local", model="x")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_ROUTE_GATING": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        # Force a low threshold so route computes to "local" even for model_hint="smart".
        # thresholds fully replaces the default table (not a partial merge — matches
        # test_router_policy.py's documented behavior), so all three keys are given.
        g._router_policy = sys.modules["_ai_router_policy"].DeterministicRouterPolicy(
            thresholds={"fast": 0.70, "structured": 0.82, "smart": 0.01}
        )
        g._call_local = fake_call_local
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")], model_hint="smart")
        self.assertTrue(captured.get("called"))
        self.assertTrue(captured.get("bypass_smart_skip"))
        self.assertEqual(result.text, "local")

    def test_route_gating_on_frontier_route_skips_local_entirely(self):
        # route_gating on, decision.route="frontier" (default high thresholds,
        # placeholder predicted_local_success): _call_local must never be invoked.
        call_local_invoked = {"called": False}

        def fake_call_local(*a, **k):
            call_local_invoked["called"] = True
            return self.gw.RouterResponse(text="local", provider="local", model="x")

        def fake_post_json(url, payload, headers, timeout=60):
            return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_ROUTE_GATING": "true",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "AI_ROUTER_WEIGHT_GEMINI": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._call_local = fake_call_local
        self.gw._post_json = fake_post_json
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        self.assertFalse(call_local_invoked["called"])
        self.assertEqual(result.text, "cloud")

    def test_complete_with_meta_normal_path_returns_tuple(self):
        def fake_post_json(url, payload, headers, timeout=60):
            return {"choices": [{"message": {"content": "hi"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "USE_LOCAL_SLM": "false",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "AI_ROUTER_WEIGHT_GEMINI": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.gw._post_json = fake_post_json
        result = g.complete_with_meta(system="s", user="u", task="cypher")
        self.assertEqual(result, ("hi", "openrouter", "openai/gpt-4o-mini"))

    def test_complete_with_meta_skip_local_bypasses_call_local(self):
        call_local_invoked = {"called": False}

        def fake_call_local(*a, **k):
            call_local_invoked["called"] = True
            return self.gw.RouterResponse(text="should not be used", provider="local", model="x")

        def fake_post_json(url, payload, headers, timeout=60):
            return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "AI_ROUTER_WEIGHT_GEMINI": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._call_local = fake_call_local
        self.gw._post_json = fake_post_json
        result = g.complete_with_meta(system="s", user="u", task="summarize", skip_local=True)
        self.assertFalse(call_local_invoked["called"])
        self.assertEqual(result[0], "cloud")

    def test_complete_still_returns_plain_text_via_delegation(self):
        def fake_post_json(url, payload, headers, timeout=60):
            return {"choices": [{"message": {"content": "plain text"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "USE_LOCAL_SLM": "false",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "AI_ROUTER_WEIGHT_GEMINI": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.gw._post_json = fake_post_json
        result = g.complete(system="s", user="u", task="general")
        self.assertEqual(result, "plain text")

    def test_verify_local_off_returns_local_response_even_if_it_would_fail_verification(self):
        # Phase 4: AI_ROUTER_VERIFY_LOCAL defaults off — the verifier never
        # loads, so a local response that WOULD fail verification (empty
        # text) is still returned as-is. Proves zero behavior change by default.
        def fake_call_local(*a, **k):
            return self.gw.RouterResponse(text="", provider="local", model="x")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.assertIsNone(g._response_verifier)
        g._call_local = fake_call_local
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        self.assertEqual(result.text, "")
        self.assertEqual(result.provider, "local")

    def test_verify_local_on_passing_verification_returns_local_response(self):
        def fake_call_local(*a, **k):
            return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_VERIFY_LOCAL": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.assertIsNotNone(g._response_verifier)
        g._call_local = fake_call_local
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        self.assertEqual(result.text, "a normal good answer")
        self.assertEqual(result.provider, "local")

    def test_verify_local_on_failing_verification_escalates_to_cloud(self):
        def fake_call_local(*a, **k):
            return self.gw.RouterResponse(text="", provider="local", model="x")   # empty -> rejected

        def fake_post_json(url, payload, headers, timeout=60):
            return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_VERIFY_LOCAL": "true",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "AI_ROUTER_WEIGHT_GEMINI": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._call_local = fake_call_local
        self.gw._post_json = fake_post_json
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        self.assertEqual(result.text, "cloud")
        self.assertEqual(result.provider, "openrouter")

    def test_verifier_exception_fails_open_returns_local_response(self):
        def fake_call_local(*a, **k):
            return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_VERIFY_LOCAL": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._response_verifier.evaluate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("verifier boom"))
        g._call_local = fake_call_local
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        self.assertEqual(result.text, "a normal good answer")

    def test_frontier_policy_defaults_to_weighted(self):
        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV}
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.assertEqual(g._frontier_policy, "weighted")
        self.assertIsNone(g._frontier_router)

    def test_frontier_capability_mode_picks_registry_informed_model(self):
        # Phase 5's one genuine new behavior: capability mode picks a specific
        # (provider, model) pair via CapabilityFrontierRouter instead of plain
        # weighted provider selection. gemini-2.5-flash has the lowest
        # cost+latency in the "structured" (fast) tier, so it should win over
        # openrouter's openai/gpt-4o-mini candidate.
        captured = {}

        def fake_post_json(url, payload, headers, timeout=60):
            captured["model"] = payload["model"]
            return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_FRONTIER_POLICY": "capability",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "USE_LOCAL_SLM": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.assertIsNotNone(g._frontier_router)
        self.gw._post_json = fake_post_json
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(captured["model"], "google/gemini-2.5-flash")

    def test_model_override_bypasses_capability_selection(self):
        captured = {}

        def fake_post_json(url, payload, headers, timeout=60):
            captured["model"] = payload["model"]
            return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_FRONTIER_POLICY": "capability",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "AI_ROUTER_WEIGHT_GEMINI": "0",
            "USE_LOCAL_SLM": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.gw._post_json = fake_post_json
        result = g.chat(
            messages=[self.gw.ChatMessage(role="user", content="hi")],
            model_override="my-custom-model",
        )
        self.assertEqual(captured["model"], "my-custom-model")

    def test_fallback_after_primary_failure_uses_hint_based_model_not_capability_pick(self):
        calls = []

        def fake_call_cloud(provider, messages, system, model_hint, max_tokens, override, tools=None):
            calls.append((provider, override))
            if provider == "gemini":
                raise RuntimeError("simulated primary failure")
            return self.gw.RouterResponse(text="fallback ok", provider=provider, model=override or "hint-model")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_FRONTIER_POLICY": "capability",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "USE_LOCAL_SLM": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._call_cloud = fake_call_cloud
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")], retries=1)
        self.assertEqual(result.text, "fallback ok")
        self.assertEqual(calls[0][0], "gemini")
        self.assertIsNotNone(calls[0][1])       # primary attempt: capability_override was passed
        self.assertEqual(calls[1][0], "openrouter")
        self.assertIsNone(calls[1][1])          # fallback attempt: uses model_override (None), not the capability pick

    def test_experience_store_off_by_default_no_file_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            trace_path = f"{d}/traces.jsonl"

            def fake_call_local(*a, **k):
                return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            self.assertIsNone(g._experience_store)
            g._call_local = fake_call_local
            result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            self.assertEqual(result.text, "a normal good answer")
            self.assertFalse(os.path.exists(trace_path))

    def test_experience_store_on_local_accepted_writes_one_record(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"

            def fake_call_local(*a, **k):
                return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            self.assertIsNotNone(g._experience_store)
            g._experience_store.path = trace_path   # redirect to a temp file for this test
            g._call_local = fake_call_local
            result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            self.assertEqual(result.text, "a normal good answer")
            lines = trace_path.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["final_source"], "local")
            self.assertFalse(parsed["escalated"])

    def test_experience_store_on_rejected_then_escalated_writes_one_record(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"

            def fake_call_local(*a, **k):
                return self.gw.RouterResponse(text="", provider="local", model="x")   # empty -> rejected

            def fake_post_json(url, payload, headers, timeout=60):
                return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_VERIFY_LOCAL": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
                "OPENROUTER_API_KEY": "sk-or-test",
                "GEMINI_ENABLED": "false",
                "AI_ROUTER_WEIGHT_GEMINI": "0",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            g._experience_store.path = trace_path
            g._call_local = fake_call_local
            self.gw._post_json = fake_post_json
            result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            self.assertEqual(result.text, "cloud")
            lines = trace_path.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["final_source"], "frontier")
            self.assertTrue(parsed["escalated"])
            self.assertIn("verification_rejected", parsed["escalation_reason"])

    def test_experience_store_write_exception_does_not_break_response(self):
        def fake_call_local(*a, **k):
            return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_EXPERIENCE_STORE": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._experience_store.record = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
        g._call_local = fake_call_local
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        self.assertEqual(result.text, "a normal good answer")

    def test_experience_redact_mode_unset_defaults_to_full(self):
        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV}
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.assertEqual(g._experience_redact_mode, "full")

    def test_experience_redact_true_false_map_to_full_none(self):
        env_true = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "AI_ROUTER_EXPERIENCE_REDACT": "true"}
        env_false = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "AI_ROUTER_EXPERIENCE_REDACT": "false"}
        with patch.dict(os.environ, env_true, clear=False):
            g_true = self.gw.AIGateway()
        with patch.dict(os.environ, env_false, clear=False):
            g_false = self.gw.AIGateway()
        self.assertEqual(g_true._experience_redact_mode, "full")
        self.assertEqual(g_false._experience_redact_mode, "none")

    def test_experience_redact_partial_writes_pii_scrubbed_content(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"

            def fake_call_local(*a, **k):
                return self.gw.RouterResponse(text="your balance is $50,000", provider="local", model="x")

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
                "AI_ROUTER_EXPERIENCE_REDACT": "partial",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            self.assertEqual(g._experience_redact_mode, "partial")
            g._experience_store.path = trace_path
            g._call_local = fake_call_local
            result = g.chat(messages=[self.gw.ChatMessage(role="user", content="what's my balance?")])
            self.assertEqual(result.text, "your balance is $50,000")   # the real response is untouched
            parsed = json.loads(trace_path.read_text().splitlines()[0])
            self.assertIn("REDACTED_AMOUNT", parsed["local_response"])
            self.assertNotIn("50,000", parsed["local_response"])
            self.assertNotIn("<redacted len=", parsed["local_response"])   # not full-mode hash either

    # ── Phase 8: live metrics + feedback loop ────────────────────────────

    def test_metrics_disabled_when_intelligent_routing_off(self):
        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AI_ROUTER_INTELLIGENT_ROUTING", None)
            g = self.gw.AIGateway()
        self.assertIsNone(g._metrics)
        self.assertEqual(g.get_metrics_snapshot(), {"enabled": False})
        self.assertFalse(g.record_feedback("some-trace-id", "incorrect"))

    def test_metrics_enabled_without_experience_store(self):
        def fake_call_local(*a, **k):
            return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.assertIsNotNone(g._metrics)       # metrics need only intelligent_routing
        self.assertIsNone(g._experience_store)  # store stays off
        g._call_local = fake_call_local
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        self.assertIsNone(result.trace_id)     # no store -> no dangling trace_id
        snap = g.get_metrics_snapshot()
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["total_requests"], 1)
        self.assertEqual(snap["route_counts"]["local"], 1)
        # feedback still needs the experience store specifically
        self.assertFalse(g.record_feedback("some-trace-id", "incorrect"))

    def test_trace_id_populated_and_feedback_recorded_when_experience_store_on(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"
            feedback_path = Path(d) / "feedback.jsonl"

            def fake_call_local(*a, **k):
                return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            self.assertIsNotNone(g._feedback_store)
            g._experience_store.path = trace_path
            g._feedback_store.path = feedback_path
            g._call_local = fake_call_local

            result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            self.assertIsNotNone(result.trace_id)
            trace = json.loads(trace_path.read_text().splitlines()[0])
            self.assertEqual(trace["trace_id"], result.trace_id)

            self.assertTrue(g.record_feedback(result.trace_id, "incorrect", note="wrong answer"))
            self.assertTrue(g.record_feedback(result.trace_id, "correct"))   # changed their mind — both preserved
            fb_lines = [json.loads(l) for l in feedback_path.read_text().splitlines()]
            self.assertEqual(len(fb_lines), 2)
            self.assertEqual(fb_lines[0]["trace_id"], result.trace_id)
            self.assertEqual(fb_lines[0]["label"], "incorrect")
            self.assertEqual(fb_lines[1]["label"], "correct")

    def test_record_feedback_invalid_label_returns_false(self):
        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_EXPERIENCE_STORE": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.assertFalse(g.record_feedback("some-trace-id", "maybe"))

    def test_metrics_snapshot_reflects_accept_and_escalate_mix(self):
        call_count = {"n": 0}

        def fake_call_local(*a, **k):
            call_count["n"] += 1
            # first call: good response (accepted); second: empty (rejected -> escalates)
            text = "a good answer" if call_count["n"] == 1 else ""
            return self.gw.RouterResponse(text=text, provider="local", model="x")

        def fake_post_json(url, payload, headers, timeout=60):
            return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_VERIFY_LOCAL": "true",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_ENABLED": "false",
            "AI_ROUTER_WEIGHT_GEMINI": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        g._call_local = fake_call_local
        self.gw._post_json = fake_post_json

        g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
        g.chat(messages=[self.gw.ChatMessage(role="user", content="hi again")])

        snap = g.get_metrics_snapshot()
        self.assertEqual(snap["total_requests"], 2)
        self.assertEqual(snap["route_counts"]["local"], 1)
        self.assertEqual(snap["route_counts"]["frontier"], 1)
        self.assertEqual(snap["verifier_counts"]["accepted"], 1)
        self.assertEqual(snap["verifier_counts"]["rejected"], 1)
        self.assertEqual(snap["by_domain"]["general"]["total"], 2)
        self.assertEqual(snap["by_domain"]["general"]["local_accepted"], 1)
        self.assertEqual(snap["by_domain"]["general"]["escalated"], 1)
        self.assertIsNotNone(snap["local_latency_ms"]["p50"])
        self.assertIsNotNone(snap["frontier_latency_ms"]["p50"])

    # ── Phase 9: versioning + canary tagging ─────────────────────────────

    def test_policy_id_verifier_id_unset_use_default_constants(self):
        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_VERIFY_LOCAL": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("AI_ROUTER_POLICY_ID", None)
            os.environ.pop("AI_ROUTER_VERIFIER_ID", None)
            g = self.gw.AIGateway()
        import router_policy as rp
        self.assertEqual(g._router_policy.policy_id, rp.ROUTER_VERSION)
        self.assertEqual(g._response_verifier.verifier_id, rp.VERIFIER_VERSION)

    def test_policy_id_verifier_id_set_reflect_in_written_trace(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"

            def fake_call_local(*a, **k):
                return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_VERIFY_LOCAL": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
                "AI_ROUTER_POLICY_ID": "canary-lower-threshold",
                "AI_ROUTER_VERIFIER_ID": "canary-verifier",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            self.assertEqual(g._router_policy.policy_id, "canary-lower-threshold")
            self.assertEqual(g._response_verifier.verifier_id, "canary-verifier")
            g._experience_store.path = trace_path
            g._call_local = fake_call_local
            g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            trace = json.loads(trace_path.read_text().splitlines()[0])
            self.assertEqual(trace["routing_decision"]["router_version"], "canary-lower-threshold")
            self.assertEqual(trace["verification"]["verifier_version"], "canary-verifier")

    def test_adapter_version_in_written_trace_is_real_version_not_name(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"

            def fake_call_local(*a, **k):
                return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
                "LOCAL_INTEL_DOMAIN": "finance",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            g._experience_store.path = trace_path
            g._call_local = fake_call_local
            g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            trace = json.loads(trace_path.read_text().splitlines()[0])
            self.assertEqual(trace["routing_decision"]["domain"], "finance")
            self.assertEqual(trace["adapter_version"], "2026-08-22")   # DomainConfig.adapter_version, not "finance" (the name)
            self.assertEqual(trace["model_versions"]["local"], "2026-08-22")

    def test_frontier_router_version_none_when_local_succeeds(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"

            def fake_call_local(*a, **k):
                return self.gw.RouterResponse(text="a normal good answer", provider="local", model="x")

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            g._experience_store.path = trace_path
            g._call_local = fake_call_local
            g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            trace = json.loads(trace_path.read_text().splitlines()[0])
            self.assertIsNone(trace["frontier_router_version"])

    def test_frontier_router_version_weighted_on_default_frontier_call(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"

            def fake_post_json(url, payload, headers, timeout=60):
                return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
                "OPENROUTER_API_KEY": "sk-or-test",
                "GEMINI_ENABLED": "false",
                "AI_ROUTER_WEIGHT_GEMINI": "0",
                "USE_LOCAL_SLM": "false",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            g._experience_store.path = trace_path
            self.gw._post_json = fake_post_json
            g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            trace = json.loads(trace_path.read_text().splitlines()[0])
            import router_policy as rp
            self.assertEqual(trace["frontier_router_version"], rp.WEIGHTED_FRONTIER_VERSION)

    def test_frontier_router_version_capability_on_capability_pick(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "traces.jsonl"

            def fake_post_json(url, payload, headers, timeout=60):
                return {"choices": [{"message": {"content": "cloud"}}], "usage": {"completion_tokens": 1}}

            env = {
                "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
                "AI_ROUTER_INTELLIGENT_ROUTING": "true",
                "AI_ROUTER_EXPERIENCE_STORE": "true",
                "AI_ROUTER_FRONTIER_POLICY": "capability",
                "OPENROUTER_API_KEY": "sk-or-test",
                "GEMINI_ENABLED": "false",
                "USE_LOCAL_SLM": "false",
            }
            with patch.dict(os.environ, env, clear=False):
                g = self.gw.AIGateway()
            g._experience_store.path = trace_path
            self.gw._post_json = fake_post_json
            g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")])
            trace = json.loads(trace_path.read_text().splitlines()[0])
            import router_policy as rp
            self.assertEqual(trace["frontier_router_version"], rp.FRONTIER_ROUTER_VERSION)

    # ── Phase 10: remote-forwarding surface (server.py's real behavior is
    # covered separately in test_server.py, against a real subprocess) ────

    def test_get_json_helper_parses_response(self):
        # Mirrors _post_json's existing test coverage pattern — mock at the
        # urllib.request.urlopen level, no real network call.
        import io
        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV}
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()

        class _FakeResp:
            def read(self_inner):
                return b'{"ok": true, "n": 3}'
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False

        with patch.object(self.gw.urllib.request, "urlopen", return_value=_FakeResp()):
            result = self.gw._get_json("http://localhost:7861/v1/metrics")
        self.assertEqual(result, {"ok": True, "n": 3})

    def test_complete_with_meta_skip_local_forwards_when_remote_url_set(self):
        captured = {}

        def fake_post_json(url, payload, headers, timeout=60):
            captured["url"] = url
            captured["payload"] = payload
            return {"text": "remote answer", "provider": "anthropic", "model": "claude-haiku-4-5"}

        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "REMOTE_GATEWAY_URL": "http://localhost:7861"}
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.gw._post_json = fake_post_json
        result = g.complete_with_meta(system="s", user="u", task="cypher", skip_local=True)
        self.assertEqual(result, ("remote answer", "anthropic", "claude-haiku-4-5"))
        self.assertEqual(captured["url"], "http://localhost:7861/v1/complete_with_meta")
        self.assertTrue(captured["payload"]["skip_local"])

    def test_complete_with_meta_skip_local_unaffected_when_remote_url_unset(self):
        # Regression: skip_local's local-dispatch path must be untouched
        # when not in remote mode.
        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "ANTHROPIC_API_KEY": "sk-ant-test"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("REMOTE_GATEWAY_URL", None)
            g = self.gw.AIGateway()

        def fake_call_cloud(prov, msgs, system, hint, max_tokens, override, tools=None):
            return self.gw.RouterResponse(text="local cloud answer", provider=prov, model="x")
        g._call_cloud = fake_call_cloud
        result = g.complete_with_meta(system="s", user="u", skip_local=True)
        self.assertEqual(result[0], "local cloud answer")

    def test_complete_with_meta_skip_local_none_on_remote_error_response(self):
        def fake_post_json(url, payload, headers, timeout=60):
            return {"error": "complete_with_meta() returned None (total failure)"}

        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "REMOTE_GATEWAY_URL": "http://localhost:7861"}
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.gw._post_json = fake_post_json
        result = g.complete_with_meta(system="s", user="u", skip_local=True)
        self.assertIsNone(result)

    def test_record_feedback_forwards_when_remote_url_set(self):
        captured = {}

        def fake_post_json(url, payload, headers, timeout=60):
            captured["url"] = url
            captured["payload"] = payload
            return {"ok": True}

        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "REMOTE_GATEWAY_URL": "http://localhost:7861"}
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.gw._post_json = fake_post_json
        result = g.record_feedback("trace-123", "incorrect", note="wrong")
        self.assertTrue(result)
        self.assertEqual(captured["url"], "http://localhost:7861/v1/record_feedback")
        self.assertEqual(captured["payload"]["trace_id"], "trace-123")

    def test_record_feedback_unaffected_when_remote_url_unset(self):
        # Regression: local record_feedback (Phase 8 behavior) is untouched.
        env = {
            "AI_GATEWAY_CONFIG": _NONEXISTENT_ENV,
            "AI_ROUTER_INTELLIGENT_ROUTING": "true",
            "AI_ROUTER_EXPERIENCE_STORE": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("REMOTE_GATEWAY_URL", None)
            g = self.gw.AIGateway()
        self.assertIsNotNone(g._feedback_store)
        g._feedback_store.record = lambda *a, **k: None   # avoid a real file write
        self.assertTrue(g.record_feedback("trace-1", "correct"))

    def test_get_metrics_snapshot_forwards_when_remote_url_set(self):
        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "REMOTE_GATEWAY_URL": "http://localhost:7861"}
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()

        class _FakeResp:
            def read(self_inner):
                return b'{"enabled": true, "total_requests": 5}'
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False

        with patch.object(self.gw.urllib.request, "urlopen", return_value=_FakeResp()):
            snap = g.get_metrics_snapshot()
        self.assertEqual(snap, {"enabled": True, "total_requests": 5})

    def test_get_metrics_snapshot_unaffected_when_remote_url_unset(self):
        # Regression: local metrics (Phase 8 behavior) is untouched.
        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "AI_ROUTER_INTELLIGENT_ROUTING": "true"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("REMOTE_GATEWAY_URL", None)
            g = self.gw.AIGateway()
        snap = g.get_metrics_snapshot()
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["total_requests"], 0)

    def test_pinned_provider_call_forwards_when_remote_url_set(self):
        # Confirms chat()'s remote-forward check runs BEFORE the provider-pin
        # check — a pinned-provider caller (e.g. portfolio_tracker's
        # stock_intelligence.py) gets remoted correctly once migrated.
        captured = {}

        def fake_post_json(url, payload, headers, timeout=60):
            captured["url"] = url
            captured["payload"] = payload
            return {"text": "remote", "provider": "anthropic", "model": "x", "tokens": 1}

        env = {"AI_GATEWAY_CONFIG": _NONEXISTENT_ENV, "REMOTE_GATEWAY_URL": "http://localhost:7861"}
        with patch.dict(os.environ, env, clear=False):
            g = self.gw.AIGateway()
        self.gw._post_json = fake_post_json
        result = g.chat(messages=[self.gw.ChatMessage(role="user", content="hi")], provider="anthropic")
        self.assertEqual(result.text, "remote")
        self.assertEqual(captured["payload"]["provider"], "anthropic")


if __name__ == "__main__":
    unittest.main()
