"""Tests for router_policy.py (Phase 1). Run via:
    python3 -m unittest discover -s tests -v
from the ai-router repo root.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import router_policy as rp


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


def _ctx(**overrides):
    defaults = dict(
        messages=[_Msg("user", "hello")],
        system="You are a helpful assistant.",
        model_hint="structured",
    )
    defaults.update(overrides)
    return rp.RequestContext(**defaults)


def _decision(**overrides):
    defaults = dict(
        domain="general", subdomain=None, task="unclassified",
        complexity=0.1, risk=0.0, predicted_local_success=0.5,
        selected_adapter=None, required_capabilities=[],
        quality_threshold=0.7, route="frontier", reason="test",
        router_version="test",
    )
    defaults.update(overrides)
    return rp.RoutingDecision(**defaults)


class TestDeterministicRouterPolicy(unittest.TestCase):

    def test_threshold_table_by_model_hint(self):
        policy = rp.DeterministicRouterPolicy()
        expected = {"fast": 0.70, "structured": 0.82, "smart": 0.95}
        for hint, threshold in expected.items():
            with self.subTest(model_hint=hint):
                decision = policy.decide(_ctx(model_hint=hint))
                self.assertEqual(decision.quality_threshold, threshold)

    def test_domain_hint_respected_when_present(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx(app_domain_hint="finance"))
        self.assertEqual(decision.domain, "finance")
        self.assertEqual(decision.selected_adapter, "finance")

    def test_domain_defaults_to_general_when_absent(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx(app_domain_hint=None))
        self.assertEqual(decision.domain, "general")

    def test_route_is_frontier_given_placeholder_success_rate(self):
        # predicted_local_success is a flat 0.5 constant in Phase 1, and every
        # default threshold is >= 0.70, so route should always be "frontier"
        # until Phase 2/7 make predicted_local_success meaningful.
        policy = rp.DeterministicRouterPolicy()
        for hint in ("fast", "structured", "smart"):
            with self.subTest(model_hint=hint):
                decision = policy.decide(_ctx(model_hint=hint))
                self.assertEqual(decision.route, "frontier")

    def test_route_flips_to_local_when_threshold_is_low_enough(self):
        # With a custom threshold below the 0.5 placeholder, route should
        # compute to "local" — proves the comparison logic itself is sound,
        # independent of the placeholder constant's specific value.
        policy = rp.DeterministicRouterPolicy(thresholds={"structured": 0.1})
        decision = policy.decide(_ctx(model_hint="structured"))
        self.assertEqual(decision.route, "local")

    def test_reason_states_actual_numbers(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx(model_hint="smart"))
        self.assertIn("0.50", decision.reason)
        self.assertIn("0.95", decision.reason)
        self.assertIn("smart", decision.reason)

    def test_required_capabilities_uses_named_constant(self):
        policy = rp.DeterministicRouterPolicy()
        with_tools = policy.decide(_ctx(tools=[{"name": "search"}]))
        without_tools = policy.decide(_ctx(tools=None))
        self.assertEqual(with_tools.required_capabilities, [rp.REQUIRES_TOOLS])
        self.assertEqual(without_tools.required_capabilities, [])

    def test_custom_thresholds_override_default_table(self):
        policy = rp.DeterministicRouterPolicy(thresholds={"fast": 0.01, "structured": 0.02, "smart": 0.03})
        decision = policy.decide(_ctx(model_hint="fast"))
        self.assertEqual(decision.quality_threshold, 0.01)
        self.assertEqual(decision.route, "local")   # 0.5 placeholder >= 0.01

    def test_subdomain_and_task_are_documented_placeholders(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx())
        self.assertIsNone(decision.subdomain)
        self.assertEqual(decision.task, "unclassified")

    def test_router_policy_protocol_isinstance(self):
        policy = rp.DeterministicRouterPolicy()
        self.assertIsInstance(policy, rp.RouterPolicy)

    def test_adapter_version_matches_domain_config_not_adapter_name(self):
        # Phase 9 bug fix regression: adapter_version must be the real
        # version string (DomainConfig.adapter_version), never the adapter
        # NAME (selected_adapter already covers that).
        policy = rp.DeterministicRouterPolicy()
        for domain, expected_version in (("retail", "v3"), ("finance", "2026-08-22"), ("health", "2026-09-15")):
            with self.subTest(domain=domain):
                decision = policy.decide(_ctx(app_domain_hint=domain, domain_mode="override"))
                self.assertEqual(decision.adapter_version, expected_version)
                self.assertNotEqual(decision.adapter_version, decision.selected_adapter)

    def test_adapter_version_none_for_domain_with_no_adapter(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx(app_domain_hint="general"))
        self.assertIsNone(decision.adapter_version)

    def test_policy_id_recorded_as_router_version(self):
        policy = rp.DeterministicRouterPolicy(policy_id="canary-lower-threshold")
        decision = policy.decide(_ctx())
        self.assertEqual(decision.router_version, "canary-lower-threshold")

    def test_policy_id_unset_falls_back_to_router_version_constant(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx())
        self.assertEqual(decision.router_version, rp.ROUTER_VERSION)


class TestDomainRegistry(unittest.TestCase):
    """Phase 2: config-driven domain registry + classification."""

    def test_default_registry_has_real_data(self):
        registry = rp.load_domain_registry()
        self.assertEqual(set(registry.keys()), {"retail", "finance", "health", "general"})
        self.assertEqual(registry["retail"].benchmark_success_rate, 0.60)
        self.assertEqual(registry["finance"].benchmark_success_rate, 0.87)
        self.assertEqual(registry["health"].benchmark_success_rate, 0.55)
        self.assertIsNone(registry["general"].benchmark_success_rate)

    def test_load_domain_registry_falls_back_when_no_yaml_file(self):
        # Covers all 3 real deployments today, which won't have config/domains.yaml.
        registry = rp.load_domain_registry(path=Path("/nonexistent/domains.yaml"))
        self.assertEqual(registry, rp.DEFAULT_DOMAIN_REGISTRY)

    def test_load_domain_registry_survives_malformed_yaml(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("domains: [this is not valid: yaml: at all: -")
            bad_path = Path(f.name)
        try:
            registry = rp.load_domain_registry(path=bad_path)
            # Must never raise, and must fall back to the real hardcoded data.
            self.assertEqual(registry["retail"].benchmark_success_rate, 0.60)
        finally:
            bad_path.unlink()

    def test_classify_domain_matches_keywords(self):
        registry = rp.DEFAULT_DOMAIN_REGISTRY
        self.assertEqual(rp._classify_domain("what's my portfolio dividend this month", registry), "finance")
        self.assertEqual(rp._classify_domain("check inventory sku for this supplier", registry), "retail")
        self.assertEqual(rp._classify_domain("patient has a cardiology appointment", registry), "health")

    def test_classify_domain_returns_none_when_no_match(self):
        registry = rp.DEFAULT_DOMAIN_REGISTRY
        self.assertIsNone(rp._classify_domain("what's the weather like today", registry))


class TestDomainMode(unittest.TestCase):
    """Phase 2: AI_ROUTER_DOMAIN_MODE semantics (override/hint/auto)."""

    def test_override_mode_ignores_classification(self):
        # domain_mode="override" (the default) must behave exactly like Phase 1:
        # the hint always wins, even when the text would classify differently.
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx(
            app_domain_hint="health",
            messages=[_Msg("user", "check inventory sku for this supplier")],   # retail-flavored text
            domain_mode="override",
        )
        decision = policy.decide(ctx)
        self.assertEqual(decision.domain, "health")

    def test_hint_mode_prefers_classification_over_hint(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx(
            app_domain_hint="health",
            messages=[_Msg("user", "what's my portfolio dividend this month")],   # finance-flavored text
            domain_mode="hint",
        )
        decision = policy.decide(ctx)
        self.assertEqual(decision.domain, "finance")

    def test_hint_mode_falls_back_to_hint_when_no_classification_match(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx(
            app_domain_hint="health",
            messages=[_Msg("user", "what's the weather like today")],   # matches nothing
            domain_mode="hint",
        )
        decision = policy.decide(ctx)
        self.assertEqual(decision.domain, "health")

    def test_auto_mode_ignores_hint_entirely(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx(
            app_domain_hint="health",
            messages=[_Msg("user", "check inventory sku for this supplier")],   # retail-flavored text
            domain_mode="auto",
        )
        decision = policy.decide(ctx)
        self.assertEqual(decision.domain, "retail")

    def test_auto_mode_falls_back_to_general_when_no_match(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx(
            app_domain_hint="health",
            messages=[_Msg("user", "what's the weather like today")],
            domain_mode="auto",
        )
        decision = policy.decide(ctx)
        self.assertEqual(decision.domain, "general")

    def test_predicted_local_success_uses_benchmark_when_available(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx(app_domain_hint="finance", domain_mode="override"))
        self.assertEqual(decision.predicted_local_success, 0.87)
        self.assertIn("benchmark", decision.reason)

    def test_predicted_local_success_falls_back_to_placeholder_for_general(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx(app_domain_hint=None, domain_mode="override"))
        self.assertEqual(decision.predicted_local_success, 0.5)
        self.assertIn("placeholder", decision.reason)

    def test_selected_adapter_uses_real_adapter_name(self):
        policy = rp.DeterministicRouterPolicy()
        decision = policy.decide(_ctx(app_domain_hint="retail", domain_mode="override"))
        self.assertEqual(decision.selected_adapter, "retail_v3")


class TestForceFrontier(unittest.TestCase):
    """Phase 3: force_frontier overrides route regardless of predicted success/threshold."""

    def test_force_frontier_forces_frontier_route(self):
        # Use a threshold low enough that route would normally be "local".
        policy = rp.DeterministicRouterPolicy(thresholds={"structured": 0.01})
        decision = policy.decide(_ctx(model_hint="structured", force_frontier=True))
        self.assertEqual(decision.route, "frontier")
        self.assertIn("force_frontier", decision.reason)

    def test_without_force_frontier_same_conditions_route_local(self):
        # Sanity check: without force_frontier, the same low threshold does
        # route local — proves force_frontier is what's flipping it above.
        policy = rp.DeterministicRouterPolicy(thresholds={"structured": 0.01})
        decision = policy.decide(_ctx(model_hint="structured", force_frontier=False))
        self.assertEqual(decision.route, "local")


class TestDeterministicVerifier(unittest.TestCase):
    """Phase 4: deterministic, domain-agnostic response verification."""

    def test_normal_response_accepted(self):
        v = rp.DeterministicVerifier()
        result = v.evaluate(None, None, "0 orders from Encompass in May 2026")
        self.assertTrue(result.accepted)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.failure_reasons, [])

    def test_empty_response_rejected(self):
        v = rp.DeterministicVerifier()
        for text in ("", "   ", None):
            with self.subTest(text=repr(text)):
                result = v.evaluate(None, None, text)
                self.assertFalse(result.accepted)
                self.assertIn("empty response", result.failure_reasons)

    def test_refusal_phrase_rejected(self):
        v = rp.DeterministicVerifier()
        result = v.evaluate(None, None, "I'm sorry, but I cannot help with that request.")
        self.assertFalse(result.accepted)
        self.assertIn("refusal phrase detected", result.failure_reasons)

    def test_error_artifact_rejected(self):
        v = rp.DeterministicVerifier()
        result = v.evaluate(None, None, "Traceback (most recent call last):\n  File ...")
        self.assertFalse(result.accepted)
        self.assertIn("error/exception artifact detected", result.failure_reasons)

    def test_confidence_is_capped_not_full(self):
        # A passing check means "no red flags found," not "this is correct" —
        # confidence must never read as 1.0 (see module docstring).
        v = rp.DeterministicVerifier()
        result = v.evaluate(None, None, "a perfectly normal answer")
        self.assertEqual(result.confidence, rp._DETERMINISTIC_CONFIDENCE)
        self.assertLess(result.confidence, 1.0)

    def test_groundedness_and_completeness_stay_undetermined(self):
        v = rp.DeterministicVerifier()
        result = v.evaluate(None, None, "a perfectly normal answer")
        self.assertIsNone(result.groundedness)
        self.assertIsNone(result.completeness)

    def test_response_verifier_protocol_isinstance(self):
        v = rp.DeterministicVerifier()
        self.assertIsInstance(v, rp.ResponseVerifier)

    def test_verifier_id_recorded_as_verifier_version(self):
        v = rp.DeterministicVerifier(verifier_id="canary-verifier")
        result = v.evaluate(None, None, "a normal answer")
        self.assertEqual(result.verifier_version, "canary-verifier")

    def test_verifier_id_unset_falls_back_to_verifier_version_constant(self):
        v = rp.DeterministicVerifier()
        result = v.evaluate(None, None, "a normal answer")
        self.assertEqual(result.verifier_version, rp.VERIFIER_VERSION)

    def test_does_not_catch_semantically_wrong_but_well_formed_answer(self):
        # Documents the known, deliberate limitation (see module docstring):
        # a confident, well-formed, non-refusal answer that's just wrong
        # passes deterministic checks — this is expected, not a bug.
        v = rp.DeterministicVerifier()
        result = v.evaluate(None, None, "0 orders from Encompass in May 2026")
        self.assertTrue(result.accepted)


class TestModelRegistry(unittest.TestCase):
    """Phase 5: config-driven model capability registry."""

    def test_default_registry_has_expected_entries(self):
        self.assertEqual(len(rp.DEFAULT_MODEL_REGISTRY), 8)
        fast = rp.DEFAULT_MODEL_REGISTRY[("gemini", "gemini-2.5-flash")]
        smart = rp.DEFAULT_MODEL_REGISTRY[("gemini", "gemini-2.5-pro")]
        self.assertEqual(fast.reasoning, 0.60)
        self.assertEqual(smart.reasoning, 0.85)

    def test_load_model_registry_falls_back_when_no_yaml_file(self):
        registry = rp.load_model_registry(path=Path("/nonexistent/models.yaml"))
        self.assertEqual(registry, rp.DEFAULT_MODEL_REGISTRY)

    def test_load_model_registry_survives_malformed_yaml(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("models: [this is not valid: yaml: at all: -")
            bad_path = Path(f.name)
        try:
            registry = rp.load_model_registry(path=bad_path)
            self.assertEqual(registry, rp.DEFAULT_MODEL_REGISTRY)
        finally:
            bad_path.unlink()


class TestCapabilityFrontierRouter(unittest.TestCase):
    """Phase 5: expected-utility scoring over a registry-informed candidate set."""

    def test_picks_cheapest_fastest_within_fast_tier(self):
        # Uniform tier placeholders mean capability score ties within a tier —
        # gemini-2.5-flash has the lowest cost AND lowest latency, so it must win.
        router = rp.CapabilityFrontierRouter()
        candidates = [
            ("openrouter", "openai/gpt-4o-mini"),
            ("gemini", "gemini-2.5-flash"),
            ("anthropic", "claude-haiku-4-5"),
        ]
        picked = router.select(_ctx(), _decision(), candidates)
        self.assertEqual(picked, ("gemini", "gemini-2.5-flash"))

    def test_respects_candidates_filter(self):
        router = rp.CapabilityFrontierRouter()
        # gemini-2.5-flash would normally win but isn't offered as a candidate.
        candidates = [("anthropic", "claude-haiku-4-5"), ("openai", "gpt-4o-mini")]
        picked = router.select(_ctx(), _decision(), candidates)
        self.assertIn(picked, candidates)
        self.assertNotEqual(picked, ("gemini", "gemini-2.5-flash"))

    def test_returns_none_when_no_candidate_in_registry(self):
        router = rp.CapabilityFrontierRouter()
        picked = router.select(_ctx(), _decision(), [("openrouter", "nonexistent-model")])
        self.assertIsNone(picked)

    def test_uses_tool_use_score_when_tools_required(self):
        router = rp.CapabilityFrontierRouter()
        decision = _decision(required_capabilities=[rp.REQUIRES_TOOLS])
        # Uniform placeholders mean tool_use == reasoning/coding/structured_output
        # within a tier, so this should not raise and should still return a valid pick.
        picked = router.select(_ctx(), decision, [("openai", "gpt-4o-mini"), ("gemini", "gemini-2.5-flash")])
        self.assertIsNotNone(picked)

    def test_frontier_router_protocol_isinstance(self):
        router = rp.CapabilityFrontierRouter()
        self.assertIsInstance(router, rp.FrontierRouter)


class TestExperienceStore(unittest.TestCase):
    """Phase 6: append-only trace of routed requests, redacted by default."""

    def test_experience_store_protocol_isinstance(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            store = rp.JSONLExperienceStore(path=Path(d) / "traces.jsonl")
            self.assertIsInstance(store, rp.ExperienceStore)

    def test_record_appends_valid_json_lines(self):
        import tempfile, json
        decision = _decision()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "traces.jsonl"
            store = rp.JSONLExperienceStore(path=path)
            trace1 = rp.build_experience_trace(context=_ctx(), decision=decision, final_source="local")
            trace2 = rp.build_experience_trace(context=_ctx(), decision=decision, final_source="frontier")
            store.record(trace1)
            store.record(trace2)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)   # appended, not overwritten
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["final_source"], "local")
            self.assertIn("trace_id", parsed)
            self.assertIn("routing_decision", parsed)   # nested dataclass serialized

    def test_record_never_raises_on_write_failure(self):
        decision = _decision()
        store = rp.JSONLExperienceStore(path=Path("/nonexistent_root_xyz/traces.jsonl"))
        trace = rp.build_experience_trace(context=_ctx(), decision=decision)
        store.record(trace)   # must not raise

    def test_redact_mode_full_hides_raw_text(self):
        decision = _decision()
        ctx = _ctx(messages=[_Msg("user", "my IRA balance is $50,000")])
        trace = rp.build_experience_trace(
            context=ctx, decision=decision,
            local_response="your IRA balance is $50,000",
            redact_mode="full",
        )
        self.assertNotIn("50,000", str(trace.request))
        self.assertNotIn("50,000", trace.local_response)
        self.assertIn("len=", trace.local_response)
        self.assertIn("message_count", trace.request)

    def test_redact_mode_none_preserves_raw_text(self):
        decision = _decision()
        ctx = _ctx(messages=[_Msg("user", "my IRA balance is $50,000")])
        trace = rp.build_experience_trace(
            context=ctx, decision=decision,
            local_response="your IRA balance is $50,000",
            redact_mode="none",
        )
        self.assertEqual(trace.local_response, "your IRA balance is $50,000")
        self.assertEqual(trace.request["messages"][0]["content"], "my IRA balance is $50,000")

    def test_redact_mode_partial_scrubs_pii_keeps_rest(self):
        decision = _decision()
        ctx = _ctx(messages=[_Msg("user", "my IRA balance is $50,000, call 555-123-4567")])
        trace = rp.build_experience_trace(
            context=ctx, decision=decision,
            local_response="your IRA balance is $50,000",
            redact_mode="partial",
        )
        self.assertNotIn("50,000", trace.local_response)
        self.assertIn("REDACTED_AMOUNT", trace.local_response)
        self.assertIn("REDACTED_PHONE", trace.request["messages"][0]["content"])
        self.assertIn("balance is", trace.request["messages"][0]["content"])   # surrounding text kept

    def test_tool_call_names_kept_arguments_dropped_when_fully_redacted(self):
        decision = _decision()
        trace = rp.build_experience_trace(
            context=_ctx(), decision=decision,
            tool_calls=[{"name": "search_web", "arguments": {"query": "sensitive query text"}}],
            redact_mode="full",
        )
        self.assertEqual(trace.tool_calls, [{"name": "search_web"}])

    def test_tool_call_arguments_scrubbed_not_dropped_when_partially_redacted(self):
        decision = _decision()
        trace = rp.build_experience_trace(
            context=_ctx(), decision=decision,
            tool_calls=[{"name": "search_web", "arguments": {"query": "call 555-123-4567"}}],
            redact_mode="partial",
        )
        self.assertEqual(trace.tool_calls[0]["name"], "search_web")
        self.assertIn("REDACTED_PHONE", trace.tool_calls[0]["arguments"])

    def test_errors_capped_not_redacted(self):
        decision = _decision()
        long_error = "x" * 500
        trace = rp.build_experience_trace(context=_ctx(), decision=decision, errors=[long_error])
        self.assertEqual(len(trace.errors[0]), 200)


class TestRedactPiiPatterns(unittest.TestCase):
    """Phase 7: regex-based structured-PII scrubbing for redact_mode='partial'."""

    def test_scrubs_dollar_amount(self):
        self.assertEqual(rp._redact_pii_patterns("balance is $50,000.25"), "balance is <REDACTED_AMOUNT>")

    def test_scrubs_ssn(self):
        self.assertEqual(rp._redact_pii_patterns("SSN 123-45-6789 on file"), "SSN <REDACTED_SSN> on file")

    def test_scrubs_phone(self):
        self.assertEqual(rp._redact_pii_patterns("call 555-123-4567 now"), "call <REDACTED_PHONE> now")

    def test_scrubs_email(self):
        self.assertEqual(rp._redact_pii_patterns("reach me at foo@bar.com please"), "reach me at <REDACTED_EMAIL> please")

    def test_scrubs_long_account_number(self):
        self.assertEqual(rp._redact_pii_patterns("account 123456789012 active"), "account <REDACTED_ACCOUNT_NUMBER> active")

    def test_leaves_surrounding_text_intact(self):
        result = rp._redact_pii_patterns("What is my IRA balance? It was $50,000 last month.")
        self.assertIn("What is my IRA balance?", result)
        self.assertIn("It was", result)
        self.assertIn("last month.", result)
        self.assertNotIn("50,000", result)

    def test_none_input_returns_none(self):
        self.assertIsNone(rp._redact_pii_patterns(None))


class TestBuildExperienceTraceId(unittest.TestCase):
    """Phase 8: trace_id threading — explicit id is used when given, auto-
    generated (Phase 1-7 behavior) when omitted."""

    def test_explicit_trace_id_used(self):
        trace = rp.build_experience_trace(context=_ctx(), decision=_decision(), trace_id="fixed-id-123")
        self.assertEqual(trace.trace_id, "fixed-id-123")

    def test_omitted_trace_id_auto_generates(self):
        trace = rp.build_experience_trace(context=_ctx(), decision=_decision())
        self.assertTrue(trace.trace_id)   # non-empty, matches pre-Phase-8 behavior


class TestVersioningFields(unittest.TestCase):
    """Phase 9: adapter_version/model_versions bug fix + frontier_router_version threading."""

    def test_adapter_version_reflects_decision_adapter_version_not_name(self):
        decision = _decision(selected_adapter="retail_v3", adapter_version="v3")
        trace = rp.build_experience_trace(context=_ctx(), decision=decision)
        self.assertEqual(trace.adapter_version, "v3")
        self.assertNotEqual(trace.adapter_version, decision.selected_adapter)
        self.assertEqual(trace.model_versions["local"], "v3")

    def test_adapter_version_none_when_decision_has_no_adapter(self):
        decision = _decision(selected_adapter=None, adapter_version=None)
        trace = rp.build_experience_trace(context=_ctx(), decision=decision)
        self.assertIsNone(trace.adapter_version)
        self.assertIsNone(trace.model_versions["local"])

    def test_frontier_router_version_threads_through_when_passed(self):
        trace = rp.build_experience_trace(
            context=_ctx(), decision=_decision(), frontier_router_version=rp.FRONTIER_ROUTER_VERSION,
        )
        self.assertEqual(trace.frontier_router_version, rp.FRONTIER_ROUTER_VERSION)

    def test_frontier_router_version_defaults_to_none(self):
        trace = rp.build_experience_trace(context=_ctx(), decision=_decision())
        self.assertIsNone(trace.frontier_router_version)


class TestFeedbackStore(unittest.TestCase):
    """Phase 8: append-only feedback loop, joined against traces at read time."""

    def test_feedback_store_protocol_isinstance(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            store = rp.JSONLFeedbackStore(path=Path(d) / "feedback.jsonl")
            self.assertIsInstance(store, rp.FeedbackStore)

    def test_record_appends_valid_json_lines(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "feedback.jsonl"
            store = rp.JSONLFeedbackStore(path=path)
            fb1 = rp.build_feedback_record(trace_id="t1", label="correct")
            fb2 = rp.build_feedback_record(trace_id="t2", label="incorrect", note="wrong Cypher date")
            store.record(fb1)
            store.record(fb2)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)   # appended, not overwritten
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["trace_id"], "t1")
            self.assertEqual(parsed["label"], "correct")
            self.assertIn("feedback_id", parsed)

    def test_record_never_raises_on_write_failure(self):
        store = rp.JSONLFeedbackStore(path=Path("/nonexistent_root_xyz/feedback.jsonl"))
        fb = rp.build_feedback_record(trace_id="t1", label="unrated")
        store.record(fb)   # must not raise

    def test_valid_labels_produce_a_record(self):
        for label in ("correct", "incorrect", "unrated"):
            with self.subTest(label=label):
                fb = rp.build_feedback_record(trace_id="t1", label=label)
                self.assertIsNotNone(fb)
                self.assertEqual(fb.label, label)

    def test_invalid_label_returns_none(self):
        self.assertIsNone(rp.build_feedback_record(trace_id="t1", label="maybe"))

    def test_redact_mode_full_hashes_note(self):
        fb = rp.build_feedback_record(trace_id="t1", label="incorrect", note="my SSN is 123-45-6789", redact_mode="full")
        self.assertNotIn("123-45-6789", fb.note)
        self.assertIn("len=", fb.note)

    def test_redact_mode_partial_scrubs_note(self):
        fb = rp.build_feedback_record(trace_id="t1", label="incorrect", note="my SSN is 123-45-6789", redact_mode="partial")
        self.assertNotIn("123-45-6789", fb.note)
        self.assertIn("REDACTED_SSN", fb.note)

    def test_redact_mode_none_preserves_note(self):
        fb = rp.build_feedback_record(trace_id="t1", label="incorrect", note="looked wrong to me", redact_mode="none")
        self.assertEqual(fb.note, "looked wrong to me")

    def test_note_none_stays_none_regardless_of_redact_mode(self):
        for mode in ("full", "partial", "none"):
            with self.subTest(mode=mode):
                fb = rp.build_feedback_record(trace_id="t1", label="unrated", redact_mode=mode)
                self.assertIsNone(fb.note)


if __name__ == "__main__":
    unittest.main()
