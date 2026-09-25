"""Tests for training/compute_metrics.py (Phase 8). Run via:
    python3 -m unittest discover -s tests -v
from the ai-router repo root.

Fixture traces/feedback are built via router_policy.py's own
build_experience_trace()/build_feedback_record() + dataclasses.asdict()
(the same path the real stores use), not hand-written JSON — keeps
fixtures faithful to the real schema, same convention as
test_generate_candidates.py.
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

import router_policy as rp
import compute_metrics as cm


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


def _ctx(**overrides):
    defaults = dict(messages=[_Msg("user", "hello")], system="s", model_hint="structured")
    defaults.update(overrides)
    return rp.RequestContext(**defaults)


def _finance_decision(**overrides):
    defaults = dict(
        domain="finance", subdomain=None, task="unclassified",
        complexity=0.1, risk=0.0, predicted_local_success=0.87,
        selected_adapter="finance", required_capabilities=[],
        quality_threshold=0.8, route="local", reason="test",
        router_version="test",
    )
    defaults.update(overrides)
    return rp.RoutingDecision(**defaults)


def _write_jsonl(path: Path, dataclass_objs: list) -> None:
    with open(path, "w") as f:
        for obj in dataclass_objs:
            f.write(json.dumps(asdict(obj), default=str) + "\n")


class TestLoadFeedback(unittest.TestCase):

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(cm.load_feedback(Path("/nonexistent/feedback.jsonl")), [])

    def test_malformed_line_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "feedback.jsonl"
            path.write_text("not valid json\n" + json.dumps({"trace_id": "abc"}) + "\n")
            records = cm.load_feedback(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["trace_id"], "abc")


class TestLatestFeedbackByTrace(unittest.TestCase):

    def test_last_record_wins_by_file_order(self):
        feedback = [
            {"trace_id": "t1", "label": "incorrect"},
            {"trace_id": "t1", "label": "correct"},   # later in file — changed their mind
            {"trace_id": "t2", "label": "unrated"},
        ]
        latest = cm._latest_feedback_by_trace(feedback)
        self.assertEqual(latest["t1"]["label"], "correct")
        self.assertEqual(latest["t2"]["label"], "unrated")


class TestDomainMetrics(unittest.TestCase):

    def _trace(self, *, final_source, escalated, accepted=None, trace_id=None):
        decision = _finance_decision()
        verification = None
        if accepted is not None:
            verification = rp.VerificationResult(accepted=accepted, score=1.0 if accepted else 0.0, confidence=0.5, failure_reasons=[])
        return asdict(rp.build_experience_trace(
            context=_ctx(), decision=decision, trace_id=trace_id,
            final_source=final_source, escalated=escalated, verification=verification,
            redact_mode="none",
        ))

    def test_verifier_accept_and_escalation_rates(self):
        traces = [
            self._trace(final_source="local", escalated=False, accepted=True, trace_id="t1"),
            self._trace(final_source="local", escalated=False, accepted=True, trace_id="t2"),
            self._trace(final_source="frontier", escalated=True, accepted=False, trace_id="t3"),
        ]
        by_domain = cm.compute_domain_metrics(traces, {})
        m = by_domain["finance"]
        self.assertEqual(m["total"], 3)
        self.assertAlmostEqual(m["verifier_accept_rate"], 2 / 3)
        self.assertAlmostEqual(m["escalation_rate"], 1 / 3)
        self.assertAlmostEqual(m["local_route_share"], 2 / 3)

    def test_benchmark_deviation_uses_trace_frozen_prediction(self):
        # _finance_decision()'s default predicted_local_success (0.87) happens
        # to match DEFAULT_DOMAIN_REGISTRY's current finance benchmark — this
        # just confirms the basic arithmetic still holds post-refactor.
        # 1 local-accepted out of 1 total -> observed=1.0 -> deviation = 1.0 - 0.87
        traces = [self._trace(final_source="local", escalated=False, trace_id="t1")]
        by_domain = cm.compute_domain_metrics(traces, {})
        self.assertAlmostEqual(by_domain["finance"]["avg_predicted_local_success"], 0.87, places=6)
        self.assertAlmostEqual(by_domain["finance"]["benchmark_deviation"], 1.0 - 0.87, places=6)

    def test_benchmark_deviation_ignores_live_registry_after_it_changes(self):
        # Phase 9 regression: the OLD implementation re-derived the baseline
        # from rp.load_domain_registry() at analysis time, which would
        # silently go wrong the moment finance's benchmark_success_rate
        # (currently 0.87 in DEFAULT_DOMAIN_REGISTRY) gets updated after
        # these traces were already recorded. Simulate that: this trace was
        # decided when the router predicted 0.70 (an older, different
        # number than today's 0.87) — deviation must be computed against
        # the trace's OWN frozen 0.70, never today's registry value.
        decision = _finance_decision(predicted_local_success=0.70)
        trace = asdict(rp.build_experience_trace(
            context=_ctx(), decision=decision, trace_id="t1",
            final_source="local", escalated=False, redact_mode="none",
        ))
        by_domain = cm.compute_domain_metrics([trace], {})
        self.assertAlmostEqual(by_domain["finance"]["avg_predicted_local_success"], 0.70, places=6)
        self.assertAlmostEqual(by_domain["finance"]["benchmark_deviation"], 1.0 - 0.70, places=6)
        # explicitly NOT the live registry's 0.87
        self.assertNotAlmostEqual(by_domain["finance"]["benchmark_deviation"], 1.0 - 0.87, places=6)

    def test_false_accept_rate_from_joined_feedback(self):
        traces = [
            self._trace(final_source="local", escalated=False, trace_id="t1"),   # incorrect
            self._trace(final_source="local", escalated=False, trace_id="t2"),   # correct
            self._trace(final_source="local", escalated=False, trace_id="t3"),   # unlabeled
        ]
        feedback_by_trace = {
            "t1": {"trace_id": "t1", "label": "incorrect"},
            "t2": {"trace_id": "t2", "label": "correct"},
        }
        m = cm.compute_domain_metrics(traces, feedback_by_trace)["finance"]
        self.assertEqual(m["local_accepted_count"], 3)
        self.assertEqual(m["feedback_labeled_count"], 2)   # t3 excluded — no feedback
        self.assertAlmostEqual(m["false_accept_rate"], 1 / 2)   # 1 incorrect of 2 labeled
        self.assertAlmostEqual(m["feedback_coverage"], 2 / 3)   # 2 of 3 local-accepted labeled

    def test_zero_feedback_anywhere_reports_null_not_zero(self):
        traces = [self._trace(final_source="local", escalated=False, trace_id="t1")]
        m = cm.compute_domain_metrics(traces, {})["finance"]
        self.assertIsNone(m["false_accept_rate"])
        self.assertEqual(m["feedback_coverage"], 0.0)   # coverage IS measurable (0 of N labeled) — that's not the same claim as the rate itself

    def test_no_local_accepted_traces_gives_none_coverage(self):
        traces = [self._trace(final_source="frontier", escalated=True, trace_id="t1")]
        m = cm.compute_domain_metrics(traces, {})["finance"]
        self.assertIsNone(m["feedback_coverage"])   # undefined over zero local-accepted traces
        self.assertIsNone(m["false_accept_rate"])


class TestRouterVersionMetrics(unittest.TestCase):
    """Phase 9: the canary-comparison tool — cohorts separated by
    RoutingDecision.router_version instead of domain."""

    def _trace(self, *, router_version, final_source, escalated, accepted=None, trace_id=None):
        decision = _finance_decision(router_version=router_version)
        verification = None
        if accepted is not None:
            verification = rp.VerificationResult(accepted=accepted, score=1.0 if accepted else 0.0, confidence=0.5, failure_reasons=[])
        return asdict(rp.build_experience_trace(
            context=_ctx(), decision=decision, trace_id=trace_id,
            final_source=final_source, escalated=escalated, verification=verification,
            redact_mode="none",
        ))

    def test_two_cohorts_cleanly_separated(self):
        traces = [
            self._trace(router_version="baseline", final_source="local", escalated=False, accepted=True, trace_id="b1"),
            self._trace(router_version="baseline", final_source="local", escalated=False, accepted=True, trace_id="b2"),
            self._trace(router_version="canary-lower-threshold", final_source="frontier", escalated=True, accepted=False, trace_id="c1"),
        ]
        by_version = cm.compute_router_version_metrics(traces, {})
        self.assertEqual(set(by_version.keys()), {"baseline", "canary-lower-threshold"})
        self.assertEqual(by_version["baseline"]["total"], 2)
        self.assertAlmostEqual(by_version["baseline"]["verifier_accept_rate"], 1.0)
        self.assertEqual(by_version["canary-lower-threshold"]["total"], 1)
        self.assertAlmostEqual(by_version["canary-lower-threshold"]["escalation_rate"], 1.0)


class TestVerifierVersionMetrics(unittest.TestCase):
    """Phase 9: same idea, grouped by which verifier config actually ran."""

    def test_traces_without_verification_excluded(self):
        decision = _finance_decision()
        with_verifier = asdict(rp.build_experience_trace(
            context=_ctx(), decision=decision, final_source="local",
            verification=rp.VerificationResult(accepted=True, score=1.0, confidence=0.5, failure_reasons=[], verifier_version="v-a"),
            redact_mode="none",
        ))
        without_verifier = asdict(rp.build_experience_trace(
            context=_ctx(), decision=decision, final_source="local", redact_mode="none",
        ))
        by_verifier = cm.compute_verifier_version_metrics([with_verifier, without_verifier], {})
        self.assertEqual(set(by_verifier.keys()), {"v-a"})   # the unverified trace has no key to group under
        self.assertEqual(by_verifier["v-a"]["total"], 1)

    def test_two_verifier_cohorts_separated(self):
        decision = _finance_decision()
        traces = [
            asdict(rp.build_experience_trace(
                context=_ctx(), decision=decision, final_source="local",
                verification=rp.VerificationResult(accepted=True, score=1.0, confidence=0.5, failure_reasons=[], verifier_version="strict"),
                redact_mode="none",
            )),
            asdict(rp.build_experience_trace(
                context=_ctx(), decision=decision, final_source="frontier", escalated=True,
                verification=rp.VerificationResult(accepted=False, score=0.0, confidence=0.5, failure_reasons=["x"], verifier_version="lenient"),
                redact_mode="none",
            )),
        ]
        by_verifier = cm.compute_verifier_version_metrics(traces, {})
        self.assertEqual(by_verifier["strict"]["verifier_accept_rate"], 1.0)
        self.assertEqual(by_verifier["lenient"]["verifier_accept_rate"], 0.0)


class TestOverallMetrics(unittest.TestCase):

    def test_latency_percentiles_and_provider_counts(self):
        decision = _finance_decision()
        traces = [
            asdict(rp.build_experience_trace(context=_ctx(), decision=decision, final_source="local",
                                              local_latency_ms=100.0, redact_mode="none")),
            asdict(rp.build_experience_trace(context=_ctx(), decision=decision, final_source="frontier",
                                              escalated=True, frontier_provider="openrouter",
                                              frontier_latency_ms=500.0, redact_mode="none")),
        ]
        overall = cm.compute_overall_metrics(traces)
        self.assertEqual(overall["total_traces"], 2)
        self.assertEqual(overall["local_latency_ms"]["p50"], 100.0)
        self.assertEqual(overall["frontier_latency_ms"]["p50"], 500.0)
        self.assertEqual(overall["provider_counts"], {"openrouter": 1})
        self.assertEqual(overall["error_count"], 0)

    def test_error_rate_computed(self):
        decision = _finance_decision()
        traces = [
            asdict(rp.build_experience_trace(context=_ctx(), decision=decision, final_source="frontier",
                                              escalated=True, errors=["boom"], redact_mode="none")),
            asdict(rp.build_experience_trace(context=_ctx(), decision=decision, final_source="local", redact_mode="none")),
        ]
        overall = cm.compute_overall_metrics(traces)
        self.assertEqual(overall["error_count"], 1)
        self.assertAlmostEqual(overall["error_rate"], 0.5)


class TestWriteReport(unittest.TestCase):

    def test_report_contains_expected_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            out_root = Path(d) / "reports"
            traces_path = Path(d) / "traces.jsonl"
            feedback_path = Path(d) / "feedback.jsonl"
            report = {"by_domain": {}, "overall": {"total_traces": 0}}
            out_dir = cm.write_report(out_root, report, traces_path, feedback_path, 5, 2)
            written = json.loads((out_dir / "report.json").read_text())
            self.assertIn("version", written)
            self.assertIn("generated_at", written)
            self.assertEqual(written["source_trace_count"], 5)
            self.assertEqual(written["source_feedback_count"], 2)
            self.assertEqual(written["by_domain"], {})


class TestEndToEnd(unittest.TestCase):

    def test_full_run_against_synthetic_fixture(self):
        decision = _finance_decision()
        accepted_v = rp.VerificationResult(accepted=True, score=1.0, confidence=0.5, failure_reasons=[])
        rejected_v = rp.VerificationResult(accepted=False, score=0.0, confidence=0.5, failure_reasons=["empty response"])

        traces = [
            rp.build_experience_trace(context=_ctx(), decision=decision, trace_id="t1",
                                       final_source="local", verification=accepted_v,
                                       local_latency_ms=120.0, redact_mode="none"),
            rp.build_experience_trace(context=_ctx(), decision=decision, trace_id="t2",
                                       final_source="local", verification=accepted_v,
                                       local_latency_ms=90.0, redact_mode="none"),
            rp.build_experience_trace(context=_ctx(), decision=decision, trace_id="t3",
                                       final_source="frontier", escalated=True, verification=rejected_v,
                                       frontier_provider="openrouter", frontier_latency_ms=600.0,
                                       redact_mode="none"),
        ]
        feedback = [
            rp.build_feedback_record(trace_id="t1", label="incorrect"),
            rp.build_feedback_record(trace_id="t2", label="correct"),
        ]

        with tempfile.TemporaryDirectory() as d:
            traces_path = Path(d) / "experience_traces.jsonl"
            feedback_path = Path(d) / "experience_feedback.jsonl"
            _write_jsonl(traces_path, traces)
            _write_jsonl(feedback_path, feedback)
            out_root = Path(d) / "reports"

            loaded_traces = cm.load_traces(traces_path)
            loaded_feedback = cm.load_feedback(feedback_path)
            feedback_by_trace = cm._latest_feedback_by_trace(loaded_feedback)

            by_domain = cm.compute_domain_metrics(loaded_traces, feedback_by_trace)
            by_router_version = cm.compute_router_version_metrics(loaded_traces, feedback_by_trace)
            by_verifier_version = cm.compute_verifier_version_metrics(loaded_traces, feedback_by_trace)
            overall = cm.compute_overall_metrics(loaded_traces)

            self.assertEqual(by_domain["finance"]["total"], 3)
            self.assertEqual(by_domain["finance"]["local_accepted_count"], 2)
            self.assertEqual(by_domain["finance"]["feedback_labeled_count"], 2)
            self.assertAlmostEqual(by_domain["finance"]["false_accept_rate"], 0.5)
            self.assertAlmostEqual(by_domain["finance"]["feedback_coverage"], 1.0)

            # _finance_decision()'s router_version="test" for all 3 traces -> one cohort
            self.assertEqual(by_router_version["test"]["total"], 3)
            # all 3 traces have a verification result (verifier_version defaults to "unset")
            self.assertEqual(by_verifier_version["unset"]["total"], 3)

            out_dir = cm.write_report(
                out_root,
                {"by_domain": by_domain, "by_router_version": by_router_version,
                 "by_verifier_version": by_verifier_version, "overall": overall},
                traces_path, feedback_path, len(loaded_traces), len(loaded_feedback),
            )
            self.assertTrue((out_dir / "report.json").exists())
            written = json.loads((out_dir / "report.json").read_text())
            self.assertEqual(written["by_domain"]["finance"]["local_accepted_count"], 2)
            self.assertEqual(written["by_router_version"]["test"]["total"], 3)
            self.assertEqual(written["by_verifier_version"]["unset"]["total"], 3)


if __name__ == "__main__":
    unittest.main()
