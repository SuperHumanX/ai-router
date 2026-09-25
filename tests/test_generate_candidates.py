"""Tests for training/generate_candidates.py (Phase 7). Run via:
    python3 -m unittest discover -s tests -v
from the ai-router repo root.

Fixture traces are built via router_policy.py's own build_experience_trace()
+ dataclasses.asdict() (the same path JSONLExperienceStore.record() uses),
not hand-written JSON — keeps fixtures faithful to the real schema.
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
import generate_candidates as gc


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


def _ctx(**overrides):
    defaults = dict(messages=[_Msg("user", "hello")], system="s", model_hint="fast")
    defaults.update(overrides)
    return rp.RequestContext(**defaults)


def _write_traces(path: Path, traces: list) -> None:
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(asdict(t), default=str) + "\n")


class TestLoadTraces(unittest.TestCase):

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(gc.load_traces(Path("/nonexistent/traces.jsonl")), [])

    def test_malformed_line_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "traces.jsonl"
            path.write_text("not valid json\n" + json.dumps({"trace_id": "abc"}) + "\n")
            traces = gc.load_traces(path)
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["trace_id"], "abc")


class TestRouterCandidates(unittest.TestCase):

    def test_works_regardless_of_redaction_level(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx(app_domain_hint="finance")
        decision = policy.decide(ctx)
        trace_full = rp.build_experience_trace(context=ctx, decision=decision, local_response="answer", final_source="local", redact_mode="full")
        candidates = gc.build_router_candidates([asdict(trace_full)])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["domain"], "finance")
        self.assertIn("label_local_succeeded", candidates[0])

    def test_label_reflects_final_source_and_escalation(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx()
        decision = policy.decide(ctx)
        t_local = asdict(rp.build_experience_trace(context=ctx, decision=decision, final_source="local", escalated=False))
        t_escalated = asdict(rp.build_experience_trace(context=ctx, decision=decision, final_source="frontier", escalated=True))
        candidates = gc.build_router_candidates([t_local, t_escalated])
        self.assertTrue(candidates[0]["label_local_succeeded"])
        self.assertFalse(candidates[1]["label_local_succeeded"])


class TestDistillCandidates(unittest.TestCase):

    def _escalated_trace(self, redact_mode, frontier_response="a good real answer here"):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx(app_domain_hint="finance")
        decision = policy.decide(ctx)
        return asdict(rp.build_experience_trace(
            context=ctx, decision=decision,
            local_response="", escalated=True, escalation_reason="verification_rejected: empty response",
            frontier_provider="openrouter", frontier_model="openai/gpt-4o-mini",
            frontier_response=frontier_response, final_source="frontier", redact_mode=redact_mode,
        ))

    def test_fully_redacted_traces_produce_zero_candidates(self):
        traces = [self._escalated_trace("full")]
        candidates, stats = gc.build_distill_candidates(traces)
        self.assertEqual(len(candidates), 0)
        self.assertEqual(stats["fully_redacted"], 1)

    def test_partial_redaction_produces_usable_candidates(self):
        traces = [self._escalated_trace("partial")]
        candidates, stats = gc.build_distill_candidates(traces)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(candidates[0]["domain"], "finance")
        self.assertIn("frontier_response", candidates[0])

    def test_none_redaction_produces_usable_candidates(self):
        traces = [self._escalated_trace("none")]
        candidates, stats = gc.build_distill_candidates(traces)
        self.assertEqual(len(candidates), 1)

    def test_non_escalated_trace_excluded(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx()
        decision = policy.decide(ctx)
        t = asdict(rp.build_experience_trace(context=ctx, decision=decision, final_source="local", escalated=False, redact_mode="none"))
        candidates, stats = gc.build_distill_candidates([t])
        self.assertEqual(len(candidates), 0)
        self.assertEqual(stats["not_escalated"], 1)

    def test_frontier_quality_gate_rejects_refusal_response(self):
        traces = [self._escalated_trace("none", frontier_response="I'm sorry, but I cannot help with that.")]
        candidates, stats = gc.build_distill_candidates(traces)
        self.assertEqual(len(candidates), 0)
        self.assertEqual(stats["failed_quality_gate"], 1)

    def test_duplicate_frontier_response_deduplicated(self):
        traces = [self._escalated_trace("none"), self._escalated_trace("none")]   # identical frontier_response + domain
        candidates, stats = gc.build_distill_candidates(traces)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(stats["duplicate"], 1)


class TestVerifierCandidates(unittest.TestCase):

    def test_captures_accepted_and_rejected_cases(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx()
        decision = policy.decide(ctx)
        accepted_verification = rp.VerificationResult(accepted=True, score=1.0, confidence=0.5, failure_reasons=[])
        rejected_verification = rp.VerificationResult(accepted=False, score=0.0, confidence=0.5, failure_reasons=["empty response"])
        t_accepted = asdict(rp.build_experience_trace(context=ctx, decision=decision, verification=accepted_verification, final_source="local", redact_mode="none"))
        t_rejected = asdict(rp.build_experience_trace(context=ctx, decision=decision, verification=rejected_verification, escalated=True, final_source="frontier", redact_mode="none"))
        t_no_verifier = asdict(rp.build_experience_trace(context=ctx, decision=decision, final_source="local", redact_mode="none"))
        candidates = gc.build_verifier_candidates([t_accepted, t_rejected, t_no_verifier])
        self.assertEqual(len(candidates), 2)   # t_no_verifier excluded — verification is None
        self.assertTrue(candidates[0]["verifier_accepted"])
        self.assertFalse(candidates[1]["verifier_accepted"])
        self.assertIsNone(candidates[0]["user_feedback"])   # no feedback mechanism exists yet


class TestWriteDatasetAndManifest(unittest.TestCase):

    def test_manifest_contains_expected_version_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            out_root = Path(d) / "datasets"
            source_path = Path(d) / "traces.jsonl"
            out_dir = gc.write_dataset(out_root, "router", [{"a": 1}], source_path, 5, {"extra": "x"})
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["dataset_type"], "router")
            self.assertEqual(manifest["source_trace_count"], 5)
            self.assertEqual(manifest["candidate_count"], 1)
            self.assertIn("version", manifest)
            self.assertIn("generated_at", manifest)
            data_lines = (out_dir / "data.jsonl").read_text().splitlines()
            self.assertEqual(len(data_lines), 1)


class TestEndToEnd(unittest.TestCase):

    def test_full_run_against_synthetic_fixture(self):
        policy = rp.DeterministicRouterPolicy()
        ctx = _ctx(app_domain_hint="finance")
        decision = policy.decide(ctx)
        verification = rp.VerificationResult(accepted=False, score=0.0, confidence=0.5, failure_reasons=["empty response"])

        traces = [
            rp.build_experience_trace(context=ctx, decision=decision, final_source="local", redact_mode="full"),
            rp.build_experience_trace(
                context=ctx, decision=decision, verification=verification, escalated=True,
                escalation_reason="verification_rejected: empty response", frontier_response="a real distillable answer",
                frontier_provider="openrouter", frontier_model="openai/gpt-4o-mini",
                final_source="frontier", redact_mode="partial",
            ),
        ]
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "experience_traces.jsonl"
            _write_traces(trace_path, traces)
            out_root = Path(d) / "datasets"

            all_traces = gc.load_traces(trace_path)
            router_c = gc.build_router_candidates(all_traces)
            distill_c, distill_stats = gc.build_distill_candidates(all_traces)
            verifier_c = gc.build_verifier_candidates(all_traces)

            self.assertEqual(len(router_c), 2)
            self.assertEqual(len(distill_c), 1)   # only the partial-redacted, escalated one qualifies
            self.assertEqual(len(verifier_c), 1)  # only the one with a verification result

            gc.write_dataset(out_root, "router", router_c, trace_path, len(all_traces), {})
            gc.write_dataset(out_root, "distill", distill_c, trace_path, len(all_traces), {"exclusion_stats": distill_stats})
            gc.write_dataset(out_root, "verifier", verifier_c, trace_path, len(all_traces), {})

            self.assertTrue((out_root / "router").exists())
            self.assertTrue((out_root / "distill").exists())
            self.assertTrue((out_root / "verifier").exists())


if __name__ == "__main__":
    unittest.main()
