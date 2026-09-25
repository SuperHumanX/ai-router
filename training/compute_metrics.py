"""
compute_metrics.py — Phase 8: offline safety/observability metrics
(Phase 9: version-cohort breakdowns + a benchmark_deviation staleness fix)
════════════════════════════════════════════════════════════════════════════
Standalone batch script, sibling to generate_candidates.py — same posture:
never imported by gateway.py/router_policy.py, never called from chat() or
any request path. Reads a Phase 6 Experience Store trace file and a Phase 8
feedback file (produced by AIGateway.record_feedback()), joins them by
trace_id, and computes the metrics the live in-process snapshot
(AIGateway.get_metrics_snapshot()) structurally can't: durable, cross-
process history, and — the headline one — a REAL false-accept rate.

Live snapshot vs. this script
──────────────────────────────
AIGateway.get_metrics_snapshot() is bounded, in-memory, content-free, and
needs only AI_ROUTER_INTELLIGENT_ROUTING — it's "what does this process
look like right now." This script needs AI_ROUTER_EXPERIENCE_STORE (for
trace history) and, for the false-accept number specifically, actual
feedback collected via record_feedback() — nothing live calls that yet
(see ARCHITECTURE.md's Phase 8 section), so this script's false_accept_rate
will legitimately be `null` (not `0.0` — see below) against any real
deployment's trace file today.

The false-accept rate, honestly
──────────────────────────────
Numerator: traces where final_source=="local" (verifier accepted it, not
escalated) whose LATEST joined feedback record has label=="incorrect".
Denominator: traces where final_source=="local" AND a feedback record
exists at all — NOT "all local-accepted traces." A rate over unlabeled
data would be fabricated, not measured. `feedback_coverage` (labeled /
all local-accepted) is always reported alongside it, and a domain with
zero coverage reports `false_accept_rate: null`, never `0.0` — "not
enough data" and "measured zero false accepts" are categorically
different claims, and reporting the former as the latter would be a
fabricated safety signal.

`benchmark_deviation` is a separate, always-available PROXY — observed
local-accepted-without-escalation rate minus the average `predicted_local_
success` that each group's own traces actually recorded at decide-time
(Phase 9: this used to re-derive the comparison baseline from today's LIVE
domain registry, which silently went wrong the moment a domain got
rebenchmarked after some traces were already written — using each trace's
own frozen prediction is correct regardless of what the registry says
today). Useful even with zero feedback, but explicitly not a substitute
for the real, feedback-backed number above.

Version-cohort breakdowns (Phase 9)
────────────────────────────────────
Alongside `by_domain`, the report also breaks the same metric set out
`by_router_version` and `by_verifier_version` — grouping by
`RoutingDecision.router_version`/`VerificationResult.verifier_version`
instead of domain. This is the direct analytical payoff of Phase 9's
`policy_id`/`verifier_id` canary tagging: run two separately-configured
`AIGateway` processes (e.g. baseline vs. `AI_ROUTER_POLICY_ID=canary-a`),
point this script at each one's trace file (or a merged one — the grouping
does the separation), and compare `false_accept_rate`/`verifier_accept_
rate` between cohorts. There is no traffic-splitting or dual-dispatch code
anywhere in ai-router — this script is the comparison tool, not the split.

Usage:
    python3 compute_metrics.py --traces <path-to-experience_traces.jsonl> \\
        [--feedback <path-to-experience_feedback.jsonl>] [--output-dir training/reports/metrics]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from generate_candidates import load_traces   # noqa: E402 — reused, not duplicated


def load_feedback(path: Path) -> list[dict]:
    """Same defensive contract as load_traces: a missing file or a malformed
    line must never abort the whole run."""
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _latest_feedback_by_trace(feedback: list[dict]) -> dict[str, dict]:
    """Last record per trace_id wins, by file order — a later write is a
    newer opinion (e.g. a user changing their mind)."""
    latest: dict[str, dict] = {}
    for record in feedback:
        trace_id = record.get("trace_id")
        if trace_id:
            latest[trace_id] = record
    return latest


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "count": 0}
    data = sorted(values)
    if len(data) < 2:
        v = data[0]
        return {"p50": v, "p95": v, "p99": v, "count": len(data)}
    import statistics
    q = statistics.quantiles(data, n=100, method="inclusive")
    def _at(pct: int) -> float:
        return q[min(max(pct - 1, 0), len(q) - 1)]
    return {"p50": _at(50), "p95": _at(95), "p99": _at(99), "count": len(data)}


def _domain_of(trace: dict) -> Optional[str]:
    return (trace.get("routing_decision") or {}).get("domain") or "unknown"


def _router_version_of(trace: dict) -> Optional[str]:
    return (trace.get("routing_decision") or {}).get("router_version")


def _verifier_version_of(trace: dict) -> Optional[str]:
    verification = trace.get("verification")
    return verification.get("verifier_version") if verification else None


def _group_metrics(traces: list[dict], feedback_by_trace: dict[str, dict], key_fn) -> dict:
    """Generic grouping — same metric set regardless of what key_fn groups
    by (domain, router_version, verifier_version, ...). See module
    docstring for exactly what each field means and does not mean.
    key_fn returning None excludes a trace from every group (e.g. a trace
    with no verification result has no verifier_version to group by)."""
    groups: dict[str, list[dict]] = {}
    for t in traces:
        key = key_fn(t)
        if key is None:
            continue
        groups.setdefault(key, []).append(t)

    out: dict = {}
    for key, gtraces in groups.items():
        total = len(gtraces)
        verifier_ran = [t for t in gtraces if t.get("verification") is not None]
        verifier_accepted = [t for t in verifier_ran if t["verification"].get("accepted")]
        escalated = [t for t in gtraces if t.get("escalated")]
        local_accepted = [t for t in gtraces if t.get("final_source") == "local" and not t.get("escalated")]

        labeled = [t for t in local_accepted if t.get("trace_id") in feedback_by_trace]
        incorrect = [t for t in labeled if feedback_by_trace[t["trace_id"]].get("label") == "incorrect"]

        # Phase 9: compares against each trace's OWN frozen prediction from
        # decide()-time, not a fresh registry lookup — see module docstring.
        predicted = [
            t["routing_decision"]["predicted_local_success"] for t in gtraces
            if t.get("routing_decision") and t["routing_decision"].get("predicted_local_success") is not None
        ]
        avg_predicted = (sum(predicted) / len(predicted)) if predicted else None
        benchmark_deviation = (
            (len(local_accepted) / total) - avg_predicted
            if avg_predicted is not None and total else None
        )

        out[key] = {
            "total": total,
            "verifier_accept_rate": (len(verifier_accepted) / len(verifier_ran)) if verifier_ran else None,
            "verifier_reject_rate": (1 - len(verifier_accepted) / len(verifier_ran)) if verifier_ran else None,
            "escalation_rate": (len(escalated) / total) if total else None,
            "local_route_share": (len(local_accepted) / total) if total else None,
            "avg_predicted_local_success": avg_predicted,
            "benchmark_deviation": benchmark_deviation,
            "false_accept_rate": (len(incorrect) / len(labeled)) if labeled else None,
            "feedback_coverage": (len(labeled) / len(local_accepted)) if local_accepted else None,
            "local_accepted_count": len(local_accepted),
            "feedback_labeled_count": len(labeled),
        }
    return out


def compute_domain_metrics(traces: list[dict], feedback_by_trace: dict[str, dict]) -> dict:
    return _group_metrics(traces, feedback_by_trace, _domain_of)


def compute_router_version_metrics(traces: list[dict], feedback_by_trace: dict[str, dict]) -> dict:
    """Phase 9: the canary-comparison tool — see module docstring."""
    return _group_metrics(traces, feedback_by_trace, _router_version_of)


def compute_verifier_version_metrics(traces: list[dict], feedback_by_trace: dict[str, dict]) -> dict:
    """Phase 9: same idea, grouped by which verifier config actually ran.
    Naturally excludes traces where no verifier ran at all."""
    return _group_metrics(traces, feedback_by_trace, _verifier_version_of)


def compute_overall_metrics(traces: list[dict]) -> dict:
    """Latency percentiles, provider distribution, error rate — aggregated
    across all domains, mirroring the live snapshot's own overall/per-domain
    split (per-domain gets rates, not percentiles — same scope decision)."""
    local_latencies = [t["local_latency_ms"] for t in traces if t.get("local_latency_ms") is not None]
    frontier_latencies = [t["frontier_latency_ms"] for t in traces if t.get("frontier_latency_ms") is not None]
    providers: dict[str, int] = {}
    for t in traces:
        p = t.get("frontier_provider")
        if p:
            providers[p] = providers.get(p, 0) + 1
    error_count = sum(1 for t in traces if t.get("errors"))
    return {
        "total_traces": len(traces),
        "local_latency_ms": _percentiles(local_latencies),
        "frontier_latency_ms": _percentiles(frontier_latencies),
        "provider_counts": providers,
        "error_count": error_count,
        "error_rate": (error_count / len(traces)) if traces else None,
    }


def write_report(output_root: Path, report: dict, traces_path: Path, feedback_path: Path, trace_count: int, feedback_count: int) -> Path:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out_dir = output_root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    full_report = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_trace_file": str(traces_path),
        "source_trace_count": trace_count,
        "source_feedback_file": str(feedback_path),
        "source_feedback_count": feedback_count,
        **report,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True, help="Path to an experience_traces.jsonl file")
    parser.add_argument("--feedback", default=None, help="Path to an experience_feedback.jsonl file (default: sibling to --traces)")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "reports" / "metrics"), help="Root directory for versioned report output")
    args = parser.parse_args()

    traces_path = Path(args.traces)
    feedback_path = Path(args.feedback) if args.feedback else traces_path.parent / "experience_feedback.jsonl"
    output_root = Path(args.output_dir)

    traces = load_traces(traces_path)
    if not traces:
        print(f"No traces found at {traces_path} — nothing to compute.")
        return
    feedback = load_feedback(feedback_path)
    feedback_by_trace = _latest_feedback_by_trace(feedback)

    by_domain = compute_domain_metrics(traces, feedback_by_trace)
    by_router_version = compute_router_version_metrics(traces, feedback_by_trace)
    by_verifier_version = compute_verifier_version_metrics(traces, feedback_by_trace)
    overall = compute_overall_metrics(traces)

    report = {
        "by_domain": by_domain,
        "by_router_version": by_router_version,
        "by_verifier_version": by_verifier_version,
        "overall": overall,
    }
    out_dir = write_report(output_root, report, traces_path, feedback_path, len(traces), len(feedback))

    print(f"Source: {traces_path} ({len(traces)} traces), {feedback_path} ({len(feedback)} feedback records)")
    print(f"Report: {out_dir / 'report.json'}")
    print(f"\nOverall: {overall['total_traces']} traces, error_rate={overall['error_rate']}, "
          f"local_latency_p50={overall['local_latency_ms']['p50']}, "
          f"frontier_latency_p50={overall['frontier_latency_ms']['p50']}")

    def _print_group(label: str, groups: dict) -> None:
        for key, m in groups.items():
            print(f"\n  {label}={key} total={m['total']} "
                  f"verifier_accept_rate={m['verifier_accept_rate']} "
                  f"escalation_rate={m['escalation_rate']} "
                  f"benchmark_deviation={m['benchmark_deviation']}")
            if m["false_accept_rate"] is None:
                print(f"    false_accept_rate=null — 0 of {m['local_accepted_count']} local-accepted traces have "
                      f"feedback; wire up record_feedback() calls from a live UI to start collecting it.")
            else:
                print(f"    false_accept_rate={m['false_accept_rate']:.3f} "
                      f"(feedback_coverage={m['feedback_coverage']:.3f}, "
                      f"{m['feedback_labeled_count']}/{m['local_accepted_count']} local-accepted traces labeled)")

    print("\n--- by domain ---")
    _print_group("domain", by_domain)
    print("\n--- by router_version (canary comparison) ---")
    _print_group("router_version", by_router_version)
    print("\n--- by verifier_version (canary comparison) ---")
    _print_group("verifier_version", by_verifier_version)


if __name__ == "__main__":
    main()
