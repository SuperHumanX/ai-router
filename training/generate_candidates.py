"""
generate_candidates.py — Phase 7: offline learning-candidate generation
═════════════════════════════════════════════════════════════════════════
Standalone batch script. NEVER called from chat() or any request path —
"Do NOT train models automatically in the request path" is a hard
constraint from the original spec. Reads a JSONL Experience Store trace
file (produced by Phase 6's JSONLExperienceStore, e.g.
<deployment>/agents/data/experience_traces.jsonl) and derives three
training-candidate datasets from it:

  router      — works regardless of trace redaction level (metadata only):
                domain/task/model_hint/complexity/risk/predicted_local_success/
                quality_threshold, labeled with the ground truth
                (did local actually succeed, i.e. final_source=="local" and
                not escalated).
  distill     — adapter/distillation candidates: requires escalated=True,
                final_source=="frontier", USABLE content (not full-redacted
                — see below), and the frontier response passing a
                deterministic quality gate (reuses router_policy.py's
                DeterministicVerifier against the frontier text).
  verifier    — verifier-training candidates: any trace where a
                verification actually ran (accepted or rejected cases).

Redaction level matters for `distill`/`verifier` (they want real content):
  - redact_mode="full"    -> content is a "<redacted len=... sha256=...>"
                              placeholder. NOT usable for distill candidates.
  - redact_mode="partial" -> structured PII scrubbed ("<REDACTED_...>"
                              markers), rest of the text real. Usable.
  - redact_mode="none"    -> raw content. Usable.
`router` candidates are metadata-only and work at every redaction level.

Dataset versioning (the spec's explicit ask): each run writes to
  training/datasets/<candidate_type>/<version>/data.jsonl
  training/datasets/<candidate_type>/<version>/manifest.json
so a future router/verifier/LoRA training run can cite exactly which
dataset version it used.

Usage:
    python3 generate_candidates.py --input <path-to-experience_traces.jsonl> [--output-dir training/datasets]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# router_policy.py lives at the repo root, one level up from training/.
sys.path.insert(0, str(Path(__file__).parent.parent))
import router_policy as rp   # noqa: E402


def _looks_fully_redacted(text) -> bool:
    return isinstance(text, str) and text.startswith("<redacted len=")


def _trace_is_usable_for_content(trace: dict) -> bool:
    """A trace is usable for distill/verifier candidates only if its content
    fields weren't fully redacted (partial/none are both fine — partial is
    scrubbed but substantively real text)."""
    for field in ("local_response", "frontier_response"):
        if _looks_fully_redacted(trace.get(field)):
            return False
    request = trace.get("request") or {}
    # full-redact mode's request shape is {"message_count":..., "total_chars":..., "content_hash":...}
    # (no "messages" key) — partial/none modes always have "messages".
    if "messages" not in request:
        return False
    return True


def load_traces(path: Path) -> list[dict]:
    traces = []
    if not path.exists():
        return traces
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except Exception:
                continue   # a malformed line must never abort the whole run
    return traces


def build_router_candidates(traces: list[dict]) -> list[dict]:
    """Works regardless of redaction level — metadata only."""
    candidates = []
    for t in traces:
        decision = t.get("routing_decision") or {}
        ctx_meta = t.get("context_metadata") or {}
        local_succeeded = (t.get("final_source") == "local") and not t.get("escalated", False)
        candidates.append({
            "trace_id": t.get("trace_id"),
            "domain": decision.get("domain"),
            "subdomain": decision.get("subdomain"),
            "task": decision.get("task"),
            "model_hint": ctx_meta.get("model_hint"),
            "complexity": decision.get("complexity"),
            "risk": decision.get("risk"),
            "predicted_local_success": decision.get("predicted_local_success"),
            "quality_threshold": decision.get("quality_threshold"),
            "route_decided": decision.get("route"),
            "label_local_succeeded": local_succeeded,
        })
    return candidates


def build_distill_candidates(traces: list[dict]) -> tuple[list[dict], dict]:
    """Requires escalated=True, final_source=="frontier", usable content,
    and the frontier response passing a deterministic quality gate. Returns
    (candidates, stats) — stats explains why traces were excluded, so a
    caller never has to guess why the output is smaller/empty than expected."""
    verifier = rp.DeterministicVerifier()
    candidates: list[dict] = []
    seen_signatures: set = set()
    stats = {"total": len(traces), "not_escalated": 0, "not_frontier": 0,
              "fully_redacted": 0, "failed_quality_gate": 0, "duplicate": 0, "accepted": 0}

    for t in traces:
        if not t.get("escalated", False):
            stats["not_escalated"] += 1
            continue
        if t.get("final_source") != "frontier":
            stats["not_frontier"] += 1
            continue
        if not _trace_is_usable_for_content(t):
            stats["fully_redacted"] += 1
            continue

        frontier_text = t.get("frontier_response") or ""
        # Reuse Phase 4's deterministic checks against the frontier side this
        # time — context/decision params are unused in the implementation,
        # confirmed safe to call this way.
        frontier_check = verifier.evaluate(None, None, frontier_text)
        if not frontier_check.accepted:
            stats["failed_quality_gate"] += 1
            continue

        signature = (t.get("routing_decision", {}).get("domain"), hashlib.sha256(frontier_text.encode()).hexdigest())
        if signature in seen_signatures:
            stats["duplicate"] += 1
            continue
        seen_signatures.add(signature)

        decision = t.get("routing_decision") or {}
        verification = t.get("verification") or {}
        candidates.append({
            "trace_id": t.get("trace_id"),
            "domain": decision.get("domain"),
            "task": decision.get("task"),
            "request": t.get("request"),
            "local_response": t.get("local_response"),
            "local_failure_reason": t.get("escalation_reason"),
            "frontier_response": t.get("frontier_response"),
            "frontier_provider": t.get("frontier_provider"),
            "frontier_model": t.get("frontier_model"),
            "quality_metadata": {
                "local_verification_score": verification.get("score"),
                "frontier_verification_score": frontier_check.score,
            },
        })
        stats["accepted"] += 1

    return candidates, stats


def build_verifier_candidates(traces: list[dict]) -> list[dict]:
    """Any trace where a verification actually ran — both accepted and
    rejected cases. user_feedback is always None today — no feedback
    collection mechanism exists yet, a stated gap, not silently omitted."""
    candidates = []
    for t in traces:
        verification = t.get("verification")
        if verification is None:
            continue
        decision = t.get("routing_decision") or {}
        candidates.append({
            "trace_id": t.get("trace_id"),
            "domain": decision.get("domain"),
            "task": decision.get("task"),
            "request": t.get("request"),
            "local_response": t.get("local_response"),
            "verifier_accepted": verification.get("accepted"),
            "verifier_score": verification.get("score"),
            "verifier_reasons": verification.get("failure_reasons"),
            "frontier_response": t.get("frontier_response") if t.get("escalated") else None,
            "user_feedback": t.get("user_feedback"),
            "final_outcome": t.get("final_source"),
        })
    return candidates


def write_dataset(output_root: Path, candidate_type: str, candidates: list[dict], source_path: Path, source_trace_count: int, extra_manifest: dict) -> Path:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out_dir = output_root / candidate_type / version
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "data.jsonl"
    with open(data_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, default=str) + "\n")

    manifest = {
        "dataset_type": candidate_type,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_trace_file": str(source_path),
        "source_trace_count": source_trace_count,
        "candidate_count": len(candidates),
        **extra_manifest,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to an experience_traces.jsonl file")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "datasets"), help="Root directory for versioned dataset output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_dir)

    traces = load_traces(input_path)
    if not traces:
        print(f"No traces found at {input_path} — nothing to generate.")
        return

    router_versions = {t.get("routing_decision", {}).get("router_version") for t in traces if t.get("routing_decision")}
    verifier_versions = {t.get("verification", {}).get("verifier_version") for t in traces if t.get("verification")}

    router_candidates = build_router_candidates(traces)
    distill_candidates, distill_stats = build_distill_candidates(traces)
    verifier_candidates = build_verifier_candidates(traces)

    router_dir = write_dataset(output_root, "router", router_candidates, input_path, len(traces),
                                {"router_versions": sorted(v for v in router_versions if v)})
    distill_dir = write_dataset(output_root, "distill", distill_candidates, input_path, len(traces),
                                 {"exclusion_stats": distill_stats})
    verifier_dir = write_dataset(output_root, "verifier", verifier_candidates, input_path, len(traces),
                                  {"verifier_versions": sorted(v for v in verifier_versions if v)})

    print(f"Source: {input_path} ({len(traces)} traces)")
    print(f"  router   -> {len(router_candidates)} candidates  ({router_dir})")
    print(f"  distill  -> {len(distill_candidates)} candidates  ({distill_dir})")
    if len(distill_candidates) == 0 and distill_stats["fully_redacted"] > 0:
        print(f"             0 distillation candidates — {distill_stats['fully_redacted']} escalated trace(s) were "
              f"fully redacted (redact_mode=\"full\"). Re-run against a deployment using "
              f"AI_ROUTER_EXPERIENCE_REDACT=partial or none to collect usable candidates.")
    print(f"  verifier -> {len(verifier_candidates)} candidates  ({verifier_dir})")


if __name__ == "__main__":
    main()
