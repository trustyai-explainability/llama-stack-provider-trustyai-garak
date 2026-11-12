"""Stream Garak observations into an atomic per-sample artifact."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..result_utils import _detector_scores, _intent_attempt_accepted, _source_stub


def _entries(report: Path):
    """Read one report record at a time without retaining the whole scan."""
    with report.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _attempts(report: Path):
    """Yield completed standard attempts with flattened, validated score arrays."""
    for entry in _entries(report):
        if entry.get("entry_type") != "attempt" or entry.get("status") != 2:
            continue
        if not isinstance(entry.get("uuid"), str) or not entry["uuid"]:
            raise ValueError("Completed attempts require unique UUIDs")
        outputs = entry.get("outputs") or []
        detectors = entry.get("detector_results") or {}
        if not outputs or not detectors:
            raise ValueError("Per-sample export requires outputs and detector scores")
        for detector, values in detectors.items():
            scores = _detector_scores(values)
            if len(scores) != len(outputs):
                raise ValueError("Detector scores must align with generated responses")
            if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) for s in scores):
                raise ValueError("Per-sample detector scores must be finite numbers")
            detectors[detector] = scores
        yield entry


def _standard_records(report: Path, threshold: float):
    """Select conservative detectors, then stream their response observations."""
    hits = defaultdict(lambda: defaultdict(int))
    seen = set()
    for entry in _attempts(report):
        if entry["uuid"] in seen:
            raise ValueError("Completed attempts require unique UUIDs")
        seen.add(entry["uuid"])
        for detector, scores in entry["detector_results"].items():
            hits[entry["probe_classname"]][detector] += sum(s >= threshold for s in scores)
    # Standard AVID's passed/total_evaluated count responses, even when an
    # attempt has multiple generations. Counting hit attempts changes ASR.
    selected = {probe: max(sorted(counts), key=counts.get) for probe, counts in hits.items()}
    for entry in _attempts(report):
        probe = entry["probe_classname"]
        detector = selected[probe]
        if detector not in entry["detector_results"]:
            raise ValueError("A probe's selected detector is missing from an attempt")
        for index, (output, score) in enumerate(zip(entry["outputs"], entry["detector_results"][detector])):
            yield {
                "sample_id": f"{entry['uuid']}:{index}",
                "probe": probe,
                "response": output,
                "detector": detector,
                "detector_score": score,
                "attack_success_rate": float(score >= threshold),
            }


def _stub_id(intent: str, stub: str) -> str:
    """Identify a source stub consistently across its probes and retries."""
    return hashlib.sha256(json.dumps([intent, stub], ensure_ascii=False).encode("utf-8")).hexdigest()


def _intent_attempts(report: Path):
    """Match the shared parser's completed/orphan snapshots and real probes."""
    completed = {
        entry["uuid"]
        for entry in _entries(report)
        if entry.get("entry_type") == "attempt" and entry.get("status") == 2 and entry.get("uuid") is not None
    }
    for entry in _entries(report):
        if entry.get("entry_type") != "attempt" or str(entry.get("probe_classname", "")).lower().endswith(
            "earlystopharness"
        ):
            continue
        if entry.get("status") == 2 or (entry.get("status") == 1 and entry.get("uuid") not in completed):
            yield entry


def _intent_records(report: Path, threshold: float):
    """Export final source-stub observations, preserving the intents estimand."""
    # Current harness summaries are authoritative, including unattempted stubs.
    has_summaries = any(e.get("entry_type") == "harness_stub_summary" for e in _entries(report))
    outcomes = {}
    if not has_summaries:
        # Older reports require an OR across attempts for each (intent, stub).
        # Retain only hashed identities and booleans, never their response bodies.
        for entry in _intent_attempts(report):
            stub = _source_stub(entry)
            if stub:
                ident = _stub_id(str(entry.get("intent") or "unknown"), stub)
                outcomes[ident] = outcomes.get(ident, False) or _intent_attempt_accepted(entry, threshold)
    seen = set()
    for entry in _entries(report) if has_summaries else _intent_attempts(report):
        if has_summaries:
            if entry.get("entry_type") != "harness_stub_summary":
                continue
            stub = str(entry.get("source_stub") or "")
        else:
            stub = _source_stub(entry)
        if not stub:
            continue
        intent = str(entry.get("intent") or "unknown")
        ident = _stub_id(intent, stub)
        if ident in seen:
            if has_summaries:
                raise ValueError("Harness stub summaries require unique source stubs")
            continue
        seen.add(ident)
        accepted = bool(entry.get("accepted", False)) if has_summaries else outcomes[ident]
        yield {
            "sample_id": ident,
            "intent": intent,
            "source_stub": stub,
            "sample_unit": "source_stub",
            "attack_success_rate": float(accepted),
        }


def save_sample_results(report: Path, target: Path, threshold: float, *, art_intents: bool = False) -> tuple[Path, int]:
    """Publish validated observations atomically, with at most one attempt in memory."""
    records = _intent_records(report, threshold) if art_intents else _standard_records(report, threshold)
    temporary = None
    try:
        # A same-directory temporary file prevents readers seeing a partial export.
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=".samples-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            count = 0
            for record in records:
                json.dump(record, stream, ensure_ascii=False)
                stream.write("\n")
                count += 1
            if not count:
                raise ValueError("No completed scored observations in the Garak report")
        temporary.replace(target)
        return target, count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
