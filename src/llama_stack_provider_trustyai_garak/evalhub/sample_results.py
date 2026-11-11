"""Persist standard Garak response observations for confidence-interval jobs."""

import json
import math
from collections import defaultdict
from pathlib import Path


def save_sample_results(report: Path, target: Path, threshold: float) -> tuple[Path, int]:
    """Export completed responses using each probe's most conservative detector."""
    attempts = []
    detector_hits = defaultdict(lambda: defaultdict(int))
    seen = set()
    for line in report.read_text().splitlines():
        entry = json.loads(line)
        # Setup records and pre-detection snapshots are not scored observations.
        if entry.get("entry_type") != "attempt" or entry.get("status") != 2:
            continue
        ident, probe = entry.get("uuid"), entry.get("probe_classname")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("Completed attempts require unique UUIDs")
        seen.add(ident)
        outputs = entry.get("outputs") or []
        detectors = entry.get("detector_results") or {}
        if not outputs or not detectors:
            raise ValueError("Per-sample export requires outputs and detector scores")
        for detector, scores in detectors.items():
            if len(scores) != len(outputs):
                raise ValueError("Detector scores must align with generated responses")
            if any(
                isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                for score in scores
            ):
                raise ValueError("Per-sample detector scores must be finite numbers")
            detector_hits[probe][detector] += sum(score >= threshold for score in scores)
        attempts.append(entry)
    if not attempts:
        raise ValueError("No completed scored responses in the Garak report")

    # AVID computes standard ASR from the detector with the fewest passed
    # responses per probe. Choose that detector globally, not per response.
    selected = {probe: max(sorted(hits), key=hits.get) for probe, hits in detector_hits.items()}
    records = []
    for entry in attempts:
        probe = entry["probe_classname"]
        detector = selected[probe]
        if detector not in entry["detector_results"]:
            raise ValueError("A probe's selected detector is missing from an attempt")
        for index, (output, score) in enumerate(zip(entry["outputs"], entry["detector_results"][detector])):
            records.append(
                {
                    "sample_id": f"{entry['uuid']}:{index}",
                    "probe": probe,
                    "response": output,
                    "detector": detector,
                    "detector_score": score,
                    "attack_success_rate": float(score >= threshold),
                }
            )
    # Write only after validation, so an invalid scan cannot leave a partial export.
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    return target, len(records)
