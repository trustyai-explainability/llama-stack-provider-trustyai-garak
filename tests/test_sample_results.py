"""Check that sample exports preserve standard Garak ASR observations."""

import json

import pytest

from llama_stack_provider_trustyai_garak.evalhub.sample_results import save_sample_results


def attempt(ident="id", scores=None, outputs=None):
    """Construct a completed attempt with aligned detector observations."""
    return {
        "entry_type": "attempt",
        "status": 2,
        "uuid": ident,
        "probe_classname": "dan.Test",
        "outputs": outputs or ["yes", "no"],
        "detector_results": scores or {"primary": [1, 0]},
    }


def export(tmp_path, entries):
    """Write a report and invoke the same exporter used before persistence."""
    source = tmp_path / "scan.report.jsonl"
    source.write_text("".join(json.dumps(x) + "\n" for x in entries))
    target, count = save_sample_results(source, tmp_path / "samples.jsonl", 0.5)
    return [json.loads(x) for x in target.read_text().splitlines()], count


def test_completed_responses_exclude_metadata_and_incomplete_snapshots(tmp_path):
    complete = attempt()
    records, count = export(tmp_path, [{"entry_type": "init"}, {**complete, "status": 1}, complete])
    assert count == 2
    assert [x["sample_id"] for x in records] == ["id:0", "id:1"]
    assert [x["attack_success_rate"] for x in records] == [1.0, 0.0]
    assert [x["response"] for x in records] == ["yes", "no"]


def test_detector_is_selected_across_the_probe_not_per_response(tmp_path):
    records, _ = export(tmp_path, [attempt("a", {"a": [1, 0], "b": [0, 1]}), attempt("b", {"a": [0, 0], "b": [1, 0]})])
    assert {x["detector"] for x in records} == {"b"}
    assert sum(x["attack_success_rate"] for x in records) == 2


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [attempt(), attempt()],
        [attempt(scores={"a": [1]})],
        [attempt(scores={"a": [None, 1]})],
        [attempt(scores={"a": [float("nan"), 1]})],
    ],
)
def test_invalid_observations_leave_no_partial_file(tmp_path, entries):
    with pytest.raises(ValueError):
        export(tmp_path, entries)
    assert not (tmp_path / "samples.jsonl").exists()


def test_nested_scores_match_shared_parser(tmp_path):
    records, count = export(tmp_path, [attempt(scores={"primary": [[1], [[0]]]})])
    assert count == 2
    assert [row["detector_score"] for row in records] == [1, 0]


def test_conservative_detector_counts_responses_not_hit_attempts(tmp_path):
    # Clustered wins on responses (3 > 2), distributed on hit attempts (2 > 1).
    records, _ = export(
        tmp_path,
        [
            attempt("a", {"clustered": [1, 1, 1], "distributed": [1, 0, 0]}, ["a", "b", "c"]),
            attempt("b", {"clustered": [0, 0, 0], "distributed": [1, 0, 0]}, ["d", "e", "f"]),
        ],
    )
    assert {row["detector"] for row in records} == {"clustered"}
    assert sum(row["attack_success_rate"] for row in records) == 3


def test_export_streams_and_publishes_atomically(tmp_path, monkeypatch):
    from pathlib import Path

    source = tmp_path / "scan.report.jsonl"
    source.write_text(json.dumps(attempt()))
    target = tmp_path / "samples.jsonl"
    target.write_text("previous export")
    original_dump = json.dump

    def checked_dump(record, stream, **kwargs):
        # The destination remains intact until the complete export is validated.
        assert target.read_bytes() == b"previous export"
        original_dump(record, stream, **kwargs)

    def reject_bulk_read(*args, **kwargs):
        raise AssertionError("Exporter must stream the source")

    monkeypatch.setattr(Path, "read_text", reject_bulk_read)
    monkeypatch.setattr(json, "dump", checked_dump)
    assert save_sample_results(source, target, 0.5)[1] == 2
    assert len(target.read_bytes().splitlines()) == 2
    assert not list(tmp_path.glob(".samples-*"))


def test_failed_export_preserves_previous_file_and_cleans_temporary(tmp_path):
    source = tmp_path / "scan.report.jsonl"
    source.write_text(json.dumps(attempt()) + "\n" + "invalid json")
    target = tmp_path / "samples.jsonl"
    target.write_text("previous export")
    with pytest.raises(ValueError):
        save_sample_results(source, target, 0.5)
    assert target.read_text() == "previous export"
    assert not list(tmp_path.glob(".samples-*"))


def test_intent_summaries_preserve_source_stub_unit(tmp_path):
    from pathlib import Path

    source = Path(__file__).parent / "_resources" / "garak_v017_earlystop_run.jsonl"
    target, count = save_sample_results(source, tmp_path / "samples.jsonl", 0.5, art_intents=True)
    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert count == 2
    assert {row["sample_unit"] for row in records} == {"source_stub"}
    assert len({row["sample_id"] for row in records}) == 2
    assert sum(row["attack_success_rate"] for row in records) / count == 0.5


def test_legacy_intents_combine_acceptance_across_attempts(tmp_path):
    source = tmp_path / "scan.report.jsonl"
    entries = [
        {**attempt("a"), "intent": "fraud", "goal": "stub", "detector_results": {"judge": [[0, 0]]}},
        {**attempt("b"), "intent": "fraud", "goal": "stub", "detector_results": {"judge": [[1, 1]]}},
        {**attempt("c"), "intent": "fraud", "goal": "other", "detector_results": {"judge": [0, 0]}},
    ]
    source.write_text("\n".join(map(json.dumps, entries)))
    target, count = save_sample_results(source, tmp_path / "samples.jsonl", 0.5, art_intents=True)
    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert count == 2
    assert {row["source_stub"]: row["attack_success_rate"] for row in records} == {"stub": 1.0, "other": 0.0}


def test_legacy_intents_ignore_superseded_snapshots_and_harness(tmp_path):
    source = tmp_path / "scan.report.jsonl"
    completed = {**attempt(), "goal": "stub", "intent": "fraud", "detector_results": {"judge": [0, 0]}}
    snapshot = {**completed, "status": 1, "detector_results": {"judge": [1, 1]}}
    harness = {**attempt("harness"), "goal": "synthetic stub", "probe_classname": "harness.EarlyStopHarness"}
    source.write_text("\n".join(map(json.dumps, [snapshot, completed, harness])))
    target, count = save_sample_results(source, tmp_path / "samples.jsonl", 0.5, art_intents=True)
    assert count == 1
    assert json.loads(target.read_text())["attack_success_rate"] == 0.0


def test_late_export_failure_does_not_publish_partial_results(tmp_path):
    source = tmp_path / "scan.report.jsonl"
    # 'primary' wins globally but is missing from the second attempt.
    entries = [attempt("a", {"primary": [1, 1]}), attempt("b", {"other": [0, 0]})]
    source.write_text("\n".join(map(json.dumps, entries)))
    target = tmp_path / "samples.jsonl"
    target.write_text("previous export")
    with pytest.raises(ValueError, match="selected detector is missing"):
        save_sample_results(source, target, 0.5)
    assert target.read_text() == "previous export"
    assert not list(tmp_path.glob(".samples-*"))
