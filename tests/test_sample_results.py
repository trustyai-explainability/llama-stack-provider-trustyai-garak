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
