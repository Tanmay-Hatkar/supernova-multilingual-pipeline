"""
Tests for filtering defective references out of the data pools.

The pipeline already refuses to let a judge grade a property a string
comparison can settle. This applies the same rule one step earlier, to
the data itself: a human reference that fails the gate is not a
translation worth training on or being measured against.

Two properties carry it. The filter must use the same checks the
language's config names, so data and output cannot drift apart. And it
must leave a record of what it removed, because a pool that silently
shrinks looks like data loss rather than a decision.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prepare_data
from prepare_data import audit_pools, filter_references


def _candidate(source: str, reference: str) -> dict:
    return {"source": source, "reference": reference, "length_band": "short"}


# --- filtering --------------------------------------------------------


def test_a_clean_reference_is_kept():
    kept, dropped = filter_references(
        [_candidate("There are 9 people.", "Hay 9 personas.")], ["numbers_preserved"]
    )
    assert len(kept) == 1
    assert dropped == {}


def test_a_reference_with_the_wrong_number_is_dropped():
    # The defect that prompted this: a reference reading 395 where its
    # source says 381. Scored as ground truth, it penalises a correct
    # translation; handed to the judge, it produces a real error report
    # that belongs to the data rather than to the model.
    kept, dropped = filter_references(
        [_candidate("We have 381 guests.", "Hay 395 invitados.")], ["numbers_preserved"]
    )
    assert kept == []
    assert dropped == {"numbers_preserved": 1}


def test_a_reference_identical_to_its_source_is_dropped():
    # not_source_copy is universal, so this is caught without any
    # language config asking for it.
    kept, dropped = filter_references([_candidate("Take my heart", "Take my heart")], [])
    assert kept == []
    assert "not_source_copy" in dropped


def test_an_empty_reference_is_dropped():
    kept, _ = filter_references([_candidate("Hello there.", "   ")], [])
    assert kept == []


def test_each_failing_check_is_counted_separately():
    kept, dropped = filter_references(
        [
            _candidate("We have 381 guests.", "Hay 395 invitados."),
            _candidate("It is here.", "Está aquí,"),
        ],
        ["numbers_preserved", "not_truncated"],
    )
    assert kept == []
    assert dropped == {"numbers_preserved": 1, "not_truncated": 1}


def test_only_the_checks_requested_are_applied():
    # A reference ending mid-clause survives when truncation is not
    # among the language's checks. The filter follows the config
    # rather than deciding for itself, so data and output are gated on
    # the same rules.
    candidates = [_candidate("It is here.", "Está aquí,")]
    assert len(filter_references(candidates, ["numbers_preserved"])[0]) == 1
    assert filter_references(candidates, ["not_truncated"])[0] == []


def test_filtering_preserves_order_and_fields():
    candidates = [
        _candidate("One person.", "Una persona."),
        _candidate("We have 381 guests.", "Hay 395 invitados."),
        _candidate("Two people.", "Dos personas."),
    ]
    kept, _ = filter_references(candidates, ["numbers_preserved"])
    assert [c["source"] for c in kept] == ["One person.", "Two people."]
    assert kept[0]["length_band"] == "short"


def test_a_clean_batch_reports_nothing_dropped():
    kept, dropped = filter_references(
        [_candidate("One person.", "Una persona.")], ["numbers_preserved", "not_truncated"]
    )
    assert len(kept) == 1
    assert dropped == {}


# --- auditing pools on disk -------------------------------------------


@pytest.fixture
def pools(tmp_path, monkeypatch):
    """Four pools on disk, with one known defect in one of them."""
    data_dir = tmp_path / "data" / "xx"
    rows = {
        "examples": [
            {"source": "There are 9 people.", "reference": "Hay 9 personas."},
            {"source": "We have 381 guests.", "reference": "Hay 395 invitados."},
        ],
        "prompt_validation": [{"source": "One person.", "reference": "Una persona."}],
        "judge_calibration": [{"source": "Two people.", "reference": "Dos personas."}],
        "final_test": [{"source": "Three people.", "reference": "Tres personas."}],
    }
    for pool, content in rows.items():
        path = (
            data_dir / "examples.json" if pool == "examples" else data_dir / pool / "examples.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(prepare_data, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(prepare_data, "checks_for", lambda code: ["numbers_preserved"])
    return data_dir


def test_audit_finds_the_defect_and_locates_it(pools):
    summary = audit_pools("xx")
    assert summary["defective"] == 1
    assert summary["total"] == 5
    assert summary["pools"]["examples"]["defective"] == 1
    assert summary["pools"]["final_test"]["defective"] == 0


def test_audit_reports_the_rate_per_pool(pools):
    summary = audit_pools("xx")
    assert summary["pools"]["examples"]["rate"] == 0.5
    assert summary["rate"] == 0.2


def test_audit_names_the_check_that_failed(pools):
    summary = audit_pools("xx")
    assert summary["pools"]["examples"]["by_check"] == {"numbers_preserved": 1}


def test_audit_carries_the_offending_rows_for_inspection(pools):
    # A rate alone cannot be acted on. The rows are what tells you
    # whether the data is defective or the check is.
    offender = audit_pools("xx")["pools"]["examples"]["examples"][0]
    assert offender["source"] == "We have 381 guests."
    assert offender["failed"] == ["numbers_preserved"]


def test_audit_skips_pools_that_do_not_exist(pools, monkeypatch):
    (pools / "final_test" / "examples.json").unlink()
    summary = audit_pools("xx")
    assert "final_test" not in summary["pools"]
    assert summary["total"] == 4
