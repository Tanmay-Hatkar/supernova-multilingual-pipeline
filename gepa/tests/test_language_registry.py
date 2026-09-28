"""
Tests for the language registry loader. These use the real
configs/languages.yaml shipped in this repo, since the registry
itself is the thing under test, not a mock of it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from language_registry import get_active_languages, get_judge_config, load_registry


def test_loads_all_entries():
    entries = load_registry()
    codes = {e.code for e in entries}
    assert codes == {"yue", "cmn", "es"}


def test_active_languages_excludes_planned():
    # Asserts the property, not today's roster: whatever the registry
    # says, the active set is exactly the entries marked active. Pinning
    # this to a specific list of codes makes the test fail every time a
    # language changes status, which is a status change, not a bug.
    entries = load_registry()
    active = {e.code for e in get_active_languages()}
    assert active == {e.code for e in entries if e.status == "active"}
    assert active.isdisjoint({e.code for e in entries if e.status == "planned"})


def test_planned_status_is_actually_excluded(tmp_path):
    registry_path = tmp_path / "languages.yaml"
    registry_path.write_text(
        f"""
languages:
  - code: yue
    name: Cantonese
    direction: en_yue
    judge_config: {load_registry()[0].judge_config_path}
    data_dir: data/yue
    status: active
  - code: xx
    name: Test Planned Language
    direction: en_xx
    judge_config: {load_registry()[0].judge_config_path}
    data_dir: data/xx
    status: planned
""",
        encoding="utf-8",
    )
    active = {e.code for e in get_active_languages(str(registry_path))}
    assert active == {"yue"}
    assert "xx" not in active


def test_rejects_duplicate_codes(tmp_path):
    bad_registry = tmp_path / "languages.yaml"
    bad_registry.write_text(
        """
languages:
  - code: yue
    name: Cantonese
    direction: en_yue
    judge_config: configs/judges/yue.yaml
    data_dir: data/yue
    status: active
  - code: yue
    name: Cantonese Again
    direction: en_yue
    judge_config: configs/judges/yue.yaml
    data_dir: data/yue
    status: active
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate language codes"):
        load_registry(str(bad_registry))


def test_rejects_missing_judge_config(tmp_path):
    bad_registry = tmp_path / "languages.yaml"
    bad_registry.write_text(
        """
languages:
  - code: xx
    name: Nonexistent
    direction: en_xx
    judge_config: configs/judges/does_not_exist.yaml
    data_dir: data/xx
    status: active
""",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_registry(str(bad_registry))


def test_get_judge_config_returns_expected_shape():
    config = get_judge_config("yue")
    assert config["language_code"] == "yue"
    assert config["scoring_method"] in {"flat", "severity_weighted"}


def test_get_judge_config_unknown_language_raises():
    with pytest.raises(KeyError):
        get_judge_config("not_a_real_language")
