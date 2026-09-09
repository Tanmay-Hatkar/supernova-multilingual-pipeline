"""
Loads and validates the language registry (configs/languages.yaml).

This is the one place the rest of the pipeline goes to answer "which
languages exist, and what does each one need" — nothing else in this
codebase should hardcode a list of supported languages or directions.
Adding a new language should never require editing this file's logic,
only adding an entry to languages.yaml plus that language's own config
and data folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGUAGES_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "languages.yaml")


@dataclass
class LanguageEntry:
    code: str
    name: str
    direction: str
    judge_config_path: str
    data_dir: str
    status: str  # "active" | "planned"

    @property
    def is_active(self) -> bool:
        return self.status == "active"


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_registry(path: str = LANGUAGES_CONFIG_PATH) -> list[LanguageEntry]:
    """
    Load every language entry from languages.yaml, active or planned.
    Raises a clear error if an entry is missing a required field or
    points at a judge config file that doesn't exist yet, so a broken
    registry fails loudly at load time, not silently mid-run.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    entries = []
    for item in raw.get("languages", []):
        for required in ("code", "name", "direction", "judge_config", "data_dir", "status"):
            if required not in item:
                raise ValueError(f"Language entry missing required field '{required}': {item}")

        judge_config_path = _resolve(item["judge_config"])
        if not os.path.exists(judge_config_path):
            raise FileNotFoundError(
                f"Language '{item['code']}' points at judge config "
                f"'{item['judge_config']}', which does not exist."
            )

        entries.append(
            LanguageEntry(
                code=item["code"],
                name=item["name"],
                direction=item["direction"],
                judge_config_path=judge_config_path,
                data_dir=_resolve(item["data_dir"]),
                status=item["status"],
            )
        )

    codes = [e.code for e in entries]
    duplicates = {c for c in codes if codes.count(c) > 1}
    if duplicates:
        raise ValueError(f"Duplicate language codes in registry: {duplicates}")

    return entries


def get_active_languages(path: str = LANGUAGES_CONFIG_PATH) -> list[LanguageEntry]:
    """The subset of the registry actually ready to run in the optimization loop."""
    return [e for e in load_registry(path) if e.is_active]


def get_judge_config(language_code: str, path: str = LANGUAGES_CONFIG_PATH) -> dict:
    """Load a specific language's judge/scoring config as a plain dict."""
    entries = {e.code: e for e in load_registry(path)}
    if language_code not in entries:
        raise KeyError(f"Unknown language code '{language_code}'. Known: {sorted(entries)}")

    with open(entries[language_code].judge_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
