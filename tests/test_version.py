"""The package version must have one value everywhere it is written."""

import re
from pathlib import Path

import scoringutils_py

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no version in pyproject.toml"
    return match.group(1)


def test_dunder_version_matches_pyproject():
    assert scoringutils_py.__version__ == _pyproject_version()


def test_changelog_has_section_for_current_version():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{_pyproject_version()}]" in changelog
