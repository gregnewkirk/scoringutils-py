"""Weekly check: has anything this port depends on moved?

Compares the scoringutils version on CRAN with the one this port was checked
against (``UPSTREAM_VERSION`` in ``src/scoringutils_py/__init__.py``), and the
newest release of each runtime dependency on PyPI with the range allowed in
``pyproject.toml``.

Writes a Markdown report to ``watch-report.md`` and pins for any dependency
release outside the allowed range to ``watch-overrides.txt``, so the workflow
can try the tests against them. Prints ``drift=true`` or ``drift=false``.

Run with: ``uv run --no-project --with packaging python scripts/upstream_watch.py``
"""

from __future__ import annotations

import json
import os
import re
import tomllib
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
CRAN_URL = "https://crandb.r-pkg.org/scoringutils"
PYPI_URL = "https://pypi.org/pypi/{name}/json"
NEWS_URL = "https://epiforecasts.io/scoringutils/news/"


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "upstream-watch"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def ported_version() -> str:
    text = (ROOT / "src" / "scoringutils_py" / "__init__.py").read_text()
    match = re.search(r'^UPSTREAM_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("UPSTREAM_VERSION not found in __init__.py")
    return match.group(1)


def out_of_range_dependencies() -> list[tuple[str, str, str]]:
    """(name, allowed range, newest release) for each capped-out dependency."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    found = []
    for line in project["dependencies"]:
        requirement = Requirement(line)
        info = _get_json(PYPI_URL.format(name=requirement.name))["info"]
        newest = Version(info["version"])
        if newest.is_prerelease or requirement.specifier.contains(newest):
            continue
        found.append((requirement.name, str(requirement.specifier), str(newest)))
    return found


def main() -> None:
    ported = os.environ.get("PORTED_VERSION") or ported_version()
    cran = _get_json(CRAN_URL)["Version"]
    capped = out_of_range_dependencies()

    lines = []
    if Version(cran) != Version(ported):
        lines += [
            f"- **scoringutils {cran} is on CRAN.** This port was checked against "
            f"{ported}. Read the [upstream changelog]({NEWS_URL}), port what "
            "changed, regenerate the fixtures with `r/generate_fixtures.R`, and "
            "update `UPSTREAM_VERSION`.",
        ]
    for name, allowed, newest in capped:
        lines.append(
            f"- **{name} {newest} is on PyPI**, outside the allowed range "
            f"`{allowed}` in `pyproject.toml`."
        )

    (ROOT / "watch-report.md").write_text("\n".join(lines) + ("\n" if lines else ""))
    (ROOT / "watch-overrides.txt").write_text(
        "".join(f"{name}=={newest}\n" for name, _, newest in capped)
    )
    print(f"ported={ported} cran={cran} capped={capped}")
    drift = "true" if lines else "false"
    print(f"drift={drift}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"drift={drift}\n")


if __name__ == "__main__":
    main()
