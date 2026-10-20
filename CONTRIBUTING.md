# Contributing

Thanks for helping. This project is an unofficial Python port of
[scoringutils](https://github.com/epiforecasts/scoringutils). Its whole value is
that it gives the same numbers as R, so every change has to keep the parity
tests green.

## Setup

```bash
uv sync
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Regenerating fixtures needs R with the pinned packages listed at the top of
`r/generate_fixtures.R`:

```bash
Rscript r/generate_fixtures.R
```

Fixtures must regenerate byte-identical unless the change is meant to alter
them, and a pull request that changes a fixture must say why.

## Making a change

1. Branch from `main`: `feat/...`, `fix/...`, `docs/...`, `test/...` or
   `chore/...`. Do not push to `main` directly.
2. Write commits in [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
   form: `feat: add get_pit_histogram`, `fix: handle NA observed in bias`,
   `docs: ...`, `test: ...`, `chore: ...`. Add `!` for a breaking change
   (`feat!: ...`) and explain it in the body.
3. Add or update tests. A ported function needs at least one test against an
   R-generated fixture.
4. Add a line under `## [Unreleased]` in `CHANGELOG.md`, in the right section
   (Added, Changed, Deprecated, Removed, Fixed, Security).
5. Update the parity table in `README.md` if a function's status changes.
6. Open a pull request. CI must pass on every supported Python version before
   it is merged. Merge with **squash**, keeping the Conventional Commit title.

## Releasing

Versions follow [Semantic Versioning](https://semver.org/). Before 1.0.0, a
minor bump (0.x.0) may change the public API and a patch bump (0.x.y) is for
fixes only.

1. On a `chore/release-X.Y.Z` branch:
   - set `version` in `pyproject.toml`;
   - rename `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`,
     add a fresh empty `## [Unreleased]` above it, update the compare links at
     the bottom, and state which scoringutils version it was checked against;
   - update the version line in `README.md`.
2. Open the pull request, wait for CI, squash-merge.
3. Tag the merge commit on `main` and push the tag:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

4. The release workflow checks the tag matches `pyproject.toml`, runs the tests,
   builds the sdist and wheel, and publishes the GitHub Release with that
   version's changelog section as its notes and the built files attached.

Publishing to PyPI is not set up yet.

## Credit

Metric definitions, defaults and test cases come from scoringutils. Keep the
credit in `README.md` and the upstream notice in `LICENSE` intact.
