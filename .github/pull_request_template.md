## What and why

<!-- One or two sentences. Link an issue if there is one. -->

## Checklist

- [ ] Branch from `main`, Conventional Commit title (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)
- [ ] Tests added or updated; ported behaviour is checked against an R-generated fixture
- [ ] `uv run pytest -q`, `uv run ruff check src tests` and `uv run ruff format --check src tests` pass
- [ ] Fixtures regenerate byte-identical, or this PR explains which changed and why
- [ ] `CHANGELOG.md` has a line under `## [Unreleased]`
- [ ] README parity table updated if a function's status changed
