# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Every release lists the upstream scoringutils version it was checked against.

## [Unreleased]

### Added

- `CHANGELOG.md`, `CONTRIBUTING.md` with the branch, commit and release
  process, a pull request template, issue templates and Dependabot updates for
  GitHub Actions and Python dependencies.
- Release workflow: pushing a `vX.Y.Z` tag checks that it matches the version
  in `pyproject.toml`, builds the sdist and wheel, and publishes a GitHub
  Release whose notes are that version's section of this file.
- CI now also builds the package on every push and pull request.

## [0.2.0] - 2026-10-19

Checked against R scoringutils 2.3.0 (CRAN, 2026-09-27) and scoringRules 1.1.3.

### Added

- Point forecasts: `as_forecast_point` (including conversion from a quantile
  forecast's median), `is_forecast_point`, default metrics and `score()`
  method.
- Binary forecasts: `as_forecast_binary`, `is_forecast_binary`, default
  metrics and `score()` method. Binary outcomes are pandas Categoricals, the
  counterpart of R factors.
- Metric functions `ae_point`, `se_point`, `ape`, `brier_score` and
  `logs_binary`.
- Bundled example data `example_point` and `example_binary` (MIT, from
  scoringutils).
- Parity fixtures for score, summarise, pairwise comparison and relative skill
  on both new types, plus 22 vector-level metric cases. Existing fixtures
  regenerate byte-identical.

### Testing

- 405 tests pass on Python 3.10 and 3.12; CI runs 3.10, 3.12 and 3.13.

## [0.1.0] - 2026-10-19

First release. Checked against R scoringutils 2.3.0 (CRAN, 2026-09-27) and
scoringRules 1.1.3.

### Added

- Quantile and sample forecasts: construction and validation
  (`as_forecast_quantile`, `as_forecast_sample`, forecast-unit detection and
  duplicate checks).
- `score()` with the scoringutils default metrics for both types, including
  the weighted interval score and its components, CRPS, log score, DSS, bias,
  dispersion, overprediction, underprediction and coverage.
- `summarise_scores()` with grouping, pairwise comparisons and relative skill
  (`add_relative_skill`), and `get_coverage()`.
- Bundled example data `example_quantile`, `example_sample_continuous` and
  `example_sample_discrete` (MIT, from scoringutils).
- `r/generate_fixtures.R`, which writes reference outputs from pinned R
  packages into `tests/fixtures`, so the Python test suite never needs R.

### Notes

- The weighted interval score is computed directly from the R definition.
  `scoringrules` 0.9 to 0.11's numpy backend for `weighted_interval_score`
  adds the weighted median instead of the weighted absolute error of the
  median.
- Estimators are passed explicitly where `scoringrules` defaults differ from
  R (`crps_ensemble(estimator="qd")`, `dssuv_ensemble(bias=True)`,
  `logs_ensemble(bw=bw.nrd(x))`).

### Testing

- 301 tests pass on Python 3.10 and 3.12.

[Unreleased]: https://github.com/gregnewkirk/scoringutils-py/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/gregnewkirk/scoringutils-py/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gregnewkirk/scoringutils-py/releases/tag/v0.1.0
