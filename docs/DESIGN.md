# Design: a faithful Python port of scoringutils (v1)

Status: approved by Greg Newkirk, 2026-10-19

This is an unofficial Python port of the R package
[scoringutils](https://github.com/epiforecasts/scoringutils) (CRAN 2.3.0,
2026-09-27, MIT, by Nikos Bosse, Sebastian Funk and contributors at
epiforecasts). It is not affiliated with or endorsed by the upstream authors.

## 1. Goal and non-goals

Goal: give Python users the scoringutils workflow (validate forecasts, score
them, summarise, compare models) with numbers that match R scoringutils 2.3.0
to about 1e-10.

v1 covers quantile and sample forecasts only.

Not in v1: binary, point, nominal, ordinal and multivariate forecasts; plots;
PIT histograms; `transform_forecasts()`; `get_forecast_counts()`;
`get_correlations()`; missing-score imputation; `filter_scores()`. These are
listed as "not yet" in the README parity table.

## 2. Package name

PyPI name `scoringutils` is deliberately left for upstream. All candidates
below returned 404 (free) on PyPI on 2026-10-19:

| Option | PyPI name | `import` name | Note |
|---|---|---|---|
| A (recommended) | `scoringutils-py` | `scoringutils_py` | Instantly recognisable, says what it is. |
| B | `pyscoringutils` | `pyscoringutils` | Same idea, one word. |
| C | `episcore` | `episcore` | Own identity, but hides the lineage. |

The rest of this document uses option A. Renaming later is a find-and-replace.

## 3. Layout and tooling

```
scoringutils-py/
  pyproject.toml           uv + hatchling, Python >= 3.10
  LICENSE                  MIT, upstream copyright kept, port copyright added
  README.md                worked example + parity table
  docs/DESIGN.md           this file
  src/scoringutils_py/
    __init__.py            public API (mirrors R export names)
    _forecast.py           forecast classes, as_forecast_*, validation, forecast unit, duplicates
    _metrics_quantile.py   wis, components, bias, coverage, ae_median, quantile_score
    _metrics_sample.py     crps, components, bias, dss, log score, mad, ae_median, se_mean
    _interval.py           quantile <-> interval helpers (internal)
    _score.py              score(), get_metrics(), select_metrics(), Scores class
    _summarise.py          summarise_scores()
    _pairwise.py           get_pairwise_comparisons(), add_relative_skill(), R-exact Wilcoxon + Holm
    _coverage.py           get_coverage()
    _rcompat.py            small exact ports of R behaviour (get_type, bw.nrd, mad)
    data/                  example_quantile, example_sample_continuous, example_sample_discrete (csv.gz)
  tests/
    fixtures/              expected outputs written by R (committed)
    test_*.py              parity tests + ported testthat cases
  r/
    generate_fixtures.R    regenerates tests/fixtures with pinned R packages
```

Dependencies: numpy, pandas (>= 2.0), scipy (normal distribution only),
scoringrules == 0.11.x. Dev: pytest, ruff, mypy-friendly type hints.

## 4. Data model (plain explanation)

R scoringutils stores a forecast as a `data.table` with an extra class tag
(`forecast_quantile`, `forecast_sample`). The tag tells `score()` which
metrics to run.

The Python equivalent is a pandas DataFrame subclass:

- `ForecastQuantile(pd.DataFrame)` and `ForecastSample(pd.DataFrame)`, both
  inheriting from `Forecast`. They behave exactly like a DataFrame (you can
  print, filter, write to CSV), plus `.forecast_type` and `.forecast_unit`.
- Any pandas operation that produces a new frame (filtering, merging) returns a
  plain DataFrame, not a forecast. R instead re-validates after every subset
  and warns if the result is broken. Returning a plain frame is simpler and
  just as safe: `score()` only accepts a validated forecast, so you call
  `as_forecast_quantile()` again after editing. Documented difference.
- `Scores(pd.DataFrame)` carries the list of metric column names in
  `.metrics` (R keeps this in `attr(scores, "metrics")`). Unlike forecasts,
  `Scores` survives filtering and copying, because R does the same, and
  `summarise_scores()` needs to know which columns are scores.

"DataFrames in and out": every function accepts a plain DataFrame where R
accepts a data.frame, and every result is a DataFrame (or subclass).

### Forecast unit

The forecast unit is the set of columns that together identify one forecast
(for example model + location + target_end_date + target_type + horizon). As
in R, it is every column that is not "protected" (`observed`, `predicted`,
`quantile_level`, `sample_id`, coverage columns and a few others) and not a
metric. Passing `forecast_unit=[...]` drops all other non-protected columns,
exactly like R's `set_forecast_unit()`.

## 5. Function mapping (R to Python)

Python names are identical to R names (snake_case already), so R code
translates almost line by line. Arguments keep R names and defaults.
`NULL` becomes `None`, character vectors become lists of strings.

### Forecast construction and validation

| R | Python | v1 |
|---|---|---|
| `as_forecast_quantile(data, forecast_unit, observed, predicted, quantile_level)` | same | ported |
| `as_forecast_sample(data, forecast_unit, observed, predicted, sample_id)` | same | ported |
| `as_forecast_quantile(<forecast_sample>, probs, type)` | `as_forecast_quantile(sample_fc, probs=..., type=7)` | ported (R quantile types 1 to 9 map to numpy methods) |
| `assert_forecast()` | same | ported |
| `is_forecast()`, `is_forecast_quantile()`, `is_forecast_sample()` | same | ported |
| `get_forecast_type()` | same | ported |
| `get_forecast_unit()` | same | ported |
| `get_duplicate_forecasts(data, forecast_unit, type, counts)` | same | ported |
| `get_forecast_type_ids()` | same | ported |
| `get_protected_columns()` | same (internal) | ported |
| `new_forecast()` | internal | ported |

Validation checks ported one-to-one: required columns, duplicate column names,
not both `quantile_level` and `sample_id`, duplicate forecasts (error), observed
constant within a forecast unit (error), differing numbers of quantiles or
samples (warning), NA rows (message, rows dropped at scoring), quantile level
rounding to 9 digits when levels differ by <= 1e-10, quantile levels in [0, 1],
numeric observed / predicted, rename collision check.

R "messages" (`cli_inform`) become Python `warnings.warn(..., ScoringutilsMessage)`
with a dedicated category so users can silence them; R warnings become
`warnings.warn(..., UserWarning)`; R errors become `ValueError` or `TypeError`.
Message text follows R closely but is not character-identical (R's cli
formatting has no Python equivalent).

### Scoring

| R | Python | v1 |
|---|---|---|
| `score(forecast, metrics = get_metrics(forecast))` | `score(forecast, metrics=None)` | ported |
| `get_metrics(forecast, select, exclude)` | same | ported |
| `get_metrics(scores)` | `scores.metrics` or `get_metrics(scores)` | ported |
| `select_metrics()` | same | ported |

Metrics are a dict of `name -> callable`, the same as R's named list. A
quantile metric is called as `f(observed, predicted, quantile_level)` with
numpy arrays (`predicted` is a 2-D matrix, one row per forecast); a sample
metric as `f(observed, predicted)`. Extra arguments are bound with
`functools.partial`, which is what R's `purrr::partial` does. As in R, a metric
that raises is dropped with a warning instead of aborting the whole call.

Default metrics, same names and order as R:

- quantile: `wis`, `overprediction`, `underprediction`, `dispersion`, `bias`,
  `interval_coverage_50`, `interval_coverage_90`, `ae_median`
- sample: `bias`, `dss`, `crps`, `overprediction`, `underprediction`,
  `dispersion`, `log_score`, `mad`, `ae_median`, `se_mean`

### Metric functions (vector / matrix level)

| R | Python | Maths source | v1 |
|---|---|---|---|
| `wis(observed, predicted, quantile_level, separate_results, weigh, count_median_twice, na.rm)` | same (`na_rm`) | own numpy port | ported |
| `dispersion_quantile`, `overprediction_quantile`, `underprediction_quantile` | same | own | ported |
| `interval_coverage(..., interval_range = 50)` | same | own | ported |
| `bias_quantile(..., na.rm = TRUE)` | same | own | ported |
| `ae_median_quantile` | same | own | ported |
| `quantile_score(..., weigh = TRUE)` | same | own (see 6) | ported |
| `crps_sample(..., separate_results)` | same | scoringrules `crps_ensemble(estimator="qd")` | ported |
| `dispersion_sample`, `overprediction_sample`, `underprediction_sample` | same | via crps_sample | ported |
| `bias_sample` | same | own | ported |
| `dss_sample` | same | scoringrules `dssuv_ensemble(bias=True)` | ported |
| `logs_sample` | same | scoringrules `logs_ensemble` with R's `bw.nrd` bandwidth | ported |
| `mad_sample` | same | own (R `stats::mad`) | ported |
| `ae_median_sample`, `se_mean_sample` | same | own | ported |

### Summaries and comparisons

| R | Python | v1 |
|---|---|---|
| `summarise_scores(scores, by = "model", fun = mean, ...)` | same; `summarize_scores` alias | ported |
| `get_pairwise_comparisons(scores, compare, by, metric, baseline, ...)` | same | ported |
| `add_relative_skill(scores, compare, by, metric, baseline, test_type)` | same | ported |
| `get_coverage(forecast, by = "model")` | same | ported |

## 6. Where the maths source matters

The brief is to use scoringrules where its definition matches scoringutils
exactly, and follow scoringutils where it does not. Findings so far, each of
which becomes a parity test:

1. CRPS for samples. R scoringutils calls `scoringRules::crps_sample()` with
   the default `"edf"` method: the CRPS of the empirical distribution
   (normalised by m squared, not the "fair" m(m-1) version). In scoringrules
   (Python) this is `crps_ensemble(estimator="qd")`. Measured: identical to
   the last bit on a test case. `"pwm"` and `"fair"` differ by about 0.026 on
   20 samples, so we always pass `estimator="qd"` explicitly, so a future
   change of default cannot silently change results.
2. Log score for samples. R uses a Gaussian kernel density with bandwidth
   `stats::bw.nrd` (1.06 x min(sd, IQR/1.34) x n^-1/5). scoringrules'
   `logs_ensemble` default bandwidth matched R on a test case (difference
   3e-16), but we port `bw.nrd` and pass it as `bw` explicitly for the same
   reason as above.
3. Quantile score. scoringutils' `quantile_score` is **2 x** the textbook
   pinball loss that `scoringrules.quantile_score` returns, so that averaging
   quantile scores gives the WIS. We follow scoringutils.
4. WIS. scoringutils builds WIS from central intervals (median weighted 0.5),
   returns its three components, handles quantile levels 0 and 1 without NaN,
   and has `weigh`, `count_median_twice` and `na.rm` options.
   `scoringrules.weighted_interval_score` has a different interface and weight
   convention, so WIS is implemented directly in numpy following the R code.
   A test checks that both agree where definitions coincide.
5. DSS. `scoringRules::dss_sample` matches
   `scoringrules.dssuv_ensemble(bias=True)` (difference 6e-17); the Python
   default `bias=False` is off by 0.05 on 20 samples. We pass `bias=True`.
6. Integer detection. Several R functions switch behaviour when predictions
   "look like integers" (`get_type()`), using `all.equal()` with tolerance
   1.5e-8. Ported exactly in `_rcompat.py`.
7. Wilcoxon p-values. Pairwise comparisons use R's `wilcox.test(paired = TRUE)`,
   which switches between an exact distribution (n < 50, no ties or zeros) and
   a normal approximation with continuity and tie correction.
   `scipy.stats.wilcoxon` uses different switching rules, so the R algorithm is
   ported directly (including an exact signed-rank distribution). Holm
   adjustment (`p.adjust` default) is ported too.
8. Permutation test (`test_type = "permutation"`). Uses random numbers; Python
   and R generators differ, so values cannot match. We port the algorithm and
   test its behaviour (valid p-value range, seed reproducibility), not the
   exact number.

## 7. Row order

R output row order comes from data.table grouping and `merge()` sorting.
The port tries to reproduce it, but parity tests compare results after sorting
by the forecast unit, so an ordering difference never hides a numeric one.
Any known order difference is listed in the README.

## 8. Testing strategy

1. **R fixtures, committed.** `r/generate_fixtures.R` runs against pinned R
   scoringutils 2.3.0 and scoringRules 1.1.3 (R 4.6.1) and writes CSV / JSON
   into `tests/fixtures/`. The Python suite only reads those files, so R is
   never needed to run tests. Modelled on CDC's
   [cfa-stf-forecasttools parity fixtures](https://github.com/CDCgov/cfa-stf-forecasttools/tree/main/tests/cfa/stf/forecasttools/test_data/scoringutils_crps).
   Doubles are written with 17 significant digits so nothing is lost in
   transit.
2. **Fixture contents.** Full `score()` output for the three example datasets;
   `summarise_scores()` by model and by model + target_type;
   `get_pairwise_comparisons()` with and without `by` and `baseline`;
   `add_relative_skill()`; `get_coverage()` with two `by` choices;
   sample-to-quantile conversion for all quantile types; every metric function
   on small hand-built vectors including edge cases (quantile levels 0 and 1,
   missing median, NA handling, ties, integer samples, `weigh = FALSE`,
   `count_median_twice = TRUE`); `get_duplicate_forecasts()` outputs.
3. **Ported testthat cases.** The relevant upstream tests (forecast classes,
   forecast unit, duplicates, quantile and sample metrics, score,
   summarise_scores, pairwise comparisons, coverage, interval helpers) are
   translated to pytest. Cases that check literal numbers are ported as is;
   cases that compare to R output use the fixtures; cases for plots or
   out-of-scope types are skipped with a note.
4. **Tolerance.** `rtol=1e-10, atol=1e-12` for floats; exact match for
   booleans, strings and integers.

## 9. Decisions (Greg, 2026-10-19)

1. Name: PyPI `scoringutils-py`, `import scoringutils_py`.
2. Ship upstream's three example datasets inside the package, with their MIT
   licence notice.
3. `requires-python >= 3.10`.

## 10. Implementation notes (v1, 2026-10-19)

Things learned while building v1 that change or sharpen the plan above.

1. scoringrules 0.11.0 requires Python >= 3.12, which conflicts with the
   approved 3.10 floor. scoringrules 0.9.0 supports 3.10, has the same API
   for everything used here and gives identical numbers, so the dependency
   is `scoringrules>=0.9,<0.12`. The suite passes on Python 3.10
   (scoringrules 0.9) and 3.12 (scoringrules 0.11).
2. `scoringrules.weighted_interval_score` (numpy backend, 0.9 to 0.11) adds
   `w_median * median` where it should add `w_median * |obs - median|`; the
   numba backend is correct. This port does not use that function (see 6.4),
   and a test checks WIS against scoringrules' `interval_score` instead.
3. With all samples equal, R's `bw.nrd` bandwidth is 0 and R's log score is
   `Inf`; scoringrules returns `NaN`. Handled explicitly in `logs_sample`.
4. R 4.6's `wilcox.test` computes exact p-values even with ties and zeros
   (n < 50), unlike older R and unlike scipy. Ported from the R source,
   including the `psignrank` flooring rule.
5. pandas' default CSV float parser is not round-trip exact (0.3 written
   with 17 digits came back as 0.2999999999999999). All CSV reads use
   `float_precision="round_trip"`.
6. Row order: the port reproduces R's row order for every fixture, so the
   parity tests check order as well as values (section 7 planned for less).
7. R's `summarise_scores(fun = median)` errors on the logical coverage
   columns when a group has an odd number of rows. The fixture uses
   `function(x) median(as.numeric(x))`; Python converts booleans first.
