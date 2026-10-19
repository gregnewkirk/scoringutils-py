"""score(), the Scores class and metric selection.

Port of ``R/score.R``, ``R/class-scores.R``, ``R/metrics.R`` and the
``score`` / ``get_metrics`` methods in ``R/class-forecast-quantile.R``,
``R/class-forecast-sample.R``, ``R/class-forecast-point.R`` and
``R/class-forecast-binary.R``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import pandas as pd

from ._forecast import (
    Forecast,
    ForecastBinary,
    ForecastPoint,
    ForecastQuantile,
    ForecastSample,
    _metrics_attr,
    assert_forecast,
    get_forecast_unit,
)
from ._metrics_binary import brier_score, logs_binary
from ._metrics_point import ae_point, ape, se_point
from ._metrics_quantile import (
    ae_median_quantile,
    bias_quantile,
    dispersion_quantile,
    interval_coverage,
    overprediction_quantile,
    underprediction_quantile,
    wis,
)
from ._metrics_sample import (
    ae_median_sample,
    bias_sample,
    crps_sample,
    dispersion_sample,
    dss_sample,
    logs_sample,
    mad_sample,
    overprediction_sample,
    se_mean_sample,
    underprediction_sample,
)

Metrics = dict[str, Callable]


class Scores(pd.DataFrame):
    """A data frame of scores that remembers which columns are scores.

    ``scores.metrics`` lists the score columns (R: ``attr(scores, "metrics")``).
    It survives filtering and copying, like R's ``[.scores`` method.
    """

    _metadata = ["metrics"]
    metrics: list[str] | None = None

    @property
    def _constructor(self):
        return Scores


def new_scores(scores: pd.DataFrame, metrics: Sequence[str]) -> Scores:
    out = Scores(pd.DataFrame(scores).copy())
    out.metrics = list(metrics)
    return out


def as_scores(scores: pd.DataFrame, metrics: Sequence[str]) -> Scores:
    """Turn a data frame into a :class:`Scores` object."""
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("Assertion on 'scores' failed: Must be of type 'data.frame'.")
    present = [m for m in metrics if m in scores.columns]
    out = new_scores(scores, present)
    assert_scores(out)
    return out


def assert_scores(scores) -> None:
    if not isinstance(scores, Scores):
        raise TypeError(
            "Assertion on 'scores' failed: Must inherit from class 'scores'."
        )
    _get_metrics_scores(scores, error=True)
    return None


def _get_metrics_scores(x: pd.DataFrame, error: bool = False) -> list[str] | None:
    if not isinstance(x, pd.DataFrame):
        raise TypeError("Assertion on 'x' failed: Must be of type 'data.frame'.")
    metrics = _metrics_attr(x)
    if error and metrics is None:
        raise ValueError(
            "Input needs an attribute `metrics` with the names of the scoring "
            "rules that were used for scoring. See `get_metrics()` for further "
            "information."
        )
    if metrics is not None:
        missing = [m for m in metrics if m not in x.columns]
        if missing:
            warnings.warn(
                "The following scores have been previously computed, but are no "
                f"longer column names of the data: {', '.join(missing)}. See "
                "`get_metrics()` for further information.",
                UserWarning,
                stacklevel=3,
            )
    return metrics


# ---------------------------------------------------------------------------
# metric lists
# ---------------------------------------------------------------------------


def _default_metrics_quantile() -> Metrics:
    return {
        "wis": wis,
        "overprediction": overprediction_quantile,
        "underprediction": underprediction_quantile,
        "dispersion": dispersion_quantile,
        "bias": bias_quantile,
        "interval_coverage_50": interval_coverage,
        "interval_coverage_90": partial(interval_coverage, interval_range=90),
        "ae_median": ae_median_quantile,
    }


def _default_metrics_sample() -> Metrics:
    return {
        "bias": bias_sample,
        "dss": dss_sample,
        "crps": crps_sample,
        "overprediction": overprediction_sample,
        "underprediction": underprediction_sample,
        "dispersion": dispersion_sample,
        "log_score": logs_sample,
        "mad": mad_sample,
        "ae_median": ae_median_sample,
        "se_mean": se_mean_sample,
    }


def _default_metrics_point() -> Metrics:
    return {"ae_point": ae_point, "se_point": se_point, "ape": ape}


def _default_metrics_binary() -> Metrics:
    return {"brier_score": brier_score, "log_score": logs_binary}


def select_metrics(
    metrics: Metrics,
    select: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
) -> Metrics:
    """Pick metrics by name. If ``select`` is given, ``exclude`` is ignored."""
    if not isinstance(metrics, dict):
        raise TypeError(
            "Assertion on 'metrics' failed: Must be of type 'list' (a dict), "
            f"not '{type(metrics).__name__}'."
        )
    select = [select] if isinstance(select, str) else select
    exclude = [exclude] if isinstance(exclude, str) else exclude
    allowed = list(metrics)
    if select is None and exclude is None:
        return dict(metrics)
    if select is None:
        _check_subset(exclude, allowed, "exclude")
        return {k: v for k, v in metrics.items() if k not in exclude}
    _check_subset(select, allowed, "select")
    return {k: metrics[k] for k in select}


def _check_subset(values, allowed, name):
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(
            f"Assertion on '{name}' failed: Must be a subset of "
            f"{{{','.join(repr(a) for a in allowed)}}}, but has additional "
            f"elements {{{','.join(repr(b) for b in bad)}}}."
        )


def get_metrics(x, select=None, exclude=None, error: bool = False):
    """Default metrics for a forecast, or the score names of a scores table.

    - For a forecast: a dict of ``name -> function``, optionally filtered by
      ``select`` / ``exclude``.
    - For a scores table: the list of score column names.
    """
    if isinstance(x, ForecastQuantile):
        return select_metrics(_default_metrics_quantile(), select, exclude)
    if isinstance(x, ForecastSample):
        return select_metrics(_default_metrics_sample(), select, exclude)
    if isinstance(x, ForecastPoint):
        return select_metrics(_default_metrics_point(), select, exclude)
    if isinstance(x, ForecastBinary):
        return select_metrics(_default_metrics_binary(), select, exclude)
    if isinstance(x, pd.DataFrame):
        return _get_metrics_scores(x, error=error)
    raise TypeError(f"No get_metrics() method for objects of type {type(x).__name__}.")


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def _validate_metrics(metrics) -> Metrics:
    if not isinstance(metrics, dict) or len(metrics) < 1:
        raise TypeError(
            "Assertion on 'metrics' failed: Must be a named list (a dict) of "
            "length >= 1."
        )
    valid = {}
    for i, (name, fun) in enumerate(metrics.items(), start=1):
        if not callable(fun):
            warnings.warn(
                f"`Metrics` element number {i} is not a valid function.",
                UserWarning,
                stacklevel=3,
            )
            continue
        valid[str(name)] = fun
    if not valid:
        raise ValueError("Assertion on 'valid metrics' failed: Must have length >= 1.")
    return valid


def _apply_metrics(frame: pd.DataFrame, metrics: Metrics, *args) -> pd.DataFrame:
    clashing = [m for m in metrics if m in frame.columns]
    if clashing:
        warnings.warn(
            f"Column(s) {', '.join(clashing)} already present in the data will be "
            "overwritten with metric results.",
            UserWarning,
            stacklevel=3,
        )
    for name, fun in metrics.items():
        try:
            result = fun(*args)
        except Exception as e:  # mirrors R's run_safely()
            warnings.warn(
                f"Computation for `{name}` failed. Error: {e}.",
                UserWarning,
                stacklevel=3,
            )
            continue
        if result is None:
            continue
        frame[name] = np.asarray(result)
    return frame


def _prepare(forecast: Forecast, metrics):
    if metrics is None:
        metrics = get_metrics(forecast)
    assert_forecast(forecast, verbose=False)
    df = pd.DataFrame(forecast).dropna().reset_index(drop=True)
    forecast_unit = get_forecast_unit(forecast)
    return df, _validate_metrics(metrics), forecast_unit


def _r_number(x: float) -> str:
    return f"{x:.15g}"


def score(forecast, metrics: Metrics | None = None) -> Scores:
    """Score a forecast with each metric, one row per forecast.

    ``metrics`` is a dict of ``name -> function`` and defaults to
    ``get_metrics(forecast)``. A metric that raises an error is skipped with a
    warning, so one failing metric does not stop the others.
    """
    if isinstance(forecast, ForecastQuantile):
        return _score_quantile(forecast, metrics)
    if isinstance(forecast, ForecastSample):
        return _score_sample(forecast, metrics)
    if isinstance(forecast, (ForecastPoint, ForecastBinary)):
        return _score_one_row_per_forecast(forecast, metrics)
    raise TypeError(
        "The input needs to be a valid forecast object. Please convert to a "
        "`forecast` object first by calling the appropriate "
        "`as_forecast_<type>()` function."
    )


def _unit_ids(df: pd.DataFrame, forecast_unit: list[str]) -> np.ndarray:
    if not forecast_unit:
        return np.zeros(len(df), dtype=int)
    return df.groupby(forecast_unit, sort=False, dropna=False).ngroup().to_numpy()


def _finalise(parts: list[pd.DataFrame], metrics: Metrics) -> Scores:
    scores = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for name in metrics:
        if name in scores.columns and scores[name].dtype == object:
            values = scores[name]
            if values.map(
                lambda v: isinstance(v, (bool, np.bool_)) or pd.isna(v)
            ).all():
                scores[name] = values.astype("boolean")
    return as_scores(scores, list(metrics))


def _score_quantile(forecast: ForecastQuantile, metrics) -> Scores:
    df, metrics, forecast_unit = _prepare(forecast, metrics)
    df["_unit"] = _unit_ids(df, forecast_unit)
    df = df.sort_values(["_unit", "quantile_level"], kind="stable")
    units = df.groupby("_unit", sort=True)
    first = units.head(1).set_index("_unit")
    levels = units["quantile_level"].agg(tuple)
    preds = units["predicted"].agg(list)
    observed = units["observed"].first()

    # forecasts are scored in groups that share the same set of quantile levels
    keys = levels.map(lambda t: ", ".join(_r_number(q) for q in t))
    parts = []
    for key in sorted(keys.unique()):
        ids = keys.index[keys == key]
        ql = np.asarray(levels.loc[ids[0]], dtype=float)
        pred = np.vstack([np.asarray(preds.loc[i], dtype=float) for i in ids])
        obs = observed.loc[ids].to_numpy(dtype=float)
        frame = first.loc[ids, forecast_unit].reset_index(drop=True)
        parts.append(_apply_metrics(frame, metrics, obs, pred, ql))
    return _finalise(parts, metrics)


def _score_sample(forecast: ForecastSample, metrics) -> Scores:
    df, metrics, forecast_unit = _prepare(forecast, metrics)
    df["_unit"] = _unit_ids(df, forecast_unit)
    units = df.groupby("_unit", sort=True)
    first = units.head(1).set_index("_unit")
    preds = units["predicted"].agg(list)
    observed = units["observed"].first()
    n_samples = preds.map(len)
    integer_input = pd.api.types.is_integer_dtype(forecast["predicted"])

    parts = []
    for n in sorted(n_samples.unique()):
        ids = n_samples.index[n_samples == n]
        pred = np.vstack([np.asarray(preds.loc[i]) for i in ids])
        if integer_input:
            pred = pred.astype(np.int64)
        obs = observed.loc[ids].to_numpy(dtype=float)
        frame = first.loc[ids, forecast_unit].reset_index(drop=True)
        parts.append(_apply_metrics(frame, metrics, obs, pred))
    return _finalise(parts, metrics)


def _score_one_row_per_forecast(forecast: Forecast, metrics) -> Scores:
    # point and binary forecasts have one row per forecast, so each metric is
    # applied to whole columns at once and rows keep their input order
    df, metrics, _ = _prepare(forecast, metrics)
    if isinstance(forecast, ForecastBinary):
        observed = pd.Categorical(df["observed"])
    else:
        observed = df["observed"].to_numpy()
    predicted = df["predicted"].to_numpy()
    frame = df.drop(columns=["predicted", "observed"])
    return _finalise([_apply_metrics(frame, metrics, observed, predicted)], metrics)
