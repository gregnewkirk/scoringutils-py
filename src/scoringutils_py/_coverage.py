"""get_coverage(). Port of ``R/get-coverage.R``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._forecast import ForecastQuantile, assert_forecast, get_forecast_unit
from ._interval import get_range_from_quantile


def get_coverage(forecast, by: str | Sequence[str] = "model") -> pd.DataFrame:
    """Interval and quantile coverage of a quantile forecast, averaged by group.

    For every quantile level (and its central interval) returns:

    - ``interval_coverage``: share of observations inside the interval,
    - ``interval_coverage_deviation``: that share minus the nominal coverage,
    - ``quantile_coverage``: share of observations at or below the quantile,
    - ``quantile_coverage_deviation``: that share minus the quantile level.
    """
    if not isinstance(forecast, ForecastQuantile):
        raise ValueError(
            "Assertion on 'get_forecast_type(forecast)' failed: Must be a subset "
            "of {'quantile'}."
        )
    assert_forecast(forecast, verbose=False)
    df = pd.DataFrame(forecast).dropna().reset_index(drop=True)
    by = [by] if isinstance(by, str) else list(by)
    by = [b for b in by if b not in ("quantile_level", "interval_range")]
    missing = [b for b in by if b not in df.columns]
    if missing:
        raise ValueError(
            f"Assertion on 'by' failed: Must be a subset of "
            f"{{{','.join(repr(c) for c in df.columns)}}}, but has additional "
            f"elements {{{','.join(repr(m) for m in missing)}}}."
        )

    df["interval_range"] = get_range_from_quantile(df["quantile_level"].to_numpy())
    forecast_cols = list(df.columns)
    forecast_unit = get_forecast_unit(df)
    keys = forecast_unit + ["interval_range"]

    # wide interval format: one row per forecast unit and interval range
    lower = df[df["quantile_level"] <= 0.5]
    upper = df[df["quantile_level"] >= 0.5]
    lower = lower[keys + ["predicted"]].rename(columns={"predicted": "lower"})
    upper = upper[keys + ["predicted"]].rename(columns={"predicted": "upper"})
    units = df[keys + ["observed"]].drop_duplicates(subset=keys)
    wide = units.merge(lower, on=keys, how="left").merge(upper, on=keys, how="left")
    obs, lo, up = (
        wide[c].to_numpy(dtype=float) for c in ("observed", "lower", "upper")
    )
    covered = ((obs <= up) & (obs >= lo)).astype(float)
    covered[np.isnan(lo) | np.isnan(up)] = np.nan
    wide["interval_coverage"] = covered
    wide["interval_coverage_deviation"] = covered - wide["interval_range"] / 100
    wide = wide.drop(columns=["lower", "upper", "observed"])

    merged = df.merge(wide, on=keys, how="inner")
    merged = merged.sort_values(keys, kind="stable", na_position="first")
    merged["quantile_coverage"] = (merged["observed"] <= merged["predicted"]).astype(
        float
    )
    merged["quantile_coverage_deviation"] = (
        merged["quantile_coverage"] - merged["quantile_level"]
    )
    new_metrics = [
        "interval_coverage",
        "interval_coverage_deviation",
        "quantile_coverage",
        "quantile_coverage_deviation",
    ]
    order = list(dict.fromkeys(forecast_cols + ["interval_range"] + new_metrics))
    merged = merged[order]

    by = list(dict.fromkeys(by + ["quantile_level", "interval_range"]))
    out = (
        merged.groupby(by, sort=False, dropna=False)[new_metrics]
        .agg(lambda s: np.mean(s.to_numpy()))
        .reset_index()
    )
    return out
