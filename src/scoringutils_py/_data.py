"""Example data from R scoringutils (MIT licence, see data/LICENSE-DATA.md).

Each access returns a fresh copy, so changing one does not affect the next.
"""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

from ._forecast import (
    ForecastBinary,
    ForecastPoint,
    ForecastQuantile,
    ForecastSample,
    new_forecast,
)

_DATE_COLS = ["target_end_date", "forecast_date"]
_STR_COLS = ["location", "location_name", "target_type", "model"]


def _read(name: str, dtype: dict | None = None) -> pd.DataFrame:
    path = files("scoringutils_py") / "data" / f"{name}.csv.gz"
    with path.open("rb") as fh:
        df = pd.read_csv(
            fh,
            compression="gzip",
            dtype={**{c: "object" for c in _STR_COLS}, **(dtype or {})},
            keep_default_na=False,
            na_values=[""],
            float_precision="round_trip",
        )
    for col in _DATE_COLS:
        df[col] = pd.to_datetime(df[col])
    return df


def example_quantile() -> ForecastQuantile:
    """Quantile forecasts of COVID-19 cases and deaths (European Forecast Hub)."""
    return new_forecast(_read("example_quantile"), "forecast_quantile")


def example_sample_continuous() -> ForecastSample:
    """Continuous sample forecasts of the same targets."""
    return new_forecast(_read("example_sample_continuous"), "forecast_sample")


def example_sample_discrete() -> ForecastSample:
    """Integer-valued sample forecasts of the same targets."""
    return new_forecast(_read("example_sample_discrete"), "forecast_sample")


def example_point() -> ForecastPoint:
    """Point forecasts of the same targets (the median of ``example_quantile``,
    replaced by a point forecast)."""
    return new_forecast(_read("example_point"), "forecast_point")


def example_binary() -> ForecastBinary:
    """Binary forecasts derived from the continuous sample forecasts.

    ``observed`` is a Categorical with categories ``"0"`` and ``"1"`` (in R a
    factor with those levels); ``predicted`` is the probability of ``"1"``.
    """
    df = _read("example_binary", dtype={"observed": "object"})
    df["observed"] = df["observed"].astype(pd.CategoricalDtype(["0", "1"]))
    return new_forecast(df, "forecast_binary")
