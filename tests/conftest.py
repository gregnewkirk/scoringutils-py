"""Shared helpers for loading R fixtures and comparing results."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scoringutils_py as su

FIXTURES = Path(__file__).parent / "fixtures"
RTOL = 1e-10
ATOL = 1e-12

DATE_COLS = ("target_end_date", "forecast_date", "date")
STR_COLS = ("location", "location_name", "target_type", "model", "compare_against")


def read_fixture(name: str) -> pd.DataFrame:
    """Read a CSV written by r/generate_fixtures.R."""
    df = pd.read_csv(
        FIXTURES / name,
        keep_default_na=False,
        na_values=[""],
        float_precision="round_trip",
        dtype={c: "object" for c in STR_COLS},
    )
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _decode(value):
    """Undo jsonlite's encoding of NA / NaN / Inf as strings."""
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if value in ("NA", "NaN"):
        return np.nan
    if value == "Inf":
        return np.inf
    if value == "-Inf":
        return -np.inf
    return value


def load_metric_cases() -> list[dict]:
    with open(FIXTURES / "metric_cases.json") as fh:
        cases = json.load(fh)
    for case in cases:
        for key in ("args", "kwargs", "result"):
            case[key] = _decode(case[key]) if case[key] != [] else {}
    return cases


def _normalise(s: pd.Series) -> list:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return [None if pd.isna(v) else float(v) for v in s]
    return [None if pd.isna(v) else str(v) for v in s]


def assert_frame_parity(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str] | None = None,
    check_order: bool = True,
    scale: dict[str, str] | None = None,
) -> None:
    """Compare a Python result to the R fixture, column by column.

    ``keys`` sorts both frames before comparing values; ``check_order`` also
    requires the original row order to match. ``scale`` maps a column to a
    reference column whose magnitude sets the absolute tolerance, for values
    computed as a difference of two larger numbers (e.g. the CRPS
    components, which are CRPS minus dispersion and can be pure rounding
    noise when the true value is 0).
    """
    scale = scale or {}
    actual = pd.DataFrame(actual).reset_index(drop=True)
    expected = pd.DataFrame(expected).reset_index(drop=True)
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)
    if keys is not None:
        if check_order:
            for key in keys:
                assert _normalise(actual[key]) == _normalise(expected[key]), (
                    f"row order differs in column {key!r}"
                )
        actual = actual.sort_values(keys, kind="stable").reset_index(drop=True)
        expected = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    for col in expected.columns:
        a, e = actual[col], expected[col]
        if pd.api.types.is_datetime64_any_dtype(e):
            assert (a.isna() == e.isna()).all(), col
            assert (a[e.notna()] == e[e.notna()]).all(), col
        elif pd.api.types.is_numeric_dtype(e) or pd.api.types.is_bool_dtype(e):
            av = a.to_numpy(dtype=float, na_value=np.nan)
            ev = e.to_numpy(dtype=float, na_value=np.nan)
            atol = ATOL
            if col in scale:
                ref = np.abs(expected[scale[col]].to_numpy(dtype=float))
                atol = np.maximum(ATOL, RTOL * np.nan_to_num(ref))
            ok = np.isclose(av, ev, rtol=RTOL, atol=atol, equal_nan=True)
            np.testing.assert_array_equal(np.isnan(av), np.isnan(ev), err_msg=col)
            assert ok.all(), (
                f"{col}: {np.sum(~ok)} mismatches, e.g. {av[~ok][:3]} vs {ev[~ok][:3]}"
            )
        else:
            assert (a.isna().to_numpy() == e.isna().to_numpy()).all(), col
            assert (a[e.notna()].astype(str) == e[e.notna()].astype(str)).all(), col


@pytest.fixture(scope="session")
def fc_quantile() -> su.ForecastQuantile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return su.as_forecast_quantile(su.example_quantile())


@pytest.fixture(scope="session")
def fc_sample_continuous() -> su.ForecastSample:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return su.as_forecast_sample(su.example_sample_continuous())


@pytest.fixture(scope="session")
def fc_sample_discrete() -> su.ForecastSample:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return su.as_forecast_sample(su.example_sample_discrete())


@pytest.fixture(scope="session")
def scores_quantile(fc_quantile) -> su.Scores:
    return su.score(fc_quantile)


@pytest.fixture(scope="session")
def scores_sample_continuous(fc_sample_continuous) -> su.Scores:
    return su.score(fc_sample_continuous)


@pytest.fixture(scope="session")
def scores_sample_discrete(fc_sample_discrete) -> su.Scores:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return su.score(fc_sample_discrete)
