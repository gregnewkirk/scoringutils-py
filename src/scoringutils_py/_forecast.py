"""Forecast objects, validation, forecast unit and duplicate detection.

Port of ``R/class-forecast.R``, ``R/class-forecast-quantile.R``,
``R/class-forecast-sample.R``, ``R/class-forecast-point.R``,
``R/class-forecast-binary.R``, ``R/forecast-unit.R``,
``R/get-duplicate-forecasts.R``, ``R/get-protected-columns.R`` and
``R/get-forecast-type.R``.

Design note: R tags a data.table with a class (``forecast_quantile``). Here a
forecast is a pandas DataFrame subclass. Any pandas operation that returns a
new frame (filtering, merging, ``copy()``) returns a plain DataFrame, so an
edited forecast has to go through ``as_forecast_*()`` again before scoring.
R instead re-validates after each subset and warns.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._checks import (
    _as_categorical,
    check_input_binary,
    check_input_point,
    inform,
)

# ---------------------------------------------------------------------------
# protected columns and forecast unit
# ---------------------------------------------------------------------------

_PROTECTED = [
    ".mv_group_id",
    ".imputed",
    "predicted",
    "observed",
    "sample_id",
    "quantile_level",
    "upper",
    "lower",
    "pit_value",
    "interval_range",
    "boundary",
    "predicted_label",
    "interval_coverage",
    "interval_coverage_deviation",
    "quantile_coverage",
    "quantile_coverage_deviation",
]


def get_protected_columns(data: pd.DataFrame | None = None) -> list[str]:
    """Columns with a special meaning that are never part of the forecast unit."""
    if data is None:
        return list(_PROTECTED)
    names = [str(c) for c in data.columns]
    protected = set(_PROTECTED)
    protected |= {c for c in names if re.search(r"_relative_skill$", c)}
    protected |= {c for c in names if "coverage_" in c}
    return [c for c in names if c in protected]


def _metrics_attr(data) -> list[str] | None:
    metrics = getattr(data, "metrics", None)
    if metrics is None and isinstance(data, pd.DataFrame):
        metrics = data.attrs.get("metrics")
    return None if metrics is None else list(metrics)


def get_forecast_unit(data: pd.DataFrame) -> list[str]:
    """Columns that together identify a single forecast.

    Every column that is not protected (``observed``, ``predicted``,
    ``quantile_level``, ...) and not a score.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Assertion on 'data' failed: Must be of type 'data.frame'.")
    excluded = set(get_protected_columns(data)) | set(_metrics_attr(data) or [])
    return [c for c in data.columns if c not in excluded]


def set_forecast_unit(data: pd.DataFrame, forecast_unit: Sequence[str]) -> pd.DataFrame:
    """Keep only the protected columns plus ``forecast_unit``, dropping duplicates."""
    data = _ensure_dataframe(data)
    forecast_unit = _as_list(forecast_unit)
    if len(forecast_unit) == 0:
        raise ValueError("Assertion on 'forecast_unit' failed: Must have length >= 1.")
    missing = [c for c in forecast_unit if c not in data.columns]
    if missing:
        raise ValueError(
            "Assertion on 'forecast_unit' failed: Must be a subset of "
            f"{{{_fmt_set(data.columns)}}}, but has additional elements "
            f"{{{_fmt_set(missing)}}}."
        )
    keep = list(dict.fromkeys(get_protected_columns(data) + forecast_unit))
    return pd.DataFrame(data[keep]).drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# forecast classes
# ---------------------------------------------------------------------------


class Forecast(pd.DataFrame):
    """Base class of validated forecast data frames."""

    forecast_type: str = ""
    type_ids: tuple[str, ...] = ()

    @property
    def _constructor(self):
        # pandas operations on a forecast return plain DataFrames
        return pd.DataFrame

    @property
    def forecast_unit(self) -> list[str]:
        return get_forecast_unit(self)

    def __repr__(self) -> str:
        try:
            unit = ", ".join(self.forecast_unit)
        except Exception:  # pragma: no cover - printing must never fail
            unit = "<could not determine forecast unit>"
        header = f"Forecast type: {self.forecast_type}\nForecast unit:\n{unit}\n\n"
        return header + pd.DataFrame(self).__repr__()


class ForecastQuantile(Forecast):
    """A validated quantile forecast (R class ``forecast_quantile``)."""

    forecast_type = "quantile"
    type_ids = ("quantile_level",)


class ForecastSample(Forecast):
    """A validated sample forecast (R class ``forecast_sample``)."""

    forecast_type = "sample"
    type_ids = ("sample_id",)


class ForecastPoint(Forecast):
    """A validated point forecast (R class ``forecast_point``)."""

    forecast_type = "point"


class ForecastBinary(Forecast):
    """A validated binary forecast (R class ``forecast_binary``).

    ``observed`` is a pandas Categorical with two categories and
    ``predicted`` the probability of the second one.
    """

    forecast_type = "binary"


_CLASSES: dict[str, type[Forecast]] = {
    "quantile": ForecastQuantile,
    "sample": ForecastSample,
    "point": ForecastPoint,
    "binary": ForecastBinary,
}


def new_forecast(data: pd.DataFrame, classname: str) -> Forecast:
    """Tag a data frame as a forecast without validating it.

    ``classname`` is an R class name such as ``"forecast_quantile"`` or just
    the type: ``"quantile"``, ``"sample"``, ``"point"`` or ``"binary"``.
    """
    key = classname.removeprefix("forecast_")
    if key not in _CLASSES:
        raise ValueError(f"Unknown forecast class {classname!r}.")
    return _CLASSES[key](pd.DataFrame(data).copy())


def is_forecast(x) -> bool:
    return isinstance(x, Forecast)


def is_forecast_quantile(x) -> bool:
    return isinstance(x, ForecastQuantile)


def is_forecast_sample(x) -> bool:
    return isinstance(x, ForecastSample)


def is_forecast_point(x) -> bool:
    return isinstance(x, ForecastPoint)


def is_forecast_binary(x) -> bool:
    return isinstance(x, ForecastBinary)


def get_forecast_type(forecast) -> str:
    """``"quantile"``, ``"sample"``, ``"point"`` or ``"binary"``."""
    if isinstance(forecast, Forecast) and forecast.forecast_type:
        return forecast.forecast_type
    raise TypeError(
        "Input is not a valid forecast object (There should be a single class "
        "beginning with `forecast_`)."
    )


def get_forecast_type_ids(data) -> list[str]:
    """Columns that, with the forecast unit, identify a row (``quantile_level``)."""
    if isinstance(data, Forecast):
        return list(data.type_ids)
    return []


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _ensure_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(data).copy()
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    try:
        return pd.DataFrame(data)
    except Exception as e:  # pragma: no cover
        raise TypeError("Assertion on 'data' failed: Must be a data.frame.") from e


def _as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _fmt_set(values) -> str:
    return ",".join(f"'{v}'" for v in values)


def _as_forecast_generic(data, forecast_unit=None, **renames) -> pd.DataFrame:
    data = _ensure_dataframe(data)
    old, new = [], []
    for target, source in renames.items():
        if source is None:
            continue
        if not isinstance(source, str):
            if isinstance(source, (list, tuple, np.ndarray)) and len(source) != 1:
                raise ValueError(f"Assertion on '{target}' failed: Must have length 1.")
            raise TypeError(
                f"Assertion on '{target}' failed: Must be of type 'character'."
            )
        if source not in data.columns:
            raise ValueError(
                f"Assertion on '{target}' failed: Must be a subset of "
                f"{{{_fmt_set(data.columns)}}}, but has additional elements "
                f"{{'{source}'}}."
            )
        old.append(source)
        new.append(target)
    if old:
        remaining = [c for c in data.columns if c not in old]
        collides = [(o, n) for o, n in zip(old, new, strict=True) if n in remaining]
        if collides:
            sources = " and ".join(f'"{o}"' for o, _ in collides)
            targets = " and ".join(f'"{n}"' for _, n in collides)
            plural = len(collides) > 1
            raise ValueError(
                f"Cannot rename column{'s' if plural else ''} {sources} to "
                f"{targets}: {'columns' if plural else 'a column'} with "
                f"{'these names' if plural else 'this name'} already "
                f"exist{'' if plural else 's'} in the data. Rename or remove the "
                f"existing column{'s' if plural else ''} first."
            )
        data = data.rename(columns=dict(zip(old, new, strict=True)))
    if forecast_unit is not None:
        data = set_forecast_unit(data, forecast_unit)
    return data


def as_forecast_quantile(
    data,
    forecast_unit: Sequence[str] | None = None,
    observed: str | None = None,
    predicted: str | None = None,
    quantile_level: str | None = None,
    *,
    probs: Sequence[float] | None = None,
    type: int = 7,
) -> ForecastQuantile:
    """Create and validate a quantile forecast.

    ``data`` needs columns ``observed``, ``predicted`` and ``quantile_level``
    (or name the columns to rename via the matching arguments). All other
    columns form the forecast unit unless ``forecast_unit`` is given.

    If ``data`` is a :class:`ForecastSample`, the samples are converted to
    quantiles instead, at levels ``probs`` (default 0.05, 0.25, 0.5, 0.75,
    0.95) using R quantile ``type`` 1-9 (default 7, numpy's default).
    """
    if isinstance(data, ForecastSample):
        return _sample_to_quantile(data, probs=probs, type=type)
    if probs is not None:
        raise TypeError(
            "`probs` only applies when converting a ForecastSample to quantiles."
        )
    data = _as_forecast_generic(
        data,
        forecast_unit,
        observed=observed,
        predicted=predicted,
        quantile_level=quantile_level,
    )
    if "quantile_level" in data.columns and pd.api.types.is_numeric_dtype(
        data["quantile_level"]
    ):
        levels = np.sort(pd.unique(data["quantile_level"].dropna()))
        if levels.size > 1 and np.any(np.diff(levels) <= 1e-10):
            warnings.warn(
                "The `quantile_level` column in your data seems to have a "
                "rounding issue (run `np.diff(np.sort(data.quantile_level.unique()))` "
                "to see this). As scoringutils does not support arbitrarily fine "
                "quantile level increments, we're going to run "
                "`round(x, digits = 9)` on the `quantile_level` column.",
                UserWarning,
                stacklevel=2,
            )
            data["quantile_level"] = data["quantile_level"].round(9)
    forecast = ForecastQuantile(data)
    assert_forecast(forecast)
    return forecast


def as_forecast_sample(
    data,
    forecast_unit: Sequence[str] | None = None,
    observed: str | None = None,
    predicted: str | None = None,
    sample_id: str | None = None,
) -> ForecastSample:
    """Create and validate a sample forecast.

    ``data`` needs columns ``observed``, ``predicted`` and ``sample_id``.
    """
    data = _as_forecast_generic(
        data, forecast_unit, observed=observed, predicted=predicted, sample_id=sample_id
    )
    forecast = ForecastSample(data)
    assert_forecast(forecast)
    return forecast


def as_forecast_point(
    data,
    forecast_unit: Sequence[str] | None = None,
    observed: str | None = None,
    predicted: str | None = None,
) -> ForecastPoint:
    """Create and validate a point forecast.

    ``data`` needs numeric columns ``observed`` and ``predicted``.

    If ``data`` is a :class:`ForecastQuantile`, the median (quantile level
    0.5) becomes the point forecast instead, as in R's
    ``as_forecast_point.forecast_quantile()``. Like R, that conversion does
    not re-validate the result.
    """
    if isinstance(data, ForecastQuantile):
        if forecast_unit is not None or observed is not None or predicted is not None:
            raise TypeError(
                "`forecast_unit`, `observed` and `predicted` do not apply when "
                "converting a ForecastQuantile to a point forecast."
            )
        return _quantile_to_point(data)
    data = _as_forecast_generic(
        data, forecast_unit, observed=observed, predicted=predicted
    )
    forecast = ForecastPoint(data)
    assert_forecast(forecast)
    return forecast


def _quantile_to_point(data: ForecastQuantile) -> ForecastPoint:
    assert_forecast(data, verbose=False)
    levels = pd.unique(data["quantile_level"])
    if not np.any(levels == 0.5):
        raise ValueError(
            "Assertion on '0.5' failed: Must be a subset of "
            f"{{{_fmt_set(levels)}}}, but has additional elements {{'0.5'}}."
        )
    df = pd.DataFrame(data)
    out = df[df["quantile_level"] == 0.5].drop(columns="quantile_level")
    return ForecastPoint(out.reset_index(drop=True))


def as_forecast_binary(
    data,
    forecast_unit: Sequence[str] | None = None,
    observed: str | None = None,
    predicted: str | None = None,
) -> ForecastBinary:
    """Create and validate a binary forecast.

    ``observed`` must be a pandas Categorical column with exactly two
    categories (R: a factor with two levels), for example
    ``pd.Categorical(df.observed, categories=[0, 1])``. ``predicted`` holds
    the probability that the outcome is the second category.
    """
    data = _as_forecast_generic(
        data, forecast_unit, observed=observed, predicted=predicted
    )
    forecast = ForecastBinary(data)
    assert_forecast(forecast)
    return forecast


# R quantile() types 1-9 and the matching numpy methods
_QUANTILE_METHODS = {
    1: "inverted_cdf",
    2: "averaged_inverted_cdf",
    3: "closest_observation",
    4: "interpolated_inverted_cdf",
    5: "hazen",
    6: "weibull",
    7: "linear",
    8: "median_unbiased",
    9: "normal_unbiased",
}


def _sample_to_quantile(data: ForecastSample, probs=None, type: int = 7):
    if probs is None:
        probs = [0.05, 0.25, 0.5, 0.75, 0.95]
    probs_arr = np.atleast_1d(np.asarray(probs, dtype=float))
    if probs_arr.size < 1 or np.any(np.isnan(probs_arr)):
        raise ValueError("Assertion on 'probs' failed: Contains missing values.")
    if np.any((probs_arr < 0) | (probs_arr > 1)):
        raise ValueError("Assertion on 'probs' failed: All elements must be in [0, 1].")
    if type not in _QUANTILE_METHODS:
        raise ValueError("`type` must be an integer between 1 and 9.")
    method = _QUANTILE_METHODS[type]
    levels = pd.unique(np.round(probs_arr, 10))

    df = pd.DataFrame(data)
    by = [c for c in df.columns if c not in ("predicted", "sample_id")]
    groups = df.groupby(by, sort=False, dropna=False)["predicted"]
    keys, values = [], []
    for key, s in groups:
        x = s.to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        q = (
            np.quantile(x, levels, method=method)
            if x.size
            else np.full(levels.size, np.nan)
        )
        keys.append(key if isinstance(key, tuple) else (key,))
        values.append(q)
    n_levels = levels.size
    out = pd.DataFrame([k for k in keys for _ in range(n_levels)], columns=by)
    out["quantile_level"] = np.tile(levels, len(keys))
    out["predicted"] = np.concatenate(values) if values else np.array([])
    for col in by:
        out[col] = out[col].astype(df[col].dtype)
    forecast = ForecastQuantile(out)
    assert_forecast(forecast)
    return forecast


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def assert_forecast(forecast, forecast_type: str | None = None, verbose: bool = True):
    """Validate a forecast object; raises on problems, returns ``None``."""
    if not isinstance(forecast, Forecast):
        raise TypeError(
            "The input needs to be a valid forecast object. Please convert to "
            "a forecast object first by calling the appropriate "
            "`as_forecast_<type>()` function."
        )
    if isinstance(forecast, ForecastQuantile):
        _assert_subset(["quantile_level"], forecast.columns, '"quantile_level"')
        _assert_forecast_generic(forecast, verbose)
        _assert_forecast_type(forecast, "quantile", forecast_type)
        _assert_numeric(forecast["quantile_level"], "quantile_level", lower=0, upper=1)
        _assert_numeric(forecast["observed"], "observed")
        _assert_numeric(forecast["predicted"], "predicted")
    elif isinstance(forecast, ForecastSample):
        _assert_forecast_generic(forecast, verbose)
        _assert_subset(["sample_id"], forecast.columns, '"sample_id"')
        _assert_forecast_type(forecast, "sample", forecast_type)
        _assert_numeric(forecast["observed"], "observed")
        _assert_numeric(forecast["predicted"], "predicted")
    elif isinstance(forecast, ForecastPoint):
        _assert_forecast_generic(forecast, verbose)
        _assert_forecast_type(forecast, "point", forecast_type)
        _check_input_column(check_input_point, forecast, "point")
    elif isinstance(forecast, ForecastBinary):
        _assert_forecast_generic(forecast, verbose)
        _assert_forecast_type(forecast, "binary", forecast_type)
        if "sample_id" in forecast.columns or "quantile_level" in forecast.columns:
            raise ValueError(
                "Checking `forecast`: Input looks like a binary forecast, but an "
                "additional column called `sample_id` or `quantile` was found. "
                "Please remove the column."
            )
        _check_input_column(check_input_binary, forecast, "binary")
    return None


def _check_input_column(check, forecast: Forecast, kind: str) -> None:
    try:
        if kind == "binary":
            observed = _as_categorical(forecast["observed"])
        else:
            _assert_numeric(forecast["observed"], "observed")
            observed = forecast["observed"].to_numpy(dtype=float, na_value=np.nan)
        _assert_numeric(forecast["predicted"], "predicted")
        predicted = forecast["predicted"].to_numpy(dtype=float, na_value=np.nan)
        check(observed, predicted)
    except (TypeError, ValueError) as e:
        raise type(e)(
            f"Checking `forecast`: Input looks like a {kind} forecast, but found "
            f"the following issue: {e}"
        ) from None


def _assert_subset(needed, available, name: str) -> None:
    missing = [c for c in needed if c not in list(available)]
    if missing:
        raise ValueError(
            f"Assertion on '{name}' failed: Must be a subset of "
            f"{{{_fmt_set(available)}}}, but has additional elements "
            f"{{{_fmt_set(missing)}}}."
        )


def _assert_forecast_type(forecast, actual: str, desired: str | None) -> None:
    if desired is not None and desired != actual:
        raise ValueError(
            f'Forecast type determined by scoringutils based on input: "{actual}". '
            f'Desired forecast type: "{desired}".'
        )


def _assert_numeric(col: pd.Series, name: str, lower=None, upper=None) -> None:
    if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        kind = (
            "character"
            if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)
            else str(col.dtype)
        )
        raise TypeError(
            f"Assertion on '{name}' failed: Must be of type 'numeric', not '{kind}'."
        )
    values = col.to_numpy(dtype=float, na_value=np.nan)
    if lower is not None and np.any(values[~np.isnan(values)] < lower):
        raise ValueError(f"Assertion on '{name}' failed: Element is not >= {lower}.")
    if upper is not None and np.any(values[~np.isnan(values)] > upper):
        raise ValueError(f"Assertion on '{name}' failed: Element is not <= {upper}.")


def _assert_forecast_generic(data: pd.DataFrame, verbose: bool = True) -> None:
    if len(data) < 1:
        raise ValueError(
            "Assertion on 'data' failed: Must have at least 1 rows, but has 0 rows."
        )
    cols = list(data.columns)
    dup_cols = list(dict.fromkeys(c for c in cols if cols.count(c) > 1))
    if dup_cols:
        raise ValueError(
            f"Found duplicate column(s) in the data: {', '.join(map(str, dup_cols))}. "
            "Column names must be unique."
        )
    _assert_subset(["observed", "predicted"], cols, 'c("observed", "predicted")')
    if "sample_id" in cols and "quantile_level" in cols:
        raise ValueError(
            "Found columns `quantile_level` and `sample_id`. Only one of these "
            "is allowed"
        )
    forecast_unit = get_forecast_unit(data)
    dup = _check_duplicates(data)
    if dup is not True:
        raise ValueError(dup)
    obs_const = _check_observed_constant(data, forecast_unit)
    if obs_const is not True:
        raise ValueError(obs_const)
    number = _check_number_per_forecast(data, forecast_unit)
    if number is not True and verbose:
        warnings.warn(number, UserWarning, stacklevel=4)
    if pd.DataFrame(data).isna().to_numpy().any():
        if len(pd.DataFrame(data).dropna()) == 0:
            raise ValueError(
                "After removing rows with NA values in the data, no forecasts are left."
            )
        if verbose:
            inform(
                "Some rows containing NA values may be removed. This is fine if "
                "not unexpected."
            )


def _group(df: pd.DataFrame, by: list[str]):
    if not by:
        return df.groupby(np.zeros(len(df)), sort=False)
    return df.groupby(by, sort=False, dropna=False)


def _check_number_per_forecast(data: pd.DataFrame, forecast_unit: list[str]):
    df = pd.DataFrame(data).dropna()
    if len(df) == 0:
        return True
    n = pd.unique(_group(df, forecast_unit)["predicted"].transform("size"))
    if len(n) > 1:
        return (
            "Some forecasts have different numbers of rows (e.g. quantiles or "
            f"samples). scoringutils found: {', '.join(str(int(v)) for v in n)}. "
            "This may be a problem (it can potentially distort scores, making it "
            "more difficult to compare them), so make sure this is intended."
        )
    return True


def _check_observed_constant(data: pd.DataFrame, forecast_unit: list[str]):
    df = pd.DataFrame(data)
    df = df[df["observed"].notna()]
    if len(df) == 0:
        return True
    n = _group(df, forecast_unit)["observed"].nunique()
    if np.any(n.to_numpy() > 1):
        return (
            "There are instances with different observed values for the same "
            "forecast. Observed values must be constant within each forecast "
            "unit. This can't be right and needs to be resolved. Maybe you need "
            "to check the unit of a single forecast and add missing columns?"
        )
    return True


def _check_duplicates(data: pd.DataFrame):
    duplicates = get_duplicate_forecasts(data)
    if len(duplicates) > 0:
        hint = ""
        if isinstance(data, Forecast):
            fu = ", ".join(f'"{c}"' for c in get_forecast_unit(data))
            hint = f', forecast_unit = [{fu}], type = "{data.forecast_type}"'
        return (
            "There are instances with more than one forecast for the same "
            "target. This can't be right and needs to be resolved. Maybe you "
            "need to check the unit of a single forecast and add missing "
            f"columns? Use `get_duplicate_forecasts(data{hint})` to identify "
            "duplicate rows"
        )
    return True


def get_duplicate_forecasts(
    data,
    forecast_unit: Sequence[str] | None = None,
    type: str | None = None,
    counts: bool = False,
) -> pd.DataFrame:
    """Rows that share a forecast unit and quantile level / sample id.

    For a plain DataFrame pass ``type="quantile"`` or ``type="sample"``.
    With ``counts=True`` returns the number of duplicate rows per forecast unit.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Assertion on 'data' failed: Must be of type 'data.frame'.")
    is_fc = isinstance(data, Forecast)
    type_ids = list(data.type_ids) if is_fc else []
    df = pd.DataFrame(data)
    if forecast_unit is not None:
        df = set_forecast_unit(df, forecast_unit)
    fu = get_forecast_unit(df)
    if is_fc:
        type_cols = type_ids
    elif type is not None:
        key = type.removeprefix("forecast_")
        if key not in _CLASSES:
            raise ValueError(f"Unknown forecast type {type!r}.")
        type_cols = list(_CLASSES[key].type_ids)
    else:
        warnings.warn(
            "Calling `get_duplicate_forecasts()` on a data.frame without "
            "specifying `type` was deprecated in scoringutils 2.3.0. Pass `type` "
            '(e.g. "quantile", "sample") to detect type-specific duplicates on '
            "plain data.frames.",
            FutureWarning,
            stacklevel=2,
        )
        known = ["sample_id", "quantile_level", "predicted_label"]
        type_cols = [c for c in known if c in df.columns]
    by = fu + [c for c in type_cols if c not in fu]
    n = _group(df, by)["predicted"].transform("size") if by else len(df)
    out = df[np.asarray(n) > 1]
    sort_cols = by + (["predicted"] if "predicted" in out.columns else [])
    if len(out) and sort_cols:
        out = out.sort_values(sort_cols, kind="stable", na_position="first")
    out = out.reset_index(drop=True)
    if counts:
        unit = get_forecast_unit(out)
        out = _group(out, unit).size().reset_index(name="n_duplicates")
    return out
