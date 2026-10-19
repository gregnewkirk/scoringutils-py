"""Input checks shared by the metric functions.

These mirror the checkmate assertions in R scoringutils
(``assert_input_quantile()``, ``assert_input_sample()``,
``assert_input_interval()``). Messages follow checkmate's wording so that
errors look familiar to R users.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class ScoringutilsMessage(UserWarning):
    """Informational note, the equivalent of an R ``message()``.

    Raised through :func:`warnings.warn` so that it can be silenced with
    ``warnings.simplefilter("ignore", ScoringutilsMessage)``.
    """


def inform(msg: str) -> None:
    warnings.warn(msg, ScoringutilsMessage, stacklevel=3)


def as_numeric(x, name: str) -> np.ndarray:
    """Convert to a float array, refusing non-numeric input like checkmate."""
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError):
            raise TypeError(
                f"Assertion on '{name}' failed: Must be of type 'numeric'."
            ) from None
    if arr.dtype.kind not in "biuf":
        kind = "character" if arr.dtype.kind in "US" else str(arr.dtype)
        raise TypeError(
            f"Assertion on '{name}' failed: Must be of type 'numeric', not '{kind}'."
        )
    if arr.dtype.kind == "b":
        raise TypeError(
            f"Assertion on '{name}' failed: Must be of type 'numeric', not 'logical'."
        )
    return arr


def check_quantile_level(quantile_level, unique: bool = True) -> np.ndarray:
    ql = as_numeric(quantile_level, "quantile_level").astype(float)
    ql = np.atleast_1d(ql)
    if ql.ndim != 1:
        raise ValueError("Assertion on 'quantile_level' failed: Must be a vector.")
    if ql.size < 1:
        raise ValueError(
            "Assertion on 'quantile_level' failed: Must have length >= 1, "
            "but has length 0."
        )
    for i, q in enumerate(ql):
        if not np.isnan(q) and q < 0:
            raise ValueError(
                f"Assertion on 'quantile_level' failed: Element {i + 1} is not >= 0."
            )
        if not np.isnan(q) and q > 1:
            raise ValueError(
                f"Assertion on 'quantile_level' failed: Element {i + 1} is not <= 1."
            )
    if unique:
        seen: set[float] = set()
        for i, q in enumerate(ql):
            if not np.isnan(q) and q in seen:
                raise ValueError(
                    "Assertion on 'quantile_level' failed: Contains duplicated "
                    f"values, position {i + 1}."
                )
            seen.add(q)
    return ql


def check_observed(observed) -> np.ndarray:
    obs = np.atleast_1d(as_numeric(observed, "observed")).astype(float)
    if obs.ndim != 1:
        raise ValueError("Assertion on 'observed' failed: Must be a vector.")
    if obs.size < 1:
        raise ValueError(
            "Assertion on 'observed' failed: Must have length >= 1, but has length 0."
        )
    return obs


def check_input_quantile(
    observed, predicted, quantile_level, unique_quantile_levels: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``assert_input_quantile()``; returns (observed, predicted 2-D, ql)."""
    obs = check_observed(observed)
    ql = check_quantile_level(quantile_level, unique=unique_quantile_levels)
    pred = as_numeric(predicted, "predicted").astype(float)
    n_obs = obs.size
    n_q = ql.size
    if n_obs == 1:
        if pred.ndim == 1 or pred.ndim == 0:
            pred = np.atleast_1d(pred)
            if pred.size < n_q:
                raise ValueError(
                    "Assertion failed. One of the following must apply:\n"
                    " * check_numeric_vector(predicted): Must have length >= "
                    f"{n_q}\n * check_matrix(predicted): Must be of type 'matrix'"
                )
            if pred.size != n_q:
                raise ValueError(
                    "Assertion on 'quantile_level' failed: Must have length "
                    f"{pred.size}, but has length {n_q}."
                )
            pred = pred.reshape(1, -1)
        elif pred.ndim == 2:
            if pred.shape != (1, n_q):
                raise ValueError(
                    "Assertion failed. One of the following must apply:\n"
                    " * check_numeric_vector(predicted): Must be a vector\n"
                    f" * check_matrix(predicted): Must have exactly 1 rows and "
                    f"{n_q} cols"
                )
        else:
            raise ValueError("Assertion on 'predicted' failed: Must be a matrix.")
    else:
        if pred.ndim != 2:
            raise ValueError(
                "Assertion on 'predicted' failed: Must be of type 'matrix', "
                "not 'double'."
            )
        if pred.shape[0] != n_obs:
            raise ValueError(
                f"Assertion on 'predicted' failed: Must have exactly {n_obs} rows, "
                f"but has {pred.shape[0]} rows."
            )
        if pred.shape[1] != n_q:
            raise ValueError(
                f"Assertion on 'predicted' failed: Must have exactly {n_q} cols, "
                f"but has {pred.shape[1]} cols."
            )
    return obs, pred, ql


def check_input_sample(observed, predicted) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``assert_input_sample()`` + ``ensure_sample_matrix()``."""
    obs = check_observed(observed)
    pred = np.asarray(predicted)
    keep_int = pred.dtype.kind in "iu"
    pred = as_numeric(pred, "predicted")
    if not keep_int:
        pred = pred.astype(float)
    n_obs = obs.size
    if n_obs == 1:
        if pred.ndim <= 1:
            pred = np.atleast_1d(pred)
            if pred.size < 1:
                raise ValueError(
                    "Assertion on 'predicted' failed: Must have length >= 1."
                )
            pred = pred.reshape(1, -1)
        elif pred.ndim == 2 and pred.shape[0] != 1:
            raise ValueError(
                "Assertion failed. One of the following must apply:\n"
                " * check_numeric_vector(predicted): Must be a vector\n"
                " * check_matrix(predicted): Must have exactly 1 rows, but has "
                f"{pred.shape[0]} rows"
            )
    else:
        if pred.ndim != 2:
            kind = "integer" if keep_int else "double"
            raise ValueError(
                "Assertion on 'predicted' failed: Must be of type 'matrix', "
                f"not '{kind}'."
            )
        if pred.shape[0] != n_obs:
            raise ValueError(
                f"Assertion on 'predicted' failed: Must have exactly {n_obs} rows, "
                f"but has {pred.shape[0]} rows."
            )
    return obs, pred


def _r_type_name(arr: np.ndarray) -> str:
    """The R type name checkmate would report for a numpy array."""
    return {"b": "logical", "i": "integer", "u": "integer", "f": "double"}.get(
        arr.dtype.kind, "character" if arr.dtype.kind in "OUS" else str(arr.dtype)
    )


def check_dims_ok_scalar(observed, predicted) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``assert_dims_ok_scalar()``.

    ``observed`` is a vector of length 1 or n. ``predicted`` is a vector of
    length 1 or n, or an (n x 1) matrix. Returns both as 1-D arrays.
    """
    obs = np.atleast_1d(np.asarray(observed))
    if obs.ndim != 1 or obs.size < 1:
        raise ValueError(
            "Assertion on 'observed' failed: Must be a vector of length >= 1."
        )
    n_obs = obs.size
    pred = np.asarray(predicted)
    if pred.ndim == 0:
        pred = pred.reshape(1)
    ok_vector = pred.ndim == 1 and pred.size >= 1
    ok_matrix = pred.ndim == 2 and pred.shape == (n_obs, 1)
    if not (ok_vector or ok_matrix):
        raise ValueError(
            "Assertion failed. One of the following must apply:\n"
            " * check_vector(predicted): Must be a vector (1-D), not an array "
            f"of shape {pred.shape}\n"
            f" * check_matrix(predicted): Must have exactly {n_obs} rows and 1 col"
        )
    pred = pred.reshape(-1)
    n_pred = pred.size
    if n_obs != 1 and n_pred != 1 and n_obs != n_pred:
        raise ValueError(
            "`observed` and `predicted` must either be of length 1 or of equal "
            f"length. Found {n_obs} and {n_pred}."
        )
    return obs, pred


def check_input_point(observed, predicted) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``assert_input_point()``; returns float arrays."""
    if isinstance(observed, pd.Categorical) or isinstance(
        getattr(observed, "dtype", None), pd.CategoricalDtype
    ):
        raise TypeError(
            "Assertion on 'observed' failed: Must be of type 'numeric', not 'factor'."
        )
    obs = as_numeric(observed, "observed")
    pred = as_numeric(predicted, "predicted")
    obs, pred = check_dims_ok_scalar(obs, pred)
    return obs.astype(float), pred.astype(float)


def _as_categorical(observed) -> pd.Categorical:
    if isinstance(observed, pd.Categorical):
        return observed
    if isinstance(getattr(observed, "dtype", None), pd.CategoricalDtype):
        return pd.Categorical(observed)
    kind = _r_type_name(np.asarray(observed))
    raise TypeError(
        f"Assertion on 'observed' failed: Must be of type 'factor', not '{kind}'. "
        "Convert it with pd.Categorical(observed, categories=[0, 1])."
    )


def _r_level(level) -> str:
    if isinstance(level, (bool, np.bool_)):
        return "TRUE" if level else "FALSE"
    return str(level)


def check_input_binary(observed, predicted) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``assert_input_binary()``.

    ``observed`` must be a pandas Categorical (R: factor) with exactly two
    categories; ``predicted`` holds probabilities for the second (highest)
    category. Returns the observed outcomes as 0.0 / 1.0 (NaN where missing)
    and the predictions as floats.
    """
    cat = _as_categorical(observed)
    if len(cat) < 1:
        raise ValueError(
            "Assertion on 'observed' failed: Must have length >= 1, but has length 0."
        )
    if len(cat.categories) != 2:
        raise ValueError("Assertion on 'observed' failed: Must have exactly 2 levels.")
    pred = np.asarray(predicted)
    if pred.dtype == object:
        raise TypeError(
            "Assertion on 'predicted' failed: Must be of type 'numeric', not 'list'."
        )
    pred = as_numeric(pred, "predicted").astype(float)
    flat = pred.reshape(-1)
    for i, p in enumerate(flat):
        if not np.isnan(p) and p < 0:
            raise ValueError(
                f"Assertion on 'predicted' failed: Element {i + 1} is not >= 0."
            )
        if not np.isnan(p) and p > 1:
            raise ValueError(
                f"Assertion on 'predicted' failed: Element {i + 1} is not <= 1."
            )
    codes = cat.codes.astype(float)
    codes[cat.codes < 0] = np.nan
    obs, pred = check_dims_ok_scalar(codes, pred)

    levels = [_r_level(c) for c in cat.categories]
    if (set(levels) == {"0", "1"} and levels[0] == "1") or (
        set(levels) == {"TRUE", "FALSE"} and levels[0] == "TRUE"
    ):
        warnings.warn(
            "Factor levels of `observed` appear to be in counterintuitive order: "
            f'"{levels[0]}" and "{levels[1]}". Predictions will be interpreted as '
            f'the probability of observing "{levels[1]}" (the highest factor '
            "level). If this is not intended, consider reordering the "
            "categories, e.g. "
            f"`observed.reorder_categories([{levels[1]!r}, {levels[0]!r}])`.",
            UserWarning,
            stacklevel=3,
        )
    return obs, pred
