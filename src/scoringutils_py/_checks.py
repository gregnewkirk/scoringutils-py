"""Input checks shared by the metric functions.

These mirror the checkmate assertions in R scoringutils
(``assert_input_quantile()``, ``assert_input_sample()``,
``assert_input_interval()``). Messages follow checkmate's wording so that
errors look familiar to R users.
"""

from __future__ import annotations

import warnings

import numpy as np


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
