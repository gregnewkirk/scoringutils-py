"""Metrics for point forecasts.

Port of the default metrics in ``get_metrics.forecast_point()``
(``R/class-forecast-point.R``, scoringutils 2.3.0). In R these are three
unnamed one-line functions, ``function(actual, predicted)``, that match the
``Metrics`` package's ``ae()``, ``se()`` and ``ape()``. Here they get names so
they can be imported and reused.

Like the R originals they do no input checking: ``observed`` and
``predicted`` are numbers or numpy arrays that broadcast against each other.
"""

from __future__ import annotations

import numpy as np


def ae_point(observed, predicted):
    """Absolute error, ``|observed - predicted|``."""
    return np.abs(
        np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    )


def se_point(observed, predicted):
    """Squared error, ``(observed - predicted) ** 2``."""
    return (np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)) ** 2


def ape(observed, predicted):
    """Absolute percentage error, ``|observed - predicted| / |observed|``.

    As in R, an observed value of 0 gives ``inf`` (or ``nan`` if the
    prediction is also 0), without a warning.
    """
    obs = np.asarray(observed, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(obs - np.asarray(predicted, dtype=float)) / np.abs(obs)
