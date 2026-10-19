"""Ported from scoringutils tests/testthat test-metrics-point.R and
test-metrics-binary.R (2.3.0).

Not ported: the checks that ``brier_score()`` without ``observed`` or
``predicted`` errors (Python raises its own ``TypeError`` for a missing
argument), and the comparison of ``logs_binary()`` against
``Metrics::ll()``, which is replaced by the same formula written out.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import scoringutils_py as su
from scoringutils_py._checks import check_input_binary, check_input_point

rng = np.random.default_rng(1)
observed = pd.Categorical(rng.binomial(1, 0.5, 10), categories=[0, 1])
predicted = np.array([0.425, 0.55, 0.541, 0.52, 0.13, 0.469, 0.86, 0.22, 0.74, 0.9])
observed_point = rng.normal(size=10)
predicted_point = rng.normal(size=10)
outcome = np.asarray(observed).astype(float)


def no_warnings(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args)


# ---------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------


def test_correct_input_works():
    no_warnings(check_input_binary, observed, predicted)
    no_warnings(check_input_point, observed_point, predicted_point)
    # observed is a single value
    no_warnings(check_input_binary, pd.Categorical([1], categories=[0, 1]), predicted)
    no_warnings(check_input_point, 1, predicted_point)
    # predicted is a single value
    no_warnings(check_input_binary, observed, 0.2)
    no_warnings(check_input_point, observed_point, 0.2)
    # predicted is an (n x 1) matrix
    no_warnings(check_input_binary, observed, predicted.reshape(-1, 1))
    no_warnings(check_input_point, observed_point, predicted_point.reshape(-1, 1))


def test_binary_observed_must_be_categorical():
    with pytest.raises(
        TypeError,
        match="Assertion on 'observed' failed: Must be of type 'factor', not 'double'",
    ):
        check_input_binary(rng.normal(size=10), predicted)
    with pytest.raises(TypeError, match="Must be of type 'factor', not 'integer'"):
        check_input_binary(np.arange(1, 11), predicted)


def test_predicted_must_be_numeric():
    with pytest.raises(
        TypeError,
        match="Assertion on 'predicted' failed: Must be of type 'numeric', not 'list'",
    ):
        check_input_binary(observed, np.array(list(predicted), dtype=object))
    with pytest.raises(TypeError, match="Assertion on 'predicted' failed"):
        check_input_point(observed_point, np.array(["a"] * 10))


def test_point_observed_must_not_be_categorical():
    with pytest.raises(
        TypeError,
        match="Assertion on 'observed' failed: Must be of type 'numeric', not 'factor'",
    ):
        check_input_point(pd.Categorical(rng.normal(size=10)), predicted)


def test_binary_observed_needs_two_levels():
    with pytest.raises(ValueError, match="Must have exactly 2 levels"):
        check_input_binary(pd.Categorical(np.arange(1, 11)), predicted)


def test_wrong_length():
    msg = "`observed` and `predicted` must either be of length 1 or of equal length"
    with pytest.raises(ValueError, match=msg):
        check_input_binary(observed, rng.uniform(size=15))
    with pytest.raises(ValueError, match=msg):
        check_input_point(observed_point, rng.uniform(size=15))


def test_predicted_outside_unit_interval():
    with pytest.raises(
        ValueError, match="Assertion on 'predicted' failed: Element 1 is not <= 1."
    ):
        check_input_binary(observed, predicted + 1)
    with pytest.raises(
        ValueError, match="Assertion on 'predicted' failed: Element 1 is not >= 0."
    ):
        check_input_binary(observed, predicted - 1)


@pytest.mark.parametrize(
    "bad",
    [np.array([[0.2]]), np.tile(predicted, (2, 1)).T],
    ids=["one_by_one_matrix", "two_column_matrix"],
)
def test_predicted_matrix_with_wrong_shape(bad):
    with pytest.raises(
        ValueError, match="(?s)Assertion failed.*check_vector.*check_matrix"
    ):
        check_input_binary(observed, bad)
    with pytest.raises(
        ValueError, match="(?s)Assertion failed.*check_vector.*check_matrix"
    ):
        check_input_point(observed_point, bad)


# ---------------------------------------------------------------------------
# factor level order
# ---------------------------------------------------------------------------


def test_warns_on_reversed_zero_one_levels():
    obs = pd.Categorical([0, 1, 1, 0, 1], categories=[1, 0])
    with pytest.warns(UserWarning, match="counterintuitive"):
        check_input_binary(obs, [0.1, 0.9, 0.8, 0.2, 0.7])


def test_no_warning_for_standard_levels():
    obs = pd.Categorical(["0", "1", "1", "0", "1"], categories=["0", "1"])
    no_warnings(check_input_binary, obs, [0.1, 0.9, 0.8, 0.2, 0.7])


def test_no_warning_for_non_numeric_levels():
    obs = pd.Categorical(["a", "b", "b", "a"], categories=["a", "b"])
    no_warnings(check_input_binary, obs, [0.3, 0.7, 0.6, 0.4])


def test_brier_score_differs_with_reversed_levels():
    correct = pd.Categorical([0, 1, 1, 0], categories=[0, 1])
    reversed_ = pd.Categorical([0, 1, 1, 0], categories=[1, 0])
    pred = [0.1, 0.9, 0.8, 0.2]
    scores_correct = su.brier_score(correct, pred)
    np.testing.assert_allclose(scores_correct, [0.01, 0.01, 0.04, 0.04])
    with pytest.warns(UserWarning, match="counterintuitive"):
        scores_reversed = su.brier_score(reversed_, pred)
    assert not np.all(scores_correct == scores_reversed)


def test_logs_binary_warns_with_reversed_levels():
    obs = pd.Categorical([0, 1, 1, 0], categories=[1, 0])
    with pytest.warns(UserWarning, match="counterintuitive"):
        su.logs_binary(obs, [0.1, 0.9, 0.8, 0.2])


def test_warns_for_true_false_in_counterintuitive_order():
    obs = pd.Categorical([True, False, True], categories=[True, False])
    with pytest.warns(UserWarning, match="counterintuitive"):
        check_input_binary(obs, [0.8, 0.2, 0.9])


# ---------------------------------------------------------------------------
# binary metrics
# ---------------------------------------------------------------------------


def test_brier_score_different_inputs():
    one = pd.Categorical([1], categories=[0, 1])
    np.testing.assert_allclose(su.brier_score(one, predicted), (1 - predicted) ** 2)
    np.testing.assert_allclose(
        su.brier_score(observed, 0.2), np.where(outcome == 1, 0.8**2, 0.2**2)
    )
    with pytest.raises(ValueError, match="(?s)Assertion failed.*check_matrix"):
        su.brier_score(observed, np.array([[0.2]]))


def test_binary_metrics_within_and_outside_score():
    df = pd.DataFrame(
        {"observed": observed, "predicted": predicted, "model": "m1", "id": range(10)}
    )
    result = su.score(su.as_forecast_binary(df))
    np.testing.assert_allclose(
        su.brier_score(observed, predicted), result["brier_score"]
    )
    np.testing.assert_allclose(su.logs_binary(observed, predicted), result["log_score"])


def _ll(actual, pred):
    # Metrics::ll()
    return -(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))


def test_logs_binary_matches_log_loss():
    np.testing.assert_allclose(
        su.logs_binary(observed, predicted), _ll(outcome, predicted)
    )
    np.testing.assert_allclose(
        su.logs_binary(observed[:1], predicted), _ll(outcome[0], predicted)
    )
    np.testing.assert_allclose(
        su.logs_binary(observed, predicted[0]), _ll(outcome, predicted[0])
    )


def test_logs_binary_certain_and_wrong_is_infinite():
    obs = pd.Categorical([0, 1], categories=[0, 1])
    np.testing.assert_array_equal(no_warnings(su.logs_binary, obs, [1.0, 0.0]), np.inf)


def test_binary_metrics_keep_missing_values():
    obs = pd.Categorical([0, None, 1], categories=[0, 1])
    result = su.brier_score(obs, [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(np.isnan(result), [False, True, False])


# ---------------------------------------------------------------------------
# point metrics (adapted from covidHubUtils via test-metrics-point.R)
# ---------------------------------------------------------------------------

y = np.array([1, -15, 22])
target_end_dates = pd.to_datetime("2020-01-01") + pd.to_timedelta([7, 14, 7], "D")
horizons = ["1", "2", "1"]
locations = ["01", "01", "02"]


def _forecasts(point_forecast, median=None):
    n = len(point_forecast)
    rows = {
        "model": "m1",
        "forecast_date": pd.Timestamp("2020-01-01"),
        "location": locations,
        "horizon": horizons,
        "temporal_resolution": "wk",
        "target_variable": "inc death",
        "target_end_date": target_end_dates,
        "type": ["point"] * n,
        "predicted": point_forecast,
    }
    df = pd.DataFrame(rows)
    if median is not None:
        df = pd.concat([df, df.assign(type="quantile", predicted=median)])
    truth = pd.DataFrame(
        {
            "target_variable": "inc death",
            "target_end_date": target_end_dates,
            "location": locations,
            "observed": y,
        }
    )
    return df.merge(truth)


def test_abs_error_point_forecast_only():
    point_forecast = np.array([5, 6, 7])
    scores = su.score(su.as_forecast_point(_forecasts(point_forecast)))
    np.testing.assert_allclose(scores["ae_point"], np.abs(y - point_forecast))


def test_abs_error_point_and_median_different():
    point_forecast = np.array([5, 6, 7])
    data = _forecasts(point_forecast, median=np.array([1, 2, 3]))
    scores = su.score(su.as_forecast_point(data))
    np.testing.assert_allclose(
        scores.loc[scores["type"] == "point", "ae_point"], np.abs(y - point_forecast)
    )


def test_abs_error_point_and_median_same():
    point_forecast = np.array([1, 2, 3])
    data = _forecasts(point_forecast, median=np.array([1, 2, 3]))
    point = su.as_forecast_point(data[data["type"] == "point"])
    scores = su.score(point)
    summary = su.summarise_scores(scores, by=["location", "target_end_date", "horizon"])
    merged = summary.merge(
        pd.DataFrame(
            {
                "location": locations,
                "target_end_date": target_end_dates,
                "expected": np.abs(y - point_forecast),
            }
        )
    )
    np.testing.assert_allclose(merged["ae_point"], merged["expected"])


def test_ape_zero_observed_is_inf_without_warning():
    result = no_warnings(su.ape, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))
    assert np.isinf(result[0])
    assert np.isnan(result[1])
    assert result[2] == 0
