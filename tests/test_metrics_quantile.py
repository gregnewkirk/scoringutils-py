"""Ported from scoringutils tests/testthat/test-metrics-quantile.R and
test-metrics-interval-range.R (2.3.0)."""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import scoringutils_py as su
from scoringutils_py._checks import ScoringutilsMessage, check_input_quantile

observed = np.array([1.0, -15, 22])
predicted = np.array([[-1.0, 0, 1, 2, 3], [-2, 1, 2, 2, 4], [-2, 0, 3, 3, 4]])
quantile_level = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
y = observed
fq_matrix = predicted
fq_probs = quantile_level


def wis_metric(**kwargs):
    return {"wis": partial(su.wis, **kwargs)}


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------


def test_input_checking_for_quantile_forecasts():
    check_input_quantile(observed, predicted, quantile_level)
    with pytest.raises(ValueError, match="Element 1 is not <= 1"):
        check_input_quantile(observed, predicted, quantile_level + 1)
    with pytest.raises(ValueError, match="Element 1 is not >= 0"):
        check_input_quantile(observed, predicted, quantile_level - 1)
    with pytest.raises(ValueError, match="Must have exactly 10 rows, but has 3 rows"):
        check_input_quantile(np.arange(1, 11), predicted, quantile_level)
    with pytest.raises(TypeError, match="Must be of type 'numeric'"):
        check_input_quantile(np.array(list("abc")), predicted, quantile_level)
    with pytest.raises(ValueError, match="(?s)check_numeric_vector.*check_matrix"):
        check_input_quantile(1, predicted, quantile_level)
    with pytest.raises(ValueError, match="Must be of type 'matrix', not 'double'"):
        check_input_quantile(observed, predicted.ravel(), quantile_level)


# ---------------------------------------------------------------------------
# wis
# ---------------------------------------------------------------------------


def test_wis_median_only():
    lower = upper = predicted_quantile = np.array([1.0, 2, 3])
    actual = su.interval_score(
        y, lower=lower, upper=upper, weigh=True, interval_range=0
    )
    actual_wis = su.wis(y, predicted_quantile.reshape(-1, 1), [0.5])
    expected = np.abs(y - lower)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_wis, expected)


def _quantile_frame(obs, levels, preds):
    n = len(obs)
    return pd.DataFrame(
        {
            "observed": np.tile(obs, len(levels)),
            "quantile_level": np.repeat(levels, n),
            "predicted": preds,
            "model": "model1",
            "date": np.tile(np.arange(1, n + 1), len(levels)),
        }
    )


def test_wis_within_score_for_median_forecast():
    data = pd.DataFrame(
        {
            "observed": [1, -15, 22],
            "predicted": [1, 2, 3],
            "quantile_level": 0.5,
            "model": "model1",
            "date": [1, 2, 3],
        }
    )
    fc = su.as_forecast_quantile(data)
    metrics = su.get_metrics(fc, select=["ae_median", "wis"])
    metrics["wis"] = partial(metrics["wis"], count_median_twice=True)
    scores = su.score(fc, metrics=metrics)
    np.testing.assert_allclose(scores["ae_median"], scores["wis"])


def test_wis_equals_interval_score_one_interval():
    lower = np.array([0.0, 1, 0])
    upper = np.array([2.0, 2, 3])
    pred = np.column_stack([lower, upper])
    alpha = 0.5
    expected = (upper - lower) * (alpha / 2) + np.array([0, 1 - (-15), 22 - 3])
    actual = su.interval_score(
        y, lower=lower, upper=upper, weigh=True, interval_range=50
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(su.wis(y, pred, [0.25, 0.75]), expected)


def test_wis_within_score_for_one_interval():
    fc = su.as_forecast_quantile(_quantile_frame(y, [0.25, 0.75], [0, 1, 0, 2, 2, 3]))
    scores = su.score(fc, metrics=wis_metric(count_median_twice=True))
    scores = su.summarise_scores(scores, by=["model", "date"])
    expected = (np.array([2.0, 2, 3]) - np.array([0.0, 1, 0])) * 0.25 + np.array(
        [0, 16, 19]
    )
    np.testing.assert_allclose(scores["wis"], expected)


def test_wis_one_interval_and_median():
    fc = su.as_forecast_quantile(
        _quantile_frame(y, [0.25, 0.5, 0.75], [0, 1, 0, 1, 2, 3, 2, 2, 3])
    )
    scores = su.score(fc, metrics=wis_metric(count_median_twice=True))
    scores = su.summarise_scores(scores, by=["model", "date"])
    q = np.array([[0.0, 1, 2], [1, 2, 2], [0, 3, 3]])
    alpha = 0.5
    expected = 0.5 * (
        np.abs(y - q[:, 1]) + (q[:, 2] - q[:, 0]) * (alpha / 2) + np.array([0, 16, 19])
    )
    actual = su.wis(y, q, [0.25, 0.5, 0.75], count_median_twice=True)
    np.testing.assert_allclose(scores["wis"], expected)
    np.testing.assert_allclose(actual, expected)


def test_wis_two_intervals_and_median():
    fc = su.as_forecast_quantile(
        _quantile_frame(y, fq_probs, [-1, -2, -2, 0, 1, 0, 1, 2, 3, 2, 2, 3, 3, 4, 4])
    )
    scores = su.score(fc, metrics=wis_metric(count_median_twice=True))
    scores = su.summarise_scores(scores, by=["model", "date"])
    q = fq_matrix
    a1, a2 = 0.2, 0.5
    expected = (1 / 3) * (
        np.abs(y - q[:, 2])
        + (q[:, 4] - q[:, 0]) * (a1 / 2)
        + np.array([0, -2 + 15, 22 - 4])
        + (q[:, 3] - q[:, 1]) * (a2 / 2)
        + np.array([0, 1 + 15, 22 - 3])
    )
    np.testing.assert_allclose(scores["wis"], expected)
    np.testing.assert_allclose(
        su.wis(y, q, fq_probs, count_median_twice=True), expected
    )


def _hub_frame(levels_idx):
    """The covidHubUtils comparison data used upstream."""
    q = fq_matrix[:, levels_idx]
    probs = fq_probs[levels_idx]
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta([7, 14, 7], unit="D")
    truth = pd.DataFrame(
        {
            "target_variable": "inc death",
            "target_end_date": dates,
            "location": ["01", "01", "02"],
            "observed": y,
        }
    )
    k = q.shape[1]
    forecasts = pd.DataFrame(
        {
            "model": "m1",
            "forecast_date": pd.to_datetime("2020-01-01"),
            "location": np.tile(["01", "01", "02"], k),
            "horizon": np.tile(["1", "2", "1"], k),
            "temporal_resolution": "wk",
            "target_variable": "inc death",
            "target_end_date": np.tile(dates, k),
            "type": "quantile",
            "quantile_level": np.repeat(probs, 3),
            "predicted": q.T.ravel(),
        }
    )
    merged = forecasts.merge(
        truth, on=["location", "target_end_date", "target_variable"]
    )
    return su.as_forecast_quantile(merged), q


HUB_BY = [
    "model",
    "location",
    "target_variable",
    "target_end_date",
    "forecast_date",
    "horizon",
]


def test_wis_median_only_covidhubutils():
    fc, q = _hub_frame([2])
    scores = su.score(fc, metrics=wis_metric(count_median_twice=True))
    expected = np.abs(y - q[:, 0])
    np.testing.assert_allclose(scores["wis"], expected)
    np.testing.assert_allclose(su.wis(y, q, [0.5], count_median_twice=False), expected)


def test_wis_one_interval_covidhubutils():
    fc, q = _hub_frame([0, 4])
    scores = su.summarise_scores(su.score(fc, metrics={"wis": su.wis}), by=HUB_BY)
    expected = (q[:, 1] - q[:, 0]) * (0.2 / 2) + np.array([0, -2 + 15, 22 - 4])
    np.testing.assert_allclose(scores["wis"], expected)
    np.testing.assert_allclose(su.wis(y, q, [0.1, 0.9]), expected)


def test_wis_two_intervals_and_median_covidhubutils():
    fc, q = _hub_frame([0, 1, 2, 3, 4])
    metrics = su.get_metrics(
        fc, exclude=["interval_coverage_50", "interval_coverage_90"]
    )
    scores = su.summarise_scores(su.score(fc, metrics=metrics), by=HUB_BY)
    a1, a2 = 0.2, 0.5
    parts = (
        (q[:, 4] - q[:, 0]) * (a1 / 2)
        + np.array([0, -2 + 15, 22 - 4])
        + (q[:, 3] - q[:, 1]) * (a2 / 2)
        + np.array([0, 1 + 15, 22 - 3])
    )
    expected = (1 / 2.5) * (0.5 * np.abs(y - q[:, 2]) + parts)
    np.testing.assert_allclose(scores["wis"], expected)
    np.testing.assert_allclose(su.wis(y, q, fq_probs), expected)

    expected2 = (1 / 3) * (np.abs(y - q[:, 2]) + parts)
    np.testing.assert_allclose(
        su.wis(y, q, fq_probs, count_median_twice=True), expected2
    )
    scores2 = su.summarise_scores(
        su.score(fc, metrics=wis_metric(count_median_twice=True)), by=HUB_BY
    )
    np.testing.assert_allclose(scores2["wis"], expected2)


@pytest.mark.parametrize("weigh", [True, False])
def test_quantile_score_and_interval_score_agree(weigh):
    np.testing.assert_allclose(
        su.quantile_score(observed, predicted, quantile_level),
        su.wis(observed, predicted, quantile_level),
    )
    rng = np.random.default_rng(1)
    obs = rng.normal(np.arange(1, 11))
    for alpha in (0.1, 0.5, 0.9):
        lower = norm.ppf(alpha / 2, rng.normal(np.arange(1, 11)))
        upper = norm.ppf(1 - alpha / 2, rng.normal(np.arange(11, 21)))
        interval = su.interval_score(
            obs, lower, upper, interval_range=(1 - alpha) * 100, weigh=weigh
        )
        w = su.wis(
            obs,
            np.column_stack([lower, upper]),
            [alpha / 2, 1 - alpha / 2],
            count_median_twice=False,
            weigh=weigh,
        )
        qs_lower = su.quantile_score(obs, lower[:, None], [alpha / 2], weigh=weigh)
        qs_upper = su.quantile_score(obs, upper[:, None], [1 - alpha / 2], weigh=weigh)
        np.testing.assert_allclose((qs_lower + qs_upper) / 2, interval)
        np.testing.assert_allclose(w, interval)


def test_wis_errors_without_valid_intervals():
    pred = [
        0.1032978,
        0.1078166,
        0.1906849,
        0.1969254,
        0.2017249,
        0.2078487,
        0.2810377,
        0.3062168,
        0.4006721,
        0.8219655,
    ]
    levels = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
    with pytest.raises(ValueError, match="No valid forecast intervals found"):
        su.wis(0.1, pred, levels)


def test_wis_with_quantile_levels_zero_and_one():
    pred = np.array([[1.0, 3, 5, 7, 9]])
    levels = [0, 0.25, 0.5, 0.75, 1]
    np.testing.assert_allclose(su.wis(5, pred, levels), 0.4)
    np.testing.assert_allclose(
        su.wis(5, pred, levels), su.quantile_score(5, pred, levels)
    )
    np.testing.assert_allclose(
        su.wis(10, pred, levels), su.quantile_score(10, pred, levels)
    )
    separate = su.wis(5, pred, levels, separate_results=True)
    assert separate["overprediction"][0] == 0
    assert separate["underprediction"][0] == 0


def test_interval_score_handles_range_100():
    weighted = su.interval_score(5, 1, 9, interval_range=100, separate_results=True)
    assert all(float(v) == 0 for v in weighted.values())
    assert su.interval_score(5, 1, 9, interval_range=100, weigh=False) == 8
    unweighted = su.interval_score(
        10, 1, 9, interval_range=100, weigh=False, separate_results=True
    )
    assert unweighted["underprediction"] == np.inf
    assert unweighted["overprediction"] == 0


def test_quantile_score_unweighted_levels_zero_and_one():
    pred = np.array([[1.0, 9]])
    assert su.quantile_score(5, pred, [0, 1], weigh=False)[0] == 0
    assert su.quantile_score(10, pred, [0, 1], weigh=False)[0] == np.inf


def test_wis_components_add_up():
    res = su.wis(y, fq_matrix, fq_probs, separate_results=True)
    np.testing.assert_allclose(
        res["wis"], res["dispersion"] + res["overprediction"] + res["underprediction"]
    )
    total = su.wis(y, fq_matrix, fq_probs)
    d = su.dispersion_quantile(y, fq_matrix, fq_probs)
    o = su.overprediction_quantile(y, fq_matrix, fq_probs)
    u = su.underprediction_quantile(y, fq_matrix, fq_probs)
    np.testing.assert_allclose(total, d + o + u)


def test_wis_incomplete_intervals_error_and_na_rm():
    with pytest.raises(ValueError, match="symmetric prediction intervals"):
        su.wis(y, fq_matrix[:, :4], fq_probs[:4])
    # dropping the incomplete interval gives the same as never having it
    np.testing.assert_allclose(
        su.wis(y, fq_matrix[:, :4], fq_probs[:4], na_rm=True),
        su.wis(y, fq_matrix[:, 1:4], fq_probs[1:4]),
    )


# ---------------------------------------------------------------------------
# interval_coverage
# ---------------------------------------------------------------------------


def test_interval_coverage_works():
    np.testing.assert_array_equal(
        su.interval_coverage(observed, predicted, quantile_level, interval_range=50),
        [True, False, False],
    )


def test_interval_coverage_floating_point_levels():
    levels = np.arange(0.05, 0.951, 0.05)  # like R's seq(0.05, 0.95, 0.05)
    pred = np.vstack([x + norm.ppf(levels) for x in (1, 0, 22)])
    np.testing.assert_array_equal(
        su.interval_coverage(observed, pred, levels, interval_range=30),
        [True, False, True],
    )
    np.testing.assert_array_equal(
        su.interval_coverage(observed, pred, levels, interval_range=50),
        [True, False, True],
    )


def test_interval_coverage_rejects_wrong_inputs():
    with pytest.raises(ValueError, match="Must have length 1"):
        su.interval_coverage(
            observed, predicted, quantile_level, interval_range=[50, 0]
        )


def test_interval_coverage_requires_quantiles():
    with pytest.raises(ValueError, match="the 0.25 and 0.75 quantiles are required"):
        su.interval_coverage(
            observed,
            np.delete(predicted, 3, axis=1),
            np.delete(quantile_level, 3),
            interval_range=50,
        )


# ---------------------------------------------------------------------------
# bias_quantile
# ---------------------------------------------------------------------------


def test_bias_quantile_works_as_expected():
    assert su.bias_quantile(2, [1, 2, 3], [0.1, 0.5, 0.9])[0] == 0
    assert su.bias_quantile(2, [0, 1, 2], [0.1, 0.5, 0.9])[0] == pytest.approx(-0.8)
    pred = [
        705.500,
        1127.000,
        4006.250,
        4341.500,
        4709.000,
        4821.996,
        5340.500,
        5451.000,
        5703.500,
        6087.014,
        6329.500,
        6341.000,
        6352.500,
        6594.986,
        6978.500,
        7231.000,
        7341.500,
        7860.004,
        7973.000,
        8340.500,
        8675.750,
        11555.000,
        11976.500,
    ]
    levels = [0.01, 0.025, *np.arange(0.05, 0.951, 0.05), 0.975, 0.99]
    assert su.bias_quantile(8062, pred, levels)[0] == pytest.approx(-0.8)


def test_bias_quantile_matrix_input():
    obs = np.linspace(10, 0, 4)
    pred = np.arange(1, 13, dtype=float).reshape(3, 4).T
    np.testing.assert_allclose(
        su.bias_quantile(obs, pred, [0.1, 0.5, 0.9]), [-1.0, -0.8, 0.8, 1.0]
    )


def test_bias_quantile_vector_too_long():
    with pytest.raises(ValueError, match="Must have length 4, but has length 3"):
        su.bias_quantile(2, [np.nan, 1, 2, 3], [0.1, 0.5, 0.9])


def test_bias_quantile_handles_na():
    assert su.bias_quantile(2, [np.nan, 1, 2], [0.1, 0.5, 0.9])[0] == pytest.approx(
        -0.8
    )
    assert su.bias_quantile(2, [0, 1, 2], [0.1, 0.5, np.nan])[0] == -1
    assert np.isnan(su.bias_quantile(2, [0, 1, 2], [0.1, 0.5, np.nan], na_rm=False)[0])


def test_bias_quantile_order_invariant():
    a = su.bias_quantile(1.5, [3, 1, 2], [0.75, 0.25, 0.5])
    assert a[0] == 0.5
    np.testing.assert_array_equal(
        a, su.bias_quantile(1.5, [1, 2, 3], [0.25, 0.5, 0.75])
    )


def test_bias_quantile_na_removing_one_side():
    assert np.isnan(su.bias_quantile(2, [np.nan, np.nan, 3], [0.25, 0.5, 0.75])[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScoringutilsMessage)
        assert np.isnan(su.bias_quantile(2, [np.nan, 3], [0.25, 0.75])[0])
    assert su.bias_quantile(1, [np.nan, 2, 3], [0.25, 0.5, 0.75])[0] == 1


def test_bias_quantile_errors_without_predictions():
    with pytest.raises(ValueError, match="Must have length >= 1, but has length 0"):
        su.bias_quantile(2, [], [])


@pytest.mark.parametrize(
    ("obs", "pred", "levels", "expected"),
    [
        (1, [1, 2, 4, 5], [0.1, 0.3, 0.7, 0.9], 0.8),
        (5, [1, 2, 4, 5], [0.1, 0.3, 0.7, 0.9], -0.8),
        (3, [1, 2, 3, 4], [0.1, 0.3, 0.5, 0.7], 0),
        (1, [2, 3, 4, 5], [0.1, 0.3, 0.7, 0.9], 1),
        (6, [1, 2, 3, 4], [0.1, 0.3, 0.5, 0.7], -1),
    ],
)
def test_bias_quantile_known_values(obs, pred, levels, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScoringutilsMessage)
        assert su.bias_quantile(obs, pred, levels)[0] == pytest.approx(expected)


def test_bias_quantile_level_checks():
    with pytest.raises(ValueError, match="Element 1 is not >= 0"):
        su.bias_quantile(3, [1, 2, 3, 4], [-0.1, 0.3, 0.5, 0.8])
    with pytest.raises(ValueError, match="must not be decreasing"):
        su.bias_quantile(3, [1, 2, 3, 4], [0.8, 0.3, 0.5, 0.9])
    with pytest.raises(ValueError, match="must not be decreasing"):
        su.bias_quantile(3, [1, 2, 4, 3], [0.1, 0.3, 0.5, 0.9])
    with pytest.raises(ValueError, match="Contains duplicated values, position 2"):
        su.bias_quantile(3, [1, 2, 3, 4], [0.3, 0.3, 0.5, 0.8])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.bias_quantile(3, [1, 2, 3, 4], [0.1, 0.3, 0.5, 0.8])
        su.bias_quantile(3, [1, 2, 3, 4], [0.3, 0.5, 0.8, 0.9])


def test_bias_quantile_only_one_message():
    with pytest.warns(ScoringutilsMessage, match="Median not available") as record:
        su.bias_quantile(
            observed, np.delete(predicted, 2, axis=1), np.delete(quantile_level, 2)
        )
    assert len([r for r in record if issubclass(r.category, ScoringutilsMessage)]) == 1


def test_bias_quantile_point_forecast():
    assert su.bias_quantile(1, [1], [0.5])[0] == 0


def test_bias_quantile_median_side_missing():
    with pytest.raises(ValueError, match=r"quantile_level >= 0.5\]' failed"):
        su.bias_quantile(15, [1, 10], [0.2, 0.4])


def test_interpolate_median():
    assert su.interpolate_median([1, 10], [0.4, 0.6]) == pytest.approx(5.5)
    assert su.interpolate_median([1, 10], [0.1, 0.9]) == pytest.approx(5.5)
    assert su.interpolate_median([1, 10], [0.3, 0.6]) == pytest.approx(
        1 + 0.2 / 0.3 * 9
    )
    assert su.interpolate_median([1, 10], [0.2, 0.6]) == pytest.approx(
        1 + 0.3 / 0.4 * 9
    )


# ---------------------------------------------------------------------------
# ae_median_quantile and quantile_score
# ---------------------------------------------------------------------------


def test_ae_median_quantile():
    obs = np.arange(1, 31, dtype=float)
    pred = np.arange(2, 32, dtype=float).reshape(-1, 1)
    np.testing.assert_array_equal(
        su.ae_median_quantile(obs, pred, [0.5]), pred[:, 0] - obs
    )
    with pytest.raises(ValueError, match="Must be of type 'matrix'"):
        su.ae_median_quantile(obs, pred[:, 0], [0.5])
    assert su.ae_median_quantile(obs[0], pred[:3, 0], [0.1, 0.5, 0.7])[0] == 2
    with pytest.raises(ValueError, match="In order to compute the absolute error"):
        su.ae_median_quantile(obs, pred, [0.6])


def test_quantile_score_vector_or_matrix():
    rng = np.random.default_rng(3)
    obs = rng.normal(2)
    alpha = [0.2, 0.4, 0.6, 0.8]
    pred = rng.normal(np.arange(1, 5))
    np.testing.assert_allclose(
        su.quantile_score(obs, pred, alpha),
        su.quantile_score(obs, pred[None, :], alpha),
    )


def test_quantile_score_is_twice_scoringrules_pinball():
    """Documented difference: scoringutils' quantile score is 2 x pinball loss."""
    import scoringrules as sr

    for q in (0.1, 0.5, 0.9):
        ours = su.quantile_score(observed, predicted[:, [2]], [q])
        theirs = np.asarray(sr.quantile_score(observed, predicted[:, 2], q))
        np.testing.assert_allclose(ours, 2 * theirs)


def test_wis_matches_scoringrules_interval_score():
    """WIS built from scoringrules' interval score equals scoringutils' WIS.

    ``scoringrules.weighted_interval_score`` itself is not used: in 0.9 to
    0.11 its numpy backend adds ``w_median * median`` instead of
    ``w_median * |obs - median|`` (the numba backend is correct).
    """
    import scoringrules as sr

    alphas = np.array([0.2, 0.5])
    lower = predicted[:, [0, 1]]
    upper = predicted[:, [4, 3]]
    interval = np.column_stack(
        [
            sr.interval_score(observed, lower[:, k], upper[:, k], a)
            for k, a in enumerate(alphas)
        ]
    )
    median = predicted[:, 2]
    expected = np.sum(alphas / 2 * interval, axis=1) + 0.5 * np.abs(observed - median)
    expected = expected / (len(alphas) + 0.5)
    np.testing.assert_allclose(su.wis(observed, predicted, quantile_level), expected)


# ---------------------------------------------------------------------------
# interval range (test-metrics-interval-range.R)
# ---------------------------------------------------------------------------


def test_assert_input_interval():
    rng = np.random.default_rng(5)
    obs = rng.normal(np.arange(1, 31))
    lower = norm.ppf(0.05, rng.normal(np.arange(1, 31)))
    upper = norm.ppf(0.95, rng.normal(np.arange(11, 41)))
    su.interval_score(obs, lower, upper, 90)
    with pytest.raises(ValueError, match="All values in `upper` need to be greater"):
        su.interval_score(obs, upper, lower, 90)
    import scoringutils_py._interval as iv

    iv._small_range_warned = False
    with pytest.warns(UserWarning, match="Found interval ranges between 0 and 1"):
        su.interval_score(obs, lower, upper, 0.5)
