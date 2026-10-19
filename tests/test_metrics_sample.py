"""Ported from scoringutils tests/testthat/test-metrics-sample.R (2.3.0).

PIT histogram tests are not ported (PIT histograms are not in v1).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import scoringutils_py as su

rng = np.random.default_rng(123)


def pois_matrix(n_rows, n_cols):
    lam = np.arange(1, n_rows + 1)
    return rng.poisson(lam[:, None], size=(n_rows, n_cols))


def test_input_handling():
    obs = rng.poisson(np.arange(1, 31))
    pred = pois_matrix(30, 20)
    assert len(su.crps_sample(obs, pred)) == 30
    with pytest.raises(ValueError, match="Must be of type 'matrix', not 'integer'"):
        su.crps_sample(obs, rng.poisson(1, size=30))
    with pytest.raises(ValueError, match="Must have exactly 30 rows, but has 31 rows"):
        su.crps_sample(obs, pois_matrix(31, 20))
    with pytest.raises(TypeError):
        su.crps_sample(predicted=pred)


def test_crps_components_add_up():
    obs = rng.poisson(np.arange(1, 31))
    pred = pois_matrix(30, 20)
    res = su.crps_sample(obs, pred, separate_results=True)
    np.testing.assert_allclose(
        res["crps"], res["dispersion"] + res["overprediction"] + res["underprediction"]
    )
    cases = (
        (obs, pred),
        (
            rng.normal(np.arange(1, 31)),
            rng.normal(np.arange(1, 31)[:, None], size=(30, 20)),
        ),
    )
    for obs, pred in cases:
        crps = su.crps_sample(obs, pred)
        total = (
            su.dispersion_sample(obs, pred)
            + su.overprediction_sample(obs, pred)
            + su.underprediction_sample(obs, pred)
        )
        np.testing.assert_allclose(crps, total)


def test_crps_components_correspond_to_wis_components():
    r = np.random.default_rng(123)
    n, m = 15, 2000
    obs = r.normal(np.arange(1, n + 1))
    pred = r.normal(np.arange(1, n + 1)[:, None], size=(n, m))
    levels = np.round(np.arange(0.01, 0.995, 0.01), 10)
    quantiles = np.quantile(pred, levels, axis=1).T
    diffs = np.concatenate(
        [
            su.dispersion_sample(obs, pred)
            - su.dispersion_quantile(obs, quantiles, levels),
            su.overprediction_sample(obs, pred)
            - su.overprediction_quantile(obs, quantiles, levels),
            su.underprediction_sample(obs, pred)
            - su.underprediction_quantile(obs, quantiles, levels),
        ]
    )
    assert np.all(np.abs(diffs) < 0.01)


def test_crps_single_observation():
    res = su.crps_sample(2.5, np.arange(1.5, 11.5), separate_results=True)
    assert len(res) == 4
    assert {len(v) for v in res.values()} == {1}


def test_bias_sample_missing_arguments():
    with pytest.raises(TypeError):
        su.bias_sample(predicted=pois_matrix(10, 50))
    with pytest.raises(TypeError):
        su.bias_sample(observed=rng.poisson(np.arange(1, 11)))


@pytest.mark.parametrize("kind", ["integer", "continuous"])
def test_bias_sample_output(kind):
    if kind == "integer":
        obs, pred = rng.poisson(np.arange(1, 11)), pois_matrix(10, 10)
    else:
        obs, pred = rng.normal(size=10), rng.normal(size=(10, 10))
    out = su.bias_sample(obs, pred)
    assert len(out) == 10
    assert out.dtype == float


def test_bias_sample_approx_equals_bias_quantile():
    r = np.random.default_rng(123)
    obs = 3
    pred = r.normal(obs, 2, size=1000)
    b_sample = su.bias_sample(obs, pred[None, :])
    levels = np.linspace(0, 1, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b_quantile = su.bias_quantile(obs, np.quantile(pred, levels), levels)
    assert b_quantile == pytest.approx(b_sample, abs=0.1)


def test_bias_sample_vector_of_samples():
    np.testing.assert_array_equal(
        su.bias_sample(1, [0, 2, 4]), su.bias_sample(1, np.array([[0, 2, 4]]))
    )
    np.testing.assert_array_equal(
        su.bias_sample(1.5, [0.5, 2.5, 4.5]),
        su.bias_sample(1.5, np.array([[0.5, 2.5, 4.5]])),
    )


def test_bias_sample_ties_give_zero():
    assert su.bias_sample(1.5, [1.5, 1.5, 1.5, 1.6])[0] == pytest.approx(1 - 2 * 0.375)
    assert su.bias_sample(1.5, [1.5, 1.5])[0] == 0


def test_ae_median_sample():
    obs = rng.normal(np.arange(1, 31))
    pred = rng.normal(np.arange(1, 31))
    np.testing.assert_allclose(
        su.ae_median_sample(obs, pred[:, None]), np.abs(obs - pred)
    )
    ae = su.ae_median_sample(1, [0, 2, 4])
    assert len(ae) == 1 and ae[0] == 1


def test_se_mean_sample_vector():
    se = su.se_mean_sample(1, [0, 2, 4])
    assert len(se) == 1 and se[0] == 1


def test_mad_sample():
    with pytest.raises(TypeError):
        su.mad_sample()
    # R: mad(c(0, 2, 4)) = 1.4826 * 2
    assert su.mad_sample(predicted=[0, 2, 4])[0] == pytest.approx(2.9652)


def test_vector_and_one_row_matrix_identical():
    obs = 2.3
    pred = rng.normal(2, size=20)
    mat = pred[None, :]
    for fn in (su.ae_median_sample, su.se_mean_sample, su.bias_sample):
        np.testing.assert_array_equal(fn(obs, pred), fn(obs, mat))
    np.testing.assert_array_equal(
        su.mad_sample(predicted=pred), su.mad_sample(predicted=mat)
    )


def test_logs_sample_warns_for_integer_predictions():
    obs = rng.poisson(np.arange(1, 31))
    pred = pois_matrix(30, 200)
    with pytest.warns(UserWarning, match="integer"):
        su.logs_sample(obs, pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = su.logs_sample(obs, pred)
    assert len(result) == 30
    assert np.all(np.isfinite(result))
    # integer-valued doubles warn too
    with pytest.warns(UserWarning, match="integer"):
        su.logs_sample(obs.astype(float), pred.astype(float))


def test_logs_sample_continuous_no_warning():
    obs = rng.normal(np.arange(1, 31))
    pred = rng.normal(np.arange(1, 31)[:, None], size=(30, 200))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = su.logs_sample(obs, pred)
    assert result.dtype == float and len(result) == 30


def test_logs_sample_zero_bandwidth_matches_r():
    # all samples equal: R returns Inf (scoringrules alone would give NaN)
    assert su.logs_sample(1.0, [0.0, 0.0, 0.0])[0] == np.inf
    assert su.logs_sample(0.0, [0.0, 0.0, 0.0])[0] == -np.inf


def test_score_warns_for_integer_samples(fc_sample_discrete):
    with pytest.warns(UserWarning, match="integer"):
        su.score(fc_sample_discrete)


def test_score_continuous_samples_no_log_score_warning(fc_sample_continuous):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.score(fc_sample_continuous)


def test_score_with_one_and_two_samples(fc_sample_continuous):
    df = pd.DataFrame(fc_sample_continuous).dropna()
    one = su.new_forecast(df[df.sample_id == 20], "sample")
    with pytest.warns(UserWarning, match="Computation for `log_score` failed"):
        scores = su.score(one)
    assert scores["dss"].isna().all()
    two = su.new_forecast(df[df.sample_id.isin([20, 21])], "sample")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.score(two)
