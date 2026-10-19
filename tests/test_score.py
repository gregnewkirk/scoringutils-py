"""Ported from scoringutils tests/testthat test-score.R, test-class-scores.R,
test-metrics.R, test-summarise_scores.R and the score() parts of
test-class-forecast-quantile.R (2.3.0)."""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest

import scoringutils_py as su
from scoringutils_py import ScoringutilsMessage


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Scores class
# ---------------------------------------------------------------------------


def test_new_and_as_scores():
    assert isinstance(su.new_scores(pd.DataFrame(), metrics=[""]), su.Scores)
    s = su.as_scores(pd.DataFrame({"wis": [1.0]}), metrics=["wis"])
    assert isinstance(s, su.Scores) and s.metrics == ["wis"]


def test_assert_scores():
    with pytest.raises(TypeError, match="Must inherit from class 'scores'"):
        su.assert_scores(pd.DataFrame())


def test_score_output_is_scores(scores_quantile, scores_sample_continuous):
    su.assert_scores(scores_quantile)
    su.assert_scores(scores_sample_continuous)


def test_subsetting_scores_keeps_metrics(scores_quantile):
    sub = scores_quantile.iloc[:9]
    assert isinstance(sub, su.Scores)
    assert sub.metrics == scores_quantile.metrics
    sub2 = scores_quantile[["model", "wis"]]
    assert sub2.metrics == scores_quantile.metrics


def test_score_rejects_non_forecasts():
    with pytest.raises(
        TypeError, match="The input needs to be a valid forecast object"
    ):
        su.score(None)
    with pytest.raises(
        TypeError, match="The input needs to be a valid forecast object"
    ):
        su.score(pd.DataFrame(su.example_quantile()))


def test_score_sample_continuous_shape(scores_sample_continuous):
    assert len(scores_sample_continuous) == 887


def test_score_errors_with_only_na():
    fc = su.new_forecast(
        pd.DataFrame(su.example_sample_continuous()).assign(predicted=np.nan), "sample"
    )
    with pytest.raises(ValueError, match="no forecasts are left"):
        su.score(fc)


# ---------------------------------------------------------------------------
# metrics handling
# ---------------------------------------------------------------------------


def test_apply_metrics_warns_about_clashes(fc_quantile):
    df = pd.DataFrame(fc_quantile).assign(wis=999.0)
    fc = quiet(su.as_forecast_quantile, df)  # "wis" becomes part of the forecast unit
    with pytest.warns(UserWarning, match="already present"):
        su.score(fc, metrics={"wis": su.wis})


def test_invalid_metrics_are_dropped_with_warning(fc_quantile):
    with pytest.warns(UserWarning, match="element number 2 is not a valid function"):
        s = su.score(fc_quantile, metrics={"wis": su.wis, "hi": "hi"})
    assert s.metrics == ["wis"]


def test_failing_metric_warns_and_is_skipped(fc_quantile):
    def broken(observed, predicted, quantile_level):
        raise RuntimeError("nope")

    with pytest.warns(
        UserWarning, match="Computation for `broken` failed. Error: nope"
    ):
        s = su.score(fc_quantile, metrics={"wis": su.wis, "broken": broken})
    assert "broken" not in s.columns
    assert s.metrics == ["wis"]


def test_select_metrics():
    m = su.get_metrics(su.example_quantile())
    assert su.select_metrics(m) == m
    assert list(su.select_metrics(m, select="wis")) == ["wis"]
    assert len(su.select_metrics(m, exclude="wis")) == len(m) - 1
    # if both are given, exclude is ignored
    assert list(su.select_metrics(m, select="wis", exclude="wis")) == ["wis"]
    with pytest.raises(TypeError, match="Must be of type 'list'"):
        su.select_metrics(su.get_metrics)


def test_get_metrics_select_and_exclude():
    with pytest.raises(ValueError, match="Must be a subset of"):
        su.get_metrics(su.example_sample_continuous(), select="not-included")
    with pytest.raises(ValueError, match="Must be a subset of"):
        su.get_metrics(su.example_quantile(), exclude="not-included")


def test_default_metric_names():
    assert list(su.get_metrics(su.example_quantile())) == [
        "wis",
        "overprediction",
        "underprediction",
        "dispersion",
        "bias",
        "interval_coverage_50",
        "interval_coverage_90",
        "ae_median",
    ]
    assert list(su.get_metrics(su.example_sample_continuous())) == [
        "bias",
        "dss",
        "crps",
        "overprediction",
        "underprediction",
        "dispersion",
        "log_score",
        "mad",
        "ae_median",
        "se_mean",
    ]


def test_get_metrics_on_scores(scores_sample_continuous):
    assert su.get_metrics(scores_sample_continuous) == scores_sample_continuous.metrics
    with pytest.raises(ValueError, match="Input needs an attribute"):
        su.get_metrics(pd.DataFrame(su.example_quantile()), error=True)
    renamed = scores_sample_continuous.rename(columns={"crps": "changed"})
    with pytest.warns(UserWarning, match="no longer column names"):
        su.get_metrics(renamed)


# ---------------------------------------------------------------------------
# score() for quantile forecasts
# ---------------------------------------------------------------------------


def test_score_quantile_separate_results_false(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    df = df[
        (df.model == "EuroCOVIDhub-ensemble")
        & (df.target_type == "Cases")
        & (df.location == "DE")
    ]
    metrics = su.get_metrics(fc_quantile)
    metrics["wis"] = partial(su.wis, separate_results=False)
    s = su.score(su.as_forecast_quantile(df), metrics=metrics)
    assert len(s) > 1
    assert all(m in s.columns for m in su.get_metrics(fc_quantile))


def test_score_quantile_produces_metrics():
    data = pd.DataFrame(
        {
            "observed": np.repeat(np.arange(1, 11), 3),
            "predicted": np.tile([-0.3, 0, 0.3], 10) + np.repeat(np.arange(1, 11), 3),
            "model": "Model 1",
            "date": np.repeat(np.arange(10), 3),
            "quantile_level": np.tile([0.1, 0.5, 0.9], 10),
        }
    )
    fc = quiet(su.as_forecast_quantile, data)
    metrics = su.get_metrics(
        fc, exclude=["interval_coverage_50", "interval_coverage_90"]
    )
    out = su.score(fc, metrics=metrics)
    for m in ("dispersion", "underprediction", "overprediction", "bias", "ae_median"):
        assert m in out.columns


def test_ae_median_matches_manual(scores_quantile):
    summ = su.summarise_scores(scores_quantile, by="model")
    df = pd.DataFrame(su.example_quantile()).dropna()
    med = df[df.quantile_level == 0.5]
    manual = (med.observed - med.predicted).abs().groupby(med.model).mean()
    np.testing.assert_allclose(np.sort(summ.ae_median), np.sort(manual.to_numpy()))


def test_wis_same_with_other_metrics_omitted(fc_quantile, scores_quantile):
    s = su.score(fc_quantile, metrics={"wis": su.wis})
    assert s.wis.sum() == pytest.approx(scores_quantile.wis.sum(), rel=1e-12)


def test_score_only_median(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    only = su.new_forecast(df[df.quantile_level == 0.5], "quantile")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = su.score(
            only,
            metrics=su.get_metrics(
                fc_quantile, exclude=["interval_coverage_50", "interval_coverage_90"]
            ),
        )
    expected = (only.dropna().observed - only.dropna().predicted).abs().to_numpy()
    np.testing.assert_allclose(s.wis.to_numpy(), expected)


def test_score_one_interval_messages(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    one = su.as_forecast_quantile(df[df.quantile_level.isin([0.25, 0.75])])
    with pytest.warns(ScoringutilsMessage, match="Median not available"):
        su.score(
            one,
            metrics=su.get_metrics(
                fc_quantile, exclude=["interval_coverage_90", "ae_median"]
            ),
        )


def test_score_asymmetric_intervals(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    asymm = su.new_forecast(df[df.quantile_level <= 0.6], "quantile")
    metrics = su.get_metrics(
        fc_quantile, exclude=["overprediction", "underprediction", "dispersion"]
    )
    metrics["wis"] = partial(su.wis, na_rm=True)
    with pytest.warns(UserWarning) as record:
        scores_a = su.summarise_scores(su.score(asymm, metrics=metrics), by="model")
    messages = [str(r.message) for r in record]
    assert any("Computation for `interval_coverage_50` failed" in m for m in messages)
    assert any("Computation for `interval_coverage_90` failed" in m for m in messages)

    with pytest.warns(
        UserWarning, match="Not all quantile levels specified form symmetric"
    ):
        su.score(asymm, metrics={"wis": su.wis})

    inner = su.new_forecast(
        df[df.quantile_level.round(10).isin([0.4, 0.45, 0.5, 0.55, 0.6])], "quantile"
    )
    scores_b = su.summarise_scores(su.score(inner, metrics={"wis": su.wis}), by="model")
    np.testing.assert_allclose(scores_a.wis, scores_b.wis)


def test_score_median_missing_for_one_model(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    no_med = df[(df.model == "epiforecasts-EpiNow2") & (df.quantile_level != 0.5)]
    test = quiet(
        su.as_forecast_quantile,
        pd.concat([df[df.model != "epiforecasts-EpiNow2"], no_med]),
    )
    with pytest.warns(UserWarning) as record:
        su.score(test)
    messages = [str(r.message) for r in record]
    assert any("Computation for `ae_median` failed" in m for m in messages)
    assert any(
        "interpolating median from the two innermost quantiles" in m for m in messages
    )


def test_coverage_columns_are_boolean(scores_quantile):
    assert scores_quantile["interval_coverage_50"].dtype == bool


# ---------------------------------------------------------------------------
# summarise_scores()
# ---------------------------------------------------------------------------


def test_summarise_returns_scores(scores_quantile):
    assert isinstance(su.summarise_scores(scores_quantile), su.Scores)


def test_summarise_by_forecast_unit_keeps_shape(scores_quantile):
    fu = su.get_forecast_unit(scores_quantile)
    s = su.summarise_scores(scores_quantile, by=fu)
    assert s.shape == scores_quantile.shape
    np.testing.assert_allclose(s.wis, scores_quantile.wis)


def test_summarise_bad_by(scores_quantile):
    for by in ("not_present", "sample_id", None):
        with pytest.raises(
            ValueError, match="Assertion on 'by' failed: Must be a subset of"
        ):
            su.summarise_scores(scores_quantile, by=by)


def test_summarise_by_metric_column_errors(scores_quantile):
    with pytest.raises(ValueError, match="metric"):
        su.summarise_scores(scores_quantile, by=["model", "wis"])
    with pytest.raises(ValueError, match='"wis" and "bias"'):
        su.summarise_scores(scores_quantile, by=["model", "wis", "bias"])


def test_summarise_metrics_attribute(scores_quantile, scores_sample_continuous):
    with pytest.raises(ValueError, match="Input needs an attribute `metrics`"):
        su.summarise_scores(pd.DataFrame(scores_quantile), by="model")
    renamed = scores_sample_continuous.rename(columns={"crps": "crp2"})
    with pytest.warns(UserWarning, match="previously computed, but are no longer"):
        su.summarise_scores(renamed, by="model")


def test_summarise_plain_dataframe_with_attrs(scores_quantile):
    df = pd.DataFrame(scores_quantile)
    df.attrs["metrics"] = scores_quantile.metrics
    su.summarise_scores(df, by="model")


def test_summarise_no_score_columns(scores_quantile):
    empty = su.new_scores(
        pd.DataFrame(scores_quantile).drop(columns=scores_quantile.metrics), metrics=[]
    )
    with pytest.raises(ValueError, match="No score columns to summarise"):
        su.summarise_scores(empty, by="model")


def test_summarise_all_metrics_failed(fc_quantile):
    with pytest.warns(UserWarning, match="interval coverage"):
        sc = su.score(
            fc_quantile,
            metrics={
                "interval_coverage_55": partial(su.interval_coverage, interval_range=55)
            },
        )
    assert not set(sc.columns) & set(sc.metrics)
    with pytest.raises(ValueError, match="No score columns to summarise"):
        su.summarise_scores(sc, by="model")


def test_summarise_does_not_partial_match(scores_quantile):
    test = scores_quantile.assign(wis_extra=0.0)
    assert "wis_extra" not in su.summarise_scores(test, by="model").columns


def test_summarise_mean_propagates_nan_like_r(scores_quantile):
    s = scores_quantile.copy()
    s.loc[s.index[0], "wis"] = np.nan
    out = su.summarise_scores(s, by="model")
    assert out.wis.isna().sum() == 1
