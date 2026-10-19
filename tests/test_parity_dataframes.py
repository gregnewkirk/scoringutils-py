"""Data-frame level parity with R scoringutils 2.3.0.

Every expected value here was produced by ``r/generate_fixtures.R``.
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
from conftest import assert_frame_parity, read_fixture

import scoringutils_py as su

FU = [
    "location",
    "target_end_date",
    "target_type",
    "location_name",
    "forecast_date",
    "model",
    "horizon",
]
CRPS_SCALE = {"overprediction": "crps", "underprediction": "crps", "dispersion": "crps"}
FU_SAMPLE = [
    "location",
    "location_name",
    "target_end_date",
    "target_type",
    "forecast_date",
    "model",
    "horizon",
]


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def test_score_quantile(scores_quantile):
    assert_frame_parity(scores_quantile, read_fixture("scores_quantile.csv"), FU)


def test_score_sample_continuous(scores_sample_continuous):
    assert_frame_parity(
        scores_sample_continuous,
        read_fixture("scores_sample_continuous.csv"),
        FU_SAMPLE,
        scale=CRPS_SCALE,
    )


def test_score_sample_discrete(scores_sample_discrete):
    assert_frame_parity(
        scores_sample_discrete,
        read_fixture("scores_sample_discrete.csv"),
        FU_SAMPLE,
        scale=CRPS_SCALE,
    )


def test_score_quantile_without_median_for_one_model(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    no_median = df[(df.model == "epiforecasts-EpiNow2") & (df.quantile_level != 0.5)]
    df = pd.concat([df[df.model != "epiforecasts-EpiNow2"], no_median])
    fc = quiet(su.as_forecast_quantile, df)
    scores = quiet(su.score, fc)
    assert_frame_parity(
        scores, read_fixture("scores_quantile_no_median.csv"), FU, check_order=False
    )


def test_score_quantile_asymmetric_with_na_rm(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    asymm = su.new_forecast(df[df.quantile_level <= 0.6], "forecast_quantile")
    metrics = su.get_metrics(
        fc_quantile, exclude=["overprediction", "underprediction", "dispersion"]
    )
    metrics["wis"] = partial(su.wis, na_rm=True)
    scores = quiet(su.score, asymm, metrics=metrics)
    assert_frame_parity(scores, read_fixture("scores_quantile_asymmetric.csv"), FU)


def test_sample_discrete_to_quantile_and_score(fc_sample_discrete):
    probs = [0.01, 0.025, *np.round(np.arange(0.05, 0.951, 0.05), 10), 0.975, 0.99]
    sample = su.new_forecast(pd.DataFrame(fc_sample_discrete).dropna(), "sample")
    q = quiet(su.as_forecast_quantile, sample, probs=probs)
    scores = quiet(su.score, q)
    expected = read_fixture("scores_sample_discrete_to_quantile.csv")
    assert_frame_parity(scores, expected, FU_SAMPLE)
    assert not scores["interval_coverage_50"].isna().any()  # issue 557


@pytest.mark.parametrize("quantile_type", range(1, 10))
def test_sample_to_quantile_all_r_types(fc_sample_continuous, quantile_type):
    df = pd.DataFrame(fc_sample_continuous).dropna()
    df = df[df.model == "EuroCOVIDhub-ensemble"]
    q = su.as_forecast_quantile(
        su.new_forecast(df, "sample"),
        probs=[0.05, 0.25, 0.5, 0.75, 0.95],
        type=quantile_type,
    )
    expected = read_fixture("sample_to_quantile_types.csv")
    expected = expected[expected.quantile_type == quantile_type].drop(
        columns="quantile_type"
    )
    expected["observed"] = expected["observed"].astype(float)
    assert_frame_parity(q, expected, FU_SAMPLE + ["quantile_level"])


# ---------------------------------------------------------------------------
# summarise_scores()
# ---------------------------------------------------------------------------


def test_summarise_by_model(scores_quantile):
    assert_frame_parity(
        su.summarise_scores(scores_quantile, by="model"),
        read_fixture("summarise_quantile_model.csv"),
        ["model"],
    )


def test_summarise_by_model_and_target(scores_quantile):
    assert_frame_parity(
        su.summarise_scores(scores_quantile, by=["model", "target_type"]),
        read_fixture("summarise_quantile_model_target.csv"),
        ["model", "target_type"],
    )


def test_summarise_with_median(scores_quantile):
    assert_frame_parity(
        su.summarise_scores(scores_quantile, by="model", fun=np.median),
        read_fixture("summarise_quantile_model_median.csv"),
        ["model"],
    )


def test_summarise_sample_by_model_and_target(scores_sample_continuous):
    assert_frame_parity(
        su.summarise_scores(scores_sample_continuous, by=["model", "target_type"]),
        read_fixture("summarise_sample_continuous_model_target.csv"),
        ["model", "target_type"],
        scale=CRPS_SCALE,
    )


def test_summarise_with_extra_arguments(scores_sample_discrete):
    # R: summarise_scores(scores, by = "model", fun = sd)
    assert_frame_parity(
        su.summarise_scores(scores_sample_discrete, by="model", fun=np.std, ddof=1),
        read_fixture("summarise_sample_discrete_model_sd.csv"),
        ["model"],
    )


# ---------------------------------------------------------------------------
# pairwise comparisons and relative skill
# ---------------------------------------------------------------------------

PAIR_KEYS = ["model", "compare_against"]


def test_pairwise_default(scores_quantile):
    assert_frame_parity(
        su.get_pairwise_comparisons(scores_quantile),
        read_fixture("pairwise_quantile.csv"),
        PAIR_KEYS,
    )


def test_pairwise_by_target_with_baseline(scores_quantile):
    result = su.get_pairwise_comparisons(
        scores_quantile, by="target_type", baseline="EuroCOVIDhub-baseline"
    )
    assert_frame_parity(
        result,
        read_fixture("pairwise_quantile_by_target_baseline.csv"),
        ["target_type", *PAIR_KEYS],
    )


def test_pairwise_other_metric_without_test(scores_quantile):
    result = su.get_pairwise_comparisons(
        scores_quantile, metric="ae_median", test_type=None
    )
    assert_frame_parity(
        result, read_fixture("pairwise_quantile_ae_median_notest.csv"), PAIR_KEYS
    )


def test_pairwise_sample_by_target(scores_sample_continuous):
    result = su.get_pairwise_comparisons(scores_sample_continuous, by="target_type")
    assert_frame_parity(
        result,
        read_fixture("pairwise_sample_continuous_by_target.csv"),
        ["target_type", *PAIR_KEYS],
    )


def test_pairwise_partial_overlap(scores_quantile):
    s = scores_quantile
    s = s[~((s.model == "EuroCOVIDhub-ensemble") & (s.location == "DE"))]
    s = s[~((s.model == "EuroCOVIDhub-ensemble") & (s.target_type == "Cases"))]
    s = s[~((s.model == "epiforecasts-EpiNow2") & (s.target_type == "Deaths"))]
    assert_frame_parity(s, read_fixture("scores_quantile_partial_overlap.csv"), FU)
    result = su.get_pairwise_comparisons(s, metric="wis")
    assert result["mean_scores_ratio"].isna().any()
    assert_frame_parity(
        result, read_fixture("pairwise_quantile_partial_overlap.csv"), PAIR_KEYS
    )


def test_add_relative_skill(scores_quantile):
    assert_frame_parity(
        su.add_relative_skill(scores_quantile),
        read_fixture("relskill_quantile.csv"),
        FU,
    )


def test_add_relative_skill_by_target_with_baseline(scores_quantile):
    rel = su.add_relative_skill(
        scores_quantile, by="target_type", baseline="EuroCOVIDhub-baseline"
    )
    assert_frame_parity(
        rel, read_fixture("relskill_quantile_by_target_baseline.csv"), FU
    )
    assert rel.metrics[-2:] == ["wis_relative_skill", "wis_scaled_relative_skill"]
    assert_frame_parity(
        su.summarise_scores(rel, by=["model", "target_type"]),
        read_fixture("relskill_quantile_by_target_baseline_summary.csv"),
        ["model", "target_type"],
    )


def test_add_relative_skill_on_summarised_scores(scores_sample_discrete):
    summ = su.summarise_scores(scores_sample_discrete, by=["model", "target_type"])
    rel = su.add_relative_skill(summ, metric="crps", baseline="EuroCOVIDhub-baseline")
    assert_frame_parity(
        rel,
        read_fixture("relskill_sample_discrete_summarised.csv"),
        ["model", "target_type"],
    )


# ---------------------------------------------------------------------------
# get_coverage()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("by", "fixture"),
    [
        ("model", "coverage_model.csv"),
        (["model", "target_type"], "coverage_model_target.csv"),
        (FU, "coverage_forecast_unit.csv"),
    ],
)
def test_get_coverage(fc_quantile, by, fixture):
    by_list = [by] if isinstance(by, str) else by
    assert_frame_parity(
        su.get_coverage(fc_quantile, by=by),
        read_fixture(fixture),
        [*by_list, "quantile_level"],
    )


def test_get_coverage_asymmetric(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    fc = su.new_forecast(df[~df.quantile_level.isin([0.2, 0.3, 0.5])], "quantile")
    assert_frame_parity(
        su.get_coverage(fc),
        read_fixture("coverage_asymmetric.csv"),
        ["model", "quantile_level"],
    )


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------


def _with_duplicates():
    df = pd.DataFrame(su.example_quantile())
    return pd.concat([df, df.iloc[999:1010]], ignore_index=True)


def test_get_duplicate_forecasts():
    result = su.get_duplicate_forecasts(_with_duplicates(), type="quantile")
    expected = read_fixture("duplicates_quantile.csv")
    expected["predicted"] = expected["predicted"].astype(float)
    assert_frame_parity(result, expected)


def test_get_duplicate_forecasts_counts():
    result = su.get_duplicate_forecasts(
        _with_duplicates(), type="quantile", counts=True
    )
    assert_frame_parity(result, read_fixture("duplicates_quantile_counts.csv"))
