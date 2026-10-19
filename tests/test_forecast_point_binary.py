"""Ported from scoringutils tests/testthat test-class-forecast-point.R and
test-class-forecast-binary.R (2.3.0), plus the point and binary cases in
test-class-forecast.R, test-score.R, test-get-forecast-type.R,
test-get-duplicate-forecasts.R, test-forecast-unit.R, test-class-scores.R
and test-pairwise_comparison.R.

Not ported: ``is_forecast_*(example_nominal)`` (nominal forecasts are not
ported yet), the ``[.forecast`` warning when a column is changed after
validation (pandas operations return plain DataFrames, see docs/DESIGN.md
section 4), the check that ``set_forecast_unit()`` keeps the forecast class
(it returns a plain DataFrame here), and the check that the internal
``test_columns_not_present()`` no longer exists (no Python equivalent).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import scoringutils_py as su

FU_BINARY = ["location", "target_end_date", "target_type", "horizon", "model"]


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture
def point_df():
    return pd.DataFrame(su.example_point()).dropna().reset_index(drop=True)


@pytest.fixture
def binary_df():
    return pd.DataFrame(su.example_binary()).dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# as_forecast_point()
# ---------------------------------------------------------------------------


def test_as_forecast_point_from_quantile_has_no_conditions(fc_quantile):
    q = quiet(su.as_forecast_quantile, pd.DataFrame(fc_quantile).dropna())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        point = su.as_forecast_point(q)
    assert su.is_forecast_point(point)
    assert "quantile_level" not in point.columns
    assert len(point) == (q["quantile_level"] == 0.5).sum()


def test_as_forecast_point_from_quantile_needs_median(fc_quantile):
    df = pd.DataFrame(fc_quantile).dropna()
    q = quiet(su.as_forecast_quantile, df[df.quantile_level != 0.5])
    with pytest.raises(ValueError, match=r"Assertion on '0.5' failed"):
        su.as_forecast_point(q)


def test_as_forecast_point_from_quantile_rejects_renames(fc_quantile):
    q = quiet(su.as_forecast_quantile, pd.DataFrame(fc_quantile).dropna())
    with pytest.raises(TypeError, match="do not apply"):
        su.as_forecast_point(q, predicted="value")


def test_as_forecast_point_renames_columns(point_df):
    df = point_df.rename(columns={"observed": "obs", "predicted": "pred"})
    point = su.as_forecast_point(df, observed="obs", predicted="pred")
    assert {"observed", "predicted"} <= set(point.columns)


# ---------------------------------------------------------------------------
# is_forecast_point() / is_forecast_binary() / is_forecast()
# ---------------------------------------------------------------------------


def test_is_forecast_point():
    assert su.is_forecast_point(su.example_point())
    assert not su.is_forecast_point(su.example_binary())
    assert not su.is_forecast_point(su.example_quantile())
    assert not su.is_forecast_point(su.example_sample_continuous())


def test_is_forecast_binary():
    assert su.is_forecast_binary(su.example_binary())
    assert not su.is_forecast_binary(su.example_point())
    assert not su.is_forecast_binary(su.example_quantile())
    assert not su.is_forecast_binary(su.example_sample_continuous())


def test_is_forecast_point_and_binary():
    assert su.is_forecast(su.example_binary())
    assert su.is_forecast(su.example_point())
    assert not su.is_forecast(pd.DataFrame(su.example_point()))


def test_get_forecast_type():
    assert su.get_forecast_type(su.example_binary()) == "binary"
    assert su.get_forecast_type(su.example_point()) == "point"


# ---------------------------------------------------------------------------
# assert_forecast() for point forecasts
# ---------------------------------------------------------------------------


def test_assert_forecast_point_character_predicted(point_df):
    point = su.as_forecast_point(point_df)
    point["predicted"] = point["predicted"].astype(str)
    with pytest.raises(
        TypeError,
        match="Input looks like a point forecast, but found the following issue",
    ):
        su.assert_forecast(point)


def test_assert_forecast_point_categorical_observed(point_df):
    point_df["observed"] = pd.Categorical(point_df["observed"])
    with pytest.raises(TypeError, match="Input looks like a point forecast"):
        su.as_forecast_point(point_df)


def test_assert_forecast_point_wrong_type(point_df):
    point = su.new_forecast(point_df, "forecast_point")
    with pytest.raises(
        ValueError, match="Forecast type determined by scoringutils based on input:"
    ):
        su.assert_forecast(point, forecast_type="quantile")


# ---------------------------------------------------------------------------
# score() for point forecasts
# ---------------------------------------------------------------------------


def test_score_point_output(scores_point, fc_point):
    assert list(scores_point.columns) == fc_point.forecast_unit + [
        "ae_point",
        "se_point",
        "ape",
    ]
    summary = su.summarise_scores(scores_point, by=["model", "target_type"])
    assert len(summary) > 1
    assert list(summary.columns) == ["model", "target_type"] + list(
        su.get_metrics(fc_point)
    )
    assert isinstance(summary, su.Scores)


def test_score_point_renamed_metric(fc_point):
    metrics = su.get_metrics(fc_point)
    metrics = {
        ("just_testing" if k == "ae_point" else k): v for k, v in metrics.items()
    }
    scores = su.score(fc_point, metrics=metrics)
    summary = su.summarise_scores(scores, by="model")
    assert list(summary.columns) == ["model", "just_testing", "se_point", "ape"]


def test_score_point_errors_with_only_na():
    only_nas = pd.DataFrame(su.example_point())
    only_nas["predicted"] = np.nan
    with pytest.raises(ValueError, match="no forecasts are left"):
        su.score(su.new_forecast(only_nas, "forecast_point"))


def test_get_metrics_point(scores_point):
    metrics = su.get_metrics(su.example_point())
    assert isinstance(metrics, dict)
    assert list(metrics) == ["ae_point", "se_point", "ape"]
    assert all(callable(f) for f in metrics.values())
    assert su.get_metrics(scores_point) == ["ae_point", "se_point", "ape"]
    assert list(su.get_metrics(su.example_point(), select="ape")) == ["ape"]


def test_point_metrics_match_metrics_package_formulas():
    # R checks against Metrics::ae(), Metrics::se() and Metrics::ape()
    observed = np.array([1, -15, 22, 0, 5.5])
    predicted = np.array([5, 6, 7, 0, 5.5])
    metrics = su.get_metrics(su.example_point())
    np.testing.assert_array_equal(
        metrics["ae_point"](observed, predicted), np.abs(observed - predicted)
    )
    np.testing.assert_array_equal(
        metrics["se_point"](observed, predicted), (observed - predicted) ** 2
    )
    with np.errstate(invalid="ignore"):
        expected_ape = np.abs(observed - predicted) / np.abs(observed)
    np.testing.assert_array_equal(metrics["ape"](observed, predicted), expected_ape)


def test_score_point_gneiting_example():
    # from the get_metrics.forecast_point() examples: reporting the mean is
    # rewarded by the squared error
    rng = np.random.default_rng(123)
    n = 500
    observed = rng.normal(5, 4, n) ** 2
    mu = observed.mean()
    df = pd.DataFrame(
        {
            "model": np.repeat(["perfect", "bad"], n),
            "predicted": np.concatenate([np.full(n, mu), mu - rng.normal(10, 2, n)]),
            "observed": np.tile(observed, 2),
            "id": np.tile(np.arange(n), 2),
        }
    )
    summary = su.summarise_scores(su.score(su.as_forecast_point(df)))
    se = summary.set_index("model")["se_point"]
    assert se["perfect"] < se["bad"]


# ---------------------------------------------------------------------------
# as_forecast_binary() and assert_forecast() for binary forecasts
# ---------------------------------------------------------------------------


def test_as_forecast_binary_output_is_accepted_by_score(binary_df):
    check = quiet(su.as_forecast_binary, su.example_binary())
    scored_na = su.score(check)
    scored = su.score(su.as_forecast_binary(binary_df))
    pd.testing.assert_frame_equal(pd.DataFrame(scored_na), pd.DataFrame(scored))


def test_as_forecast_binary_rejects_sample_id(binary_df):
    binary_df["sample_id"] = np.arange(len(binary_df))
    with pytest.raises(
        ValueError,
        match="Input looks like a binary forecast, but an additional column "
        "called `sample_id` or `quantile` was found.",
    ):
        su.as_forecast_binary(binary_df)


def test_as_forecast_binary_rejects_quantile_level(binary_df):
    binary_df["quantile_level"] = 0.5
    with pytest.raises(ValueError, match="but an additional column"):
        su.as_forecast_binary(binary_df)


def test_as_forecast_binary_rejects_probabilities_above_one(binary_df):
    binary_df["predicted"] = binary_df["predicted"] + 1
    with pytest.raises(
        ValueError,
        match="Input looks like a binary forecast, but found the following issue",
    ):
        su.as_forecast_binary(binary_df)


def test_as_forecast_binary_rejects_non_categorical_observed(binary_df):
    binary_df["observed"] = binary_df["observed"].astype(int)
    with pytest.raises(TypeError, match="Must be of type 'factor', not 'integer'"):
        su.as_forecast_binary(binary_df)


def test_assert_forecast_binary_accepts_example():
    assert su.assert_forecast(su.example_binary(), verbose=False) is None


def test_as_forecast_binary_rename_onto_existing_column_errors():
    df = pd.DataFrame(
        {
            "model": "m",
            "id": [1, 2],
            "observed": pd.Categorical([0, 1]),
            "predicted": [0.9, 0.9],
            "prob": [0.3, 0.7],
        }
    )
    with pytest.raises(
        ValueError, match='rename column "prob" to "predicted".*already exists'
    ):
        su.as_forecast_binary(df, predicted="prob")


def test_as_forecast_binary_rename_onto_existing_columns_plural():
    df = pd.DataFrame(
        {"model": "m", "id": [1, 2], "observed": 1, "obs": 2, "predicted": 3, "prob": 4}
    )
    with pytest.raises(
        ValueError,
        match=r'rename\s+columns\s+"obs"\s+and\s+"prob"\s+to\s+"observed"'
        r'\s+and\s+"predicted".*already\s+exist\s+in\s+the\s+data',
    ):
        su.as_forecast_binary(df, observed="obs", predicted="prob")


def test_as_forecast_binary_identity_rename():
    df = pd.DataFrame(
        {
            "model": "m",
            "id": [1, 2],
            "observed": pd.Categorical([0, 1]),
            "predicted": [0.3, 0.7],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.as_forecast_binary(df, observed="observed", predicted="predicted")


def test_as_forecast_binary_warns_on_reversed_levels():
    df = pd.DataFrame(
        {
            "model": "m1",
            "id": [1, 2, 3, 4],
            "observed": pd.Categorical([0, 1, 1, 0], categories=[1, 0]),
            "predicted": [0.1, 0.9, 0.8, 0.2],
        }
    )
    with pytest.warns(UserWarning, match="counterintuitive"):
        su.as_forecast_binary(df)


def test_score_binary_standard_levels_do_not_warn(binary_df):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.score(su.as_forecast_binary(binary_df))


# ---------------------------------------------------------------------------
# score() for binary forecasts
# ---------------------------------------------------------------------------


def test_score_binary_output(scores_binary, fc_binary):
    assert list(scores_binary.columns) == fc_binary.forecast_unit + [
        "brier_score",
        "log_score",
    ]
    summary = su.summarise_scores(scores_binary, by=["model", "target_type"])
    assert len(summary) > 1
    assert list(summary.columns) == [
        "model",
        "target_type",
        "brier_score",
        "log_score",
    ]
    assert isinstance(summary, su.Scores)


def test_score_binary_errors_with_only_na():
    only_nas = pd.DataFrame(su.example_binary())
    only_nas["predicted"] = np.nan
    with pytest.raises(ValueError, match="no forecasts are left"):
        su.score(su.new_forecast(only_nas, "forecast_binary"))


def test_score_binary_same_as_brier_score(scores_binary):
    ex = su.example_binary()
    manual = su.brier_score(ex["observed"], ex["predicted"].to_numpy())
    np.testing.assert_allclose(
        scores_binary["brier_score"].to_numpy(), manual[~np.isnan(manual)]
    )


def test_score_binary_custom_function(binary_df):
    df = binary_df[
        (binary_df.model == "EuroCOVIDhub-ensemble")
        & (binary_df.target_type == "Cases")
        & (binary_df.location == "DE")
    ]
    fc = su.as_forecast_binary(df)
    result = su.score(fc, metrics={"identity": lambda x, y: y})
    np.testing.assert_array_equal(
        result["identity"].to_numpy(), fc["predicted"].to_numpy()
    )


def test_score_binary_warns_on_metric_clash():
    ex = pd.DataFrame(su.example_binary())
    ex["brier_score"] = 999
    with pytest.warns(UserWarning, match="brier_score"):
        su.score(quiet(su.as_forecast_binary, ex))


def test_score_binary_invalid_metrics(binary_df):
    def test_fun(x, y, **kwargs):
        return y

    fc = su.as_forecast_binary(binary_df)
    with pytest.warns(UserWarning) as record:
        su.score(fc, metrics={"test1": test_fun, "test": test_fun, "hi": "hi", "2": 3})
    messages = [str(r.message) for r in record]
    assert "`Metrics` element number 3 is not a valid function." in messages
    assert "`Metrics` element number 4 is not a valid function." in messages


def test_assert_scores_point_and_binary(scores_point, scores_binary):
    assert su.assert_scores(scores_point) is None
    assert su.assert_scores(scores_binary) is None
    assert "brier_score" in su.get_metrics(scores_binary)


def test_subsetting_point_scores_keeps_metrics(scores_point):
    assert scores_point.iloc[:10].metrics == scores_point.metrics
    assert scores_point[["model", "ae_point"]].metrics == scores_point.metrics


def test_get_metrics_binary():
    metrics = su.get_metrics(su.example_binary())
    assert isinstance(metrics, dict)
    assert list(metrics) == ["brier_score", "log_score"]
    assert list(su.get_metrics(su.example_binary(), select="brier_score")) == [
        "brier_score"
    ]
    assert list(su.get_metrics(su.example_binary(), exclude="log_score")) == [
        "brier_score"
    ]


# ---------------------------------------------------------------------------
# duplicates, forecast unit, relative skill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "rows"),
    [
        (lambda: su.as_forecast_binary(su.example_binary()), slice(999, 1010)),
        (lambda: su.as_forecast_point(su.example_point()), slice(1009, 1020)),
    ],
)
def test_get_duplicate_forecasts(make, rows):
    fc = quiet(make)
    assert len(su.get_duplicate_forecasts(fc)) == 0
    df = pd.DataFrame(fc)
    dup = su.new_forecast(pd.concat([df, df.iloc[rows]]), fc.forecast_type)
    assert len(su.get_duplicate_forecasts(dup)) == 22


def test_get_forecast_type_ids_point_binary(fc_point, fc_binary):
    assert su.get_forecast_type_ids(fc_binary) == []
    assert su.get_forecast_type_ids(fc_point) == []


def test_binary_duplicates_are_detected(fc_binary):
    df = pd.DataFrame(fc_binary)
    dup = pd.concat([df.iloc[999:1002], df.iloc[999:1002]])
    with pytest.raises(
        ValueError, match="There are instances with more than one forecast"
    ):
        su.as_forecast_binary(dup)


def test_set_and_get_forecast_unit_binary():
    ex = su.set_forecast_unit(su.example_binary(), FU_BINARY)
    assert su.get_forecast_unit(ex) == FU_BINARY


def test_add_relative_skill_binary_summarised(scores_binary):
    summary = su.summarise_scores(scores_binary, by=["model", "target_type"])
    rel = su.add_relative_skill(summary)
    assert "brier_score_relative_skill" in rel.columns


def test_add_relative_skill_point(scores_point):
    rel = su.add_relative_skill(scores_point, metric="se_point")
    rel = su.summarise_scores(rel, by="model")
    manual = su.get_pairwise_comparisons(scores_point, metric="se_point")
    expected = manual.drop_duplicates("model").set_index("model")
    np.testing.assert_allclose(
        rel.set_index("model")["se_point_relative_skill"],
        expected.loc[rel["model"], "se_point_relative_skill"],
    )
