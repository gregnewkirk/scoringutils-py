"""Ported from scoringutils tests/testthat test-class-forecast.R,
test-class-forecast-quantile.R, test-class-forecast-sample.R,
test-forecast-unit.R and test-get-duplicate-forecasts.R (2.3.0).

Binary and point cases are in test_forecast_point_binary.py. Not ported:
nominal / ordinal / multivariate cases (not ported yet), R print snapshots
and data.table ``:=`` autoprint behaviour (no Python equivalent), and
``[.forecast`` re-validation (see docs/DESIGN.md section 4).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import scoringutils_py as su
from scoringutils_py import ScoringutilsMessage

FU = [
    "location",
    "target_end_date",
    "target_type",
    "location_name",
    "forecast_date",
    "model",
    "horizon",
]


@pytest.fixture
def quantile_df():
    return pd.DataFrame(su.example_quantile()).dropna().reset_index(drop=True)


@pytest.fixture
def sample_df():
    return pd.DataFrame(su.example_sample_continuous()).dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# as_forecast_generic (renaming)
# ---------------------------------------------------------------------------


def test_rename_onto_existing_column_errors():
    df = pd.DataFrame(
        {
            "model": "m",
            "target": "t",
            "observed": 5.0,
            "predicted": [1.0, 5, 9],
            "quantile_level": [0.1, 0.5, 0.9],
            "q": [0.1, 0.5, 0.9],
        }
    )
    with pytest.raises(
        ValueError, match='rename column "q" to "quantile_level".*already exists'
    ):
        su.as_forecast_quantile(df, quantile_level="q")
    multi = pd.DataFrame(
        {
            "model": "m",
            "id": [1, 2],
            "observed": 1.0,
            "obs": 2.0,
            "predicted": 3.0,
            "pred": 4.0,
            "sample_id": [1, 2],
        }
    )
    with pytest.raises(
        ValueError,
        match='rename columns "obs" and "pred" to "observed" and "predicted"'
        ".*already exist in the data",
    ):
        su.as_forecast_sample(multi, observed="obs", predicted="pred")


def test_identity_renames_allowed(quantile_df):
    su.as_forecast_quantile(
        quantile_df,
        observed="observed",
        predicted="predicted",
        quantile_level="quantile_level",
    )


# ---------------------------------------------------------------------------
# is_forecast / assert_forecast / new_forecast
# ---------------------------------------------------------------------------


def test_is_forecast(quantile_df):
    assert su.is_forecast(su.example_quantile())
    assert su.is_forecast(su.example_sample_continuous())
    assert not su.is_forecast(np.arange(10))
    assert not su.is_forecast(quantile_df)


def test_is_forecast_quantile_and_sample():
    assert su.is_forecast_quantile(su.example_quantile())
    assert not su.is_forecast_quantile(su.example_sample_continuous())
    assert su.is_forecast_sample(su.example_sample_continuous())
    assert not su.is_forecast_sample(su.example_quantile())
    assert not su.is_forecast_sample(np.arange(10))


def test_assert_forecast_rejects_plain_data():
    with pytest.raises(
        TypeError, match="The input needs to be a valid forecast object"
    ):
        su.assert_forecast(pd.DataFrame({"x": range(10)}))


def test_duplicate_column_names_error():
    df = pd.DataFrame(
        [["m", 0.25, 1.0, 0.9, 1.0]],
        columns=["model", "quantile_level", "observed", "predicted", "predicted"],
    )
    with pytest.raises(ValueError, match="duplicate"):
        su.assert_forecast(su.ForecastQuantile(df))


def test_new_forecast(quantile_df):
    fc = su.new_forecast(quantile_df, "forecast_quantile")
    assert isinstance(fc, su.ForecastQuantile)
    assert su.get_forecast_type(fc) == "quantile"


def test_pandas_operations_return_plain_frames(quantile_df):
    # documented difference from R: no silent re-validation after subsetting
    fc = su.as_forecast_quantile(quantile_df)
    assert type(fc[fc.location == "DE"]) is pd.DataFrame
    assert type(fc.copy()) is pd.DataFrame


def test_repr_shows_type_and_unit(quantile_df):
    text = repr(su.as_forecast_quantile(quantile_df))
    assert "Forecast type: quantile" in text
    assert "Forecast unit:" in text


# ---------------------------------------------------------------------------
# check_number_per_forecast / check_observed_constant / NA handling
# ---------------------------------------------------------------------------


def test_different_numbers_of_quantiles_warn(quantile_df):
    with pytest.warns(
        UserWarning, match="Some forecasts have different numbers of rows"
    ):
        su.as_forecast_quantile(quantile_df.drop(index=999))


def test_observed_constant_checks():
    ok = pd.DataFrame(
        {
            "model": "m1",
            "target": "t1",
            "quantile_level": [0.25, 0.5, 0.75],
            "predicted": [1, 2, 3],
            "observed": 10,
        }
    )
    su.as_forecast_quantile(ok)
    bad = ok.assign(observed=[10, 10, 20])
    with pytest.raises(ValueError, match="different observed values"):
        su.as_forecast_quantile(bad)
    samples = pd.DataFrame(
        {
            "model": "m1",
            "target": "t1",
            "sample_id": [1, 2, 3, 4],
            "predicted": [1.0, 2, 3, 4],
            "observed": [5, 5, 5, 7],
        }
    )
    with pytest.raises(ValueError, match="different observed values"):
        su.as_forecast_sample(samples)


def test_observed_constant_ignores_na():
    dt = pd.DataFrame(
        {
            "model": "m1",
            "target": "t1",
            "sample_id": [1, 2, 3, 4],
            "predicted": [1.0, 2, 3, 4],
            "observed": [5, 5, 5, np.nan],
        }
    )
    with pytest.warns(ScoringutilsMessage, match="NA values"):
        fc = su.as_forecast_sample(dt)
    assert isinstance(fc, su.ForecastSample)


def test_removing_na_rows():
    assert len(pd.DataFrame(su.example_quantile()).dropna()) == 20401


def test_only_na_predictions_error():
    df = pd.DataFrame(su.example_quantile()).assign(predicted=np.nan)
    with pytest.raises(ValueError, match="no forecasts are left"):
        su.as_forecast_quantile(df)


# ---------------------------------------------------------------------------
# as_forecast_quantile
# ---------------------------------------------------------------------------


def test_as_forecast_quantile_works(quantile_df):
    fc = su.as_forecast_quantile(quantile_df)
    assert isinstance(fc, su.ForecastQuantile)
    with pytest.raises(TypeError, match="Must be of type 'character'"):
        su.as_forecast_quantile(quantile_df, observed=3)
    with pytest.raises(ValueError, match="Must have length 1"):
        su.as_forecast_quantile(quantile_df, quantile_level=["1", "2"])
    with pytest.raises(ValueError, match="Must be a subset of"):
        su.as_forecast_quantile(quantile_df, observed="missing")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.as_forecast_quantile(
            quantile_df,
            observed="observed",
            predicted="predicted",
            forecast_unit=[
                "location",
                "model",
                "target_type",
                "target_end_date",
                "horizon",
            ],
            quantile_level="quantile_level",
        )


def test_as_forecast_quantile_raw_example_data():
    with pytest.warns(ScoringutilsMessage):
        assert isinstance(
            su.as_forecast_quantile(su.example_quantile()), su.ForecastQuantile
        )


def test_quantile_and_sample_id_together_error(quantile_df):
    with pytest.raises(
        ValueError, match="Found columns `quantile_level` and `sample_id`"
    ):
        su.as_forecast_quantile(quantile_df.assign(sample_id=1))


def test_duplicate_forecasts_error(quantile_df):
    df = pd.concat([quantile_df, quantile_df.iloc[999:1010]])
    with pytest.raises(ValueError, match="more than one forecast for the same target"):
        su.as_forecast_quantile(df)


@pytest.mark.parametrize(
    "drop", [["predicted"], ["observed"], ["observed", "predicted"]]
)
def test_missing_required_columns(quantile_df, drop):
    with pytest.raises(ValueError, match="Must be a subset of"):
        su.as_forecast_quantile(quantile_df.drop(columns=drop))


def test_samples_to_quantiles():
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(1, 11), unit="D")
    samples = pd.DataFrame(
        {
            "date": np.tile(dates, 5),
            "model": "model1",
            "observed": np.tile(np.arange(1, 11), 5),
            "predicted": np.concatenate(
                [
                    np.zeros(10),
                    np.arange(2, 12),
                    np.arange(3, 13),
                    np.arange(4, 14),
                    np.full(10, 100),
                ]
            ),
            "sample_id": np.repeat(np.arange(1, 6), 10),
        }
    )
    samples = su.as_forecast_sample(samples)
    expected = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "model": "model1",
            "observed": np.repeat(np.arange(1, 11), 2),
            "quantile_level": np.tile([0.25, 0.75], 10),
            "predicted": np.repeat(np.arange(2, 12), 2) + np.tile([0, 2], 10),
        }
    )
    q = su.as_forecast_quantile(samples, probs=[0.25, 0.75])
    pd.testing.assert_frame_equal(pd.DataFrame(q), expected, check_dtype=False)

    # a column called "type" must not clash with the `type` argument
    q3 = su.as_forecast_quantile(
        su.as_forecast_sample(pd.DataFrame(samples).assign(type="test")),
        probs=[0.25, 0.75],
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(q3).drop(columns="type"), pd.DataFrame(q)
    )


@pytest.fixture
def hundred_samples():
    return su.as_forecast_sample(
        pd.DataFrame(
            {
                "model": "model1",
                "observed": 50,
                "predicted": np.arange(1, 101),
                "sample_id": np.arange(1, 101),
            }
        )
    )


def test_samples_to_single_asymmetric_prob(hundred_samples):
    q = su.as_forecast_quantile(hundred_samples, probs=0.4)
    assert len(q) == 1
    assert q.quantile_level.iloc[0] == 0.4
    assert q.predicted.iloc[0] == np.quantile(np.arange(1, 101), 0.4)


def test_samples_to_multiple_asymmetric_probs(hundred_samples):
    q = su.as_forecast_quantile(hundred_samples, probs=[0.1, 0.2])
    np.testing.assert_array_equal(q.quantile_level, [0.1, 0.2])
    np.testing.assert_array_equal(
        q.predicted, np.quantile(np.arange(1, 101), [0.1, 0.2])
    )


def test_samples_to_quantiles_deduplicates(hundred_samples):
    q = su.as_forecast_quantile(hundred_samples, probs=[0.5, 0.5])
    assert len(q) == 1


@pytest.mark.parametrize("probs", [[0.5, 1.5], [0.5, np.nan]])
def test_samples_to_quantiles_bad_probs(hundred_samples, probs):
    with pytest.raises(ValueError, match="Assertion on 'probs' failed"):
        su.as_forecast_quantile(hundred_samples, probs=probs)


def test_as_forecast_quantile_does_not_modify_input():
    df = pd.DataFrame(
        {"my_quantile": [0.25, 0.5], "forecast_value": [1, 2], "observed_value": [5, 5]}
    )
    before = list(df.columns)
    su.as_forecast_quantile(
        df,
        predicted="forecast_value",
        observed="observed_value",
        quantile_level="my_quantile",
    )
    assert list(df.columns) == before


def test_quantile_rounding_issue():
    df = pd.DataFrame(
        {
            "quantile_level": [0.25, 0.75 + 1e-13, 0.25, 0.75],
            "predicted": [1, 2, 3, 4],
            "observed": 5,
            "location": [1, 1, 2, 2],
        }
    )
    with pytest.warns(UserWarning, match="rounding issue"):
        fc = su.as_forecast_quantile(df)
    with pytest.warns(UserWarning, match="digits = 9"):
        su.as_forecast_quantile(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.score(fc, metrics={"wis": su.wis})


def test_non_numeric_observed_and_predicted():
    df = pd.DataFrame(
        {
            "observed": ["1", "1"],
            "predicted": ["0.5", "1.5"],
            "quantile_level": [0.25, 0.75],
            "model": "m1",
        }
    )
    with pytest.raises(TypeError, match="Must be of type 'numeric', not 'character'"):
        su.as_forecast_quantile(df)
    with pytest.raises(TypeError, match="Must be of type 'numeric', not 'character'"):
        su.as_forecast_quantile(df.assign(observed=[1, 1]))
    samples = pd.DataFrame(
        {
            "observed": ["5", "5", "5"],
            "predicted": ["4", "5", "6"],
            "sample_id": [1, 2, 3],
            "model": "m1",
        }
    )
    with pytest.raises(TypeError, match="Must be of type 'numeric', not 'character'"):
        su.as_forecast_sample(samples)


# ---------------------------------------------------------------------------
# as_forecast_sample
# ---------------------------------------------------------------------------


def test_as_forecast_sample_renames(sample_df):
    df = sample_df.rename(
        columns={"observed": "obs", "predicted": "pred", "sample_id": "sample"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.as_forecast_sample(
            df,
            observed="obs",
            predicted="pred",
            sample_id="sample",
            forecast_unit=[
                "location",
                "model",
                "target_type",
                "target_end_date",
                "horizon",
            ],
        )


def test_as_forecast_sample_twice_is_identical(sample_df):
    once = su.as_forecast_sample(sample_df)
    twice = su.as_forecast_sample(once)
    pd.testing.assert_frame_equal(pd.DataFrame(once), pd.DataFrame(twice))


def test_get_metrics_types(fc_quantile, fc_sample_continuous):
    assert isinstance(su.get_metrics(fc_quantile), dict)
    assert isinstance(su.get_metrics(fc_sample_continuous), dict)


# ---------------------------------------------------------------------------
# forecast unit
# ---------------------------------------------------------------------------


def test_get_forecast_unit(scores_quantile):
    assert su.get_forecast_unit(su.example_quantile()) == FU
    assert su.get_forecast_unit(scores_quantile) == FU
    assert su.get_forecast_unit(pd.DataFrame(su.example_quantile())) == FU


def test_set_forecast_unit_gives_same_scores(scores_quantile):
    ex2 = su.set_forecast_unit(
        su.example_quantile(),
        ["location", "target_end_date", "target_type", "horizon", "model"],
    )
    scores2 = su.score(su.as_forecast_quantile(ex2.dropna()))
    keys = ["location", "target_end_date", "target_type", "horizon", "model"]
    a = pd.DataFrame(scores_quantile).sort_values(keys)["wis"].to_numpy()
    b = pd.DataFrame(scores2).sort_values(keys)["wis"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_set_forecast_unit_plain_frame():
    df = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]})
    assert list(su.set_forecast_unit(df, ["a", "b"]).columns) == ["a", "b"]
    assert type(su.set_forecast_unit(df, ["a", "b"])) is pd.DataFrame


def test_set_forecast_unit_missing_column():
    with pytest.raises(
        ValueError, match="Assertion on 'forecast_unit' failed: Must be a subset of"
    ):
        su.set_forecast_unit(su.example_quantile(), ["location", "test1"])


def test_get_and_set_forecast_unit_together():
    fu = ["location", "target_end_date", "target_type", "horizon", "model"]
    assert su.get_forecast_unit(su.set_forecast_unit(su.example_quantile(), fu)) == fu


# ---------------------------------------------------------------------------
# get_duplicate_forecasts
# ---------------------------------------------------------------------------


def _dup(fc, start, stop):
    df = pd.DataFrame(fc)
    return type(fc)(pd.concat([df, df.iloc[start:stop]], ignore_index=True))


def test_duplicates_quantile():
    fc = su.example_quantile()
    assert len(su.get_duplicate_forecasts(fc)) == 0
    assert len(su.get_duplicate_forecasts(_dup(fc, 999, 1010))) == 22


def test_duplicates_sample():
    fc = su.example_sample_continuous()
    assert len(su.get_duplicate_forecasts(fc)) == 0
    assert len(su.get_duplicate_forecasts(_dup(fc, 1039, 1050))) == 22


def test_duplicates_with_type_on_raw_data():
    df = pd.DataFrame(su.example_quantile())
    bad = pd.concat([df, df.iloc[999:1010]])
    assert len(su.get_duplicate_forecasts(bad, type="quantile")) == 22
    assert len(su.get_duplicate_forecasts(df, type="quantile")) == 0


def test_duplicates_warn_without_type():
    with pytest.warns(FutureWarning, match="without specifying `type` was deprecated"):
        su.get_duplicate_forecasts(pd.DataFrame(su.example_quantile()))
    df = pd.DataFrame(su.example_quantile())
    with pytest.warns(FutureWarning):
        assert len(su.get_duplicate_forecasts(pd.concat([df, df.iloc[999:1010]]))) == 22


def test_duplicates_ignore_type_on_forecast():
    fc = su.example_quantile()
    assert len(su.get_duplicate_forecasts(fc, type="sample")) == len(
        su.get_duplicate_forecasts(fc)
    )


def test_duplicates_respect_forecast_unit():
    fc = su.example_quantile()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        su.get_duplicate_forecasts(
            fc,
            forecast_unit=[
                "location",
                "target_end_date",
                "target_type",
                "location_name",
                "forecast_date",
                "model",
            ],
        )


def test_duplicates_counts():
    dup = su.get_duplicate_forecasts(_dup(su.example_quantile(), 100, 110), counts=True)
    assert len(dup) == 2
    assert set(dup.n_duplicates) == {10}


def test_get_forecast_type_ids():
    assert su.get_forecast_type_ids(su.example_quantile()) == ["quantile_level"]
    assert su.get_forecast_type_ids(su.example_sample_continuous()) == ["sample_id"]
    assert su.get_forecast_type_ids(pd.DataFrame({"x": [1]})) == []


def test_duplicate_message_mentions_type_and_unit():
    df = pd.DataFrame(su.example_quantile()).iloc[999:1010]
    with pytest.raises(ValueError) as err:
        su.as_forecast_quantile(pd.concat([df, df]))
    msg = str(err.value)
    assert 'type = "quantile"' in msg
    assert "forecast_unit" in msg
    assert "get_duplicate_forecasts" in msg
