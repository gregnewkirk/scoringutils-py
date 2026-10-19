"""Ported from scoringutils tests/testthat/test-pairwise_comparison.R and
test-get-coverage.R (2.3.0). Plot tests are not ported."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import scoringutils_py as su
from scoringutils_py import ScoringutilsMessage
from scoringutils_py._rcompat import p_adjust_holm, wilcox_paired_pvalue


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# reference implementation by Johannes Bracher (ported from the R test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_scores():
    values = np.concatenate(
        [
            sorted([4, 1, 5, 5, 6]),
            sorted([12, 4, 5, 12, 53]),
            sorted([8, 6, 3, 1, 46]),
            sorted([6, 3, 5, 8, 5]),
            sorted([3, 1, 5, 7, 7]),
            sorted([3, 2, 6, 8, 5]),
        ]
    ).astype(float)
    truth = pd.DataFrame(
        {
            "location": np.repeat([f"location_{i}" for i in (1, 2, 3)], 10),
            "target_end_date": pd.to_datetime("2020-01-01")
            + pd.to_timedelta(np.tile(np.arange(1, 11), 3), unit="D"),
            "target_variable": "inc death",
            "observed": values,
        }
    )
    rng = np.random.default_rng(123)
    rows = []
    for sd in (1, 2, 3, 4):
        for _, t in truth.iterrows():
            draws = np.abs(rng.normal(t.observed, sd, size=100))
            for q, v in zip(
                [0.05, 0.25, 0.5, 0.75, 0.95],
                np.quantile(draws, [0.05, 0.25, 0.5, 0.75, 0.95]),
                strict=True,
            ):
                rows.append(
                    {
                        "model": f"m{sd}",
                        "location": t.location,
                        "target_end_date": t.target_end_date,
                        "target_variable": t.target_variable,
                        "quantile_level": q,
                        "predicted": v,
                        "observed": t.observed,
                    }
                )
    fc = su.as_forecast_quantile(pd.DataFrame(rows))
    return su.score(fc)


def _bracher_ratios(scores, subset=None):
    df = pd.DataFrame(scores)
    if subset is not None:
        df = df[subset(df)]
    models = [f"m{i}" for i in range(1, 5)]
    ratio = pd.DataFrame(np.nan, index=models, columns=models)
    for mx in models:
        for my in models:
            a = df[df.model == mx]
            b = df[df.model == my]
            sub = a.merge(b, on=["target_end_date", "location"])
            ratio.loc[mx, my] = sub.wis_x.sum() / sub.wis_y.sum()
    return ratio


def test_relative_skill_matches_bracher_reference(toy_scores):
    by = ["model", "location", "target_end_date", "target_variable"]
    summ = su.summarise_scores(toy_scores, by=by)
    without = quiet(su.add_relative_skill, summ)
    with_base = quiet(su.add_relative_skill, summ, baseline="m1")

    ratios = _bracher_ratios(summ)
    geo = np.exp(np.log(ratios).mean(axis=1))
    rel = without.groupby("model").wis_relative_skill.first()
    np.testing.assert_allclose(rel.loc[geo.index], geo.to_numpy())

    geo_nb = np.exp(np.log(ratios.drop(columns="m1")).mean(axis=1))
    scaled = geo_nb / geo_nb["m1"]
    rel_b = with_base.groupby("model").wis_scaled_relative_skill.first()
    np.testing.assert_allclose(rel_b.loc[scaled.index], scaled.to_numpy())

    # by location, compared with a subset computation
    ratios3 = _bracher_ratios(toy_scores, subset=lambda d: d.location == "location_3")
    geo3 = np.exp(np.log(ratios3.drop(columns="m1")).mean(axis=1))
    scaled3 = geo3 / geo3["m1"]
    rel3 = su.add_relative_skill(toy_scores, by="location", baseline="m1")
    rel3 = su.summarise_scores(rel3, by=["model", "location"])
    rel3 = (
        rel3[rel3.location == "location_3"].set_index("model").wis_scaled_relative_skill
    )
    np.testing.assert_allclose(rel3.loc[scaled3.index], scaled3.to_numpy())


# ---------------------------------------------------------------------------
# basic behaviour
# ---------------------------------------------------------------------------


def test_relative_skill_with_integer_samples(scores_sample_discrete):
    summ = su.summarise_scores(scores_sample_discrete, by=["model", "target_type"])
    assert "crps_relative_skill" in su.add_relative_skill(summ).columns


def test_pairwise_output_columns():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "model": np.repeat(["model1", "model2", "model3"], 10),
            "date": np.tile(np.repeat(np.arange(5), 2), 3),
            "location": np.tile([1, 2], 15),
            "wis": np.abs(rng.normal(size=30)),
            "ae_median": np.abs(rng.normal(size=30)),
        }
    )
    df.attrs["metrics"] = ["wis", "ae_median"]
    res = quiet(su.get_pairwise_comparisons, df, baseline="model1")
    for col in (
        "model",
        "compare_against",
        "mean_scores_ratio",
        "pval",
        "adj_pval",
        "wis_relative_skill",
        "wis_scaled_relative_skill",
    ):
        assert col in res.columns
    assert type(res) is pd.DataFrame


def test_pairwise_and_add_relative_skill_agree(scores_sample_continuous):
    pw = su.get_pairwise_comparisons(scores_sample_continuous, metric="crps")
    rel = su.summarise_scores(
        su.add_relative_skill(scores_sample_continuous), by="model"
    )
    np.testing.assert_allclose(
        np.sort(pw.crps_relative_skill.unique()), np.sort(rel.crps_relative_skill)
    )


def test_missing_baseline(scores_quantile):
    with pytest.raises(
        ValueError, match="Assertion on 'baseline' failed: Must be a subset of"
    ):
        su.get_pairwise_comparisons(scores_quantile, baseline="missing_model")


def test_add_relative_skill_input_checks(scores_sample_continuous):
    ev = scores_sample_continuous
    with pytest.raises(
        ValueError, match="Not all columns specified in `by` are present"
    ):
        su.add_relative_skill(ev, by="missing", metric="crps")
    with pytest.raises(ValueError, match=r"Must be disjunct from \{'model'\}"):
        su.add_relative_skill(ev, by=["model", "target_type"], metric="crps")
    with pytest.raises(ValueError, match="Assertion on 'baseline' failed"):
        su.add_relative_skill(ev, baseline="missing", metric="crps")

    few = ev[ev.model.isin(["EuroCOVIDhub-ensemble", "EuroCOVIDhub-baseline"])]
    su.add_relative_skill(few, metric="crps")
    su.add_relative_skill(few, baseline="EuroCOVIDhub-baseline", metric="crps")

    one = ev[ev.model == "EuroCOVIDhub-baseline"]
    with pytest.raises(ValueError, match="not enough comparators"):
        su.add_relative_skill(one, baseline="EuroCOVIDhub-baseline", metric="crps")

    with pytest.raises(ValueError):
        su.add_relative_skill(ev, metric="missing")
    with pytest.raises(
        ValueError, match="Assertion on 'metric' failed: Must be a subset of"
    ):
        quiet(su.add_relative_skill, ev.drop(columns="crps"))

    with pytest.raises(ValueError, match="Must be a subset of"):
        su.add_relative_skill(ev.drop(columns="model"), by="target_type", metric="crps")

    with pytest.raises(ValueError, match="needs an attribute `metrics`"):
        su.add_relative_skill(pd.DataFrame(ev), metric="crps")

    nas = ev.copy()
    nas.loc[nas.index[:10], "crps"] = np.nan
    with pytest.warns(UserWarning, match="Some values for the metric `crps` are NA"):
        su.add_relative_skill(nas, metric="crps")
    nas["crps"] = np.nan
    with pytest.raises(ValueError, match='After removing "NA" values for `crps`'):
        su.add_relative_skill(nas, metric="crps")

    diff = ev.copy()
    diff.loc[diff.index[:10], "crps"] = -diff.loc[diff.index[:10], "crps"]
    with pytest.raises(ValueError, match="must have the same sign"):
        su.add_relative_skill(diff, metric="crps")

    fu = su.get_forecast_unit(ev)
    with pytest.warns(ScoringutilsMessage, match="relative skill can only be computed"):
        su.add_relative_skill(ev, by=[c for c in fu if c != "model"], metric="crps")

    summ = su.summarise_scores(ev, by="model")
    with pytest.warns(
        UserWarning, match="`compare` is set to the unit of a single forecast"
    ):
        su.add_relative_skill(summ, metric="crps")


def test_one_group_errors(scores_sample_continuous):
    from scoringutils_py._pairwise import _pairwise_one_group

    ev = scores_sample_continuous
    with pytest.raises(ValueError, match="pairwise comparisons require a column"):
        _pairwise_one_group(ev.drop(columns="model"), "crps", None, "model", [])
    with pytest.raises(ValueError, match="not enough comparators"):
        _pairwise_one_group(ev[ev.model == "impossible"], "crps", None, "model", [])
    with pytest.raises(ValueError, match="not enough comparators"):
        _pairwise_one_group(
            ev[ev.model == "EuroCOVIDhub-ensemble"], "crps", None, "model", []
        )
    with pytest.raises(ValueError, match="Baseline comparator `missing` missing"):
        _pairwise_one_group(ev, "crps", "missing", "model", [])


def test_relative_skill_values(scores_quantile):
    rel = su.summarise_scores(su.add_relative_skill(scores_quantile), by="model")
    np.testing.assert_allclose(
        rel.wis_relative_skill, [1.6, 0.81, 0.75, 1.03], atol=0.01
    )
    rel = su.summarise_scores(
        su.add_relative_skill(scores_quantile, metric="ae_median"), by="model"
    )
    np.testing.assert_allclose(
        rel.ae_median_relative_skill, [1.6, 0.78, 0.77, 1.04], atol=0.01
    )


def test_permutation_test_runs_and_is_reproducible(scores_quantile):
    a = su.get_pairwise_comparisons(
        scores_quantile,
        test_type="permutation",
        one_sided=True,
        n_permutations=50,
        rng=1,
    )
    b = su.get_pairwise_comparisons(
        scores_quantile,
        test_type="permutation",
        one_sided=True,
        n_permutations=50,
        rng=1,
    )
    pd.testing.assert_frame_equal(a, b)
    off = a[a.model != a.compare_against]
    assert ((off.pval > 0) & (off.pval <= 1)).all()


def test_no_warning_without_pvalues():
    fc = su.as_forecast_quantile(
        pd.DataFrame(su.example_quantile()).dropna(),
        forecast_unit=[
            "location",
            "forecast_date",
            "target_end_date",
            "target_type",
            "model",
            "horizon",
        ],
    )
    scores = su.score(fc, metrics=su.get_metrics(fc, "ae_median"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = su.add_relative_skill(
            scores, by="location", metric="ae_median", test_type=None
        )
    assert "ae_median_relative_skill" in out.columns
    assert not out.ae_median_relative_skill.isna().any()


def test_pivot_scores(scores_quantile):
    from scoringutils_py._pairwise import _pivot_scores

    s = scores_quantile[
        ~(
            (scores_quantile.model == "EuroCOVIDhub-ensemble")
            & (scores_quantile.location == "DE")
        )
    ]
    m = _pivot_scores(s, compare="model", metric="wis")
    fu = [c for c in su.get_forecast_unit(s) if c != "model"]
    assert m.shape[0] == len(s[fu].drop_duplicates())
    assert set(m.columns) == set(s.model.unique())
    assert (
        m["EuroCOVIDhub-ensemble"].notna().sum()
        == (s.model == "EuroCOVIDhub-ensemble").sum()
    )
    assert m.notna().to_numpy().sum() == len(s)


def test_duplicated_scores_error(scores_quantile):
    from scoringutils_py._pairwise import _pivot_scores

    extra = scores_quantile.iloc[:5].copy()
    extra["wis"] = extra["wis"] + 1
    dup = su.new_scores(pd.concat([scores_quantile, extra]), scores_quantile.metrics)
    with pytest.raises(
        ValueError, match="more than one score for the same forecast unit"
    ):
        _pivot_scores(dup, compare="model", metric="wis")
    with pytest.raises(
        ValueError, match="more than one score for the same forecast unit"
    ):
        su.get_pairwise_comparisons(dup, metric="wis")
    with pytest.raises(
        ValueError, match="more than one score for the same forecast unit"
    ):
        su.add_relative_skill(dup, metric="wis")
    exact = su.new_scores(
        pd.concat([scores_quantile, scores_quantile.iloc[:5]]), scores_quantile.metrics
    )
    su.get_pairwise_comparisons(exact, metric="wis")
    su.add_relative_skill(exact, metric="wis")


def test_pairwise_matches_per_pair_reference(scores_quantile):
    s = scores_quantile
    s = s[~((s.model == "EuroCOVIDhub-ensemble") & (s.location == "DE"))]
    s = s[~((s.model == "EuroCOVIDhub-ensemble") & (s.target_type == "Cases"))]
    s = s[~((s.model == "epiforecasts-EpiNow2") & (s.target_type == "Deaths"))]
    fu = [c for c in su.get_forecast_unit(s) if c != "model"]
    pw = su.get_pairwise_comparisons(s, metric="wis")
    for a, b in combinations(s.model.unique(), 2):
        merged = s[s.model == a].merge(s[s.model == b], on=fu)
        row = pw[(pw.model == a) & (pw.compare_against == b)].iloc[0]
        if merged.empty:
            assert np.isnan(row.mean_scores_ratio)
            continue
        assert row.mean_scores_ratio == pytest.approx(
            merged.wis_x.sum() / merged.wis_y.sum()
        )
        assert row.pval == pytest.approx(
            wilcox_paired_pvalue(merged.wis_x, merged.wis_y)
        )
    notest = su.get_pairwise_comparisons(s, metric="wis", test_type=None)
    assert notest[notest.model != notest.compare_against].pval.isna().all()


def test_factor_compare_column(scores_quantile):
    s = scores_quantile.copy()
    s["model"] = s["model"].astype("category")
    a = su.get_pairwise_comparisons(s, metric="wis").sort_values(
        ["model", "compare_against"]
    )
    b = su.get_pairwise_comparisons(scores_quantile, metric="wis").sort_values(
        ["model", "compare_against"]
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_add_relative_skill_skips_test_by_default(scores_quantile):
    without = su.add_relative_skill(scores_quantile, metric="wis")
    assert "pval" not in without.columns
    with_test = su.add_relative_skill(
        scores_quantile, metric="wis", test_type="non_parametric"
    )
    pd.testing.assert_frame_equal(pd.DataFrame(without), pd.DataFrame(with_test))


# ---------------------------------------------------------------------------
# R statistics helpers
# ---------------------------------------------------------------------------


def test_holm_adjustment():
    np.testing.assert_allclose(
        p_adjust_holm([0.01, 0.04, 0.03, 0.2, np.nan, 0.5]),
        [0.05, 0.12, 0.12, 0.4, np.nan, 0.5],
    )


def test_wilcoxon_large_sample_matches_scipy():
    # for n >= 50 without ties R and scipy use the same normal approximation
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(0)
    x, y = rng.normal(size=80), rng.normal(0.3, size=80)
    expected = wilcoxon(x, y, correction=True, method="approx").pvalue
    assert wilcox_paired_pvalue(x, y) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# get_coverage() (test-get-coverage.R)
# ---------------------------------------------------------------------------


def test_get_coverage_by_forecast_unit(fc_quantile):
    fu = su.get_forecast_unit(fc_quantile)
    cov = su.get_coverage(fc_quantile, by=fu)
    assert sorted(cov.columns) == sorted(
        fu
        + [
            "interval_range",
            "quantile_level",
            "interval_coverage",
            "interval_coverage_deviation",
            "quantile_coverage",
            "quantile_coverage_deviation",
        ]
    )
    assert len(cov) == len(pd.DataFrame(fc_quantile).dropna())
    assert type(cov) is pd.DataFrame


def test_get_coverage_non_symmetric_intervals(fc_quantile):
    df = pd.DataFrame(fc_quantile)
    test = su.new_forecast(df[~df.quantile_level.isin([0.2, 0.3, 0.5])], "quantile")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cov = su.get_coverage(test)
    ranges = su.get_range_from_quantile([0.2, 0.3, 0.5])
    assert cov[cov.interval_range.isin(ranges)].interval_coverage.isna().all()
    assert not cov[~cov.interval_range.isin(ranges)].isna().any().any()

    # NA predictions give the same result, because rows with NA are dropped
    df2 = df.copy()
    df2.loc[df2.quantile_level.isin([0.2, 0.3, 0.5]), "predicted"] = np.nan
    cov2 = su.get_coverage(su.new_forecast(df2, "quantile"))
    pd.testing.assert_frame_equal(cov, cov2)


def test_get_coverage_requires_quantile_forecast(fc_sample_continuous):
    with pytest.raises(ValueError, match="quantile"):
        su.get_coverage(fc_sample_continuous)
