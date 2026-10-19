"""scoringutils-py: an unofficial Python port of the R package scoringutils.

R scoringutils is by Nikos Bosse, Sebastian Funk and contributors
(epiforecasts, MIT licence): https://github.com/epiforecasts/scoringutils.
This port is not affiliated with or endorsed by the upstream authors.

Function names and arguments follow the R package, so R code translates
almost line by line::

    import scoringutils_py as su

    forecast = su.as_forecast_quantile(df)
    scores = su.score(forecast)
    su.summarise_scores(scores, by=["model", "target_type"])
"""

from ._checks import ScoringutilsMessage
from ._coverage import get_coverage
from ._data import example_quantile, example_sample_continuous, example_sample_discrete
from ._forecast import (
    Forecast,
    ForecastQuantile,
    ForecastSample,
    as_forecast_quantile,
    as_forecast_sample,
    assert_forecast,
    get_duplicate_forecasts,
    get_forecast_type,
    get_forecast_type_ids,
    get_forecast_unit,
    get_protected_columns,
    is_forecast,
    is_forecast_quantile,
    is_forecast_sample,
    new_forecast,
    set_forecast_unit,
)
from ._interval import get_range_from_quantile, interval_score
from ._metrics_quantile import (
    ae_median_quantile,
    bias_quantile,
    dispersion_quantile,
    interpolate_median,
    interval_coverage,
    overprediction_quantile,
    quantile_score,
    underprediction_quantile,
    wis,
)
from ._metrics_sample import (
    ae_median_sample,
    bias_sample,
    crps_sample,
    dispersion_sample,
    dss_sample,
    logs_sample,
    mad_sample,
    overprediction_sample,
    se_mean_sample,
    underprediction_sample,
)
from ._pairwise import (
    add_relative_skill,
    compare_scores,
    get_pairwise_comparisons,
    permutation_test,
)
from ._score import (
    Scores,
    as_scores,
    assert_scores,
    get_metrics,
    new_scores,
    score,
    select_metrics,
)
from ._summarise import summarise_scores, summarize_scores

__version__ = "0.1.0"
UPSTREAM_VERSION = "2.3.0"

__all__ = [
    "Forecast",
    "ForecastQuantile",
    "ForecastSample",
    "Scores",
    "ScoringutilsMessage",
    "UPSTREAM_VERSION",
    "add_relative_skill",
    "ae_median_quantile",
    "ae_median_sample",
    "as_forecast_quantile",
    "as_forecast_sample",
    "as_scores",
    "assert_forecast",
    "assert_scores",
    "bias_quantile",
    "bias_sample",
    "compare_scores",
    "crps_sample",
    "dispersion_quantile",
    "dispersion_sample",
    "dss_sample",
    "example_quantile",
    "example_sample_continuous",
    "example_sample_discrete",
    "get_coverage",
    "get_duplicate_forecasts",
    "get_forecast_type",
    "get_forecast_type_ids",
    "get_forecast_unit",
    "get_metrics",
    "get_pairwise_comparisons",
    "get_protected_columns",
    "get_range_from_quantile",
    "interpolate_median",
    "interval_coverage",
    "interval_score",
    "is_forecast",
    "is_forecast_quantile",
    "is_forecast_sample",
    "logs_sample",
    "mad_sample",
    "new_forecast",
    "new_scores",
    "overprediction_quantile",
    "overprediction_sample",
    "permutation_test",
    "quantile_score",
    "score",
    "se_mean_sample",
    "select_metrics",
    "set_forecast_unit",
    "summarise_scores",
    "summarize_scores",
    "underprediction_quantile",
    "underprediction_sample",
    "wis",
]
