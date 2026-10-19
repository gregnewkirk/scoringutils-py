"""Vector / matrix level parity with R scoringutils 2.3.0.

Each case in ``fixtures/metric_cases.json`` stores the R function name, its
inputs and R's result. The inputs (including random draws) were generated in
R, so both languages compute on identical numbers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from conftest import ATOL, RTOL, load_metric_cases

import scoringutils_py as su
from scoringutils_py import _rcompat

CASES = load_metric_cases()

# R function name -> Python callable
FUNCTIONS = {
    "bw.nrd": _rcompat.bw_nrd,
    "get_type": _rcompat.get_type,
    "p.adjust": _rcompat.p_adjust_holm,
}
# R argument name -> Python argument name
ARG_NAMES = {"na.rm": "na_rm"}


def _call(case):
    fn = FUNCTIONS.get(case["fn"]) or getattr(su, case["fn"])
    args = {ARG_NAMES.get(k, k): _as_array(v) for k, v in case["args"].items()}
    kwargs = {ARG_NAMES.get(k, k): v for k, v in case["kwargs"].items()}
    if case["id"].endswith("_integer"):
        # keep R's integer storage type, which get_type() looks at
        args = {k: np.asarray(v, dtype=np.int64) for k, v in args.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(**args, **kwargs)


def _as_array(v):
    if isinstance(v, list):
        return np.asarray(v, dtype=float)
    return v


def _compare(actual, expected, label):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), label
        for key, value in expected.items():
            _compare(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, str):
        assert actual == expected, label
        return
    a = np.atleast_1d(np.asarray(actual, dtype=float))
    e = np.atleast_1d(np.asarray(expected, dtype=float))
    np.testing.assert_allclose(a, e, rtol=RTOL, atol=ATOL, err_msg=label)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_metric_case_matches_r(case):
    _compare(_call(case), case["result"], case["id"])


def test_every_case_is_covered():
    # guard against silently skipping cases if the fixture grows
    assert len(CASES) >= 100
