"""Exact ports of small pieces of base R behaviour that scoringutils relies on.

Each function documents the R function it reproduces. These exist because the
"obvious" numpy / scipy equivalent differs from R in some edge case, and the
point of this package is numeric parity with R scoringutils.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

# R's default tolerance for all.equal() on numeric vectors.
_ALL_EQUAL_TOL = 1.5e-8
_INT_MAX = 2**31 - 1


def get_type(x: np.ndarray) -> str:
    """Port of scoringutils:::get_type() for numeric input.

    Returns "integer" if every value is (close to) a whole number, otherwise
    "continuous". R decides this with ``all.equal(x, as.integer(x))``, which
    uses a relative tolerance of 1.5e-8 and treats values outside the 32-bit
    integer range as NA (so they count as continuous).
    """
    x = np.asarray(x)
    if x.dtype == bool:
        raise TypeError("Assertion on 'x' failed: Must be of type 'numeric'.")
    if np.issubdtype(x.dtype, np.integer):
        return "integer"
    x = np.asarray(x, dtype=float).ravel()
    if np.all(np.isnan(x)):
        raise ValueError("Can't get type: all values of are NA.")
    as_int = np.trunc(x)
    as_int[np.abs(x) > _INT_MAX] = np.nan
    as_int[np.isnan(x)] = np.nan
    if _all_equal(x, as_int) and not np.all(np.isnan(as_int)):
        return "integer"
    return "continuous"


def _all_equal(target: np.ndarray, current: np.ndarray) -> bool:
    """Port of base::all.equal.numeric() returning TRUE / not TRUE."""
    na_t = np.isnan(target)
    na_c = np.isnan(current)
    if np.any(na_t != na_c):
        return False
    t = target[~na_t]
    c = current[~na_t]
    if t.size == 0:
        return True
    xy = np.sum(np.abs(t - c)) / t.size
    xn = np.sum(np.abs(t)) / t.size
    if np.isfinite(xn) and xn > _ALL_EQUAL_TOL:
        xy = xy / xn
    return bool(xy <= _ALL_EQUAL_TOL)


def bw_nrd(x: np.ndarray) -> float:
    """Port of stats::bw.nrd(): Scott's rule-of-thumb kernel bandwidth.

    ``1.06 * min(sd(x), IQR(x) / 1.34) * n^(-1/5)`` with R's default (type 7)
    quantiles and the n - 1 standard deviation.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 data points")
    q25, q75 = np.quantile(x, [0.25, 0.75])
    h = (q75 - q25) / 1.34
    return float(1.06 * min(np.sqrt(np.var(x, ddof=1)), h) * x.size ** (-1 / 5))


def r_mad(x: np.ndarray, constant: float = 1.4826) -> float:
    """Port of stats::mad() with its defaults (median centre, 1.4826)."""
    x = np.asarray(x, dtype=float)
    centre = np.median(x)
    return float(constant * np.median(np.abs(x - centre)))


def p_adjust_holm(p: np.ndarray) -> np.ndarray:
    """Port of stats::p.adjust(p, method = "holm"), the R default.

    NA values are kept in place and do not count towards the number of tests.
    """
    p = np.asarray(p, dtype=float)
    out = p.copy()
    ok = ~np.isnan(p)
    pv = p[ok]
    n = pv.size
    if n <= 1:
        return out
    order = np.argsort(pv, kind="stable")
    i = np.arange(1, n + 1)
    adj = np.minimum(1.0, np.maximum.accumulate((n + 1 - i) * pv[order]))
    res = np.empty(n)
    res[order] = adj
    out[ok] = res
    return out


# ---------------------------------------------------------------------------
# Wilcoxon signed rank test, as in stats::wilcox.test(x, y, paired = TRUE)
# from R 4.6 (two-sided, mu = 0, exact = NULL, correct = TRUE).
# ---------------------------------------------------------------------------


def _subset_sum_distribution(scores: np.ndarray) -> np.ndarray:
    """P(sum of a random subset of ``scores`` == k) for k = 0..sum(scores).

    Equivalent to R's internal C_dpermdist1 / csignrank: each score enters the
    sum with probability 1/2, independently.
    """
    total = int(np.sum(scores))
    counts = np.zeros(total + 1, dtype=float)
    counts[0] = 1.0
    for s in scores:
        s = int(s)
        if s == 0:
            counts = counts * 2
            continue
        shifted = np.zeros_like(counts)
        shifted[s:] = counts[:-s]
        counts = counts + shifted
    return counts / (2.0 ** len(scores))


def _psignrank(q: float, n: int) -> float:
    """stats::psignrank(q, n): P(V <= floor(q)) for the signed rank statistic."""
    x = np.floor(q + 1e-7)
    if x < 0:
        return 0.0
    if x >= n * (n + 1) / 2:
        return 1.0
    d = _subset_sum_distribution(np.arange(1, n + 1))
    return float(np.sum(d[: int(x) + 1]))


def _psignrank_ties(q: float, z: np.ndarray, lower_tail: bool = True) -> float:
    """stats:::.psignrank(q, n, z): exact distribution with tied / zero ranks."""
    f = 2 if np.any(z != np.floor(z)) else 1
    scores = np.sort((f * z).astype(int))
    d = _subset_sum_distribution(scores)
    s = np.arange(d.size) / f  # possible values of V
    p = float(np.sum(d[s < q + 1e-8]))
    return p if lower_tail else 1.0 - p


def wilcox_paired_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided p-value of R's ``wilcox.test(x, y, paired = TRUE)$p.value``.

    R 4.6 uses the exact null distribution when n < 50 (n counts zero
    differences), handling ties and zeros exactly, and otherwise a normal
    approximation with continuity and tie correction after dropping zeros.
    ``scipy.stats.wilcoxon`` switches between methods differently, which is
    why this is ported rather than imported.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    d = x[ok] - y[ok]
    if d.size < 1:
        raise ValueError("not enough (non-missing) 'x' observations")
    n = d.size
    if n < 50:
        is_zero = d == 0
        r = rankdata(np.abs(d), method="average")
        ties = np.unique(r).size != r.size
        zero = bool(np.any(is_zero))
        q = float(np.sum(r[d > 0]))
        if ties or zero:
            z = r[~is_zero]
            m = np.sum(z) / 2
            if q > m:
                p = _psignrank_ties(q - 0.25, z, lower_tail=False)
            else:
                p = _psignrank_ties(q, z)
        else:
            m = n * (n + 1) / 4
            if q > m:
                p = 1.0 - _psignrank(q - 0.25, n)
            else:
                p = _psignrank(q, n)
        return float(min(2 * p, 1.0))

    d = d[d != 0]
    n = d.size
    r = rankdata(np.abs(d), method="average")
    stat = np.sum(r[d > 0])
    mean = n * (n + 1) / 4
    _, nties = np.unique(r, return_counts=True)
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 - np.sum(nties**3 - nties) / 48)
    z = stat - mean
    z = (z - np.sign(z) * 0.5) / sigma
    p = norm.cdf(z)
    return float(2 * min(p, 1 - p))
