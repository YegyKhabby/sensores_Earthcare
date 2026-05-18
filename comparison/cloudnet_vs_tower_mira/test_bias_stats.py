"""Tests for bias_stats.compute_stats."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comparison.cloudnet_vs_tower_mira.bias_stats import compute_stats


RNG = np.random.default_rng(0)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_pair(n=500, offset=0.0, noise=2.0, seed=1):
    rng = np.random.default_rng(seed)
    ze_fwd = rng.uniform(5, 40, n)
    ze_rad = ze_fwd + offset + rng.normal(0, noise, n)
    return ze_fwd, ze_rad


# ── basic correctness ─────────────────────────────────────────────────────────

def test_known_offset_mean():
    """Mean bias should recover the known constant offset."""
    ze_fwd, ze_rad = make_pair(n=1000, offset=-6.5, noise=1.0)
    s = compute_stats(ze_fwd, ze_rad)
    assert abs(s['mean'] - (-6.5)) < 0.15, f"mean={s['mean']:.3f}, expected ≈ -6.5"


def test_known_offset_ci_contains_true():
    """95% CI on mean should contain the true offset."""
    ze_fwd, ze_rad = make_pair(n=500, offset=-6.5, noise=2.0)
    s = compute_stats(ze_fwd, ze_rad)
    assert s['ci_lo'] <= -6.5 <= s['ci_hi'], (
        f"CI [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] does not contain -6.5")


def test_zero_bias():
    """Identical arrays → zero mean bias, RMSE, std; slope=1, intercept=0."""
    x = np.linspace(5, 40, 200)
    s = compute_stats(x, x)
    assert s['N'] == 200
    assert abs(s['mean'])      < 1e-10
    assert abs(s['median'])    < 1e-10
    assert abs(s['rmse'])      < 1e-10
    assert abs(s['slope'] - 1) < 1e-6
    assert abs(s['intercept']) < 1e-6


def test_rmse_formula():
    """RMSE = sqrt(mean(diff²)) must hold exactly."""
    ze_fwd, ze_rad = make_pair(n=300, offset=-3.0, noise=3.0)
    s = compute_stats(ze_fwd, ze_rad)
    expected_rmse = float(np.sqrt(np.mean((ze_rad - ze_fwd) ** 2)))
    assert abs(s['rmse'] - expected_rmse) < 1e-9


def test_std_ddof1():
    """Std should use ddof=1 (sample std)."""
    ze_fwd, ze_rad = make_pair(n=100, offset=-5.0, noise=2.5)
    s = compute_stats(ze_fwd, ze_rad)
    expected = float(np.std(ze_rad - ze_fwd, ddof=1))
    assert abs(s['std'] - expected) < 1e-9


def test_ci_ordering():
    """ci_lo must be strictly less than ci_hi."""
    ze_fwd, ze_rad = make_pair(n=200, offset=-7.0, noise=4.0)
    s = compute_stats(ze_fwd, ze_rad)
    assert s['ci_lo'] < s['ci_hi']


def test_ci_width_shrinks_with_n():
    """Larger N should give a narrower CI."""
    s_small = compute_stats(*make_pair(n=50,   offset=-6.0, noise=3.0))
    s_large = compute_stats(*make_pair(n=5000, offset=-6.0, noise=3.0))
    width_small = s_small['ci_hi'] - s_small['ci_lo']
    width_large = s_large['ci_hi'] - s_large['ci_lo']
    assert width_large < width_small, (
        f"CI width did not shrink: small={width_small:.3f}, large={width_large:.3f}")


def test_pearson_r_perfect():
    """Perfect linear relationship → r ≈ 1."""
    x = np.linspace(5, 40, 300)
    s = compute_stats(x, 2 * x - 5)
    assert abs(s['r'] - 1.0) < 1e-6


def test_regression_slope_intercept():
    """Slope and intercept should match scipy.stats.linregress directly."""
    from scipy import stats as sp_stats
    ze_fwd, ze_rad = make_pair(n=400, offset=-5.0, noise=2.0)
    s = compute_stats(ze_fwd, ze_rad)
    lr = sp_stats.linregress(ze_fwd, ze_rad)
    assert abs(s['slope']     - lr.slope)     < 1e-9
    assert abs(s['intercept'] - lr.intercept) < 1e-9


# ── NaN and edge cases ────────────────────────────────────────────────────────

def test_all_nan_returns_nan():
    """All-NaN input → N=0, all stats NaN."""
    nan = np.full(100, np.nan)
    s = compute_stats(nan, nan)
    assert s['N'] == 0
    assert np.isnan(s['mean'])
    assert np.isnan(s['rmse'])
    assert np.isnan(s['slope'])


def test_partial_nan_ignored():
    """NaN rows should be silently dropped; valid rows used normally."""
    ze_fwd = np.array([10.0, np.nan, 20.0, np.nan, 30.0,
                        15.0, 25.0, 18.0, 22.0, 12.0])
    ze_rad = ze_fwd.copy()
    ze_rad[~np.isnan(ze_fwd)] -= 5.0  # constant -5 dB offset on valid rows
    s = compute_stats(ze_fwd, ze_rad)
    assert s['N'] == 8
    assert abs(s['mean'] - (-5.0)) < 1e-9


def test_n_below_5_returns_nan():
    """N < 5 valid pairs → all statistics are NaN."""
    ze_fwd = np.array([10.0, 15.0, 20.0, np.nan])
    ze_rad = np.array([8.0,  13.0, 18.0, np.nan])
    s = compute_stats(ze_fwd, ze_rad)
    assert s['N'] == 3
    assert np.isnan(s['mean'])
    assert np.isnan(s['ci_lo'])


def test_exactly_5_does_not_return_nan():
    """N == 5 is the minimum for valid output."""
    ze_fwd = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    ze_rad = ze_fwd - 6.0
    s = compute_stats(ze_fwd, ze_rad)
    assert s['N'] == 5
    assert not np.isnan(s['mean'])


def test_asymmetric_distribution_median_ne_mean():
    """Skewed bias distribution → median and mean should differ noticeably."""
    rng = np.random.default_rng(7)
    ze_fwd = rng.uniform(5, 40, 500)
    # Add a right skew via a few large positive outliers
    noise = rng.normal(-6.0, 2.0, 500)
    noise[:20] = 20.0   # outliers
    ze_rad = ze_fwd + noise
    s = compute_stats(ze_fwd, ze_rad)
    assert abs(s['mean'] - s['median']) > 0.5, (
        f"mean={s['mean']:.2f} median={s['median']:.2f} — expected meaningful difference")


def test_reproducibility():
    """Same seed → identical CI bounds across two calls."""
    ze_fwd, ze_rad = make_pair(n=300, offset=-6.0, noise=3.0)
    s1 = compute_stats(ze_fwd, ze_rad, seed=42)
    s2 = compute_stats(ze_fwd, ze_rad, seed=42)
    assert s1['ci_lo'] == s2['ci_lo']
    assert s1['ci_hi'] == s2['ci_hi']


def test_different_seeds_give_different_ci():
    """Different seeds should (almost always) give slightly different CI bounds."""
    ze_fwd, ze_rad = make_pair(n=100, offset=-6.0, noise=3.0)
    s1 = compute_stats(ze_fwd, ze_rad, seed=1)
    s2 = compute_stats(ze_fwd, ze_rad, seed=99)
    # Not guaranteed but true with overwhelming probability for N=100
    assert s1['ci_lo'] != s2['ci_lo'] or s1['ci_hi'] != s2['ci_hi']
