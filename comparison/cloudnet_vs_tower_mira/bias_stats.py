"""
bias_stats.py
=============
Single function that computes all bias statistics for one (ze_fwd, ze_rad) pair.
Both inputs are 1-D arrays in dBZ. NaN values are ignored.
"""

import numpy as np
from scipy import stats


def compute_stats(ze_fwd, ze_rad, n_boot=5000, ci=95, seed=42):
    """
    Compute bias statistics between a forward-simulated Ze and a radar Ze.

    Parameters
    ----------
    ze_fwd : array-like, dBZ
    ze_rad : array-like, dBZ
    n_boot : int   Number of bootstrap resamples for CI
    ci     : float Confidence interval width (default 95)
    seed   : int   Random seed for reproducibility

    Returns
    -------
    dict with keys:
        N          – number of valid (finite) pairs
        mean       – mean(Ze_rad − Ze_fwd)  [dB]
        median     – median(Ze_rad − Ze_fwd)  [dB]
        std        – std(Ze_rad − Ze_fwd)  [dB]
        ci_lo      – lower bound of bootstrap CI on the mean  [dB]
        ci_hi      – upper bound of bootstrap CI on the mean  [dB]
        rmse       – sqrt(mean((Ze_rad − Ze_fwd)²))  [dB]
        slope      – OLS slope of  Ze_rad ~ Ze_fwd
        intercept  – OLS intercept  [dB]
        r          – Pearson correlation
        p          – two-tailed p-value for r
    All float values are np.nan when N < 5.
    """
    ze_fwd = np.asarray(ze_fwd, dtype=float)
    ze_rad = np.asarray(ze_rad, dtype=float)

    mask = np.isfinite(ze_fwd) & np.isfinite(ze_rad)
    xf, xr = ze_fwd[mask], ze_rad[mask]
    N = int(mask.sum())

    nan_result = dict(N=N, mean=np.nan, median=np.nan, std=np.nan,
                      ci_lo=np.nan, ci_hi=np.nan, rmse=np.nan,
                      slope=np.nan, intercept=np.nan, r=np.nan, p=np.nan)
    if N < 5:
        return nan_result

    diff = xr - xf
    mean_b   = float(np.mean(diff))
    median_b = float(np.median(diff))
    std_b    = float(np.std(diff, ddof=1))
    rmse     = float(np.sqrt(np.mean(diff ** 2)))

    # Bootstrap CI on the mean
    rng  = np.random.default_rng(seed)
    boot = rng.choice(diff, size=(n_boot, N), replace=True)
    boot_means = boot.mean(axis=1)
    alpha = (100 - ci) / 2
    ci_lo = float(np.percentile(boot_means, alpha))
    ci_hi = float(np.percentile(boot_means, 100 - alpha))

    # OLS regression  Ze_rad = slope * Ze_fwd + intercept
    lr = stats.linregress(xf, xr)

    # Pearson r
    r, p = stats.pearsonr(xf, xr)

    return dict(N=N, mean=mean_b, median=median_b, std=std_b,
                ci_lo=ci_lo, ci_hi=ci_hi, rmse=rmse,
                slope=float(lr.slope), intercept=float(lr.intercept),
                r=float(r), p=float(p))
