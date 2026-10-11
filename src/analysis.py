"""Computations behind the Assignment 3 figures (multivariate, temporal, spatial).

Kept separate from the notebook so the numbers quoted in the report are tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import moving_average

NUMERIC = {
    "access": "Electricity access (%)",
    "gdp_pc": "GDP per capita (US$)",
    "urban": "Urban population (%)",
    "life_exp": "Life expectancy (yrs)",
    "renewables": "Renewables share (%)",
    "population": "Population",
}


# --------------------------------------------------------------------------- multivariate
def correlation_matrices(df: pd.DataFrame, cols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Three Spearman correlation matrices that answer different questions.

    - ``pooled``: all country-years together (mixes between-country and over-time variation).
    - ``between``: country means (do richer/more urban *countries* have more access?).
    - ``within``: country-demeaned values (when a country grows richer/more urban, does *its* access rise?).
      These still share the common upward time trend, so they overstate co-movement.
    - ``changes``: year-on-year changes, centred on each country's average change; this removes
      both the shared trend and country size, so it asks "in years when X rose faster than usual,
      did access also rise faster than usual?".
    Renewables are restricted to 2000-2021 by pairwise deletion of the missing 2022-2024 values.
    """
    cols = cols or list(NUMERIC)
    d = df[["iso3"] + cols].copy()
    d["gdp_pc"] = np.log10(d["gdp_pc"])
    between = d.groupby("iso3")[cols].mean()
    within = d[cols] - d.groupby("iso3")[cols].transform("mean")
    diffs = d.groupby("iso3")[cols].diff()
    changes = diffs - diffs.groupby(d["iso3"]).transform("mean")  # each country's change vs its usual change
    return {
        "pooled": d[cols].corr(method="spearman"),
        "between": between.corr(method="spearman"),
        "within": within.corr(method="spearman"),
        "changes": changes.corr(method="spearman"),
    }


# --------------------------------------------------------------------------- temporal
def smoothed(df: pd.DataFrame, col: str = "access", ks: tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """Raw series plus trailing moving averages MA_t = (1/k) * sum_{i=0}^{k-1} x_{t-i}."""
    out = df[["iso3", "country", "year", col]].copy()
    for k in ks:
        out[f"ma{k}"] = out.groupby("iso3")[col].transform(lambda s, k=k: moving_average(s, k))
    return out


def linear_trend(years: pd.Series, values: pd.Series) -> tuple[float, float]:
    """OLS slope (units per year) and R^2."""
    x, y = years.to_numpy(float), values.to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(slope), 1 - ss_res / ss_tot if ss_tot else float("nan")


def best_breakpoint(years: pd.Series, values: pd.Series, min_seg: int = 5) -> dict:
    """Continuous two-segment (piecewise linear) fit; the breakpoint minimising SSE.

    Model: y = a + b*t + c*max(0, t - tau). Slope before = b, after = b + c.
    """
    x, y = years.to_numpy(float), values.to_numpy(float)
    best = None
    for tau in x[min_seg:-min_seg]:
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - tau)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ coef) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"year": int(tau), "slope_before": float(coef[1]), "slope_after": float(coef[1] + coef[2]), "sse": sse}
    one_slope, _ = linear_trend(years, values)
    X1 = np.column_stack([np.ones_like(x), x])
    c1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    best["sse_single_line"] = float(((y - X1 @ c1) ** 2).sum())
    best["sse_reduction"] = 1 - best["sse"] / best["sse_single_line"]
    return best


def temporal_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, d in df.groupby("iso3"):
        d = d.sort_values("year")
        slope, r2 = linear_trend(d["year"], d["access"])
        bp = best_breakpoint(d["year"], d["access"])
        yoy = d["access"].diff()
        rows.append({
            "iso3": iso, "country": d["country"].iloc[0],
            "access_2000": d["access"].iloc[0], "access_2024": d["access"].iloc[-1],
            "trend_pp_per_year": slope, "trend_r2": r2,
            "break_year": bp["year"], "slope_before": bp["slope_before"], "slope_after": bp["slope_after"],
            "break_sse_reduction": bp["sse_reduction"],
            "yoy_sd": float(yoy.std()), "n_drops_gt3": int((yoy < -3).sum()),
        })
    return pd.DataFrame(rows).sort_values("trend_pp_per_year", ascending=False).reset_index(drop=True)


def anomalies(df: pd.DataFrame, z: float = 3.0, window: int = 2) -> pd.DataFrame:
    """Years that depart from what their neighbours predict.

    For each year t, a straight line is fitted to the other years within +/- ``window`` (leave-one-out)
    and used to predict t; a local line, unlike a centred median, is not biased at the ends of a
    trending series. Residuals are scaled by ONE robust SD pooled over all countries
    (1.4826 x MAD), so a volatile country cannot hide its own outliers. |z| > ``z`` is an anomaly.
    """
    parts = []
    for _, d in df.groupby("iso3"):
        d = d.sort_values("year").reset_index(drop=True)
        years, vals = d["year"].to_numpy(float), d["access"].to_numpy(float)
        pred = np.full(len(d), np.nan)
        for i, t in enumerate(years):
            m = (np.abs(years - t) <= window) & (years != t)
            if m.sum() >= 3:
                b, c = np.polyfit(years[m], vals[m], 1)
                pred[i] = b * t + c
        parts.append(d.assign(expected=pred, residual=vals - pred))
    allr = pd.concat(parts, ignore_index=True)
    r = allr["residual"].dropna()
    scale = 1.4826 * float((r - r.median()).abs().median())
    allr["robust_z"] = allr["residual"] / scale
    out = allr[allr["robust_z"].abs() > z][["country", "year", "access", "expected", "residual", "robust_z"]]
    return out.sort_values(["country", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------- spatial
def rates_vs_counts(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Rank countries by the rate (access %) and by the count (people without electricity)."""
    d = df[df["year"] == year][["iso3", "country", "access", "pop_without_m", "population"]].copy()
    d["rank_by_rate_deficit"] = d["access"].rank(ascending=True, method="min").astype(int)
    d["rank_by_count"] = d["pop_without_m"].rank(ascending=False, method="min").astype(int)
    d["share_of_region_deficit"] = d["pop_without_m"] / d["pop_without_m"].sum()
    return d.sort_values("pop_without_m", ascending=False).reset_index(drop=True)
