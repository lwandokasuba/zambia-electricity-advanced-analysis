import numpy as np
import pandas as pd
import pytest

from src import analysis, data


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return data.load()


def test_data_cleaning(df):
    assert len(df) == 225
    assert set(df.loc[df["iso3"] == "NAM", "iso2"]) == {"NA"}
    assert df.loc[df["renewables"].isna(), "year"].min() == 2022


def test_moving_average_formula():
    s = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    ma3 = data.moving_average(s, 3)
    assert ma3.isna().sum() == 2  # k-1 leading values are lost
    assert ma3.iloc[2] == pytest.approx(4.0)
    assert data.moving_average(s, 5).iloc[4] == pytest.approx(6.0)


def test_linear_trend_recovers_slope():
    years = pd.Series(range(2000, 2010))
    slope, r2 = analysis.linear_trend(years, 3 + 2.0 * (years - 2000))
    assert slope == pytest.approx(2.0)
    assert r2 == pytest.approx(1.0)


def test_breakpoint_recovers_known_kink():
    years = pd.Series(range(2000, 2025))
    values = np.where(years < 2012, 10 + 0.5 * (years - 2000), 16 + 3.0 * (years - 2012))
    bp = analysis.best_breakpoint(years, pd.Series(values))
    assert bp["year"] == 2012
    assert bp["slope_before"] == pytest.approx(0.5, abs=1e-6)
    assert bp["slope_after"] == pytest.approx(3.0, abs=1e-6)


def test_correlation_views(df):
    m = analysis.correlation_matrices(df)
    assert set(m) == {"pooled", "between", "within", "changes"}
    # wealth-access association is a between-country pattern, not a year-to-year one
    assert m["between"].loc["access", "gdp_pc"] > 0.9
    assert abs(m["changes"].loc["access", "gdp_pc"]) < 0.2


def test_anomalies_flag_known_drops(df):
    a = analysis.anomalies(df)
    flagged = set(zip(a["country"], a["year"]))
    assert ("Zambia", 2007) in flagged
    assert ("Zimbabwe", 2014) in flagged
    assert not (a["year"] == 2024).any()  # no end-of-series artefacts


def test_rates_vs_counts(df):
    r = analysis.rates_vs_counts(df, 2024).set_index("country")
    assert r.loc["Tanzania", "rank_by_count"] == 1
    assert r.loc["Malawi", "rank_by_rate_deficit"] == 1
    assert r["share_of_region_deficit"].sum() == pytest.approx(1.0)
