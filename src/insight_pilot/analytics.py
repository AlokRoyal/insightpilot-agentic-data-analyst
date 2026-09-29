"""Deterministic analysis tools exposed to the language model."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _subset(frame: pd.DataFrame, indicator: str, countries: list[str] | None = None) -> pd.DataFrame:
    available = frame["indicator"].dropna().unique().tolist()
    if indicator not in available:
        raise ValueError(f"Unknown indicator. Choose one of: {', '.join(available)}")
    result = frame.loc[frame["indicator"] == indicator].copy()
    if countries:
        known = set(frame["country"].dropna().unique())
        invalid = set(countries).difference(known)
        if invalid:
            raise ValueError(f"Unknown countries: {', '.join(sorted(invalid))}")
        result = result[result["country"].isin(countries)]
    return result


def summarize_indicator(frame: pd.DataFrame, indicator: str, countries: list[str] | None = None) -> str:
    data = _subset(frame, indicator, countries)
    latest = data.sort_values("year").groupby("country", as_index=False).tail(1)
    result = {
        "indicator": indicator,
        "units": "as defined by the World Bank indicator; GDP per capita is current US$, population is persons",
        "observations": int(len(data)),
        "latest_by_country": latest[["country", "year", "value"]].round({"value": 3}).to_dict("records"),
        "period_min": int(data["year"].min()),
        "period_max": int(data["year"].max()),
    }
    return json.dumps(result)


def trend_indicator(frame: pd.DataFrame, indicator: str, country: str, start_year: int | None = None, end_year: int | None = None) -> str:
    data = _subset(frame, indicator, [country]).sort_values("year")
    if start_year is not None:
        data = data[data["year"] >= start_year]
    if end_year is not None:
        data = data[data["year"] <= end_year]
    if data.empty:
        raise ValueError("No observations exist for that country and period.")
    first, last = data.iloc[0], data.iloc[-1]
    absolute = float(last["value"] - first["value"])
    change_pct = (absolute / first["value"] * 100) if first["value"] else None
    result = {
        "indicator": indicator,
        "country": country,
        "start": {"year": int(first["year"]), "value": float(first["value"])},
        "end": {"year": int(last["year"]), "value": float(last["value"])},
        "absolute_change": absolute,
        "percent_change": change_pct,
        "observations": int(len(data)),
    }
    return json.dumps(result)


def compare_countries(frame: pd.DataFrame, indicator: str, year: int, countries: list[str]) -> str:
    if len(countries) < 2:
        raise ValueError("Select at least two countries to compare.")
    data = _subset(frame, indicator, countries)
    data = data[data["year"] == year].sort_values("value", ascending=False)
    return json.dumps({
        "indicator": indicator,
        "year": year,
        "available": int(len(data)),
        "requested": countries,
        "results": data[["country", "value"]].round({"value": 3}).to_dict("records"),
    })


def correlation_check(frame: pd.DataFrame, indicator_a: str, indicator_b: str, year: int | None = None) -> str:
    a = _subset(frame, indicator_a)[["country_code", "country", "year", "value"]].rename(columns={"value": "a"})
    b = _subset(frame, indicator_b)[["country_code", "year", "value"]].rename(columns={"value": "b"})
    joined = a.merge(b, on=["country_code", "year"], how="inner")
    if year is not None:
        joined = joined[joined["year"] == year]
    if len(joined) < 3:
        raise ValueError("At least three matched country-year observations are needed.")
    corr = float(joined["a"].corr(joined["b"]))
    result: dict[str, Any] = {
        "indicator_a": indicator_a,
        "indicator_b": indicator_b,
        "pearson_correlation": corr,
        "matched_observations": int(len(joined)),
        "year_filter": year,
        "caution": "Correlation is descriptive and does not establish causation.",
    }
    return json.dumps(result)
