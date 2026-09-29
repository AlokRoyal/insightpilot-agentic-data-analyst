"""Curated, deterministic feature implementations used by the no-key daily agent."""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"country_code", "country", "year", "indicator", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    result = frame[list(required)].copy()
    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result.dropna(subset=["country_code", "country", "year", "indicator", "value"])


def _latest(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("year").groupby(["country_code", "country", "indicator"], as_index=False).tail(1)


def run_feature(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    """Run one allow-listed insight calculation without model-generated code."""
    data = _valid(frame)
    latest = _latest(data)

    if feature == "trend_velocity":
        ordered = data.sort_values("year").copy()
        ordered["annual_change"] = ordered.groupby(["country_code", "indicator"])["value"].diff()
        ordered["previous_value"] = ordered.groupby(["country_code", "indicator"])["value"].shift()
        ordered["annual_change_pct"] = ordered["annual_change"].div(ordered["previous_value"].abs().replace(0, pd.NA)) * 100
        recent = ordered.dropna(subset=["annual_change"]).groupby(["country", "indicator"], as_index=False).tail(1)
        recent = recent.sort_values("annual_change_pct", key=lambda s: s.abs(), ascending=False).head(12)
        return {"feature": feature, "largest_recent_moves": recent[["country", "indicator", "year", "annual_change", "annual_change_pct"]].round(3).to_dict("records"), "note": "Absolute changes are shown alongside percent changes; rates can be unstable near zero."}

    if feature == "country_benchmarks":
        summary = latest.groupby("indicator")["value"].agg(median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75)).reset_index()
        result = latest.merge(summary, on="indicator")
        result["vs_median_pct"] = (result["value"] - result["median"]).div(result["median"].abs().replace(0, pd.NA)) * 100
        result["benchmark_year"] = result["year"]
        return {"feature": feature, "benchmarks": result[["country", "indicator", "benchmark_year", "value", "median", "q1", "q3", "vs_median_pct"]].round(3).to_dict("records"), "note": "Each country uses its latest available year, so compare years before interpreting rankings."}

    if feature == "coverage_audit":
        rows = []
        for (country, indicator), group in data.groupby(["country", "indicator"]):
            years = sorted(set(group["year"].astype(int)))
            expected = set(range(years[0], years[-1] + 1))
            gaps = sorted(expected.difference(years))
            rows.append({"country": country, "indicator": indicator, "start_year": years[0], "end_year": years[-1], "observations": len(years), "missing_year_count": len(gaps), "missing_years_preview": gaps[:15]})
        rows.sort(key=lambda r: r["missing_year_count"], reverse=True)
        return {"feature": feature, "coverage": rows, "note": "A missing year may mean the source did not publish an observation."}

    if feature == "indicator_relationships":
        wide = data.pivot_table(index=["country_code", "year"], columns="indicator", values="value", aggfunc="mean")
        relationships = []
        for left, right in itertools.combinations(wide.columns, 2):
            pair = wide[[left, right]].dropna()
            if len(pair) >= 3:
                relationships.append({"indicator_a": left, "indicator_b": right, "pearson_r": float(pair[left].corr(pair[right])), "matched_observations": int(len(pair))})
        relationships.sort(key=lambda item: abs(item["pearson_r"]), reverse=True)
        return {"feature": feature, "top_relationships": relationships[:15], "note": "Descriptive correlations do not establish causation and can reflect time trends."}

    if feature == "distribution_profile":
        profiles = []
        for indicator, group in data.groupby("indicator"):
            values = group["value"]
            profiles.append({"indicator": indicator, "observations": int(values.count()), "mean": float(values.mean()), "median": float(values.median()), "std_dev": float(values.std()) if len(values) > 1 else None, "q05": float(values.quantile(.05)), "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75)), "q95": float(values.quantile(.95)), "skew": float(values.skew()) if len(values) > 2 else None})
        return {"feature": feature, "profiles": profiles, "note": "Pooled distributions combine countries and years with different scales and coverage."}

    if feature == "change_points":
        ordered = data.sort_values("year").copy()
        groups = ordered.groupby(["country_code", "indicator"], group_keys=False)
        ordered["change"] = groups["value"].diff()
        ordered["prior_year"] = groups["year"].shift()
        candidates = []
        for (country, indicator), group in ordered.groupby(["country", "indicator"]):
            changes = group["change"].dropna()
            if len(changes) < 4:
                continue
            median = changes.median()
            mad = (changes - median).abs().median()
            threshold = 3 * 1.4826 * mad
            if threshold == 0:
                threshold = changes.std() * 2 if len(changes) > 1 else 0
            for row in group.dropna(subset=["change"]).itertuples():
                deviation = abs(float(row.change) - float(median))
                if threshold > 0 and deviation > threshold:
                    candidates.append({"country": country, "indicator": indicator, "from_year": int(row.prior_year), "to_year": int(row.year), "change": float(row.change), "median_change": float(median), "robust_threshold": float(threshold)})
        candidates.sort(key=lambda item: abs(item["change"] - item["median_change"]), reverse=True)
        return {"feature": feature, "unusual_changes": candidates[:20], "note": "Statistical flags are review prompts, not evidence of errors or causal events."}

    if feature == "peer_groups":
        result = latest.copy()
        result["peer_band"] = result.groupby("indicator")["value"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["lower quartile", "lower-middle", "upper-middle", "upper quartile"]))
        return {"feature": feature, "peers": result[["country", "indicator", "year", "value", "peer_band"]].to_dict("records"), "note": "Quartile groups are relative to this dataset and use each country's latest available year."}

    raise ValueError(f"Unknown daily feature: {feature}")
