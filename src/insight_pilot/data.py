"""Fetch and validate public country-level indicators from the World Bank API."""

from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DATA_PATH = RAW_DIR / "world_bank_indicators.csv"
COUNTRIES = {
    "IND": "India",
    "BGD": "Bangladesh",
    "BRA": "Brazil",
    "CHN": "China",
    "DEU": "Germany",
    "IDN": "Indonesia",
    "JPN": "Japan",
    "USA": "United States",
}
INDICATORS = {
    "GDP per capita (current US$)": "NY.GDP.PCAP.CD",
    "Population, total": "SP.POP.TOTL",
    "Life expectancy at birth, total (years)": "SP.DYN.LE00.IN",
    "Individuals using the Internet (% of population)": "IT.NET.USER.ZS",
    "Unemployment, total (% of labor force)": "SL.UEM.TOTL.ZS",
}


def fetch_world_bank(start_year: int = 2000, end_year: int | None = None) -> pd.DataFrame:
    """Fetch five indicators for eight countries in parallel, then return tidy rows."""
    end_year = end_year or datetime.now(timezone.utc).year
    country_codes = ";".join(COUNTRIES)

    def fetch_indicator(indicator_name: str, indicator_code: str) -> list[dict[str, object]]:
        url = (
            f"https://api.worldbank.org/v2/country/{country_codes}/indicator/"
            f"{indicator_code}?format=json&per_page=20000&date={start_year}:{end_year}"
        )
        response = requests.get(
            url,
            headers={"User-Agent": "InsightPilot-portfolio/1.0"},
            timeout=(5, 30),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return []
        result = []
        for item in payload[1]:
            value = item.get("value")
            country_code = item.get("countryiso3code")
            if value is None or country_code not in COUNTRIES:
                continue
            result.append(
                {
                    "country_code": country_code,
                    "country": COUNTRIES[country_code],
                    "year": int(item["date"]),
                    "indicator": indicator_name,
                    "indicator_code": indicator_code,
                    "value": float(value),
                }
            )
        return result

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(INDICATORS)) as pool:
        futures = [
            pool.submit(fetch_indicator, indicator_name, indicator_code)
            for indicator_name, indicator_code in INDICATORS.items()
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    if not rows:
        raise RuntimeError("The World Bank API returned no observations.")
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["country_code", "year", "indicator_code"]
    )
    frame = frame.sort_values(["indicator", "country", "year"]).reset_index(drop=True)
    return frame


def refresh_data(output_path: Path = DATA_PATH) -> Path:
    """Download the latest available observations and replace the CSV snapshot."""
    frame = fetch_world_bank()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def load_data() -> pd.DataFrame:
    """Load the last snapshot, refreshing automatically if one is not present."""
    if not DATA_PATH.exists():
        refresh_data()
    frame = pd.read_csv(DATA_PATH)
    required = {"country_code", "country", "year", "indicator", "indicator_code", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame
