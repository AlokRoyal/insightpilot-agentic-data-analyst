"""Dataset quality checks for missing values, duplicate keys, and outliers."""

from __future__ import annotations

import pandas as pd


def run(frame: pd.DataFrame) -> dict[str, object]:
    key_columns = [column for column in ("country_code", "year", "indicator_code") if column in frame]
    duplicates = int(frame.duplicated(key_columns).sum()) if len(key_columns) == 3 else None
    value = pd.to_numeric(frame.get("value"), errors="coerce")
    numeric = value.dropna()
    if numeric.empty:
        outliers = 0
    else:
        q1, q3 = numeric.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
    return {
        "rows": int(len(frame)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_observation_keys": duplicates,
        "iqr_outlier_values": outliers,
        "note": "IQR flags are review prompts, not proof of bad data.",
    }
