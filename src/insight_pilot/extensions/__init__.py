"""Plug-in insight modules use a common ``run(frame) -> dict`` interface."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

import pandas as pd


def run_all(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Load each insight plug-in independently so one failure cannot hide others."""
    results: dict[str, dict[str, Any]] = {}
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        runner = getattr(module, "run", None)
        if not callable(runner):
            continue
        try:
            results[module_info.name] = runner(frame)
        except Exception as exc:
            results[module_info.name] = {"error": str(exc)}
    return results
