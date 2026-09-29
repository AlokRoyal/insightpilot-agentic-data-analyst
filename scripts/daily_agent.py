"""Generate one reviewable, isolated insight plug-in from the enhancement queue."""

from __future__ import annotations

import ast
import json
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "config" / "daily_enhancements.json"
OUTPUT_DIR = ROOT / "src" / "insight_pilot" / "extensions"


def next_task(today: date | None = None) -> dict[str, str]:
    tasks = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    if not tasks:
        raise ValueError("The daily enhancement queue is empty.")
    current_day = today or date.today()
    day_number = current_day.toordinal()
    task = dict(tasks[day_number % len(tasks)])
    task["feature"] = task["slug"]
    task["slug"] = f"{task['slug']}_{current_day:%Y%m%d}"
    task["purpose"] = f"{task['purpose']} (daily iteration dated {current_day.isoformat()})"
    return task


def validate_source(source: str) -> None:
    """Reject invalid Python and imports/calls with filesystem or process access."""
    tree = ast.parse(source)
    allowed_imports = {"pandas", "statistics", "math", "collections", "typing"}
    allowed_project_import = "insight_pilot.extensions.daily_features"
    forbidden_calls = {"exec", "eval", "compile", "__import__", "open", "input"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name.split(".")[0] not in allowed_imports and item.name != allowed_project_import:
                    raise ValueError(f"Import is not allowed in a generated plug-in: {item.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] not in allowed_imports and node.module != allowed_project_import:
                raise ValueError(f"Import is not allowed in a generated plug-in: {node.module}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            raise ValueError(f"Call is not allowed in a generated plug-in: {node.func.id}")
    run_functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"]
    if len(run_functions) != 1:
        raise ValueError("Generated plug-in must define exactly one top-level run(frame) function.")


def main() -> None:
    task = next_task()
    target = OUTPUT_DIR / f"{task['slug']}.py"
    if target.exists():
        print(f"Already implemented: {target.relative_to(ROOT)}")
        return
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            instructions=(
                "Write a small, deterministic Python analytics extension for a portfolio application. "
                "Return ONLY the complete Python source code. It must implement exactly one top-level "
                "function run(frame) -> dict and use only pandas and Python standard-library imports. "
                "Do not access files, environment variables, network, subprocesses, eval, exec, or the "
                "OpenAI API. Validate columns and return JSON-serializable numbers, strings, lists, and dicts. "
                "Do not claim correlation is causation; document limitations in the returned dict."
            ),
            input=(
                f"Extension name: {task['slug']}\nPurpose: {task['purpose']}\n"
                "Dataset columns are country_code, country, year, indicator, indicator_code, value."
            ),
            max_output_tokens=1800,
        )
        source = (response.output_text or "").strip()
        if source.startswith("```"):
            source = source.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        mode = "AI-generated"
    else:
        source = (
            '"""Curated daily analytics extension generated without a cloud API key."""\n\n'
            "from insight_pilot.extensions.daily_features import run_feature\n\n\n"
            f"def run(frame):\n    return run_feature(frame, {task['feature']!r})\n"
        )
        mode = "curated no-key"
    validate_source(source)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(source + "\n", encoding="utf-8")
    print(f"Generated {target.relative_to(ROOT)} ({mode}) for {task['purpose']}")


if __name__ == "__main__":
    main()
