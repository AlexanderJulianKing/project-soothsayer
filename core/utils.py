"""Shared utility functions used across all benchmarks.

Consolidates duplicated logic for file discovery, model loading, and
JSON/flag parsing that was previously copy-pasted into 5-8 files each.
"""

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional


def get_latest_file(pattern: str) -> str:
    """Get the most recent file matching a date pattern (YYYYMMDD).

    Scans filenames for date strings and returns the one with the latest
    date.  Falls back to the first file found if no date pattern is present.

    Raises:
        ValueError: If no files match *pattern* at all.
    """
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    latest_file = None
    latest_date_str = ""

    for f_path in files:
        match = re.search(r'(\d{4}[-/]?\d{2}[-/]?\d{2})', os.path.basename(f_path))
        if match:
            date_str = re.sub(r'[-/]', '', match.group(1))
            if date_str > latest_date_str:
                latest_date_str = date_str
                latest_file = f_path

    if latest_file is None:
        return files[0]
    return latest_file


def discover_openbench_csv(script_dir: str) -> str:
    """Find the latest openbench_*.csv using standard search paths.

    Looks in (in order):
      1. <project_root>/benchmark_combiner/benchmarks/openbench_*.csv
      2. openbench_*.csv in the current working directory

    Args:
        script_dir: The directory of the calling script (used to resolve
            the project root via ``..``).

    Returns:
        Absolute path to the latest openbench CSV.

    Raises:
        ValueError: If no openbench CSV is found in any search path.
    """
    patterns = [
        os.path.join(script_dir, 'benchmark_combiner', 'benchmarks', 'openbench_*.csv'),
        os.path.join(script_dir, '..', 'benchmark_combiner', 'benchmarks', 'openbench_*.csv'),
        'openbench_*.csv',
    ]
    for pat in patterns:
        try:
            return get_latest_file(pat)
        except ValueError:
            continue
    raise ValueError("No openbench CSV found in benchmark_combiner/benchmarks/ or local directory.")


def load_models(csv_path: str) -> List[Dict[str, Any]]:
    """Load model data from an OpenBench CSV.

    Returns a list of dicts with keys ``name``, ``id``, and ``Reasoning``.
    Models without an ``openbench_id`` are dropped.

    The ``Reasoning`` column is always returned with a capital-R key,
    normalizing soothsayer_style's lowercase variant.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if 'Model' not in df.columns or 'openbench_id' not in df.columns:
        raise ValueError("CSV must contain 'Model' and 'openbench_id' columns.")

    df.dropna(subset=['openbench_id'], inplace=True)

    # Normalize column names — soothsayer_style used lowercase 'reasoning'
    if 'reasoning' in df.columns and 'Reasoning' not in df.columns:
        df.rename(columns={'reasoning': 'Reasoning'}, inplace=True)

    models = df.rename(columns={'Model': 'name', 'openbench_id': 'id'})
    return models[['name', 'id', 'Reasoning']].to_dict('records')


def normalize_reasoning_flag(value) -> bool:
    """Normalize a reasoning flag from CSV to a Python bool.

    Handles string variants ("true", "1", "yes") and passthrough for bools.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def extract_json_payload(text: str) -> Optional[dict]:
    """Extract a JSON object from LLM response text.

    Handles bare JSON, ```json fenced blocks, and nested braces.
    Returns None if no valid JSON object can be extracted.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None
