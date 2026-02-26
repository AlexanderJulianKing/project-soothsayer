#!/usr/bin/env python3
"""
Fetch LLMs from Artificial Analysis' LLMs endpoint and write a CSV containing:
- id
- name
- all pricing fields (flattened as pricing_<key>)
- all evaluation fields (flattened as eval_<key>)

Usage:
  python aa_models_grabber.py --api-key YOUR_KEY [-o llms.csv]

Alternatively, set the environment variable ARTIFICIAL_ANALYSIS_API_KEY and omit --api-key.
"""
import argparse
import csv
import os
import time
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
import sys
from typing import Dict, List, Set, Any, Tuple, Optional

import requests


ENDPOINT = "https://artificialanalysis.ai/api/v2/data/llms/models"


def fetch_models(
    api_key: str,
    endpoint: str = ENDPOINT,
    timeout: float = 120.0,
    retries: int = 2,
    backoff_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    headers = {"x-api-key": api_key}
    last_error: Optional[Exception] = None
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(endpoint, headers=headers, timeout=(5.0, timeout))
            break
        except requests.RequestException as e:
            last_error = e
            if attempt == attempts:
                print(f"Network error: {e}", file=sys.stderr)
                sys.exit(2)

            wait_for = backoff_seconds * attempt
            print(
                f"Network error (attempt {attempt}/{attempts}): {e}. "
                f"Retrying in {wait_for:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait_for)

    if resp.status_code == 401:
        print("Error: Unauthorized (401). Check your API key.", file=sys.stderr)
        sys.exit(3)
    if resp.status_code == 429:
        print("Error: Rate limit exceeded (429). Try again later.", file=sys.stderr)
        sys.exit(4)
    if resp.status_code >= 400:
        print(f"HTTP error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(5)

    try:
        payload = resp.json()
    except ValueError:
        print("Error: Response was not valid JSON.", file=sys.stderr)
        sys.exit(6)

    # Expected shape: { "status": 200, "data": [ ...models... ] }
    data = payload.get("data")
    if not isinstance(data, list):
        print("Error: Unexpected API response format; 'data' is missing or not a list.", file=sys.stderr)
        sys.exit(7)

    return data


def collect_column_keys(models: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    pricing_keys: Set[str] = set()
    eval_keys: Set[str] = set()

    for m in models:
        pricing = m.get("pricing") or {}
        evaluations = m.get("evaluations") or {}

        if isinstance(pricing, dict):
            pricing_keys.update(pricing.keys())
        if isinstance(evaluations, dict):
            eval_keys.update(evaluations.keys())

    # Stable ordering for CSV columns
    pricing_cols = sorted(pricing_keys)
    eval_cols = sorted(eval_keys)
    return pricing_cols, eval_cols


def model_to_row(m: Dict[str, Any], pricing_cols: List[str], eval_cols: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": m.get("id"),
        "name": m.get("name"),
    }

    pricing = m.get("pricing") or {}
    evaluations = m.get("evaluations") or {}

    for k in pricing_cols:
        row[f"pricing_{k}"] = pricing.get(k)
    for k in eval_cols:
        row[f"eval_{k}"] = evaluations.get(k)

    return row


def write_csv(models: List[Dict[str, Any]], out_path: str) -> int:
    pricing_cols, eval_cols = collect_column_keys(models)

    fieldnames = ["id", "name"]
    fieldnames += [f"pricing_{k}" for k in pricing_cols]
    fieldnames += [f"eval_{k}" for k in eval_cols]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in models:
            row = model_to_row(m, pricing_cols, eval_cols)
            writer.writerow(row)

    return len(models)


def main():
    parser = argparse.ArgumentParser(description="Export Artificial Analysis LLMs (id, name, pricing, evaluations) to CSV.")
    parser.add_argument(
        "-k",
        "--api-key",
        #default=os.getenv("ARTIFICIAL_ANALYSIS_API_KEY"),
        default="aa_pLhKRQgkzqvHEXCYxNuFkGlTiRhcqbRp",
        help="API key for Artificial Analysis (or set ARTIFICIAL_ANALYSIS_API_KEY env var).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: benchmarks/artificialanalysis_YYYYMMDD.csv)",
    )
    parser.add_argument(
        "--tz",
        default="America/Los_Angeles",
        help="IANA timezone for date stamp (default: America/Los_Angeles)",
    )
    parser.add_argument("--endpoint", default=ENDPOINT, help="Override the LLMs endpoint URL if needed.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Read timeout in seconds for the API request (default: 120).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries for network errors (default: 2).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Seconds to wait between retries, multiplied by the attempt number (default: 2).",
    )
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        print("Error: Provide an API key via --api-key or ARTIFICIAL_ANALYSIS_API_KEY.", file=sys.stderr)
        sys.exit(1)

    timeout = args.timeout if args.timeout and args.timeout > 0 else 120.0
    retries = max(0, args.retries)
    backoff = max(0.0, args.retry_backoff)

    models = fetch_models(
        api_key=api_key,
        endpoint=args.endpoint,
        timeout=timeout,
        retries=retries,
        backoff_seconds=backoff,
    )
    # Determine output path
    if args.tz and ZoneInfo:
        now = datetime.now(ZoneInfo(args.tz))
    else:
        now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    out_path = args.output or f"benchmarks/artificialanalysis_{date_str}.csv"
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    count = write_csv(models, out_path)
    print(f"Wrote {count} models to {out_path}")


if __name__ == "__main__":
    main()
