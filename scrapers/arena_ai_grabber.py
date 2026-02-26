#!/usr/bin/env python3
"""
Scrape arena.ai leaderboard for LMSYS/LMArena ELO scores.

Primary strategy: HTTP fetch + RSC (React Server Component) Flight data parsing.
Fallback: Playwright headless browser.

Outputs:
  - benchmarks/lmsys_YYYYMMDD.csv   (no style control — raw arena scores)
  - benchmarks/lmarena_YYYYMMDD.csv  (style control ON — length-adjusted scores)
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import re
import sys
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default page has styleControl=true (style-adjusted scores)
ARENA_STYLE_ON_URL = "https://arena.ai/leaderboard/text"
# This URL has styleControl=false (raw scores, no style penalty)
ARENA_NO_STYLE_URL = "https://arena.ai/leaderboard/text/overall-no-style-control"

# Mapping (verified against historical data):
#   lmsys_*.csv  = no style control (raw arena scores)    -> ARENA_NO_STYLE_URL
#   lmarena_*.csv = style control ON (length-adjusted)    -> ARENA_STYLE_ON_URL
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
TIMEOUT = 30
MIN_EXPECTED_MODELS = 200

CSV_HEADER = ["Rank", "Rank Spread", "Model", "Score", "95% CI (\xb1)", "Votes", "Organization", "License"]

BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_combiner", "benchmarks")


# ---------------------------------------------------------------------------
# RSC parsing helpers
# ---------------------------------------------------------------------------
def fetch_html(url: str) -> str:
    """Fetch page HTML with a realistic user-agent."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.text


def extract_rsc_payloads(html: str) -> list[str]:
    """Extract RSC Flight payloads from self.__next_f.push() script tags."""
    # The payloads look like: self.__next_f.push([1,"...escaped string..."])
    pattern = r'self\.__next_f\.push\(\s*\[.*?,\s*"((?:[^"\\]|\\.)*)"\s*\]\s*\)'
    raw_chunks = re.findall(pattern, html, re.DOTALL)

    payloads = []
    for chunk in raw_chunks:
        # Unescape JS string escapes
        try:
            unescaped = chunk.encode("utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            unescaped = chunk.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
        payloads.append(unescaped)
    return payloads


def _recursive_find_entries(obj, depth: int = 0) -> Optional[list[dict]]:
    """Recursively search a parsed JSON structure for the leaderboard entries array."""
    if depth > 20:
        return None

    if isinstance(obj, dict):
        # Check if this dict looks like a model entry
        if "rating" in obj and "modelDisplayName" in obj:
            return [obj]

        # Check if it has an "entries" key with a list value
        if "entries" in obj and isinstance(obj["entries"], list):
            entries = obj["entries"]
            if entries and isinstance(entries[0], dict) and "rating" in entries[0]:
                return entries

        # Recurse into values
        for v in obj.values():
            result = _recursive_find_entries(v, depth + 1)
            if result and len(result) >= MIN_EXPECTED_MODELS:
                return result
    elif isinstance(obj, list):
        # Check if this list IS the entries array
        model_entries = []
        for item in obj:
            if isinstance(item, dict) and "rating" in item and "modelDisplayName" in item:
                model_entries.append(item)
        if len(model_entries) >= MIN_EXPECTED_MODELS:
            return model_entries

        # Recurse into list items
        for item in obj:
            result = _recursive_find_entries(item, depth + 1)
            if result and len(result) >= MIN_EXPECTED_MODELS:
                return result

    return None


def parse_rsc_flight_data(payloads: list[str]) -> list[dict]:
    """Parse RSC Flight chunks to find the leaderboard entries array."""
    all_entries = []

    for payload in payloads:
        # RSC Flight format: each line is "<chunkId>:<data>"
        # Try splitting by lines and parsing each
        lines = payload.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Strip chunk ID prefix (e.g., "b:", "1a:", "0:")
            colon_idx = line.find(":")
            if colon_idx > 0 and colon_idx < 10:
                data_part = line[colon_idx + 1:]
            else:
                data_part = line

            # Try to parse as JSON
            try:
                parsed = json.loads(data_part)
                result = _recursive_find_entries(parsed)
                if result and len(result) > len(all_entries):
                    all_entries = result
            except (json.JSONDecodeError, ValueError):
                pass

        # Also try parsing the entire payload as one JSON blob
        try:
            parsed = json.loads(payload)
            result = _recursive_find_entries(parsed)
            if result and len(result) > len(all_entries):
                all_entries = result
        except (json.JSONDecodeError, ValueError):
            pass

    return all_entries


def parse_entries_from_html(html: str) -> list[dict]:
    """Full pipeline: extract RSC payloads from HTML and parse model entries."""
    # Strategy 1: RSC Flight payloads
    payloads = extract_rsc_payloads(html)
    if payloads:
        entries = parse_rsc_flight_data(payloads)
        if entries and len(entries) >= MIN_EXPECTED_MODELS:
            return entries
        print(f"  RSC parsing found only {len(entries)} entries from {len(payloads)} payloads")

    # Strategy 2: Look for JSON arrays directly in script tags
    # Sometimes data is in <script> tags as plain JSON or __NEXT_DATA__
    json_pattern = r'<script[^>]*>\s*(\{.*?"entries"\s*:\s*\[.*?\].*?\})\s*</script>'
    for match in re.finditer(json_pattern, html, re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            result = _recursive_find_entries(parsed)
            if result and len(result) >= MIN_EXPECTED_MODELS:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Brute-force search for JSON arrays with model data
    # Find anything that looks like [{"rank":..., "rating":...}, ...]
    array_pattern = r'\[\s*\{[^{]*?"rating"\s*:\s*\d+[^{]*?"modelDisplayName"\s*:'
    for match in re.finditer(array_pattern, html):
        start = match.start()
        # Try to find the matching closing bracket
        bracket_depth = 0
        for i in range(start, min(start + 500000, len(html))):
            if html[i] == "[":
                bracket_depth += 1
            elif html[i] == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    try:
                        parsed = json.loads(html[start : i + 1])
                        result = _recursive_find_entries(parsed)
                        if result and len(result) >= MIN_EXPECTED_MODELS:
                            return result
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    return []


# ---------------------------------------------------------------------------
# CSV formatting
# ---------------------------------------------------------------------------
def entries_to_csv_rows(entries: list[dict]) -> list[list[str]]:
    """Convert RSC model entries to CSV rows matching the existing format."""
    rows = []

    for entry in entries:
        rank = entry.get("rank", "")
        rank_upper = entry.get("rankUpper", rank)
        rank_lower = entry.get("rankLower", rank)
        model_name = entry.get("modelDisplayName", "")
        rating = entry.get("rating", 0)
        rating_upper = entry.get("ratingUpper", rating)
        rating_lower = entry.get("ratingLower", rating)
        votes = entry.get("votes", 0)
        organization = entry.get("modelOrganization", "")
        license_str = entry.get("license", "")

        # Skip entries without essential data
        if not model_name or not rating:
            continue

        # Format fields to match existing CSV style
        score = round(rating)
        ci = round((rating_upper - rating_lower) / 2)
        ci_str = f"\xb1{ci}"  # Latin-1 ± character

        # Rank spread: "upper___lower"
        rank_spread = f"{rank_upper}___{rank_lower}"

        # Format votes with commas
        if isinstance(votes, (int, float)):
            votes_str = f"{int(votes):,}"
        else:
            votes_str = str(votes)

        rows.append([
            str(rank),
            rank_spread,
            model_name,
            str(score),
            ci_str,
            votes_str,
            organization,
            license_str,
        ])

    return rows


def write_csv(filepath: str, rows: list[list[str]]) -> None:
    """Write CSV with Latin-1 encoding and CRLF line endings to match existing files."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # csv.writer uses \r\n by default when newline="" (per csv module docs)
    with open(filepath, "w", encoding="latin-1", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Playwright helpers (lazy import)
# ---------------------------------------------------------------------------
def _get_playwright():
    """Lazy import of playwright."""
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Install with: pip install playwright && playwright install chromium")
        sys.exit(1)


def fetch_with_playwright(url: str) -> str:
    """Fetch page HTML using headless Chromium (fallback when RSC parsing fails)."""
    sync_playwright = _get_playwright()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        # Use domcontentloaded — networkidle can hang on long-polling connections
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(8000)  # Wait for RSC hydration
        html = page.content()
        browser.close()
        return html


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_entries(entries: list[dict], label: str) -> bool:
    """Validate extracted entries for sanity."""
    if len(entries) < MIN_EXPECTED_MODELS:
        print(f"  WARN [{label}]: Only {len(entries)} entries (expected >= {MIN_EXPECTED_MODELS})")
        return False

    issues = 0
    for entry in entries:
        name = entry.get("modelDisplayName", "")
        rating = entry.get("rating", 0)
        rank = entry.get("rank", 0)

        if not name:
            issues += 1
            continue
        if not (600 <= rating <= 1800):
            print(f"  WARN [{label}]: Unusual rating {rating} for {name}")
            issues += 1
        if not (1 <= rank <= 500):
            issues += 1

    if issues > 10:
        print(f"  WARN [{label}]: {issues} entries with issues")
        return False

    return True


# ---------------------------------------------------------------------------
# Fetch + parse helpers for each variant
# ---------------------------------------------------------------------------
def _fetch_and_parse(url: str, label: str) -> list[dict]:
    """Fetch HTML from url, parse RSC entries, fall back to Playwright if needed."""
    # Primary: plain HTTP
    print(f"  [{label}] Fetching via HTTP...")
    try:
        html = fetch_html(url)
        print(f"  [{label}] Fetched {len(html):,} bytes")
        entries = parse_entries_from_html(html)
        print(f"  [{label}] Parsed {len(entries)} model entries")
        if len(entries) >= MIN_EXPECTED_MODELS:
            return entries
    except Exception as e:
        print(f"  [{label}] HTTP fetch failed: {e}")
        entries = []

    # Fallback: Playwright
    print(f"  [{label}] RSC parsing insufficient ({len(entries)} models). "
          f"Trying Playwright fallback...")
    try:
        pw_html = fetch_with_playwright(url)
        entries = parse_entries_from_html(pw_html)
        print(f"  [{label}] Playwright found {len(entries)} model entries")
    except Exception as e:
        print(f"  [{label}] Playwright fallback also failed: {e}")

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    lmsys_path = os.path.join(BENCHMARKS_DIR, f"lmsys_{date_str}.csv")
    lmarena_path = os.path.join(BENCHMARKS_DIR, f"lmarena_{date_str}.csv")

    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    print("=" * 60)
    print("Arena.ai Leaderboard Scraper")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Fetch lmsys data (NO style control = raw arena scores)
    # URL: /leaderboard/text/overall-no-style-control  (styleControl: false)
    # -----------------------------------------------------------------------
    print("\n[1/4] Fetching lmsys data (no style control)...")
    entries_lmsys = _fetch_and_parse(ARENA_NO_STYLE_URL, "lmsys")

    if not entries_lmsys:
        print("\nERROR: Could not extract lmsys entries. Aborting.")
        sys.exit(1)

    first = entries_lmsys[0]
    print(f"  Top model: {first.get('modelDisplayName')} "
          f"(rating={first.get('rating')}, rank={first.get('rank')})")

    # -----------------------------------------------------------------------
    # Step 2: Write lmsys CSV
    # -----------------------------------------------------------------------
    validate_entries(entries_lmsys, "lmsys")
    rows_lmsys = entries_to_csv_rows(entries_lmsys)
    write_csv(lmsys_path, rows_lmsys)
    print(f"\n[2/4] Wrote {lmsys_path} ({len(rows_lmsys)} data rows)")

    # -----------------------------------------------------------------------
    # Step 3: Fetch lmarena data (style control ON = length-adjusted)
    # URL: /leaderboard/text  (styleControl: true, the default)
    # -----------------------------------------------------------------------
    print("\n[3/4] Fetching lmarena data (style control ON)...")
    entries_lmarena = _fetch_and_parse(ARENA_STYLE_ON_URL, "lmarena")

    if not entries_lmarena:
        print("  WARNING: Could not extract lmarena entries. "
              "Falling back to lmsys data.")
        entries_lmarena = entries_lmsys

    first = entries_lmarena[0]
    print(f"  Top model: {first.get('modelDisplayName')} "
          f"(rating={first.get('rating')}, rank={first.get('rank')})")

    # -----------------------------------------------------------------------
    # Step 4: Write lmarena CSV
    # -----------------------------------------------------------------------
    validate_entries(entries_lmarena, "lmarena")
    rows_lmarena = entries_to_csv_rows(entries_lmarena)
    write_csv(lmarena_path, rows_lmarena)
    print(f"\n[4/4] Wrote {lmarena_path} ({len(rows_lmarena)} data rows)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  lmsys  (no style ctrl): {len(rows_lmsys):>3d} models -> {lmsys_path}")
    print(f"  lmarena (style ctrl ON): {len(rows_lmarena):>3d} models -> {lmarena_path}")

    # Sanity check: scores should differ between the two variants
    if entries_lmsys and entries_lmarena and entries_lmsys is not entries_lmarena:
        # Find a common model to compare
        lmsys_top = {e["modelDisplayName"]: e["rating"] for e in entries_lmsys[:5]}
        lmarena_top = {e["modelDisplayName"]: e["rating"] for e in entries_lmarena[:5]}
        common = set(lmsys_top) & set(lmarena_top)
        if common:
            model = next(iter(common))
            diff = abs(lmsys_top[model] - lmarena_top[model])
            if diff < 0.01:
                print(f"  NOTE: Scores appear identical — style control toggle "
                      f"may not have worked.")
            else:
                print(f"  Score diff for {model}: "
                      f"{lmsys_top[model]:.1f} vs {lmarena_top[model]:.1f} "
                      f"(delta={diff:.1f}) — looks good!")

    print("\nDone!")


if __name__ == "__main__":
    main()
