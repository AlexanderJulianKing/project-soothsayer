#!/usr/bin/env python3
"""
Scrape Artificial Analysis evaluation pages for three benchmarks missing
from the v2 API: GDPval-AA, AA-Omniscience, and CritPt.

All three are Next.js pages with data embedded in RSC Flight payloads
(self.__next_f.push()).  The extraction pattern is adapted from
arena_ai_grabber.py.

Outputs:
  - benchmarks/aa_gdpval_YYYYMMDD.csv
  - benchmarks/aa_omniscience_YYYYMMDD.csv
  - benchmarks/aa_critpt_YYYYMMDD.csv
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://artificialanalysis.ai"
GDPVAL_URL = f"{BASE_URL}/evaluations/gdpval-aa"
OMNISCIENCE_URL = f"{BASE_URL}/evaluations/omniscience"
CRITPT_URL = f"{BASE_URL}/evaluations/critpt"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
TIMEOUT = 30

BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_combiner", "benchmarks"
)

# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def fetch_html(url: str) -> str:
    """Fetch page HTML with a realistic user-agent."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# RSC payload extraction (shared with arena_ai_grabber.py pattern)
# ---------------------------------------------------------------------------

def extract_rsc_payloads(html: str) -> list[str]:
    """Extract RSC Flight payloads from self.__next_f.push() script tags."""
    pattern = r'self\.__next_f\.push\(\s*\[.*?,\s*"((?:[^"\\]|\\.)*)"\s*\]\s*\)'
    raw_chunks = re.findall(pattern, html, re.DOTALL)

    payloads: list[str] = []
    for chunk in raw_chunks:
        try:
            unescaped = chunk.encode("utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            unescaped = (
                chunk.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        payloads.append(unescaped)
    return payloads


def _parse_rsc_lines(payloads: list[str]):
    """Yield parsed JSON objects from RSC Flight chunk lines."""
    for payload in payloads:
        lines = payload.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip chunk ID prefix (e.g. "b:", "1a:", "0:")
            colon_idx = line.find(":")
            if 0 < colon_idx < 10:
                data_part = line[colon_idx + 1:]
            else:
                data_part = line
            try:
                yield json.loads(data_part)
            except (json.JSONDecodeError, ValueError):
                pass


# ---------------------------------------------------------------------------
# defaultData finder — recursively locates the largest defaultData array
# ---------------------------------------------------------------------------

def _find_default_data_arrays(obj: Any, depth: int = 0) -> list[list[dict]]:
    """Recursively find all defaultData arrays in a parsed RSC object."""
    results: list[list[dict]] = []
    if depth > 20:
        return results
    if isinstance(obj, dict):
        if "defaultData" in obj and isinstance(obj["defaultData"], list):
            results.append(obj["defaultData"])
        for v in obj.values():
            results.extend(_find_default_data_arrays(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_find_default_data_arrays(item, depth + 1))
    return results


def find_default_data(payloads: list[str], min_entries: int = 10) -> Optional[list[dict]]:
    """Parse RSC payloads and return the largest defaultData array found."""
    best: Optional[list[dict]] = None
    for parsed in _parse_rsc_lines(payloads):
        for dd in _find_default_data_arrays(parsed):
            if len(dd) >= min_entries:
                if best is None or len(dd) > len(best):
                    best = dd
    return best


# ---------------------------------------------------------------------------
# Playwright fallback (lazy import)
# ---------------------------------------------------------------------------

def _get_playwright():
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError:
        return None


def fetch_with_playwright(url: str) -> str:
    """Fetch page HTML using headless Chromium."""
    sync_playwright = _get_playwright()
    if sync_playwright is None:
        raise RuntimeError("playwright not installed")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(8000)
        html = page.content()
        browser.close()
        return html


def fetch_and_extract(url: str, label: str, min_entries: int = 10) -> Optional[list[dict]]:
    """Fetch URL, extract RSC payloads, find defaultData. Playwright fallback."""
    print(f"  [{label}] Fetching via HTTP...")
    try:
        html = fetch_html(url)
        print(f"  [{label}] Fetched {len(html):,} bytes")
        payloads = extract_rsc_payloads(html)
        print(f"  [{label}] Extracted {len(payloads)} RSC payloads")
        dd = find_default_data(payloads, min_entries=min_entries)
        if dd:
            print(f"  [{label}] Found defaultData with {len(dd)} entries")
            return dd
        print(f"  [{label}] No sufficient defaultData in HTTP response")
    except Exception as e:
        print(f"  [{label}] HTTP fetch failed: {e}")

    # Playwright fallback
    print(f"  [{label}] Trying Playwright fallback...")
    try:
        html = fetch_with_playwright(url)
        payloads = extract_rsc_payloads(html)
        dd = find_default_data(payloads, min_entries=min_entries)
        if dd:
            print(f"  [{label}] Playwright found defaultData with {len(dd)} entries")
            return dd
        print(f"  [{label}] Playwright also found no sufficient defaultData")
    except Exception as e:
        print(f"  [{label}] Playwright fallback failed: {e}")

    return None


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def write_csv(filepath: str, header: list[str], rows: list[list]) -> None:
    """Write a CSV file with UTF-8 encoding."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# GDPval-AA scraper
# ---------------------------------------------------------------------------

GDPVAL_HEADER = [
    "Model", "Creator", "ELO", "CI_Lower", "CI_Upper", "Rank", "isReasoning",
]


def scrape_gdpval(data: list[dict]) -> list[list]:
    """Extract GDPval-AA leaderboard rows from defaultData."""
    scored: list[dict] = []
    for entry in data:
        sg = entry.get("safeGdpval")
        if not sg or not sg.get("elo"):
            continue
        scored.append(entry)

    # Sort by ELO descending, assign rank
    scored.sort(key=lambda e: e["safeGdpval"]["elo"], reverse=True)

    rows: list[list] = []
    for rank, entry in enumerate(scored, 1):
        sg = entry["safeGdpval"]
        name = entry.get("name") or entry.get("short_name", "")
        creator = entry.get("model_creators", {}).get("name", "")
        elo = round(sg["elo"], 2)
        ci_lower = round(sg.get("lower95ci", 0), 2)
        ci_upper = round(sg.get("upper95ci", 0), 2)
        is_reasoning = entry.get("reasoning_model", False)
        rows.append([name, creator, elo, ci_lower, ci_upper, rank, is_reasoning])

    return rows


# ---------------------------------------------------------------------------
# Omniscience scraper
# ---------------------------------------------------------------------------

OMNISCIENCE_HEADER = [
    "Model", "Creator", "OmniscienceIndex", "OmniscienceAccuracy",
    "OmniscienceHallucinationRate", "isReasoning",
]


def scrape_omniscience(data: list[dict]) -> list[list]:
    """Extract Omniscience leaderboard rows from defaultData."""
    rows: list[list] = []
    for entry in data:
        omni = entry.get("omniscience")
        if omni is None:
            continue

        name = entry.get("name") or entry.get("short_name", "")
        creator = entry.get("model_creators", {}).get("name", "")
        is_reasoning = entry.get("reasoning_model", False)

        # Extract accuracy and hallucination rate from breakdown
        breakdown = entry.get("omniscience_breakdown") or {}
        total = breakdown.get("total", {})
        accuracy = total.get("accuracy")
        hallucination_rate = total.get("hallucination_rate")

        rows.append([
            name,
            creator,
            round(omni, 3),
            round(accuracy, 4) if accuracy is not None else "",
            round(hallucination_rate, 4) if hallucination_rate is not None else "",
            is_reasoning,
        ])

    # Sort by OmniscienceIndex descending
    rows.sort(key=lambda r: r[2] if isinstance(r[2], (int, float)) else -999, reverse=True)
    return rows


# ---------------------------------------------------------------------------
# CritPt scraper
# ---------------------------------------------------------------------------

CRITPT_HEADER = [
    "Model", "Creator", "CritPtScore", "Rank", "isReasoning",
]


def scrape_critpt(data: list[dict]) -> list[list]:
    """Extract CritPt leaderboard rows from defaultData."""
    scored: list[dict] = []
    for entry in data:
        cp = entry.get("critpt")
        if cp is None or cp <= 0:
            continue
        scored.append(entry)

    # Sort by CritPt score descending, assign rank
    scored.sort(key=lambda e: e["critpt"], reverse=True)

    rows: list[list] = []
    for rank, entry in enumerate(scored, 1):
        name = entry.get("name") or entry.get("short_name", "")
        creator = entry.get("model_creators", {}).get("name", "")
        cp = round(entry["critpt"], 4)
        is_reasoning = entry.get("reasoning_model", False)
        rows.append([name, creator, cp, rank, is_reasoning])

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    gdpval_path = os.path.join(BENCHMARKS_DIR, f"aa_gdpval_{date_str}.csv")
    omniscience_path = os.path.join(BENCHMARKS_DIR, f"aa_omniscience_{date_str}.csv")
    critpt_path = os.path.join(BENCHMARKS_DIR, f"aa_critpt_{date_str}.csv")

    print("=" * 60)
    print("AA Evaluations Scraper (GDPval-AA, Omniscience, CritPt)")
    print("=" * 60)

    results: dict[str, int] = {}

    # -------------------------------------------------------------------
    # [1/3] CritPt  (fetched first — its defaultData contains the full
    #                model list with omniscience + critpt + gdpval fields)
    # -------------------------------------------------------------------
    print("\n[1/3] CritPt...")
    full_data = fetch_and_extract(CRITPT_URL, "CritPt", min_entries=50)
    if full_data:
        rows = scrape_critpt(full_data)
        if rows:
            write_csv(critpt_path, CRITPT_HEADER, rows)
            results["CritPt"] = len(rows)
            print(f"  Wrote {len(rows)} models -> {critpt_path}")
            print(f"  Top: {rows[0][0]} (Score={rows[0][2]})")
        else:
            print("  ERROR: No valid CritPt entries extracted")
    else:
        print("  ERROR: Could not fetch CritPt data")

    # -------------------------------------------------------------------
    # [2/3] Omniscience  (extracted from CritPt page's full dataset,
    #                     which has ~268 models vs only ~31 on the
    #                     Omniscience page's own defaultData)
    # -------------------------------------------------------------------
    print("\n[2/3] Omniscience...")
    omni_source = full_data  # prefer full dataset from CritPt page
    if not omni_source:
        # fall back to the Omniscience page itself
        omni_source = fetch_and_extract(OMNISCIENCE_URL, "Omniscience", min_entries=10)
    if omni_source:
        rows = scrape_omniscience(omni_source)
        if rows:
            write_csv(omniscience_path, OMNISCIENCE_HEADER, rows)
            results["Omniscience"] = len(rows)
            print(f"  Wrote {len(rows)} models -> {omniscience_path}")
            print(f"  Top: {rows[0][0]} (Index={rows[0][2]})")
        else:
            print("  ERROR: No valid Omniscience entries extracted")
    else:
        print("  ERROR: Could not fetch Omniscience data")

    # -------------------------------------------------------------------
    # [3/3] GDPval-AA  (needs its own page — uses safeGdpval with ELO/CI
    #                   which is only on the GDPval page's defaultData)
    # -------------------------------------------------------------------
    print("\n[3/3] GDPval-AA...")
    gdpval_data = fetch_and_extract(GDPVAL_URL, "GDPval-AA", min_entries=50)
    if gdpval_data:
        rows = scrape_gdpval(gdpval_data)
        if rows:
            write_csv(gdpval_path, GDPVAL_HEADER, rows)
            results["GDPval-AA"] = len(rows)
            print(f"  Wrote {len(rows)} models -> {gdpval_path}")
            print(f"  Top: {rows[0][0]} (ELO={rows[0][2]})")
        else:
            print("  ERROR: No valid GDPval entries extracted")
    else:
        print("  ERROR: Could not fetch CritPt data")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for bench, count in results.items():
        print(f"  {bench:15s}: {count:>3d} models")
    if len(results) < 3:
        failed = {"GDPval-AA", "Omniscience", "CritPt"} - set(results)
        for f in failed:
            print(f"  {f:15s}: FAILED")
    print("\nDone!")

    return 0 if len(results) == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
