"""
Scrape the Yupp text leaderboard and the coding-category leaderboard,
then write a single CSV with columns: model, text_score, coding_score.

Requirements:
  pip install playwright beautifulsoup4
  python -m playwright install chromium
"""

import csv
import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


BASE_URL = "https://yupp.ai/leaderboard/text"
CODING_URL = BASE_URL + "?category_names=coding"
OUT_DIR = Path("benchmarks")
REQUEST_PAUSE = 3.5  # seconds to wait after each navigation (be gentle to the site)
MAX_PAGES = 25  # safety cap in case pagination changes unexpectedly
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def parse_table(html: str):
    """Return headers list and list of row lists for the first table in the HTML."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return None, []
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.tbody.find_all("tr"):
        row = [td.get_text(strip=True) for td in tr.find_all("td")]
        if any(cell for cell in row):  # skip any blank placeholder rows
            rows.append(row)
    return headers, rows


def fetch_leaderboard(context, url: str, label: str):
    headers = None
    all_rows = []
    page_idx = 1
    page = context.new_page()

    try:
        page.goto(url, wait_until="networkidle", timeout=60_000)
        page.wait_for_selector("table tbody tr", timeout=30_000)
        page.wait_for_timeout(int(REQUEST_PAUSE * 1000))
    except PlaywrightTimeoutError:
        print(f"[{label}] Timed out while loading the leaderboard table.")
        page.close()
        return [], []

    while page_idx <= MAX_PAGES:
        current_headers, rows = parse_table(page.content())
        if not rows:
            print(f"[{label}] Page {page_idx}: no rows found, stopping.")
            break

        if headers is None:
            headers = current_headers
        all_rows.extend(rows)
        print(f"[{label}] Captured page {page_idx} with {len(rows)} rows")

        next_btn = page.get_by_role("button", name="Next page")
        if next_btn.is_disabled():
            break

        next_btn.click()
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(int(REQUEST_PAUSE * 1000))
        page_idx += 1

    page.close()
    return headers or [], all_rows


def extract_scores(headers, rows):
    """Map model name -> VIBEScore (string)."""
    header_lookup = [h.lower() for h in headers]
    def find_idx(target):
        for idx, val in enumerate(header_lookup):
            if val == target:
                return idx
        return None

    name_idx = find_idx("name")
    score_idx = find_idx("vibescore")
    if name_idx is None or score_idx is None:
        raise ValueError("Required columns ('Name' and 'VIBEScore') not found.")

    scores = {}
    for row in rows:
        if len(row) <= max(name_idx, score_idx):
            continue
        name = row[name_idx]
        score = row[score_idx]
        if not name:
            continue
        scores[name] = score
    return scores


def save_scores_csv(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    filename = OUT_DIR / f"yupp_text_coding_scores_{date_str}.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "text_score", "coding_score"])
        writer.writerows(rows)
    return filename


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1400, "height": 2000})

        text_headers, text_rows = fetch_leaderboard(context, BASE_URL, "text")
        coding_headers, coding_rows = fetch_leaderboard(context, CODING_URL, "coding")

        browser.close()

    if not text_rows and not coding_rows:
        print("No data scraped; CSV not written.")
        return

    text_scores = extract_scores(text_headers, text_rows) if text_rows else {}
    coding_scores = extract_scores(coding_headers, coding_rows) if coding_rows else {}

    ordered_models = list(text_scores.keys()) + [m for m in coding_scores.keys() if m not in text_scores]
    combined_rows = [[model, text_scores.get(model, ""), coding_scores.get(model, "")] for model in ordered_models]

    csv_path = save_scores_csv(combined_rows)
    print(f"Saved {len(combined_rows)} models to {csv_path}")


if __name__ == "__main__":
    main()
