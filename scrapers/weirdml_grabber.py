import os
from datetime import datetime

import requests


WEIRDML_URL = "https://htihle.github.io/data/weirdml_data.csv"
OUT_DIR = "benchmarks"


def today_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def fetch_csv(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def save_csv(csv_text: str):
    ensure_dir(OUT_DIR)
    outpath = os.path.join(OUT_DIR, f"weirdml_{today_stamp()}.csv")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(csv_text)
    print(f"Saved weirdml CSV to {outpath}")


def main():
    csv_text = fetch_csv(WEIRDML_URL)
    save_csv(csv_text)


if __name__ == "__main__":
    main()
