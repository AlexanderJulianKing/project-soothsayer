import requests
import re
import os
import pandas as pd
from datetime import datetime

def extract_table(markdown):
    """
    Extracts a Markdown table from the provided content.
    Returns a pandas DataFrame.
    """
    lines = markdown.splitlines()
    # Select lines that start and end with a pipe character
    table_lines = [line.strip() for line in lines if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(table_lines) < 3:
        raise ValueError("No valid table found in the Markdown content")
    
    # First row is header, second is delimiter; skip the delimiter row.
    header = [col.strip() for col in table_lines[0].strip("|").split("|")]
    data_rows = []
    for line in table_lines[2:]:
        row = [col.strip() for col in line.strip("|").split("|")]
        if len(row) == len(header):
            data_rows.append(row)
        else:
            print("Warning: Skipping row due to column mismatch:", row)
    return pd.DataFrame(data_rows, columns=header)


def extract_first_table(markdown):
    """
    Extracts the first Markdown table from the provided content.
    Returns a pandas DataFrame.
    """
    lines = markdown.splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        # Check if line appears to be part of a Markdown table.
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
            in_table = True
        else:
            # If we already started capturing table lines and encounter a non-table line, break.
            if in_table:
                break

    if len(table_lines) < 3:
        raise ValueError("No valid table found in the Markdown content")
    
    # The first row is header, second is delimiter; skip the delimiter row.
    header = [col.strip() for col in table_lines[0].strip("|").split("|")]
    data_rows = []
    for line in table_lines[2:]:
        row = [col.strip() for col in line.strip("|").split("|")]
        if len(row) == len(header):
            data_rows.append(row)
        else:
            print("Warning: Skipping row due to column mismatch:", row)
    
    return pd.DataFrame(data_rows, columns=header)

def extract_latest_update(markdown):
    """
    Extracts the latest update date from the "## Updates and Other Benchmarks" section.
    The date should be in the format "Mon DD, YYYY" (e.g., "Feb 27, 2025").
    Returns a datetime object or None if no date is found.
    """
    match = re.search(r"## Updates and Other Benchmarks(.*?)(\n##|\Z)", markdown, re.DOTALL)
    if not match:
        return None
    section = match.group(1)
    # Find all dates in the section
    date_strings = re.findall(r"([A-Z][a-z]{2} \d{1,2}, \d{4})", section)
    dates = []
    for ds in date_strings:
        try:
            dt = datetime.strptime(ds, "%b %d, %Y")
            dates.append(dt)
        except ValueError:
            print(f"Skipping unrecognized date format: {ds}")
    return max(dates) if dates else None

def prefix_columns(df, prefix):
    """
    Prefixes all columns of the dataframe except for 'Model' with the provided prefix.
    """
    new_columns = {}
    for col in df.columns:
        if col != "Model":
            new_columns[col] = f"{prefix}_{col}"
    return df.rename(columns=new_columns)

# URL for the NYT Connections README.md — the ONLY Lechmazur board we keep.
# Dropped 2026-05-28 (confab/gen/elim/step/writing): stale (last pull Dec-2025)
# and low-coverage. nyt-connections is the comprehensive (81 models), actively
# maintained one — see docs/FINDINGS.md.
url_nytcon = "https://raw.githubusercontent.com/lechmazur/nyt-connections/refs/heads/master/README.md"

# Download the README.md file.
resp_nytcon = requests.get(url_nytcon)
if resp_nytcon.status_code != 200:
    raise Exception(f"Failed to download nyt-connections README.md: {resp_nytcon.status_code}")
md_nytcon = resp_nytcon.text

# Extract the table.
df_nytcon = extract_first_table(md_nytcon)


# Normalize the Model names by removing a trailing asterisk, if present.
def normalize_model_names(df):
    if 'Model' in df.columns:
        df["Model"] = df["Model"].str.replace(r'\*$', '', regex=True)
    else:
        df["Model"] = df["LLM"].str.replace(r'\*$', '', regex=True)
        df = df.drop('LLM', axis=1)
    df["Model"] = df["Model"].str.replace("o3-mini (high reasoning)", 'o3-mini-high')
    df["Model"] = df["Model"].str.replace("o3-mini (medium reasoning)", 'o3-mini')
    df["Model"] = df["Model"].str.replace("DeepSeek V3", 'DeepSeek-V3')
    df["Model"] = df["Model"].str.replace("Gemini 2.5 Pro Exp 03-24", 'Gemini 2.5 Pro Exp 03-25')
    df["Model"] = df["Model"].str.replace("Gemini 2.0 Flash Think Exp 01-21", 'Gemini 2.0 Flash Thinking Exp 01-21')
    df["Model"] = df["Model"].str.replace("Gemini 2.0 Flash Think Exp Old", 'Gemini 2.0 Flash Thinking Exp Old')
    df["Model"] = df["Model"].str.replace("o1 (medium reasoning)", 'o1')
    df["Model"] = df["Model"].str.replace("Gemini 2.5 Pro Preview 03-25", 'Gemini 2.5 Pro Exp 03-25')
    df["Model"] = df["Model"].str.replace("Grok 3 Beta (no reasoning)", 'Grok 3 Beta (No reasoning)')
    df["Model"] = df["Model"].str.replace("Grok 3 Mini Beta (high)", 'Grok 3 Mini Beta (High)')
    df["Model"] = df["Model"].str.replace("Grok 3 Mini Beta (low)", 'Grok 3 Mini Beta (Low)')
    df["Model"] = df["Model"].str.replace("Gemini 2.5 Flash Preview 24K", 'Gemini 2.5 Flash Preview (24K)')
    df["Model"] = df["Model"].str.replace("Gemini 2.5 Flash Preview (24k)", 'Gemini 2.5 Flash Preview (24K)')
    df["Model"] = df["Model"].str.replace("Qwen3 235B A22B", 'Qwen 3 235B A22B')
    df["Model"] = df["Model"].str.replace("Qwen3 30B A3B", 'Qwen 3 30B A3B')
    df["Model"] = df["Model"].str.replace("Gemini 2.5 Pro Preview (2025-05-06)", 'Gemini-2.5-Pro-Preview-05-06')
    df["Model"] = df["Model"].str.replace("Gemini 2.5 Pro Preview 06-05", 'Gemini 2.5 Pro')
    df["Model"] = df["Model"].str.replace("Grok 3 Mini Beta (high reasoning)", 'Grok 3 Mini Beta (High)')
    df["Model"] = df["Model"].str.replace("GPT-OSS-120B (medium reasoning)", 'GPT-OSS-120B')

    # df["Model"] = df["Model"].str.replace("Claude Sonnet 4 Thinking 64K", 'Claude Sonnet 4 Thinking 16K')
    return df

df_nytcon = normalize_model_names(df_nytcon)


# Prefix columns (except 'Model').
df_nytcon = prefix_columns(df_nytcon, "nytcon")

df_combined = df_nytcon
print(df_combined.columns)
# Extract the latest update date from the nyt-connections README.
date_nytcon = extract_latest_update(md_nytcon)

dates = [d for d in [date_nytcon] if d is not None]
latest_date = max(dates) if dates else None

# Format the date as YYYYMMDD or use "unknown" if not found.
date_str = latest_date.strftime("%Y%m%d") if latest_date else "unknown"
if date_str == 'unknown':
    date_str = datetime.now().strftime("%Y%m%d")

# Construct the CSV filename.
csv_filename = f"benchmarks/lechmazur_combined_{date_str}.csv"
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

# Save the combined table to CSV.
df_combined.to_csv(csv_filename, index=False)
print(f"Combined table saved to {csv_filename}")