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

# URLs for the README.md files.
url_confab = "https://raw.githubusercontent.com/lechmazur/confabulations/refs/heads/master/README.md"
url_gen    = "https://raw.githubusercontent.com/lechmazur/generalization/refs/heads/main/README.md"
url_elim   = "https://raw.githubusercontent.com/lechmazur/elimination_game/refs/heads/main/README.md"
url_step   = "https://raw.githubusercontent.com/lechmazur/step_game/refs/heads/main/README.md"
url_nytcon = "https://raw.githubusercontent.com/lechmazur/nyt-connections/refs/heads/master/README.md"
# url_writing= "https://raw.githubusercontent.com/lechmazur/writing/refs/heads/main/README.md"

# Download the README.md files.
resp_confab = requests.get(url_confab)
if resp_confab.status_code != 200:
    raise Exception(f"Failed to download confabulations README.md: {resp_confab.status_code}")
md_confab = resp_confab.text

resp_gen = requests.get(url_gen)
if resp_gen.status_code != 200:
    raise Exception(f"Failed to download generalization README.md: {resp_gen.status_code}")
md_gen = resp_gen.text

resp_elim = requests.get(url_elim)
if resp_elim.status_code != 200:
    raise Exception(f"Failed to download elimination_game README.md: {resp_elim.status_code}")
md_elim = resp_elim.text

resp_step = requests.get(url_step)
if resp_step.status_code != 200:
    raise Exception(f"Failed to download step_game README.md: {resp_step.status_code}")
md_step = resp_step.text

resp_nytcon = requests.get(url_nytcon)
if resp_nytcon.status_code != 200:
    raise Exception(f"Failed to download nyt-connections README.md: {resp_nytcon.status_code}")
md_nytcon = resp_nytcon.text


# resp_writing = requests.get(url_writing)
# if resp_writing.status_code != 200:
#     raise Exception(f"Failed to download writing README.md: {resp_writing.status_code}")
# md_writing = resp_writing.text

# Extract the tables.
df_confab = extract_table(md_confab)
df_gen    = extract_table(md_gen)
df_gen = df_gen.drop_duplicates(subset="Model", keep='first')



# df_elim   = extract_table(md_elim)
df_step   = extract_table(md_step)
df_nytcon = extract_first_table(md_nytcon)
# df_writing = extract_first_table(md_writing)


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

df_confab = normalize_model_names(df_confab)
df_gen    = normalize_model_names(df_gen)
# df_elim   = normalize_model_names(df_elim)
df_step   = normalize_model_names(df_step)
df_nytcon = normalize_model_names(df_nytcon)
# df_writing = normalize_model_names(df_writing)


# Prefix columns for each DataFrame (except for 'Model').
df_confab = prefix_columns(df_confab, "confab")
df_gen    = prefix_columns(df_gen, "gen")
# df_elim   = prefix_columns(df_elim, "elim")
df_step   = prefix_columns(df_step, "step")
df_nytcon = prefix_columns(df_nytcon, "nytcon")
# df_writing = prefix_columns(df_writing, "writing")

# Merge all four tables on the "Model" column using outer joins.
df_temp = pd.merge(df_confab, df_gen, on="Model", how="outer")
# df_temp = pd.merge(df_temp, df_elim, on="Model", how="outer")
df_temp = pd.merge(df_temp, df_step, on="Model", how="outer")
df_combined = pd.merge(df_temp, df_nytcon, on="Model", how="outer")
# df_combined = pd.merge(df_temp, df_writing, on="Model", how="outer")
# print(df_writing.columns)
print(df_combined.columns)
# Extract the latest update dates from all README.md files.
date_confab = extract_latest_update(md_confab)
date_gen    = extract_latest_update(md_gen)
# date_elim   = extract_latest_update(md_elim)
date_step   = extract_latest_update(md_step)
date_nytcon = extract_latest_update(md_nytcon)
# date_writing = extract_latest_update(md_writing)

dates = [d for d in [date_confab, date_gen, date_step, date_nytcon] if d is not None]
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