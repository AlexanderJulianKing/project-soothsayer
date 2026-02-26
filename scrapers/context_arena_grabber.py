import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import re
import os

# Headers to mimic a browser request
request_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_and_clean_table(url):
    print(f"\nFetching HTML from {url}...")
    response = requests.get(url, headers=request_headers)
    response.raise_for_status()
    print("HTML fetched successfully.")

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.select_one('table[class^="svelte-"]')
    if not table:
        raise RuntimeError("Could not find the leaderboard table. The website structure might have changed.")

    print("Leaderboard table found. Parsing data manually...")

    # --- Manual Parsing with BeautifulSoup (The Correct Method) ---

    # 1. Parse Headers
    headers = []
    header_row = table.find('thead').find('tr')
    for th in header_row.find_all('th'):
        # Find the specific span with the label to get clean text
        label_span = th.find('span', class_='th-label')
        if label_span:
            headers.append(label_span.get_text(strip=True))
        else:
            # Fallback for any headers without that specific span
            headers.append(th.get_text(separator=' ', strip=True))

    print("Parsed Headers:", headers)

    # 2. Parse Data Rows
    data = []
    body = table.find('tbody')
    for row in body.find_all('tr'):
        cols = row.find_all('td')
        row_data = []

        # --- SPECIAL HANDLING FOR THE FIRST 'MODEL' CELL ---
        model_cell = cols[0]
        # Find the main span containing the model name
        model_name_span = model_cell.find('span', class_='td-value')

        # Get the base model name (the first piece of text, before any icons)
        base_name = model_name_span.find(string=True, recursive=False).strip()

        # The 'reasoning' version has the 'thinking-icon' but NOT the 'reason-capable-icon-wrapper'.
        is_reasoning_version = model_cell.find('span', class_='thinking-icon') and \
                               not model_cell.find('span', class_='reason-capable-icon-wrapper')

        if is_reasoning_version:
            # Append a suffix to make it unique
            model_name = f"{base_name} (Reasoning)"
        else:
            model_name = base_name

        row_data.append(model_name)
        # --- END OF SPECIAL HANDLING ---

        # Process the rest of the cells normally
        for td in cols[1:]:
            # Get text, using a space for separators to handle complex cells
            row_data.append(td.get_text(separator=' ', strip=True))

        if len(row_data) == len(headers):
            data.append(row_data)
        else:
            print(f"Warning: Skipping row with mismatching column count ({len(row_data)} vs {len(headers)}): {row_data}")

    # Create the DataFrame from our manually parsed data
    df = pd.DataFrame(data, columns=headers)
    pd.set_option('display.max_columns', None)
    print(df)

    print("\n--- Initial Parsed DataFrame ---")
    print(df[['Model']].head().to_string())

    # --- Data Cleaning ---

    # The manual parsing already gives us clean model names, so no renaming is needed.
    # The 'timeline' and other icon text is also gone.

    # 1. Extract numerical percentages from the score columns
    # Let's find the columns to process dynamically by looking for '(%)'
    cols_to_process = [col for col in df.columns if '(%)' in col]
    print(f"\nCleaning percentage columns: {cols_to_process}")

    def grab_percent(value):
        """
        Extracts a percentage value from a string like '#4 article 98.2%'.
        Returns a float or pd.NA if not found.
        """
        # Return NA for non-string or null values
        if pd.isna(value) or not isinstance(value, str):
            return pd.NA

        # The regex looks for a number (digits and a decimal) followed by a '%'
        # The parentheses () capture the number part.
        match = re.search(r'([\d\.]+)%', value)

        if match:
            # match.group(1) is the captured part (the number)
            return float(match.group(1))

        # Return NA if no match was found
        return pd.NA

    for col_name in cols_to_process:
        df[col_name] = df[col_name].apply(grab_percent)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # 2. Clean up cost columns by removing '$' and handling 'WARN'
    for col_name in ['Input Cost ($)', 'Output Cost ($)']:
        if col_name in df.columns:
            df[col_name] = df[col_name].str.replace('$', '', regex=False)
            df[col_name] = df[col_name].str.replace('WARN', '', regex=False).str.strip()
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # 3. Clean 'Max Ctx'
    if 'Max Ctx' in df.columns:
        df['Max Ctx'] = df['Max Ctx'].str.replace(',', '').str.strip()
        df['Max Ctx'] = pd.to_numeric(df['Max Ctx'], errors='coerce')

    print("\n--- Final Cleaned DataFrame (First 5 Rows) ---")
    print(df.head().to_string())

    return df

def find_metric_column(df, target_label, context_label):
    if target_label in df.columns:
        return target_label

    stripped_matches = [col for col in df.columns if col.strip() == target_label]
    if len(stripped_matches) == 1:
        return stripped_matches[0]
    if len(stripped_matches) > 1:
        raise ValueError(f"Multiple stripped matches for '{target_label}' in {context_label}: {stripped_matches}")

    partial_matches = [col for col in df.columns if target_label in col]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        raise ValueError(f"Multiple partial matches for '{target_label}' in {context_label}: {partial_matches}")

    raise ValueError(f"Could not find column '{target_label}' in {context_label}. Available columns: {list(df.columns)}")

try:
    targets = [
        {
            "label": "default",
            "url": "https://contextarena.ai/?showExtra=true",
        },
        {
            "label": "needles=8",
            "url": "https://contextarena.ai/?needles=8&showExtra=true",
        },
    ]

    # Save the final, clean DataFrame to a CSV
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs('benchmarks', exist_ok=True)

    saved = {
        "default": fetch_and_clean_table(targets[0]["url"]),
        "needles=8": fetch_and_clean_table(targets[1]["url"]),
    }

    metric_label = "8k (%)"
    default_metric_col = find_metric_column(saved["default"], metric_label, "default")
    needles_metric_col = find_metric_column(saved["needles=8"], metric_label, "needles=8")

    default_out = saved["default"][["Model", default_metric_col]].rename(
        columns={default_metric_col: "8k (%) 2 needles"}
    )
    needles_out = saved["needles=8"][["Model", needles_metric_col]].rename(
        columns={needles_metric_col: "8k (%) 8 needles"}
    )

    combined_df = pd.merge(default_out, needles_out, on="Model", how="outer", sort=False)
    model_order = list(default_out["Model"])
    for model in needles_out["Model"]:
        if model not in model_order:
            model_order.append(model)
    combined_df["Model"] = pd.Categorical(combined_df["Model"], categories=model_order, ordered=True)
    combined_df = combined_df.sort_values("Model").reset_index(drop=True)

    filename = f'benchmarks/contextarena_{current_date}.csv'
    combined_df.to_csv(filename, index=False)
    print(f"\nSaved combined table to {filename}")

    if "default" in saved and "needles=8" in saved:
        default_models = set(saved["default"]["Model"].dropna())
        needles_models = set(saved["needles=8"]["Model"].dropna())
        missing_from_needles = default_models - needles_models
        extra_in_needles = needles_models - default_models

        if missing_from_needles or extra_in_needles:
            print("\nWarning: Model lists differ between default and needles=8 tables.")
            if missing_from_needles:
                print(f"Missing from needles=8 ({len(missing_from_needles)}): {sorted(missing_from_needles)}")
            if extra_in_needles:
                print(f"Extra in needles=8 ({len(extra_in_needles)}): {sorted(extra_in_needles)}")
        else:
            print("\nModel lists match between default and needles=8 tables.")

except requests.exceptions.RequestException as e:
    print(f"Error fetching URL: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    # For debugging, print the traceback
    import traceback
    traceback.print_exc()
