from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
from bs4 import BeautifulSoup
import pandas as pd
import datetime


# Manual model name canonicalization for LiveBench.
MODEL_NAME_MAP = {
    "Claude 4.5 Opus High Effort": "Claude 4.5 Opus",
    "Claude 4.5 Opus Medium Effort": "Claude 4.5 Opus",
    "Claude 4.5 Opus Low Effort": "Claude 4.5 Opus",
}


def canonicalize_model_name(name: str) -> str:
    cleaned = name.strip()
    return MODEL_NAME_MAP.get(cleaned, cleaned)


def first_non_empty(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value) and value != "":
            return value
    return ""


# === Setup Selenium WebDriver ===
# Replace '/path/to/chromedriver' with the actual path to your ChromeDriver executable.
# service = Service('/path/to/chromedriver')
# options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # run in headless mode for efficiency
# driver = webdriver.Chrome(service=service, options=options)


# Initialize Selenium WebDriver (e.g., for Chrome)
driver = webdriver.Chrome()  # Make sure ChromeDriver is in your PATH
# driver.get("YOUR_LIVEBENCH_LEADERBOARD_URL") # Replace with the actual URL

# Update the URL to point to the LiveBench leaderboard page you wish to scrape.
url = "https://livebench.ai/#/"  # Change to the correct URL if needed.
driver.get(url)

# Wait for the table element to load (adjust timeout as needed)
wait = WebDriverWait(driver, 20)
table_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".table-wrap table")))

# === Toggle "Show Subcategories" Checkboxes ===
# Find all checkboxes that are associated with "Show Subcategories" labels and click them if they are not checked.
subcat_checkboxes = driver.find_elements(By.XPATH, "//label[contains(text(),'Show Subcategories')]/input")
i = 0
categories = ['Reasoning', 'Coding', 'Mathematics', 'Data Analysis', 'Language', 'IF']
merged_df = []
for checkbox in subcat_checkboxes:
    if not checkbox.is_selected():
        checkbox.click()
        # Wait a moment for the table to update after toggling
        time.sleep(1)

# # Optional: wait a bit more to ensure the table updates
# time.sleep(2)

        # === Extract the Table HTML and Parse with BeautifulSoup ===
        table_html = driver.find_element(By.CSS_SELECTOR, ".table-wrap table").get_attribute("outerHTML")
        soup = BeautifulSoup(table_html, "html.parser")

        # Extract headers from the table
        headers = []
        header_row = soup.find("thead").find("tr")
        for th in header_row.find_all("th"):
            headers.append(th.get_text(strip=True))

        # Extract data rows from the table body
        data = []
        for row in soup.find("tbody").find_all("tr"):
            cols = row.find_all("td")
            row_data = [col.get_text(strip=True) for col in cols]
            data.append(row_data)
        print(headers)

        df = pd.DataFrame(data, columns=headers)
        df["Model"] = df["Model"].apply(canonicalize_model_name)
        if df["Model"].duplicated().any():
            df = df.groupby("Model", as_index=False).agg(first_non_empty)

        if len(merged_df) == 0:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Model', how='outer', suffixes=('', '_dup'))
        # # === Write the Data to a CSV File ===
        # csv_filename = "livebench_table.csv"
        # with open(csv_filename, "w", newline='', encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(headers)
        #     writer.writerows(data)

        # print(f"Data successfully written to {csv_filename}")

        # Close the browser
# Remove columns where the column name contains "organization" (case insensitive)
merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('organization', case=False)]
current_date = datetime.datetime.now().strftime("%Y%m%d")
filename = 'benchmarks/livebench_{}.csv'.format(current_date)
merged_df.to_csv(filename)
driver.quit()
