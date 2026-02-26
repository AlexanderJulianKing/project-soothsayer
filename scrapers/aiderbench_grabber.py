import requests
from bs4 import BeautifulSoup
import csv
import io  # for StringIO (in-memory CSV)
import datetime

url = "https://aider.chat/docs/leaderboards/"  # Replace with the actual URL
try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
    html_string = response.text
except requests.exceptions.RequestException as e:
    print(f"Error fetching URL: {e}")
    exit()

soup = BeautifulSoup(html_string, 'html.parser')
table = soup.find('table')

if table is None:
    print("Error: Table not found on the webpage.")
    exit()

headers = [th.text for th in table.thead.find_all('th')]
data_rows = []

for row in table.tbody.find_all('tr'):
    pleaseskip = False
    cells = [td.text for td in row.find_all('td')]

    for cell in cells:
        if "# via hyperbolic" in cell:
            pleaseskip = True
    if len(cells) == 1:
        pleaseskip = True
    if not pleaseskip:
        data_rows.append(cells)

csv_output = io.StringIO()
csv_writer = csv.writer(csv_output)
csv_writer.writerow(headers)
csv_writer.writerows(data_rows)

csv_string = csv_output.getvalue()
# print(csv_string)

#To save to a file:
current_date = datetime.datetime.now().strftime("%Y%m%d")
filename = 'benchmarks/aiderbench_{}.csv'.format(current_date)
with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    csvfile.write(csv_string)
print("CSV data saved to ", filename)