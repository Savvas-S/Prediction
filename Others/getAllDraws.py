import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import time

# Base URL
base_url = "https://opap.org.cy/el/joker"

# Step 1: Detect latest draw ID automatically
response = requests.get(base_url)
if response.status_code != 200:
    raise SystemExit(f"Failed to fetch main page: status {response.status_code}")

soup = BeautifulSoup(response.text, "html.parser")
latest_draw = None

for a in soup.find_all("a", href=True):
    if "gameid=" in a["href"]:
        try:
            gameid = int(a["href"].split("gameid=")[1].split("#")[0])
            latest_draw = max(latest_draw or 0, gameid)
        except ValueError:
            pass

if not latest_draw:
    raise SystemExit("Could not determine latest draw ID from page.")

print(f"Latest draw ID detected: {latest_draw}")

# Step 2: Define range for last 2800 draws
total_draws = 2800
start_id = latest_draw
end_id = start_id - (total_draws - 1)

url_template = "https://opap.org.cy/el/joker?gameid={gameid}#results-show"
results = []

start_time = time.time()

for count, gameid in enumerate(range(start_id, end_id - 1, -1), start=1):
    url = url_template.format(gameid=gameid)
    response = requests.get(url)
    if response.status_code != 200:
        print(f"{datetime.now().strftime('%H:%M:%S')} | [{count}/{total_draws}] GameID {gameid}: Failed ({response.status_code})")
        continue

    soup = BeautifulSoup(response.text, "html.parser")
    numbers_ul = soup.select_one("ul#winnerNumbers.circles")
    if numbers_ul:
        numbers = [li.text.strip() for li in numbers_ul.find_all('li') if li.text.strip()]
        results.append(numbers)
        elapsed = time.time() - start_time
        avg_per_draw = elapsed / count
        remaining_draws = total_draws - count
        eta_sec = avg_per_draw * remaining_draws
        eta_min = eta_sec / 60
        print(f"{datetime.now().strftime('%H:%M:%S')} | [{count}/{total_draws}] GameID {gameid} saved: {numbers} | ETA: {eta_min:.1f} min")
    else:
        print(f"{datetime.now().strftime('%H:%M:%S')} | [{count}/{total_draws}] GameID {gameid}: No numbers found")

    time.sleep(0.2)

# Reverse so oldest first
results.reverse()

with open("joker_results2.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Number1", "Number2", "Number3", "Number4", "Number5", "Joker"])
    for row in results:
        writer.writerow(row + [""] * (6 - len(row)))

print(f"Done. Results saved to joker_results2.csv ({len(results)} draws, oldest → newest).")
