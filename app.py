import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import time

start_id = 2944
end_id = 2000   # inclusive
url_template = "https://opap.org.cy/el/joker?gameid={gameid}#results-show"
results = []

for gameid in range(start_id, end_id - 1, -1):
    url = url_template.format(gameid=gameid)
    response = requests.get(url)
    if response.status_code != 200:
        print(f"{datetime.now().strftime('%H:%M:%S')} | GameID {gameid}: Failed to fetch page, status {response.status_code}")
        continue
    soup = BeautifulSoup(response.text, "html.parser")
    numbers_ul = soup.select_one("ul#winnerNumbers.circles")
    if numbers_ul:
        numbers = [li.text.strip() for li in numbers_ul.find_all('li') if li.text.strip()]
        results.append(numbers)
        print(f"{datetime.now().strftime('%H:%M:%S')} | GameID {gameid}: {numbers}")
    else:
        print(f"{datetime.now().strftime('%H:%M:%S')} | GameID {gameid}: No numbers found")
    time.sleep(0.2)  # Don't hammer the server. Adjust as you want

with open("joker_results.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Number1", "Number2", "Number3", "Number4", "Number5", "Bonus"])
    for row in results:
        writer.writerow(row + [""] * (6 - len(row)))

print("Done. Results saved to joker_results.csv")
