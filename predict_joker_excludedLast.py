#!/usr/bin/env python3
import csv
import random
from collections import Counter

# ----------------------
# CONFIG
# ----------------------
CSV_FILE = "joker_results.csv"
HOT_COUNT = 2
COLD_COUNT = 1
MIXED_COUNT = 1
RANDOM_COUNT = 2
TOP_N_HOT = 10
TOP_N_COLD = 10
COLD_POOL_SIZE = 15
PICK_MAIN = 5
MAIN_RANGE = range(1, 46)
JOKER_RANGE = range(1, 21)

# ----------------------
# LOAD CSV
# ----------------------
with open(CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)  # skip header
    results = [row for row in reader if len(row) >= 6]

if not results:
    raise SystemExit(f"No valid draws found in {CSV_FILE}")

# Get exclusion numbers from the first row in CSV (most recent draw)
try:
    exclude_main = set(map(int, results[0][:5]))
    exclude_joker = {int(results[0][5])}
except ValueError:
    raise SystemExit("First row in CSV contains invalid numbers.")

# Remove excluded numbers from all picks
main_numbers = []
joker_numbers = []
for row in results:
    try:
        mains = [int(n) for n in row[:5] if int(n) not in exclude_main]
        joker = int(row[5])
        if joker not in exclude_joker:
            joker_numbers.append(joker)
        main_numbers.extend(mains)
    except ValueError:
        continue

main_counter = Counter(main_numbers)
joker_counter = Counter(joker_numbers)

# Allowed pools after exclusion
allowed_main = [n for n in MAIN_RANGE if n not in exclude_main]
allowed_joker = [j for j in JOKER_RANGE if j not in exclude_joker]

# ----------------------
# PICK FUNCTIONS
# ----------------------
def pick_hot(counter, k):
    candidates = [num for num, _ in counter.most_common() if num in allowed_main]
    return sorted(candidates[:k])

def pick_cold(counter, k):
    least_common = [num for num, _ in counter.most_common() if num in allowed_main][-COLD_POOL_SIZE:]
    return sorted(random.sample(least_common, k))

def pick_mixed(counter, k):
    hot_pool = [num for num, _ in counter.most_common(TOP_N_HOT) if num in allowed_main]
    cold_pool = [num for num, _ in counter.most_common() if num in allowed_main][-TOP_N_COLD:]
    picks = set(random.sample(hot_pool, min(3, len(hot_pool))) +
                random.sample(cold_pool, min(2, len(cold_pool))))
    while len(picks) < k:
        picks.add(random.choice(allowed_main))
    return sorted(picks)

def pick_random(k):
    return sorted(random.sample(allowed_main, k))

# ----------------------
# GENERATE SETS
# ----------------------
prediction_sets = []

# Hot sets
for _ in range(HOT_COUNT):
    main = pick_hot(main_counter, PICK_MAIN)
    joker = joker_counter.most_common(1)[0][0] if joker_counter else random.choice(allowed_joker)
    prediction_sets.append((main, joker))

# Cold sets
for _ in range(COLD_COUNT):
    main = pick_cold(main_counter, PICK_MAIN)
    cold_jokers = [num for num, _ in joker_counter.most_common() if num in allowed_joker][-5:]
    joker = random.choice(cold_jokers) if cold_jokers else random.choice(allowed_joker)
    prediction_sets.append((main, joker))

# Mixed sets
for _ in range(MIXED_COUNT):
    main = pick_mixed(main_counter, PICK_MAIN)
    joker = random.choice(allowed_joker)
    prediction_sets.append((main, joker))

# Random sets
for _ in range(RANDOM_COUNT):
    main = pick_random(PICK_MAIN)
    joker = random.choice(allowed_joker)
    prediction_sets.append((main, joker))

# ----------------------
# PRINT RESULTS
# ----------------------
print("\n--- 6 SUGGESTED JOKER SETS (exclusions applied) ---")
print(f"Excluded main numbers: {sorted(exclude_main)} | Excluded Joker: {sorted(exclude_joker)}")
for idx, (main, joker) in enumerate(prediction_sets, 1):
    print(f"Set {idx}: {main} | Joker: {joker}")

likely_eight = [num for num, _ in main_counter.most_common(8) if num in allowed_main]
print("\n--- 8-NUMBER POOL SUGGESTION (exclusions applied) ---")
print(sorted(likely_eight))
