#!/usr/bin/env python3
import csv
import random
from collections import Counter

# ----------------------
# CONFIG
# ----------------------
CSV_FILE = "joker_results2_reversed.csv"
HOT_COUNT = 2      # number of hot sets
COLD_COUNT = 1     # number of cold sets
MIXED_COUNT = 1    # number of mixed sets
RANDOM_COUNT = 2   # number of random sets
TOP_N_HOT = 10     # how many top numbers considered "hot" for mixed sets
TOP_N_COLD = 10    # how many bottom numbers considered "cold" for mixed sets
COLD_POOL_SIZE = 15  # how many bottom numbers to consider for cold picks
PICK_MAIN = 5
MAIN_RANGE = range(1, 46)
JOKER_RANGE = range(1, 21)

# ----------------------
# LOAD CSV
# ----------------------
with open(CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader, None)  # skip header
    results = [row for row in reader if len(row) >= 6]

if not results:
    raise SystemExit(f"No valid draws found in {CSV_FILE}")

main_numbers = []
joker_numbers = []

for row in results:
    try:
        mains = list(map(int, row[:5]))
        joker = int(row[5])
    except ValueError:
        continue  # skip invalid rows
    main_numbers.extend(mains)
    joker_numbers.append(joker)

main_counter = Counter(main_numbers)
joker_counter = Counter(joker_numbers)

# ----------------------
# PICK FUNCTIONS
# ----------------------
def pick_hot(counter, k):
    return sorted({num for num, _ in counter.most_common(k)})

def pick_cold(counter, k):
    least_common = [num for num, _ in counter.most_common()][-COLD_POOL_SIZE:]
    return sorted(random.sample(least_common, k))

def pick_mixed(counter, k):
    hot_pool = [num for num, _ in counter.most_common(TOP_N_HOT)]
    cold_pool = [num for num, _ in counter.most_common()][-TOP_N_COLD:]
    picks = set(random.sample(hot_pool, min(3, len(hot_pool))) +
                random.sample(cold_pool, min(2, len(cold_pool))))
    while len(picks) < k:
        picks.add(random.choice(list(MAIN_RANGE)))
    return sorted(picks)

def pick_random(k):
    return sorted(random.sample(list(MAIN_RANGE), k))

# ----------------------
# GENERATE SETS
# ----------------------
prediction_sets = []

# Hot sets
for _ in range(HOT_COUNT):
    main = pick_hot(main_counter, PICK_MAIN)
    joker = joker_counter.most_common(1)[0][0]
    prediction_sets.append((main, joker))

# Cold sets
for _ in range(COLD_COUNT):
    main = pick_cold(main_counter, PICK_MAIN)
    cold_jokers = [num for num, _ in joker_counter.most_common()][-5:]
    joker = random.choice(cold_jokers)
    prediction_sets.append((main, joker))

# Mixed sets
for _ in range(MIXED_COUNT):
    main = pick_mixed(main_counter, PICK_MAIN)
    joker = random.choice(list(JOKER_RANGE))
    prediction_sets.append((main, joker))

# Random sets
for _ in range(RANDOM_COUNT):
    main = pick_random(PICK_MAIN)
    joker = random.choice(list(JOKER_RANGE))
    prediction_sets.append((main, joker))

# ----------------------
# PRINT RESULTS
# ----------------------
print("\n--- 6 SUGGESTED JOKER SETS ---")
for idx, (main, joker) in enumerate(prediction_sets, 1):
    print(f"Set {idx}: {main} | Joker: {joker}")

likely_eight = [num for num, _ in main_counter.most_common(8)]
print("\n--- 8-NUMBER POOL SUGGESTION ---")
print(sorted(likely_eight))
