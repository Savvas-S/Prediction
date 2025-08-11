#!/usr/bin/env python3
import csv
import argparse
import random
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# CLI
# ----------------------
parser = argparse.ArgumentParser(description="JOKER ML predictor with auto-exclusion from a CSV row.")
parser.add_argument("--csv", default="joker_results.csv", help="CSV with draws (header + 6 cols: 5 mains + Joker).")
parser.add_argument("--draws", type=int, default=2800, help="How many most-recent draws to use.")
parser.add_argument("--candidates", type=int, default=4000, help="How many candidate tickets to generate.")
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument("--exclude-main", type=int, nargs="*", default=[], help="Extra main numbers to exclude.")
parser.add_argument("--exclude-joker", type=int, nargs="*", default=[], help="Extra joker numbers to exclude.")
parser.add_argument("--exclude-row-index", type=int, default=0,
                    help="Which CSV data row to auto-exclude from (0=first row after header; -1=last row).")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

MAIN_MIN, MAIN_MAX = 1, 45
JOKER_MIN, JOKER_MAX = 1, 20
all_main = list(range(MAIN_MIN, MAIN_MAX + 1))
all_joker = list(range(JOKER_MIN, JOKER_MAX + 1))

# ----------------------
# Load CSV (assumes 6 numeric columns per row)
# ----------------------
rows = []
with open(args.csv, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        if len(row) >= 6:
            try:
                vals = list(map(int, row[:6]))
                rows.append(vals)
            except ValueError:
                pass

if not rows:
    raise SystemExit("No data in CSV.")

# Auto-exclusion from specified row
# Supports negative index (e.g., -1 for last row)
ex_idx = args.exclude_row_index
if not (-len(rows) <= ex_idx < len(rows)):
    raise SystemExit(f"--exclude-row-index {ex_idx} is out of range for CSV with {len(rows)} data rows.")

auto_ex_row = rows[ex_idx]
auto_exclude_main = set(auto_ex_row[:5])
auto_exclude_joker = {auto_ex_row[5]}

# Merge manual excludes
exclude_main = set(args.exclude_main) | auto_exclude_main
exclude_joker = set(args.exclude_joker) | auto_exclude_joker

# Keep only most recent draws (regardless of order in file)
if len(rows) > args.draws:
    rows = rows[-args.draws:]

N = len(rows)

# Build indicator arrays
main_ind = np.zeros((MAIN_MAX, N), dtype=np.int16)
joker_ind = np.zeros((JOKER_MAX, N), dtype=np.int16)

for t, draw in enumerate(rows):
    mains = set(draw[:5])
    j = draw[5]
    for n in mains:
        if 1 <= n <= MAIN_MAX:
            main_ind[n-1, t] = 1
    if 1 <= j <= JOKER_MAX:
        joker_ind[j-1, t] = 1

# ----------------------
# Feature builders
# ----------------------
def window_freq(ind_row, t_end, W, denom_per_draw):
    if t_end <= 0:
        return 0.0
    start = max(0, t_end - W)
    count = ind_row[start:t_end].sum()
    draws = t_end - start
    denom = max(1, denom_per_draw * draws)
    return float(count) / denom

def recency_gap(ind_row, t_end):
    if t_end <= 0:
        return 1.0
    seen_idx = np.where(ind_row[:t_end] == 1)[0]
    if seen_idx.size == 0:
        return 1.0
    gap = (t_end - 1) - seen_idx[-1]
    gmax = max(1, t_end)
    return math.sqrt(gap) / math.sqrt(gmax)

def short_streak(ind_row, t_end, W=10, denom_per_draw=5):
    return window_freq(ind_row, t_end, W, denom_per_draw)

def build_main_features_label(t):
    X, y = [], []
    present = set(rows[t][:5])
    for n in all_main:
        row = main_ind[n-1, :]
        f50  = window_freq(row, t, 50, 5)
        f200 = window_freq(row, t, 200, 5)
        f800 = window_freq(row, t, 800, 5)
        rec  = recency_gap(row, t)
        stk  = short_streak(row, t, 10, 5)
        X.append([n, f50, f200, f800, rec, stk])
        y.append(1 if n in present else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)

def build_joker_features_label(t):
    X, y = [], []
    present = rows[t][5]
    for j in all_joker:
        row = joker_ind[j-1, :]
        f50  = window_freq(row, t, 50, 1)
        f200 = window_freq(row, t, 200, 1)
        f800 = window_freq(row, t, 800, 1)
        rec  = recency_gap(row, t)
        stk  = short_streak(row, t, 10, 1)
        X.append([j, f50, f200, f800, rec, stk])
        y.append(1 if j == present else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)

# ----------------------
# Build training sets (walk-forward)
# ----------------------
min_history = 60
X_main_all, y_main_all, X_jok_all, y_jok_all = [], [], [], []

for t in range(min_history, N):
    Xm, ym = build_main_features_label(t)
    Xj, yj = build_joker_features_label(t)
    X_main_all.append(Xm); y_main_all.append(ym)
    X_jok_all.append(Xj);  y_jok_all.append(yj)

X_main = np.vstack(X_main_all)
y_main = np.concatenate(y_main_all)
X_jok  = np.vstack(X_jok_all)
y_jok  = np.concatenate(y_jok_all)

num_id_main = X_main[:, 0].astype(int)
num_id_jok  = X_jok[:, 0].astype(int)
X_main_feat = X_main[:, 1:]
X_jok_feat  = X_jok[:, 1:]

# ----------------------
# Train models
# ----------------------
rf_main = RandomForestClassifier(
    n_estimators=500, min_samples_leaf=2,
    class_weight="balanced", n_jobs=-1, random_state=args.seed
)
rf_jok = RandomForestClassifier(
    n_estimators=400, min_samples_leaf=1,
    class_weight="balanced", n_jobs=-1, random_state=args.seed
)

rf_main.fit(X_main_feat, y_main)
rf_jok.fit(X_jok_feat, y_jok)

# ----------------------
# Predict for NEXT draw
# ----------------------
def build_main_features_for_time(t):
    X, ids = [], []
    for n in all_main:
        row = main_ind[n-1, :]
        f50  = window_freq(row, t, 50, 5)
        f200 = window_freq(row, t, 200, 5)
        f800 = window_freq(row, t, 800, 5)
        rec  = recency_gap(row, t)
        stk  = short_streak(row, t, 10, 5)
        X.append([f50, f200, f800, rec, stk])
        ids.append(n)
    return np.array(X, dtype=float), ids

def build_joker_features_for_time(t):
    X, ids = [], []
    for j in all_joker:
        row = joker_ind[j-1, :]
        f50  = window_freq(row, t, 50, 1)
        f200 = window_freq(row, t, 200, 1)
        f800 = window_freq(row, t, 800, 1)
        rec  = recency_gap(row, t)
        stk  = short_streak(row, t, 10, 1)
        X.append([f50, f200, f800, rec, stk])
        ids.append(j)
    return np.array(X, dtype=float), ids

Xm_next, ids_m = build_main_features_for_time(N)
Xj_next, ids_j = build_joker_features_for_time(N)

probs_main = rf_main.predict_proba(Xm_next)[:, 1]
probs_jok  = rf_jok.predict_proba(Xj_next)[:, 1]

# Apply exclusions: union of manual + auto (from chosen CSV row)
for i, n in enumerate(ids_m):
    if n in exclude_main:
        probs_main[i] = 0.0
for i, j in enumerate(ids_j):
    if j in exclude_joker:
        probs_jok[i] = 0.0

def norm_probs(ids, probs, allow_set):
    for i, v in enumerate(ids):
        if v not in allow_set:
            probs[i] = 0.0
    s = probs.sum()
    return probs / s if s > 0 else np.ones_like(probs) / len(probs)

allow_main = [n for n in ids_m if n not in exclude_main]
allow_jok  = [j for j in ids_j if j not in exclude_joker]

probs_main = norm_probs(ids_m, probs_main.copy(), set(allow_main))
probs_jok  = norm_probs(ids_j, probs_jok.copy(), set(allow_jok))

id_to_p_main = dict(zip(ids_m, probs_main))
id_to_p_jok  = dict(zip(ids_j, probs_jok))

# ----------------------
# Candidate generation
# ----------------------
def weighted_sample_without_replacement(items, pmap, k):
    # Efraimidis–Spirakis style keys
    keys = []
    for x in items:
        p = float(pmap.get(x, 0.0))
        if p <= 0: 
            continue
        u = random.random()
        keys.append((math.log(u) / p, x))
    keys.sort()
    pick = [x for _, x in keys[:k]]
    if len(pick) < k:
        remaining = [x for x in items if x not in pick]
        if remaining:
            pick += random.sample(remaining, k - len(pick))
    return sorted(pick)

def sample_joker_one():
    r, acc = random.random(), 0.0
    for j in sorted(allow_jok):
        acc += id_to_p_jok.get(j, 0.0)
        if r <= acc:
            return j
    return random.choice(allow_jok)

def ticket_score(nums):
    base = sum(id_to_p_main.get(n, 0.0) for n in nums)
    rng = (max(nums) - min(nums)) / 44.0
    return base + 0.15 * rng

candidates = []
for _ in range(args.candidates):
    nums = weighted_sample_without_replacement(allow_main, id_to_p_main, 5)
    candidates.append((ticket_score(nums), nums))

candidates.sort(reverse=True)

def jaccard(a, b):
    sa, sb = set(a), set(b)
    inter = len(sa & sb); union = len(sa | sb)
    return inter / union if union else 0.0

picked = []
for sc, nums in candidates:
    if all(jaccard(nums, pnums) <= 0.4 for _, pnums in picked):
        picked.append((sc, nums))
    if len(picked) == 6:
        break
while len(picked) < 6:
    for cand in candidates:
        if cand not in picked:
            picked.append(cand)
            break

final_sets = [(sorted(nums), sample_joker_one()) for _, nums in picked[:6]]
top8 = sorted([n for n, _ in sorted(id_to_p_main.items(), key=lambda kv: kv[1], reverse=True)
               if n not in exclude_main][:8])

# ----------------------
# OUTPUT
# ----------------------
print(f"\nExcluding from CSV row index {ex_idx}: mains {sorted(auto_exclude_main)} | joker {sorted(auto_exclude_joker)}")
if args.exclude_main or args.exclude_joker:
    print(f"Plus manual excludes: mains {sorted(set(args.exclude_main))} | joker {sorted(set(args.exclude_joker))}")

print("\n--- ML: 6 SUGGESTED SETS (exclusions applied) ---")
for idx, (nums, jk) in enumerate(final_sets, 1):
    print(f"Set {idx}: {nums} | Joker: {jk}")

print("\n--- ML: 8-NUMBER POOL (exclusions applied) ---")
print(top8)
