#!/usr/bin/env python3
import csv
import argparse
import random
import math
from collections import Counter, defaultdict
from itertools import combinations

# ----------------------
# CLI
# ----------------------
parser = argparse.ArgumentParser(description="JOKER predictor (stats-weighted + co-occurrence + diversity).")
parser.add_argument("--csv", default="joker_results.csv", help="Path to CSV with 1000 draws (6 columns: 5 + Joker).")
parser.add_argument("--draws", type=int, default=1000, help="How many most-recent draws to use (<= rows in CSV).")
parser.add_argument("--candidates", type=int, default=2500, help="How many candidate tickets to generate.")
parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
parser.add_argument("--exclude-main", type=int, nargs="*", default=[], help="Main numbers to exclude.")
parser.add_argument("--exclude-joker", type=int, nargs="*", default=[], help="Joker numbers to exclude.")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

# ----------------------
# Load data
# ----------------------
rows = []
with open(args.csv, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        if len(row) >= 6:
            try:
                nums = list(map(int, row[:6]))
                rows.append(nums)
            except ValueError:
                continue

# keep most recent 'draws' (assuming file is ordered from oldest->newest; if not, reverse as needed)
if len(rows) > args.draws:
    rows = rows[-args.draws:]

# main balls 1..45, joker 1..20
MAIN_MIN, MAIN_MAX = 1, 45
JOKER_MIN, JOKER_MAX = 1, 20

all_main = list(range(MAIN_MIN, MAIN_MAX + 1))
all_joker = list(range(JOKER_MIN, JOKER_MAX + 1))

exclude_main = set(args.exclude_main)
exclude_joker = set(args.exclude_joker)

# ----------------------
# Stats helpers
# ----------------------
def window_slice(data, n):
    return data[-n:] if len(data) >= n else data

def normalize(vec):
    s = sum(vec.values())
    if s <= 0:
        return {k: 1.0/len(vec) for k in vec}
    return {k: v/s for k, v in vec.items()}

def zscore_dict(d):
    vals = list(d.values())
    if not vals:
        return d
    m = sum(vals)/len(vals)
    var = sum((x - m)**2 for x in vals)/len(vals) if len(vals) > 1 else 0.0
    sd = math.sqrt(var)
    if sd == 0:
        return {k: 0.0 for k in d}
    return {k: (v - m)/sd for k, v in d.items()}

# ----------------------
# Build features
# ----------------------
lastN = rows  # already capped

# Frequencies across multiple windows (weights tuned)
windows = [
    (50, 0.50),
    (200, 0.30),
    (len(lastN), 0.20),
]

# Main frequencies
freq_scores = {n: 0.0 for n in all_main}
for W, w in windows:
    sl = window_slice(lastN, W)
    c = Counter()
    for draw in sl:
        for n in draw[:5]:
            c[n] += 1
    # normalize by total main draws in window (5 per draw)
    denom = max(1, 5 * len(sl))
    for n in all_main:
        freq_scores[n] += w * (c.get(n, 0) / denom)

# Joker frequencies
freq_scores_j = {j: 0.0 for j in all_joker}
for W, w in windows:
    sl = window_slice(lastN, W)
    c = Counter(draw[5] for draw in sl)
    denom = max(1, len(sl))
    for j in all_joker:
        freq_scores_j[j] += w * (c.get(j, 0) / denom)

# Recency (“due”): larger gap since last seen => higher score (capped)
def recency_scores_main():
    last_seen = {n: None for n in all_main}
    for idx in range(len(lastN)-1, -1, -1):
        for n in lastN[idx][:5]:
            if last_seen[n] is None:
                last_seen[n] = len(lastN)-1 - idx
    gaps = {}
    for n in all_main:
        gap = last_seen[n] if last_seen[n] is not None else len(lastN)  # never seen in window
        gaps[n] = gap
    # sqrt to soften huge gaps
    gmax = max(gaps.values()) if gaps else 1
    return {n: math.sqrt(gaps[n])/math.sqrt(gmax) for n in all_main}

def recency_scores_joker():
    last_seen = {j: None for j in all_joker}
    for idx in range(len(lastN)-1, -1, -1):
        jn = lastN[idx][5]
        if last_seen[jn] is None:
            last_seen[jn] = len(lastN)-1 - idx
    gaps = {}
    for j in all_joker:
        gap = last_seen[j] if last_seen[j] is not None else len(lastN)
        gaps[j] = gap
    gmax = max(gaps.values()) if gaps else 1
    return {j: math.sqrt(gaps[j])/math.sqrt(gmax) for j in all_joker}

rec_scores = recency_scores_main()
rec_scores_j = recency_scores_joker()

# Short-term streak (last 10 draws)
streakW = 10
sl10 = window_slice(lastN, streakW)
streak_c = Counter()
streak_j = Counter()
for d in sl10:
    for n in d[:5]:
        streak_c[n] += 1
    streak_j[d[5]] += 1
streak_scores = {n: streak_c.get(n, 0)/max(1, 5*len(sl10)) for n in all_main}
streak_scores_j = {j: streak_j.get(j, 0)/max(1, len(sl10)) for j in all_joker}

# Pairwise co-occurrence (last 200 draws)
pairW = 200
sl_pair = window_slice(lastN, pairW)
pair_count = defaultdict(int)
for d in sl_pair:
    for a, b in combinations(sorted(d[:5]), 2):
        pair_count[(a, b)] += 1
# normalize pair counts
max_pair = max(pair_count.values()) if pair_count else 1
pair_score = {p: pair_count[p]/max_pair for p in pair_count}

# ----------------------
# Combine number-level scores into a single score per number
# weights tuned empirically
w_freq = 0.60
w_rec  = 0.25
w_strk = 0.15

num_score = {}
for n in all_main:
    num_score[n] = (w_freq * freq_scores[n] +
                    w_rec  * rec_scores[n] +
                    w_strk * streak_scores[n])

num_score_j = {}
for j in all_joker:
    num_score_j[j] = (w_freq * freq_scores_j[j] +
                      w_rec  * rec_scores_j[j] +
                      w_strk * streak_scores_j[j])

# Remove excluded numbers by setting to tiny score
for n in exclude_main:
    if n in num_score:
        num_score[n] = 0.0
for j in exclude_joker:
    if j in num_score_j:
        num_score_j[j] = 0.0

# Turn into probabilities for weighted sampling
def to_probs(score_map, allowed):
    # clamp negatives to zero
    vals = {k: max(0.0, score_map.get(k, 0.0)) for k in allowed}
    s = sum(vals.values())
    if s <= 0:
        # fallback to uniform
        return {k: 1.0/len(allowed) for k in allowed}
    return {k: v/s for k, v in vals.items()}

allowed_main = [n for n in all_main if n not in exclude_main]
allowed_joker = [j for j in all_joker if j not in exclude_joker]

main_probs = to_probs(num_score, allowed_main)
joker_probs = to_probs(num_score_j, allowed_joker)

# ----------------------
# Candidate generation
# ----------------------
def weighted_sample_without_replacement(items, probs, k):
    # Efraimidis-Spirakis algorithm
    keys = []
    for x in items:
        p = probs.get(x, 0.0)
        if p <= 0:
            continue
        u = random.random()
        keys.append((math.log(u) / p, x))
    keys.sort()  # smallest first
    return sorted([x for _, x in keys[:k]])

def ticket_objective(nums5):
    # Objective = number scores + pair synergy + spread regularization
    # pair synergy
    ps = 0.0
    for a, b in combinations(sorted(nums5), 2):
        if a > b:
            a, b = b, a
        ps += pair_score.get((a, b), 0.0)

    # number base scores
    base = sum(num_score.get(n, 0.0) for n in nums5)

    # spread: prefer not-too-bunched sets (range & variance)
    rng = max(nums5) - min(nums5)
    # Normalize range to [0..1] roughly over [0..44]
    spread = rng / 44.0

    # Weights for objective
    alpha = 1.0   # base number quality
    beta  = 0.60  # pair synergy
    gamma = 0.20  # spread

    return alpha * base + beta * ps + gamma * spread

def sample_joker():
    # sample 1 joker by probabilities
    r = random.random()
    acc = 0.0
    for j, p in sorted(joker_probs.items()):
        acc += p
        if r <= acc:
            return j
    return random.choice(allowed_joker)

# Generate many candidates
candidates = []
for _ in range(args.candidates):
    nums = weighted_sample_without_replacement(allowed_main, main_probs, 5)
    if len(nums) < 5:
        # fallback uniform if something went wrong
        nums = sorted(random.sample(allowed_main, 5))
    score = ticket_objective(nums)
    candidates.append((score, nums))

# sort best first
candidates.sort(reverse=True, key=lambda x: x[0])

# ----------------------
# Pick top 6 with diversity (limit overlap)
# ----------------------
def jaccard(a, b):
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

picked = []
min_jaccard = 0.4  # encourage difference between sets
for sc, nums in candidates:
    if all(jaccard(nums, pnums) <= min_jaccard for _, pnums in picked):
        picked.append((sc, nums))
    if len(picked) == 6:
        break

# fallback if not enough (shouldn’t happen often)
idx = 0
while len(picked) < 6 and idx < len(candidates):
    cand = candidates[idx]
    if cand not in picked:
        picked.append(cand)
    idx += 1

# assign Jokers per set (weighted)
final_sets = []
for sc, nums in picked[:6]:
    final_sets.append((sorted(nums), sample_joker()))

# ----------------------
# 8-number pool (top by per-number score, excluding banned)
# ----------------------
top8 = sorted(allowed_main, key=lambda n: num_score.get(n, 0.0), reverse=True)[:8]
top8 = sorted(top8)

# ----------------------
# OUTPUT
# ----------------------
print("\n--- SMART 6 SUGGESTED SETS (exclusions applied) ---")
for i, (nums, jn) in enumerate(final_sets, 1):
    print(f"Set {i}: {nums} | Joker: {jn}")

print("\n--- 8-NUMBER POOL ---")
print(top8)

# Diagnostics (optional; uncomment if you want to see the top-scoring mains & jokers)
# mains_sorted = sorted(allowed_main, key=lambda n: num_score.get(n,0.0), reverse=True)
# jokers_sorted = sorted(allowed_joker, key=lambda j: num_score_j.get(j,0.0), reverse=True)
# print("\nTop mains by score:", mains_sorted[:15])
# print("Top jokers by score:", jokers_sorted[:10])
