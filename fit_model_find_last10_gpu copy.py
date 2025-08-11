#!/usr/bin/env python3
"""
fit_model_find_last10.py (GPU)

- Searches random hyperparams until one sequentially "finds" the last 10 draws
  (10th -> 1st), adding each found draw back to training before the next step.
- Saves the winning config as JSON.
- If --predict-next is passed, immediately retrains on ALL data with that config
  and prints 6 suggested sets for the NEXT (not-yet-seen) draw, plus an 8-number pool.

CSV format: header + 6 ints: 5 mains (1..45) + joker (1..20)
Ordering: latest draw is the FIRST row in the CSV.
"""

import csv
import argparse
import random
import math
from itertools import combinations
from dataclasses import dataclass, asdict
import json
import time

import numpy as np
import xgboost as xgb  # GPU learner

# ---------------- CLI ----------------
p = argparse.ArgumentParser(description="GPU search to hit last-10 sequentially; optional predict-next.")
p.add_argument("--csv", default="joker_results.csv", help="CSV with header + 6 cols (5 mains + joker).")
p.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
p.add_argument("--max-trials", type=int, default=500, help="Max random configs to try.")
p.add_argument("--verbose", action="store_true", help="Verbose progress.")
p.add_argument("--save-best", default="winning_model_config.json", help="Where to write winning config JSON.")
p.add_argument("--min-history", type=int, default=80, help="Minimum draws before we start modeling.")

# Hard caps (keep any sampled config inside these)
p.add_argument("--hard-cap-candidates", type=int, default=40000, help="Cap on candidates per seed.")
p.add_argument("--hard-cap-seeds", type=int, default=400, help="Cap on seeds explored.")
p.add_argument("--hard-cap-topk", type=int, default=24, help="Cap on top-K mains for deterministic scan.")
p.add_argument("--det-top-jokers", type=int, default=5, help="Top-M jokers to try for each 5-combo deterministically.")

# Predict-next options (used only when --predict-next is passed)
p.add_argument("--predict-next", action="store_true", help="After success, retrain on full data and predict next draw.")
p.add_argument("--sets", type=int, default=6, help="How many sets to output when predicting next draw.")
p.add_argument("--diversity", type=float, default=0.4, help="Jaccard max overlap among suggested sets (0..1).")

args = p.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

MAIN_MIN, MAIN_MAX = 1, 45
JOKER_MIN, JOKER_MAX = 1, 20
ALL_MAIN = list(range(MAIN_MIN, MAIN_MAX + 1))
ALL_JOKER = list(range(JOKER_MIN, JOKER_MAX + 1))

# ---------------- Data ----------------
rows = []
with open(args.csv, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    _ = next(r, None)  # header
    for row in r:
        if len(row) >= 6:
            try:
                vals = list(map(int, row[:6]))
                rows.append(vals)
            except ValueError:
                pass

if len(rows) < 200:
    raise SystemExit("Not enough rows. Need a few hundred at least.")

# Latest-first -> chronological for modeling:
chron = list(reversed(rows))
N = len(chron)
if N < args.min_history + 10:
    raise SystemExit("Not enough data to hold out last 10 and still meet min-history.")

# Holdout sequence (chronological order inside the 10)
held10 = chron[-10:]        # [10th, 9th, ..., 1st] oldest->newest within last 10
train_prefix = chron[:-10]  # training base (chronological)

# ---------------- Model Config ----------------
@dataclass
class ModelConfig:
    # feature knobs
    decay_half: float      # exponential half-life (draws) for decayed freq
    cooldown: int          # penalize numbers seen in last K draws
    # XGBoost sizes
    xgb_main_estimators: int
    xgb_jok_estimators: int
    xgb_main_depth: int
    xgb_jok_depth: int
    xgb_main_min_child_weight: float
    xgb_jok_min_child_weight: float
    # temperatures
    temp_main: float
    temp_joker: float
    # search knobs
    topk_combo: int        # deterministic C(K,5) combos from top-K mains
    seeds: int             # stochastic exploration seeds
    candidates: int        # candidates per seed
    # scoring knobs
    pair_window: int       # window for pair co-occurrence
    pair_weight: float     # weight in ticket_score
    spread_weight: float   # spread bonus weight

def sample_config() -> ModelConfig:
    # Reasonable random ranges; tuned for quality on GPU
    return ModelConfig(
        decay_half = random.uniform(80.0, 200.0),
        cooldown   = random.randint(4, 8),
        xgb_main_estimators = random.randint(600, 900),
        xgb_jok_estimators  = random.randint(450, 750),
        xgb_main_depth = random.randint(4, 7),
        xgb_jok_depth  = random.randint(4, 7),
        xgb_main_min_child_weight = random.choice([1.0, 2.0]),
        xgb_jok_min_child_weight  = random.choice([1.0, 2.0]),
        temp_main = random.uniform(1.3, 1.7),
        temp_joker= random.uniform(1.3, 1.7),
        topk_combo = random.randint(16, min(args.hard_cap_topk, 24)),
        seeds = random.randint(80, min(args.hard_cap_seeds, 250)),
        candidates = random.randint(9000, min(args.hard_cap_candidates, 30000)),
        pair_window = random.randint(300, 600),
        pair_weight = random.uniform(0.45, 0.85),
        spread_weight = random.uniform(0.10, 0.22),
    )

# ---------------- Feature helpers ----------------
def build_indicators(train_rows):
    N = len(train_rows)
    main_ind = np.zeros((MAIN_MAX, N), dtype=np.int16)
    joker_ind = np.zeros((JOKER_MAX, N), dtype=np.int16)
    for t, draw in enumerate(train_rows):
        mains = set(draw[:5]); j = draw[5]
        for n in mains:
            if 1 <= n <= MAIN_MAX: main_ind[n-1, t] = 1
        if 1 <= j <= JOKER_MAX: joker_ind[j-1, t] = 1
    return main_ind, joker_ind

def window_freq(ind_row, t_end, W, denom_per_draw):
    if t_end <= 0: return 0.0
    s = max(0, t_end - W); c = ind_row[s:t_end].sum(); d = t_end - s
    return float(c) / max(1, denom_per_draw * d)

def recency_gap(ind_row, t_end):
    if t_end <= 0: return 1.0
    idx = np.where(ind_row[:t_end] == 1)[0]
    if idx.size == 0: return 1.0
    gap = (t_end - 1) - idx[-1]
    return math.sqrt(gap) / math.sqrt(max(1, t_end))

def exp_decay_freq(ind_row, t_end, half_life, denom_per_draw):
    if t_end <= 0: return 0.0
    steps = np.arange(t_end)[::-1]
    w = 0.5 ** (steps / max(1e-9, half_life))
    w = w[:t_end]
    val = (ind_row[:t_end][::-1] * w).sum()
    norm = w.sum() * denom_per_draw
    return float(val) / max(1e-9, norm)

def short_streak(ind_row, t_end, W=12, denom_per_draw=5):
    return window_freq(ind_row, t_end, W, denom_per_draw)

def build_Xy(train_rows, main_ind, joker_ind, min_history, cfg: ModelConfig):
    N = len(train_rows)
    Xm_all, ym_all, Xj_all, yj_all = [], [], [], []
    for t in range(min_history, N):
        # mains
        present = set(train_rows[t][:5])
        Xm, ym = [], []
        for n in ALL_MAIN:
            row = main_ind[n-1, :]
            f50  = window_freq(row, t, 50,  5)
            f200 = window_freq(row, t, 200, 5)
            f800 = window_freq(row, t, 800, 5)
            fexp = exp_decay_freq(row, t, cfg.decay_half, 5)
            rec  = recency_gap(row, t)
            stk  = short_streak(row, t, 12, 5)
            Xm.append([n, f50, f200, f800, fexp, rec, stk])
            ym.append(1 if n in present else 0)
        Xm_all.append(np.array(Xm, float)); ym_all.append(np.array(ym, int))

        # joker
        present_j = train_rows[t][5]
        Xj, yj = [], []
        for j in ALL_JOKER:
            row = joker_ind[j-1, :]
            f50  = window_freq(row, t, 50,  1)
            f200 = window_freq(row, t, 200, 1)
            f800 = window_freq(row, t, 800, 1)
            fexp = exp_decay_freq(row, t, cfg.decay_half, 1)
            rec  = recency_gap(row, t)
            stk  = short_streak(row, t, 12, 1)
            Xj.append([j, f50, f200, f800, fexp, rec, stk])
            yj.append(1 if j == present_j else 0)
        Xj_all.append(np.array(Xj, float)); yj_all.append(np.array(yj, int))

    X_main = np.vstack(Xm_all); y_main = np.concatenate(ym_all)
    X_jok  = np.vstack(Xj_all); y_jok  = np.concatenate(yj_all)

    num_id_main = X_main[:, 0].astype(int); Xmf = X_main[:, 1:]
    num_id_jok  = X_jok[:, 0].astype(int);  Xjf = X_jok[:, 1:]
    return Xmf, y_main, Xjf, y_jok, num_id_main, num_id_jok

# ---------------- XGBoost GPU models ----------------
def train_models_gpu_xgb(Xmf, y_main, Xjf, y_jok, cfg, seed):
    """
    Throttled GPU training (~70–85% utilization on a 4070, typical).
    Tune n_estimators/max_bin/subsample/colsample_bytree to move load up/down.
    """
    common = dict(
        tree_method="hist",
        device="cuda",
        # throttle knobs ↓ (reduce these to lower load; increase to raise)
        max_bin=160,            # default 256; lower = less GPU work
        subsample=0.85,         # <1.0 uses a subset of rows per tree
        colsample_bytree=0.75,  # <1.0 uses a subset of features per tree
        learning_rate=0.08,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=0,
    )

    clf_main = xgb.XGBClassifier(
        n_estimators=min(getattr(cfg, "xgb_main_estimators", 700), 700),  # cap a bit for stability
        max_depth=min(getattr(cfg, "xgb_main_depth", 6), 6),
        min_child_weight=max(1.5, float(getattr(cfg, "xgb_main_min_child_weight", 1.0))),
        **common
    )

    clf_jok = xgb.XGBClassifier(
        n_estimators=min(getattr(cfg, "xgb_jok_estimators", 550), 550),
        max_depth=min(getattr(cfg, "xgb_jok_depth", 6), 6),
        min_child_weight=max(1.5, float(getattr(cfg, "xgb_jok_min_child_weight", 1.0))),
        **common
    )

    clf_main.fit(Xmf, y_main)
    clf_jok.fit(Xjf, y_jok)
    return clf_main, clf_jok

def apply_cooldown_probs(prob_vec, ids, ind_mat, k_recent):
    if k_recent <= 0: return prob_vec
    t_end = ind_mat.shape[1]
    cool = np.ones_like(prob_vec)
    for i, v in enumerate(ids):
        row = ind_mat[v-1,:]
        c = row[max(0, t_end-k_recent):t_end].sum()
        cool[i] = 1.0 / (1.0 + c)
    out = prob_vec * cool
    s = out.sum()
    return out / s if s > 0 else np.ones_like(out)/len(out)

def apply_temperature(probs, temp):
    if temp <= 1e-6: return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    logits /= max(1e-6, temp)
    x = np.exp(logits - logits.max())
    x /= x.sum()
    return x

def build_pair_scores(train_rows, cfg: ModelConfig):
    Wp = min(cfg.pair_window, len(train_rows))
    counts = {}
    if Wp > 0:
        for d in train_rows[-Wp:]:
            ms = sorted(d[:5])
            for a,b in combinations(ms, 2):
                counts[(a,b)] = counts.get((a,b), 0) + 1
    maxc = max(counts.values()) if counts else 1
    return {k: v/maxc for k,v in counts.items()}

def ticket_score(mains, p_main, pair_score, cfg: ModelConfig):
    base = sum(p_main.get(n,0.0) for n in mains)
    # pair synergy
    ps = 0.0
    ms = sorted(mains)
    for a,b in combinations(ms,2):
        if a>b: a,b=b,a
        ps += pair_score.get((a,b), 0.0)
    ps *= cfg.pair_weight
    spread = (max(ms) - min(ms)) / 44.0
    return base + ps + cfg.spread_weight * spread

def weighted_sample_no_replace(items, pmap, k):
    keys=[]
    for x in items:
        p=float(pmap.get(x,0.0))
        if p<=0: continue
        u=random.random()
        keys.append((math.log(u)/p, x))
    keys.sort()
    pick=[x for _,x in keys[:k]]
    if len(pick)<k:
        rem=[x for x in items if x not in pick]
        if rem: pick += random.sample(rem, k-len(pick))
    return tuple(sorted(pick))

def sample_joker(p_jok):
    r,acc=random.random(),0.0
    for j in sorted(ALL_JOKER):
        acc += p_jok.get(j, 0.0)
        if r<=acc: return j
    return random.choice(ALL_JOKER)

def predict_sets_once(train_rows, cfg: ModelConfig, seed_base: int):
    # Build indicators, features, train XGB models on GPU
    main_ind, joker_ind = build_indicators(train_rows)
    if len(train_rows) < args.min_history:
        return set()  # not enough history

    Xmf, y_main, Xjf, y_jok, _, _ = build_Xy(train_rows, main_ind, joker_ind, args.min_history, cfg)
    model_main, model_jok = train_models_gpu_xgb(Xmf, y_main, Xjf, y_jok, cfg, seed_base)

    # Predict next probs
    XmN, ids_m, XjN, ids_j = build_next_features(train_rows, main_ind, joker_ind, cfg)
    pm = model_main.predict_proba(XmN)[:,1]
    pj = model_jok.predict_proba(XjN)[:,1]

    # cooldown + temperature
    pm = apply_cooldown_probs(pm, ids_m, main_ind, cfg.cooldown)
    pj = apply_cooldown_probs(pj, ids_j, joker_ind, cfg.cooldown)
    pm = apply_temperature(pm, cfg.temp_main)
    pj = apply_temperature(pj, cfg.temp_joker)

    # normalize maps
    pm = pm / pm.sum() if pm.sum() > 0 else np.ones_like(pm)/len(pm)
    pj = pj / pj.sum() if pj.sum() > 0 else np.ones_like(pj)/len(pj)
    p_main = dict(zip(ids_m, pm))
    p_jok  = dict(zip(ids_j, pj))

    # pair synergy
    pair_score = build_pair_scores(train_rows, cfg)

    # deterministic scan (top-K mains × top-M jokers)
    found = set()
    K = max(5, min(cfg.topk_combo, len(ALL_MAIN)))
    topk = [n for n,_ in sorted(p_main.items(), key=lambda kv: kv[1], reverse=True)[:K]]
    top_jokers = [j for j,_ in sorted(p_jok.items(), key=lambda kv: kv[1], reverse=True)[:max(1, args.det_top_jokers)]]
    for comb in combinations(topk, 5):
        mains = tuple(sorted(comb))
        for jk in top_jokers:
            found.add((mains, jk))

    # stochastic exploration
    for s in range(cfg.seeds):
        random.seed(seed_base + s)
        np.random.seed(seed_base + s)
        for _ in range(cfg.candidates):
            mains = weighted_sample_no_replace(ALL_MAIN, p_main, 5)
            jk = sample_joker(p_jok)
            found.add((mains, jk))

    return found, p_main, p_jok, pair_score

def exact_match_in(found_sets, target_draw):
    mains = tuple(sorted(target_draw[:5])); jk = target_draw[5]
    return (mains, jk) in found_sets

def jaccard(a, b):
    sa, sb = set(a), set(b)
    inter = len(sa & sb); union = len(sa | sb)
    return inter / union if union else 0.0

def rank_and_pick(found_sets, p_main, p_jok, pair_score, cfg: ModelConfig, sets_count: int, diversity: float):
    ranked = sorted(
        found_sets,
        key=lambda ms: (ticket_score(ms[0], p_main, pair_score, cfg), p_jok.get(ms[1], 0.0)),
        reverse=True
    )
    picked = []
    for mains, jk in ranked:
        if all(jaccard(mains, pm) <= diversity for pm, _ in picked):
            picked.append((mains, jk))
        if len(picked) == sets_count:
            break
    return picked

# ---------------- Main search loop ----------------
start = time.time()
winning_cfg = None

for trial in range(1, args.max_trials + 1):
    cfg = sample_config()

    # keep random configs within the caps
    cfg.candidates = min(cfg.candidates, args.hard_cap_candidates)
    cfg.seeds = min(cfg.seeds, args.hard_cap_seeds)
    cfg.topk_combo = min(cfg.topk_combo, args.hard_cap_topk)

    train = train_prefix[:]  # base history without last 10
    ok = True
    if args.verbose:
        print(f"\n[Trial {trial}] config={asdict(cfg)}")

    for idx, target in enumerate(held10, start=1):
        found_sets, _, _, _ = predict_sets_once(train, cfg, seed_base=args.seed + trial*1000 + idx)
        if exact_match_in(found_sets, target):
            train.append(target)  # add back before moving to the next target
            if args.verbose:
                print(f"  Step {idx}/10: FOUND {target}")
        else:
            ok = False
            if args.verbose:
                print(f"  Step {idx}/10: NOT FOUND {target} -> restart")
            break

    if ok:
        winning_cfg = cfg
        elapsed = time.time() - start
        print("\nSUCCESS: Found a model that hits all 10 in sequence.")
        print(f"Tried {trial} configs in {elapsed/60:.1f} min.")
        print("\nWinning config:")
        print(json.dumps(asdict(cfg), indent=2))
        with open(args.save_best, "w", encoding="utf-8") as out:
            json.dump(asdict(cfg), out, indent=2)
        print(f"\nSaved to {args.save_best}")
        break

if winning_cfg is None:
    elapsed = time.time() - start
    print(f"\nFAILED: Tried {args.max-trials if hasattr(args,'max-trials') else args.max_trials} configs in {elapsed/60:.1f} min, no model hit all 10.")
    raise SystemExit(0)

# ---------------- Predict-next (optional) ----------------
if args.predict_next:
    # Retrain on FULL data using the winning config and output predicted sets
    full_train = chron[:]  # all historical draws
    found_sets, p_main, p_jok, pair_score = predict_sets_once(
        full_train, winning_cfg, seed_base=args.seed + 999999
    )
    picks = rank_and_pick(found_sets, p_main, p_jok, pair_score, winning_cfg, sets_count=args.sets, diversity=args.diversity)

    print("\n--- NEXT-DRAW: 6 SUGGESTED SETS (model-only, GPU) ---")
    for i, (mains, jk) in enumerate(picks, 1):
        print(f"Set {i}: {list(mains)} | Joker: {jk}")

    pool8 = sorted([n for n, _ in sorted(p_main.items(), key=lambda kv: kv[1], reverse=True)[:8]])
    print("\n--- 8-NUMBER POOL ---")
    print(pool8)
