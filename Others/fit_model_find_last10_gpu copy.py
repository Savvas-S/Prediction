#!/usr/bin/env python3
"""
fit_model_find_last10.py (FAST, GPU train + cached features; robust to messy CSV)

- Random-search hyperparams until one sequentially "finds" the last 10 draws
  (10th -> 1st), adding each found draw back to training before the next step.
- Saves the winning config as JSON.
- With --predict-next, retrains on ALL history using the winning config
  and prints suggested sets for the NEXT draw.

CSV: header + 6 ints per row: 5 mains (1..45) + joker (1..20)
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
import xgboost as xgb

# ---------------- CLI ----------------
p = argparse.ArgumentParser(description="FAST GPU model search for last-10; optional predict-next.")
p.add_argument("--csv", default="joker_results2_reversed.csv", help="CSV with header + 6 cols (5 mains + joker).")
p.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
p.add_argument("--max-trials", type=int, default=500, help="Max random configs to try.")
p.add_argument("--verbose", action="store_true", help="Verbose progress.")
p.add_argument("--save-best", default="winning_model_config.json", help="Where to write winning config JSON.")
p.add_argument("--min-history", type=int, default=80, help="Minimum draws before we start modeling.")

# Hard caps
p.add_argument("--hard-cap-candidates", type=int, default=40000, help="Cap on candidates per seed.")
p.add_argument("--hard-cap-seeds", type=int, default=400, help="Cap on seeds explored.")
p.add_argument("--hard-cap-topk", type=int, default=24, help="Cap on top-K mains for deterministic scan.")
p.add_argument("--det-top-jokers", type=int, default=5, help="Top-M jokers to try for each 5-combo deterministically.")

# Predict-next options
p.add_argument("--predict-next", action="store_true", help="After success, retrain on full data and predict next draw.")
p.add_argument("--sets", type=int, default=6, help="How many sets to output when predicting next draw.")
p.add_argument("--diversity", type=float, default=0.4, help="Jaccard max overlap among suggested sets (0..1).")

args = p.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

MAIN_MIN, MAIN_MAX = 1, 45
JOKER_MIN, JOKER_MAX = 1, 20
ALL_MAIN = np.arange(MAIN_MIN, MAIN_MAX + 1, dtype=np.int16)
ALL_JOKER = np.arange(JOKER_MIN, JOKER_MAX + 1, dtype=np.int16)

# ---------------- Robust CSV loader ----------------
def load_clean_rows(csv_path):
    raw = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        _ = next(r, None)  # header
        for row in r:
            if not row:
                continue
            vals = []
            for x in row[:6]:
                try:
                    vals.append(int(x))
                except Exception:
                    vals.append(None)
            # must have at least 6 ints
            if len(vals) < 6 or any(v is None for v in vals[:6]):
                continue
            mains, jk = vals[:5], vals[5]
            # enforce valid ranges
            if not all(MAIN_MIN <= n <= MAIN_MAX for n in mains):
                continue
            if not (JOKER_MIN <= jk <= JOKER_MAX):
                continue
            raw.append([*mains, jk])
    return raw

rows = load_clean_rows(args.csv)

if len(rows) < 200:
    raise SystemExit(f"Not enough usable rows in {args.csv}. Got {len(rows)}, need >= 200.")

# Latest-first -> chronological
chron = list(reversed(rows))
N = len(chron)
if N < args.min_history + 10:
    raise SystemExit("Not enough data to hold out last 10 and still meet min-history.")

# last 10 (chronological inside those)
held10 = chron[-10:]        # 10th, 9th, ..., 1st among the last 10
train_prefix = chron[:-10]  # base history

if args.verbose:
    print(f"[DATA] Total OK rows: {N} | Train prefix: {len(train_prefix)} | Held-out last10: {len(held10)}")

# ---------------- Model Config ----------------
@dataclass
class ModelConfig:
    decay_half: float
    cooldown: int
    # XGB sizes
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
    topk_combo: int
    seeds: int
    candidates: int
    # scoring
    pair_window: int
    pair_weight: float
    spread_weight: float

def sample_config() -> ModelConfig:
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

# ---------------- Indicators (once) ----------------
def build_indicators(train_rows):
    N = len(train_rows)
    main_ind = np.zeros((MAIN_MAX, N), dtype=np.int8)   # 45 x N
    joker_ind = np.zeros((JOKER_MAX, N), dtype=np.int8) # 20 x N
    for t, draw in enumerate(train_rows):
        mains = draw[:5]
        jk = draw[5]
        for n in mains:
            if MAIN_MIN <= n <= MAIN_MAX:
                main_ind[n-1, t] = 1
        if JOKER_MIN <= jk <= JOKER_MAX:
            joker_ind[jk-1, t] = 1
    return main_ind, joker_ind

# ---------------- Feature cache (per-trial) ----------------
class FeatureCache:
    """
    Precomputes all window sums, recency, and decay for a given decay_half.
    Lets us build features up to any time t with only slicing (no recompute).
    """
    def __init__(self, train_rows, decay_half, min_history):
        self.train_rows = train_rows
        self.N = len(train_rows)
        self.min_history = min_history

        self.main_ind, self.joker_ind = build_indicators(train_rows)
        self.cs_main = np.cumsum(self.main_ind, axis=1)
        self.cs_jok  = np.cumsum(self.joker_ind, axis=1)

        # Recency (since last seen) for mains/jokers at each t (using history < t)
        self.rec_main = np.zeros((MAIN_MAX, self.N+1), dtype=np.float32)
        self.rec_jok  = np.zeros((JOKER_MAX, self.N+1), dtype=np.float32)
        last_seen_main = -np.ones(MAIN_MAX, dtype=np.int32)
        last_seen_jok  = -np.ones(JOKER_MAX, dtype=np.int32)
        for t in range(1, self.N+1):
            gaps_m = np.where(last_seen_main >= 0, (t - 1) - last_seen_main, t - 1)
            gaps_j = np.where(last_seen_jok  >= 0, (t - 1) - last_seen_jok,  t - 1)
            norm = math.sqrt(max(1, t))
            self.rec_main[:, t] = np.sqrt(gaps_m) / norm
            self.rec_jok[:,  t] = np.sqrt(gaps_j) / norm

            # update last_seen with draw t-1 (guard ranges)
            d = train_rows[t-1]
            for n in d[:5]:
                if MAIN_MIN <= n <= MAIN_MAX:
                    last_seen_main[n-1] = t-1
            j = d[5]
            if JOKER_MIN <= j <= JOKER_MAX:
                last_seen_jok[j-1] = t-1

        # Decay (EWMA) for mains/jokers for this trial’s half-life
        alpha = 0.5 ** (1.0 / max(1e-9, float(decay_half)))
        self.ew_main = np.zeros_like(self.main_ind, dtype=np.float32)
        self.ew_jok  = np.zeros_like(self.joker_ind, dtype=np.float32)
        for t in range(self.N):
            if t == 0:
                self.ew_main[:, t] = self.main_ind[:, t]
                self.ew_jok[:, t]  = self.joker_ind[:, t]
            else:
                self.ew_main[:, t] = alpha * self.ew_main[:, t-1] + self.main_ind[:, t]
                self.ew_jok[:,  t] = alpha * self.ew_jok[:,  t-1] + self.joker_ind[:, t]
        # weights denom for history < t
        self.denom_w = np.empty(self.N+1, dtype=np.float64)
        self.denom_w[0] = 1.0
        for t in range(1, self.N+1):
            self.denom_w[t] = (1.0 - alpha**t) / (1.0 - alpha)

    @staticmethod
    def _win_sum(cs, t, W):
        if t <= 0:
            return np.zeros(cs.shape[0], dtype=np.int32)
        s = max(0, t - W)
        right = cs[:, t-1]
        left  = cs[:, s-1] if s > 0 else 0
        return right - left

    def build_Xy_up_to(self, t_end):
        """
        Build X,y using draws [0..t_end-1] for training *labels at all times < t_end*.
        Returns (Xmf, y_main, Xjf, y_jok) ready for training.
        """
        N = t_end
        assert N >= self.min_history, "not enough history"

        Xm_list, ym_list = [], []
        Xj_list, yj_list = [], []

        for t in range(self.min_history, N):
            draws_50  = min(t, 50)
            draws_200 = min(t, 200)
            draws_800 = min(t, 800)
            draws_12  = min(t, 12)

            f50  = self._win_sum(self.cs_main, t, 50)  / max(1, 5 * draws_50)
            f200 = self._win_sum(self.cs_main, t, 200) / max(1, 5 * draws_200)
            f800 = self._win_sum(self.cs_main, t, 800) / max(1, 5 * draws_800)
            fexp = (self.ew_main[:, t-1] / max(1e-9, self.denom_w[t] * 5)) if t > 0 else np.zeros(MAIN_MAX, dtype=np.float32)
            rec  = self.rec_main[:, t]
            stk  = self._win_sum(self.cs_main, t, 12) / max(1, 5 * draws_12)

            Xm = np.column_stack([f50, f200, f800, fexp, rec, stk]).astype(np.float32)

            present = np.zeros(MAIN_MAX, dtype=np.int8)
            row = self.train_rows[t] if t < len(self.train_rows) else None
            if row and len(row) >= 5:
                for n in row[:5]:
                    if MAIN_MIN <= n <= MAIN_MAX:
                        present[n-1] = 1

            Xm_list.append(Xm)
            ym_list.append(present)

            f50j  = self._win_sum(self.cs_jok, t, 50)  / max(1, 1 * draws_50)
            f200j = self._win_sum(self.cs_jok, t, 200) / max(1, 1 * draws_200)
            f800j = self._win_sum(self.cs_jok, t, 800) / max(1, 1 * draws_800)
            fexpj = (self.ew_jok[:, t-1] / max(1e-9, self.denom_w[t] * 1)) if t > 0 else np.zeros(JOKER_MAX, dtype=np.float32)
            recj  = self.rec_jok[:, t]
            stkj  = self._win_sum(self.cs_jok, t, 12) / max(1, 1 * draws_12)

            Xj = np.column_stack([f50j, f200j, f800j, fexpj, recj, stkj]).astype(np.float32)

            yj = np.zeros(JOKER_MAX, dtype=np.int8)
            if row and len(row) >= 6:
                jk = row[5]
                if JOKER_MIN <= jk <= JOKER_MAX:
                    yj[jk-1] = 1

            Xj_list.append(Xj)
            yj_list.append(yj)

        Xmf = np.vstack(Xm_list)   # (45*(N-min_hist), 6)
        y_main = np.concatenate(ym_list)
        Xjf = np.vstack(Xj_list)   # (20*(N-min_hist), 6)
        y_jok = np.concatenate(yj_list)
        return Xmf, y_main, Xjf, y_jok

    def next_features(self, t_end):
        """Features to predict draw at t_end (using history < t_end)."""
        t = t_end
        draws_50  = min(t, 50)
        draws_200 = min(t, 200)
        draws_800 = min(t, 800)
        draws_12  = min(t, 12)

        f50  = self._win_sum(self.cs_main, t, 50)  / max(1, 5 * draws_50)
        f200 = self._win_sum(self.cs_main, t, 200) / max(1, 5 * draws_200)
        f800 = self._win_sum(self.cs_main, t, 800) / max(1, 5 * draws_800)
        fexp = (self.ew_main[:, t-1] / max(1e-9, self.denom_w[t] * 5)) if t > 0 else np.zeros(MAIN_MAX, dtype=np.float32)
        rec  = self.rec_main[:, t]
        stk  = self._win_sum(self.cs_main, t, 12) / max(1, 5 * draws_12)
        XmN = np.column_stack([f50, f200, f800, fexp, rec, stk]).astype(np.float32)

        f50j  = self._win_sum(self.cs_jok, t, 50)  / max(1, 1 * draws_50)
        f200j = self._win_sum(self.cs_jok, t, 200) / max(1, 1 * draws_200)
        f800j = self._win_sum(self.cs_jok, t, 800) / max(1, 1 * draws_800)
        fexpj = (self.ew_jok[:, t-1] / max(1e-9, self.denom_w[t] * 1)) if t > 0 else np.zeros(JOKER_MAX, dtype=np.float32)
        recj  = self.rec_jok[:, t]
        stkj  = self._win_sum(self.cs_jok, t, 12) / max(1, 1 * draws_12)
        XjN = np.column_stack([f50j, f200j, f800j, fexpj, recj, stkj]).astype(np.float32)

        return XmN, XjN

# ---------------- GPU models ----------------
def train_models_gpu_xgb(Xmf, y_main, Xjf, y_jok, cfg, seed):
    common = dict(
        tree_method="hist",
        device="cuda",
        predictor="gpu_predictor",   # keep prediction on GPU
        max_bin=256,
        subsample=0.92,
        colsample_bytree=0.88,
        learning_rate=0.08,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=0,
        verbosity=0,
    )
    clf_main = xgb.XGBClassifier(
        n_estimators=min(getattr(cfg, "xgb_main_estimators", 850), 900),
        max_depth=min(getattr(cfg, "xgb_main_depth", 7), 7),
        min_child_weight=max(1.0, float(getattr(cfg, "xgb_main_min_child_weight", 1.0))),
        **common
    )
    clf_jok = xgb.XGBClassifier(
        n_estimators=min(getattr(cfg, "xgb_jok_estimators", 700), 750),
        max_depth=min(getattr(cfg, "xgb_jok_depth", 7), 7),
        min_child_weight=max(1.0, float(getattr(cfg, "xgb_jok_min_child_weight", 1.0))),
        **common
    )
    clf_main.fit(Xmf, y_main)
    clf_jok.fit(Xjf, y_jok)
    return clf_main, clf_jok

# ---------------- Helpers ----------------
def apply_cooldown_probs(prob_vec, ind_mat, k_recent):
    if k_recent <= 0: return prob_vec
    t_end = ind_mat.shape[1]
    tail = ind_mat[:, max(0, t_end-k_recent):t_end].sum(axis=1).astype(np.float32)  # (45,) or (20,)
    cool = 1.0 / (1.0 + tail)
    out = prob_vec * cool
    s = out.sum()
    return out / s if s > 0 else np.ones_like(out) / len(out)

def apply_temperature(probs, temp):
    if temp <= 1e-6: return probs
    logits = np.log(np.clip(probs, 1e-9, 1.0)) / max(1e-6, temp)
    x = np.exp(logits - logits.max())
    return x / x.sum()

def build_pair_scores(train_rows, Wp):
    Wp = min(Wp, len(train_rows))
    counts = {}
    if Wp > 0:
        for d in train_rows[-Wp:]:
            ms = sorted(d[:5])
            for a, b in combinations(ms, 2):
                counts[(a, b)] = counts.get((a, b), 0) + 1
    maxc = max(counts.values()) if counts else 1
    return {k: v / maxc for k, v in counts.items()}

def ticket_score(mains, p_main, pair_score, pair_weight, spread_weight):
    base = sum(p_main.get(n, 0.0) for n in mains)
    ps = 0.0
    ms = sorted(mains)
    for a, b in combinations(ms, 2):
        if a > b: a, b = b, a
        ps += pair_score.get((a, b), 0.0)
    ps *= pair_weight
    spread = (ms[-1] - ms[0]) / 44.0
    return base + ps + spread_weight * spread

def _gumbel_keys(logp, n, rng):
    g = rng.gumbel(size=(n, logp.shape[0])).astype(np.float32)
    return logp[None, :] + g

def generate_candidates(p_main, p_jok, n_candidates, rng_seed, topk_main_for_det, topm_jokers_det):
    rng = np.random.default_rng(rng_seed)
    mains = ALL_MAIN
    pjoks = ALL_JOKER

    pm = np.array([p_main.get(int(i), 0.0) for i in mains], dtype=np.float32)
    pj = np.array([p_jok.get(int(i), 0.0)  for i in pjoks], dtype=np.float32)
    pm /= pm.sum() if pm.sum() > 0 else 1.0
    pj /= pj.sum() if pj.sum() > 0 else 1.0

    logpm = np.log(pm + 1e-12).astype(np.float32)
    k_m = _gumbel_keys(logpm, n_candidates, rng)
    idx5 = np.argpartition(k_m, -5, axis=1)[:, -5:]
    mains5 = np.sort(mains[idx5], axis=1)

    logpj = np.log(pj + 1e-12).astype(np.float32)
    k_j = _gumbel_keys(logpj, n_candidates, rng)
    jk_idx = np.argmax(k_j, axis=1)
    jokers = pjoks[jk_idx]

    found = {(tuple(row.tolist()), int(jk)) for row, jk in zip(mains5, jokers)}

    if topk_main_for_det and topm_jokers_det:
        topk = [n for n, _ in sorted(p_main.items(), key=lambda kv: kv[1], reverse=True)[:topk_main_for_det]]
        topmj = [j for j, _ in sorted(p_jok.items(),  key=lambda kv: kv[1], reverse=True)[:topm_jokers_det]]
        for comb in combinations(topk, 5):
            for jk in topmj:
                found.add((tuple(sorted(comb)), jk))
    return found

def exact_match_in(found_sets, target_draw):
    mains = tuple(sorted(target_draw[:5])); jk = target_draw[5]
    return (mains, jk) in found_sets

def jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0

def rank_and_pick(found_sets, p_main, p_jok, pair_score, cfg, sets_count, diversity):
    ranked = sorted(
        found_sets,
        key=lambda ms: (ticket_score(ms[0], p_main, pair_score, cfg.pair_weight, cfg.spread_weight),
                        p_jok.get(ms[1], 0.0)),
        reverse=True
    )
    out = []
    for mains, jk in ranked:
        if all(jaccard(mains, pm) <= diversity for pm, _ in out):
            out.append((mains, jk))
        if len(out) == sets_count:
            break
    return out

# ---------------- One prediction cycle ----------------
def predict_sets_once(cache, cfg: ModelConfig, t_end: int, seed_base: int):
    # Training data = history up to t_end (exclusive)
    Xmf, y_main, Xjf, y_jok = cache.build_Xy_up_to(t_end)

    model_main, model_jok = train_models_gpu_xgb(Xmf, y_main, Xjf, y_jok, cfg, seed=seed_base)

    # Predict probs for draw at t_end
    XmN, XjN = cache.next_features(t_end)
    pm = model_main.predict_proba(XmN)[:, 1]
    pj = model_jok.predict_proba(XjN)[:, 1]

    # cooldown + temperature
    pm = apply_cooldown_probs(pm, cache.main_ind, cfg.cooldown)
    pj = apply_cooldown_probs(pj, cache.joker_ind, cfg.cooldown)
    pm = apply_temperature(pm, cfg.temp_main)
    pj = apply_temperature(pj, cfg.temp_joker)

    # normalize to dicts
    pm = pm / pm.sum() if pm.sum() > 0 else np.ones_like(pm) / len(pm)
    pj = pj / pj.sum() if pj.sum() > 0 else np.ones_like(pj) / len(pj)
    p_main = {int(n): float(p) for n, p in zip(ALL_MAIN, pm)}
    p_jok  = {int(j): float(p) for j, p in zip(ALL_JOKER, pj)}

    pair_score = build_pair_scores(cache.train_rows[:t_end], cfg.pair_window)

    # deterministic + stochastic candidates
    K = max(5, min(cfg.topk_combo, len(ALL_MAIN)))
    topk = [n for n, _ in sorted(p_main.items(), key=lambda kv: kv[1], reverse=True)[:K]]
    topmj = [j for j, _ in sorted(p_jok.items(),  key=lambda kv: kv[1], reverse=True)[:max(1, args.det_top_jokers)]]
    found = set()
    for comb in combinations(topk, 5):
        for jk in topmj:
            found.add((tuple(sorted(comb)), jk))

    found |= generate_candidates(
        p_main, p_jok,
        n_candidates=cfg.seeds * cfg.candidates,
        rng_seed=seed_base,
        topk_main_for_det=cfg.topk_combo,
        topm_jokers_det=args.det_top_jokers
    )

    return found, p_main, p_jok, pair_score

# ---------------- Main search ----------------
start = time.time()
winning_cfg = None

for trial in range(1, args.max_trials + 1):
    cfg = sample_config()

    # clamp inside caps
    cfg.candidates = min(cfg.candidates, args.hard_cap_candidates)
    cfg.seeds      = min(cfg.seeds,      args.hard_cap_seeds)
    cfg.topk_combo = min(cfg.topk_combo, args.hard_cap_topk)

    if args.verbose:
        print(f"\n[Trial {trial}] config={asdict(cfg)}")

    # Build per-trial cache ONCE (decay_half-specific)
    # cache = FeatureCache(train_prefix, decay_half=cfg.decay_half, min_history=args.min_history)
    cache_rows = train_prefix + held10      # <-- include the 10 held draws so t_end can advance safely
    cache = FeatureCache(cache_rows, decay_half=cfg.decay_half, min_history=args.min_history)
    ok = True
    # Steps: find held10[0] at t_end=len(train_prefix), add; then held10[1], ...
    t_end = len(train_prefix)
    for idx, target in enumerate(held10, start=1):
        if args.verbose:
            print(f"  -> Step {idx}/10: training with {t_end} draws ...", flush=True)
        found_sets, _, _, _ = predict_sets_once(cache, cfg, t_end, seed_base=args.seed + trial*1000 + idx)
        if exact_match_in(found_sets, target):
            t_end += 1  # add the draw
            if args.verbose:
                print(f"     FOUND {target}")
        else:
            ok = False
            if args.verbose:
                print(f"     NOT FOUND {target} -> restart")
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
    print(f"\nFAILED: Tried {args.max_trials} configs in {elapsed/60:.1f} min, no model hit all 10.")
    raise SystemExit(0)

# ---------------- Predict-next (optional) ----------------
if args.predict_next:
    if args.verbose:
        print("\n[Predict-next] Rebuilding features on FULL history ...", flush=True)
    cache_full = FeatureCache(chron, decay_half=winning_cfg.decay_half, min_history=args.min_history)
    t_end_full = len(chron)

    found_sets, p_main, p_jok, pair_score = predict_sets_once(
        cache_full, winning_cfg, t_end_full, seed_base=args.seed + 999999
    )
    picks = rank_and_pick(found_sets, p_main, p_jok, pair_score, winning_cfg,
                          sets_count=args.sets, diversity=args.diversity)

    print("\n--- NEXT-DRAW: 6 SUGGESTED SETS (GPU-trained) ---")
    for i, (mains, jk) in enumerate(picks, 1):
        print(f"Set {i}: {list(mains)} | Joker: {jk}")

    pool8 = sorted([n for n, _ in sorted(p_main.items(), key=lambda kv: kv[1], reverse=True)[:8]])
    print("\n--- 8-NUMBER POOL ---")
    print(pool8)
