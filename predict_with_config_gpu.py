#!/usr/bin/env python3
import csv, json, argparse, random, math
from itertools import combinations
from dataclasses import dataclass
import numpy as np
import xgboost as xgb

MAIN_MIN, MAIN_MAX = 1, 45
JOKER_MIN, JOKER_MAX = 1, 20
ALL_MAIN = list(range(MAIN_MIN, MAIN_MAX+1))
ALL_JOKER = list(range(JOKER_MIN, JOKER_MAX+1))

@dataclass
class ModelConfig:
    decay_half: float
    cooldown: int
    xgb_main_estimators: int
    xgb_jok_estimators: int
    xgb_main_depth: int
    xgb_jok_depth: int
    xgb_main_min_child_weight: float
    xgb_jok_min_child_weight: float
    temp_main: float
    temp_joker: float
    topk_combo: int
    seeds: int
    candidates: int
    diversity: float
    pair_window: int
    pair_weight: float
    spread_weight: float

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return ModelConfig(**d)

def window_freq(ind_row, t_end, W, denom_per_draw):
    if t_end <= 0: return 0.0
    s = max(0, t_end-W); c = ind_row[s:t_end].sum(); d = t_end - s
    return float(c) / max(1, denom_per_draw * d)

def recency_gap(ind_row, t_end):
    if t_end <= 0: return 1.0
    idx = np.where(ind_row[:t_end] == 1)[0]
    if idx.size == 0: return 1.0
    gap = (t_end - 1) - idx[-1]
    return (gap ** 0.5) / (max(1, t_end) ** 0.5)

def exp_decay_freq(ind_row, t_end, half_life, denom_per_draw):
    if t_end <= 0: return 0.0
    steps = np.arange(t_end)[::-1]
    w = 0.5 ** (steps / max(1e-9, half_life))
    val = (ind_row[:t_end][::-1] * w).sum()
    return float(val) / max(1e-9, w.sum() * denom_per_draw)

def build_indicators(rows):
    N = len(rows)
    M = np.zeros((MAIN_MAX, N), dtype=np.int8)
    J = np.zeros((JOKER_MAX, N), dtype=np.int8)
    for t, d in enumerate(rows):
        for n in d[:5]:
            if 1 <= n <= MAIN_MAX: M[n-1, t] = 1
        j = d[5]
        if 1 <= j <= JOKER_MAX: J[j-1, t] = 1
    return M, J

def build_training(rows, M, J, cfg: ModelConfig, min_history: int):
    N = len(rows)
    Xm, ym, Xj, yj = [], [], [], []
    for t in range(min_history, N):
        present = set(rows[t][:5])
        for n in ALL_MAIN:
            r = M[n-1, :]
            f50  = window_freq(r, t, 50, 5)
            f200 = window_freq(r, t, 200, 5)
            f800 = window_freq(r, t, 800, 5)
            fexp = exp_decay_freq(r, t, cfg.decay_half, 5)
            rec  = recency_gap(r, t)
            stk  = window_freq(r, t, 12, 5)
            Xm.append([f50, f200, f800, fexp, rec, stk]); ym.append(1 if n in present else 0)
        pj = rows[t][5]
        for j in ALL_JOKER:
            r = J[j-1, :]
            f50  = window_freq(r, t, 50, 1)
            f200 = window_freq(r, t, 200, 1)
            f800 = window_freq(r, t, 800, 1)
            fexp = exp_decay_freq(r, t, cfg.decay_half, 1)
            rec  = recency_gap(r, t)
            stk  = window_freq(r, t, 12, 1)
            Xj.append([f50, f200, f800, fexp, rec, stk]); yj.append(1 if j == pj else 0)
    return np.array(Xm, float), np.array(ym, int), np.array(Xj, float), np.array(yj, int)

def train_gpu_xgb(Xm, ym, Xj, yj, cfg: ModelConfig, seed: int):
    common = dict(tree_method="hist", device="cuda", subsample=0.9, colsample_bytree=0.8,
                  learning_rate=0.08, reg_lambda=1.0, reg_alpha=0.0, random_state=seed,
                  eval_metric="logloss", n_jobs=0)
    m_main = xgb.XGBClassifier(n_estimators=cfg.xgb_main_estimators, max_depth=cfg.xgb_main_depth,
                               min_child_weight=cfg.xgb_main_min_child_weight, **common)
    m_jok  = xgb.XGBClassifier(n_estimators=cfg.xgb_jok_estimators,  max_depth=cfg.xgb_jok_depth,
                               min_child_weight=cfg.xgb_jok_min_child_weight,  **common)
    m_main.fit(Xm, ym)
    m_jok.fit(Xj, yj)
    return m_main, m_jok

def predict_next_probs(rows, M, J, cfg: ModelConfig, model_main, model_jok, min_history: int):
    t = len(rows)  # predict the next after last row
    XmN, ids_m, XjN, ids_j = [], [], [], []
    for n in ALL_MAIN:
        r = M[n-1, :]
        f50  = window_freq(r, t, 50, 5)
        f200 = window_freq(r, t, 200, 5)
        f800 = window_freq(r, t, 800, 5)
        fexp = exp_decay_freq(r, t, cfg.decay_half, 5)
        rec  = recency_gap(r, t)
        stk  = window_freq(r, t, 12, 5)
        XmN.append([f50, f200, f800, fexp, rec, stk]); ids_m.append(n)
    for j in ALL_JOKER:
        r = J[j-1, :]
        f50  = window_freq(r, t, 50, 1)
        f200 = window_freq(r, t, 200, 1)
        f800 = window_freq(r, t, 800, 1)
        fexp = exp_decay_freq(r, t, cfg.decay_half, 1)
        rec  = recency_gap(r, t)
        stk  = window_freq(r, t, 12, 1)
        XjN.append([f50, f200, f800, fexp, rec, stk]); ids_j.append(j)
    XmN = np.array(XmN, float); XjN = np.array(XjN, float)
    pm = model_main.predict_proba(XmN)[:, 1]
    pj = model_jok.predict_proba(XjN)[:, 1]
    # cooldown
    def apply_cooldown(p, ids, Ind, k):
        if k <= 0: return p
        t_end = Ind.shape[1]
        cool = np.ones_like(p)
        for i, v in enumerate(ids):
            c = Ind[v-1, max(0,t_end-k):t_end].sum()
            cool[i] = 1.0/(1.0+c)
        p2 = p * cool; s = p2.sum()
        return p2 / s if s > 0 else np.ones_like(p)/len(p)
    pm = apply_cooldown(pm, ids_m, M, cfg.cooldown)
    pj = apply_cooldown(pj, ids_j, J, cfg.cooldown)
    # temperature
    def temp_softmax(p, temp):
        logits = np.log(np.clip(p, 1e-12, 1.0))/max(1e-6, temp)
        x = np.exp(logits - logits.max()); x /= x.sum()
        return x
    pm = temp_softmax(pm, cfg.temp_main)
    pj = temp_softmax(pj, cfg.temp_joker)
    return dict(zip(ids_m, pm)), dict(zip(ids_j, pj))

def build_pair_scores(rows, cfg: ModelConfig):
    W = min(cfg.pair_window, len(rows))
    cnt = {}
    for d in rows[-W:]:
        ms = sorted(d[:5])
        for a,b in combinations(ms,2):
            cnt[(a,b)] = cnt.get((a,b), 0) + 1
    mx = max(cnt.values(), default=1)
    return {k: v/mx for k,v in cnt.items()}

def ticket_score(mains, p_main, pair_score, cfg: ModelConfig):
    base = sum(p_main.get(n,0.0) for n in mains)
    ps = 0.0
    ms = sorted(mains)
    for a,b in combinations(ms,2):
        if a>b: a,b=b,a
        ps += pair_score.get((a,b), 0.0)
    ps *= cfg.pair_weight
    spread = (max(ms)-min(ms))/44.0
    return base + ps + cfg.spread_weight*spread

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
        acc += p_jok.get(j,0.0)
        if r<=acc: return j
    return random.choice(ALL_JOKER)

def jaccard(a,b):
    sa,sb=set(a),set(b)
    u=len(sa|sb); i=len(sa&sb)
    return i/u if u else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="joker_results.csv")
    ap.add_argument("--config", default="winning_model_config.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--min-history", type=int, default=80)
    ap.add_argument("--top-jokers", type=int, default=5, help="Deterministic scan uses top-M jokers per combo.")
    ap.add_argument("--sets", type=int, default=6)
    args2 = ap.parse_args()

    cfg = load_config(args2.config)
    # load data (latest first)
    rows = []
    with open(args2.csv, "r", encoding="utf-8") as f:
        r = csv.reader(f); _ = next(r, None)
        for row in r:
            if len(row) >= 6:
                try: rows.append(list(map(int, row[:6])))
                except: pass
    if len(rows) < args2.min_history + 10:
        raise SystemExit("Not enough data.")

    # chronological
    chron = list(reversed(rows))
    M, J = build_indicators(chron)

    # training (use ALL draws now)
    Xm, ym, Xj, yj = build_training(chron, M, J, cfg, args2.min_history)
    model_main, model_jok = train_gpu_xgb(Xm, ym, Xj, yj, cfg, seed=123)

    # probabilities for next draw
    p_main, p_jok = predict_next_probs(chron, M, J, cfg, model_main, model_jok, args2.min_history)

    # pair synergy
    pair_score = build_pair_scores(chron, cfg)

    # deterministic scan: top-K mains × top-M jokers
    K = max(5, min(cfg.topk_combo, len(ALL_MAIN)))
    topk_mains = [n for n,_ in sorted(p_main.items(), key=lambda kv: kv[1], reverse=True)[:K]]
    topj = [j for j,_ in sorted(p_jok.items(), key=lambda kv: kv[1], reverse=True)[:max(1,args2.top_jokers)]]

    found = set()
    for comb in combinations(topk_mains, 5):
        for jk in topj:
            found.add((tuple(sorted(comb)), jk))

    # stochastic
    random.seed(123); np.random.seed(123)
    for s in range(cfg.seeds):
        random.seed(123+s); np.random.seed(123+s)
        for _ in range(cfg.candidates):
            mains = weighted_sample_no_replace(ALL_MAIN, p_main, 5)
            jk = sample_joker(p_jok)
            found.add((mains, jk))

    ranked = sorted(found, key=lambda ms: (ticket_score(ms[0], p_main, pair_score, cfg),
                                           p_jok.get(ms[1],0.0)), reverse=True)

    # pick diverse sets
    picked=[]
    for mains, jk in ranked:
        if all(jaccard(mains, pm)<=cfg.diversity for pm,_ in picked):
            picked.append((mains, jk))
        if len(picked) == args2.sets: break

    print("\n--- Predicted sets (model-only, GPU) ---")
    for i,(m,jk) in enumerate(picked,1):
        print(f"Set {i}: {list(m)} | Joker: {jk}")

    pool8 = sorted([n for n,_ in sorted(p_main.items(), key=lambda kv: kv[1], reverse=True)[:8]])
    print("\n--- 8-number pool ---")
    print(pool8)

if __name__ == "__main__":
    main()
