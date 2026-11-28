"""FLIDS Day 3 Extension: Chronological Temporal Holdout (NF-UNSW-NB15-v3) + Matched Feature Ablation.

Part 1 — Strict Chronological Holdout (`temporal_cache.npz`, 100,276 flows, 5 time blocks):
    - Fit: Blocks 0-1 (3 Dirichlet alpha=0.5 FL clients)
    - Early Stopping / Model Selection: Block 2
    - Source Calibration: Block 3
    - Future Target Test: Block 4 (strictly forward in time, zero overlap)
    - Methods (8 x 5 seeds = 40 runs):
      LogisticRegression, RandomForest, LightGBM, FedAvg, FedNova, SCAFFOLD, FedLogReg, FedBN_TTA

Part 2 — Matched, Retuned NetFlow Feature Ablation (`nf_cache.npz`, 93,810 flows):
    - Feature variants:
      * no_ports (39 features: drops L4_SRC_PORT, L4_DST_PORT)
      * no_ports_ttl (37 features: drops L4_SRC_PORT, L4_DST_PORT, MIN_TTL, MAX_TTL)
    - Exact same split digests, 4-candidate inner LOSO selection, and 150-round training protocol
    - Methods (5 core x 2 variants x 4 domains x 5 seeds = 200 runs):
      LogisticRegression, LightGBM, FedLogReg, FedAvg, FedNova

Usage:
    python experiments/day3/day3_temporal_ablation.py --input /path/to/FLIDS_Master_V3_Fresh_Results.zip --work ./day3_work
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from scipy.stats import binom
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

SEEDS = [0, 42, 3407, 123, 2026]
DOMAINS = ["NF-UNSW-NB15", "NF-ToN-IoT", "NF-BoT-IoT", "NF-CSE-CIC-IDS2018"]
FEATURES = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]

TEMPORAL_METHODS = [
    "LogisticRegression", "RandomForest", "LightGBM",
    "FedAvg", "FedNova", "SCAFFOLD", "FedLogReg", "FedBN_TTA",
]
ABLATION_METHODS = ["LogisticRegression", "LightGBM", "FedLogReg", "FedAvg", "FedNova"]
ABLATION_VARIANTS = {
    "no_ports": ["L4_SRC_PORT", "L4_DST_PORT"],
    "no_ports_ttl": ["L4_SRC_PORT", "L4_DST_PORT", "MIN_TTL", "MAX_TTL"],
}


def make_seed(*args) -> int:
    return int.from_bytes(hashlib.sha256(repr(args).encode()).digest()[:4], "little")


def make_rng(*args) -> np.random.Generator:
    return np.random.default_rng(make_seed(*args))


def signed_log(x: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(x, dtype=np.float64), -1e12, 1e12)
    return np.clip(np.sign(c) * np.log1p(np.abs(c)), -18.0, 18.0).astype(np.float32)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def npwrite(path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **kw)
    tmp.replace(path)


def emp_cut(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if not len(s):
        return float("inf")
    k = math.floor(alpha * len(s))
    return float(np.nextafter(s[len(s) - k - 1], np.inf))


def wilks_cut(scores: np.ndarray, alpha: float, delta: float = 0.05) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    k = int(binom.ppf(1.0 - delta, n, 1.0 - alpha)) + 1
    return float(np.nextafter(s[k - 1], np.inf)) if k <= n else float("inf")


def pauc(y: np.ndarray, s: np.ndarray, max_fpr=None) -> float:
    return float(roc_auc_score(y, s, max_fpr=max_fpr))


def eval_op(y: np.ndarray, s: np.ndarray, thr: float):
    n = s[y == 0]
    p = s[y == 1]
    fp = int((n >= thr).sum())
    tp = int((p >= thr).sum())
    return fp / len(n), tp / len(p), fp, tp


def groupsplit(ids, y, s, frac):
    a, b = train_test_split(np.asarray(ids), test_size=frac, random_state=s, stratify=y[ids])
    return np.sort(a), np.sort(b)


def split_outer_nf(y, d, held, s):
    src = np.flatnonzero(~np.isin(d, held))
    test = np.flatnonzero(np.isin(d, held))
    domains = sorted(set(d[src]))
    fit, stop, cal, st = [], [], [], []
    for dom in domains:
        ids = src[d[src] == dom]
        a, b = groupsplit(ids, y, make_seed(s, dom, "outer"), 0.30)
        c, e = groupsplit(b, y, make_seed(s, dom, "cal"), 2 / 3)
        g, h = groupsplit(e, y, make_seed(s, dom, "st"), 0.5)
        fit.extend(a); stop.extend(c); cal.extend(g); st.extend(h)
    return tuple(np.array(v, dtype=int) for v in (fit, stop, cal, st, test))


def dirichlet_clients(ids, y, s, K=3, alpha=0.5):
    r = make_rng(s, "dirichlet_temporal")
    buckets = [[] for _ in range(K)]
    for lab in (0, 1):
        a = r.permutation(ids[y[ids] == lab])
        p = r.dirichlet(np.full(K, alpha))
        chunks = np.split(a, (np.cumsum(p)[:-1] * len(a)).astype(int))
        for j, c in enumerate(chunks):
            buckets[j].extend(c)
    return [np.asarray(b, dtype=int) for b in buckets if len(b) >= 4]


def candidate_grid(method: str):
    if method == "LogisticRegression":
        return [dict(C=v) for v in (0.01, 0.1, 1.0, 10.0)]
    if method == "RandomForest":
        return [dict(max_depth=d, min_samples_leaf=l) for d, l in [(12, 1), (None, 1), (16, 5), (None, 5)]]
    if method == "LightGBM":
        return [dict(learning_rate=l, num_leaves=n) for l, n in [(0.03, 31), (0.1, 31), (0.03, 63), (0.1, 63)]]
    if method == "FedLogReg":
        return [dict(lr=0.01, wd=1e-2), dict(lr=0.01, wd=1e-3), dict(lr=0.03, wd=1e-3), dict(lr=0.03, wd=1e-4)]
    if method == "SCAFFOLD":
        return [dict(lr=l) for l in (0.001, 0.003, 0.01, 0.03)]
    return [dict(lr=l) for l in (0.0003, 0.001, 0.003, 0.01)]


class LinearLogReg(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPBN(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def score_model(net: nn.Module, xx: torch.Tensor, ids: np.ndarray) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for chunk in np.array_split(ids, max(1, math.ceil(len(ids) / 4096))):
            out.append(torch.sigmoid(net(xx[chunk])).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def adapt_bn_unsupervised(nets, xx: torch.Tensor, unlabeled_ids: np.ndarray, batch_size: int = 512):
    adapted = []
    for src_net in nets:
        net = MLPBN(xx.shape[1]).to(xx.device)
        net.load_state_dict({k: v.detach().clone() for k, v in src_net.state_dict().items()})
        for m in net.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.reset_running_stats()
                m.momentum = None
        net.train()
        with torch.no_grad():
            for chunk in np.array_split(unlabeled_ids, max(1, math.ceil(len(unlabeled_ids) / batch_size))):
                if len(chunk) >= 2:
                    _ = net(xx[chunk])
        net.eval()
        adapted.append(net)
    return adapted


def fit_method(raw, y, d, tr, va, method, cfg, s, opts, clients_override=None):
    t0 = time.perf_counter()
    trans = signed_log(raw)
    sc = StandardScaler().fit(trans[tr])
    X = np.clip(sc.transform(trans).astype(np.float32), -20.0, 20.0)

    if method in ("LogisticRegression", "RandomForest", "LightGBM"):
        if method == "LogisticRegression":
            m = LogisticRegression(C=cfg["C"], max_iter=2000, random_state=s)
            m.fit(X[tr], y[tr])
        elif method == "RandomForest":
            m = RandomForestClassifier(n_estimators=10 if opts["smoke"] else 300, n_jobs=2, random_state=s, **cfg)
            m.fit(X[tr], y[tr])
        else:
            m = LGBMClassifier(n_estimators=15 if opts["smoke"] else 1500, verbosity=-1, n_jobs=2, random_state=s, **cfg)
            m.fit(
                X[tr], y[tr],
                eval_set=[(X[va], y[va])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        return (lambda ids: m.predict_proba(X[ids])[:, 1].astype(np.float64)), None, dict(seconds=time.perf_counter() - t0)

    device = opts["device"]
    xx = torch.as_tensor(X, device=device)
    yy = torch.as_tensor(y, dtype=torch.float32, device=device)
    torch.manual_seed(s)
    r = make_rng(s, method)

    clients = clients_override if clients_override is not None else [tr[d[tr] == dom] for dom in sorted(set(d[tr]))]
    rounds = opts["rounds"]
    local_steps = opts["local_steps"]
    check = opts["check_every"]
    patience = opts["patience"]
    min_round = opts["min_round"]
    B = opts["batch"]

    best = -np.inf
    best_state = None
    wait = 0

    if method == "FedBN_TTA":
        nets = [MLPBN(X.shape[1]).to(device) for _ in clients]
        init = nets[0].state_dict()
        for n in nets:
            n.load_state_dict(init)
        shared = [k for k in init if k.startswith(("net.0.", "net.3.", "net.6."))]
        state = {k: init[k].detach().clone() for k in shared}

        def sync():
            for n in nets:
                st = n.state_dict(); st.update(state); n.load_state_dict(st)

        def pred_src(ids):
            return np.mean([score_model(n, xx, ids) for n in nets], axis=0)

        for rd in range(1, rounds + 1):
            sync()
            w = np.array([len(c) for c in clients], dtype=float); w /= w.sum()
            updates = []
            for i, ids in enumerate(clients):
                n = nets[i]; n.train()
                opt = torch.optim.Adam(n.parameters(), lr=cfg["lr"])
                for _ in range(local_steps):
                    b = r.choice(ids, max(2, min(B, len(ids))), replace=len(ids) < 2)
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(n(xx[b]), yy[b])
                    loss.backward(); opt.step()
                updates.append({k: n.state_dict()[k].detach().clone() for k in shared})
            state = {k: sum(float(a) * u[k] for a, u in zip(w, updates)) for k in shared}
            sync()
            if rd % check == 0 or rd == rounds:
                v = pauc(y[va], pred_src(va), 0.05)
                if v > best + 1e-5:
                    best = v
                    best_state = [{k: val.detach().cpu().clone() for k, val in n.state_dict().items()} for n in nets]
                    wait = 0
                else:
                    wait += 1
            if rd >= min_round and wait >= patience:
                break
        for n, st in zip(nets, best_state):
            n.load_state_dict(st)

        def pred_tta(ids, adapt_ids=None):
            if adapt_ids is None:
                return pred_src(ids)
            adapted = adapt_bn_unsupervised(nets, xx, adapt_ids)
            return np.mean([score_model(n, xx, ids) for n in adapted], axis=0)

        return pred_src, pred_tta, dict(seconds=time.perf_counter() - t0)

    def build_net():
        return LinearLogReg(X.shape[1]).to(device) if method == "FedLogReg" else MLP(X.shape[1]).to(device)

    model = build_net()
    mean_client_len = float(np.mean([len(c) for c in clients]))
    cglobal = [torch.zeros_like(p) for p in model.parameters()]
    clocal = [[torch.zeros_like(p) for p in model.parameters()] for _ in clients]

    for rd in range(1, rounds + 1):
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        weights = np.array([len(c) for c in clients], dtype=float); weights /= weights.sum()
        deltas, counts, dc = [], [], []
        for i, ids in enumerate(clients):
            local = build_net(); local.load_state_dict(state); local.train()
            if method in ("FedNova", "SCAFFOLD"):
                opt = torch.optim.SGD(local.parameters(), lr=cfg["lr"])
            elif method == "FedLogReg":
                opt = torch.optim.Adam(local.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            else:
                opt = torch.optim.Adam(local.parameters(), lr=cfg["lr"])
            nsteps = local_steps if method != "FedNova" else max(1, min(4 * local_steps, round(local_steps * len(ids) / mean_client_len)))
            ref = [p.detach().clone() for p in local.parameters()]
            control = [a - b for a, b in zip(cglobal, clocal[i])] if method == "SCAFFOLD" else None
            for _ in range(nsteps):
                b = r.choice(ids, min(B, len(ids)), replace=False)
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(local(xx[b]), yy[b])
                loss.backward()
                if control is not None:
                    for p, c_vec in zip(local.parameters(), control):
                        p.grad.add_(c_vec)
                opt.step()
            deltas.append({k: v.detach() - state[k] for k, v in local.state_dict().items()})
            counts.append(nsteps)
            if method == "SCAFFOLD":
                new_c = [old - g + (q - p.detach()) / (nsteps * cfg["lr"]) for old, g, q, p in zip(clocal[i], cglobal, ref, local.parameters())]
                dc.append([a - b for a, b in zip(new_c, clocal[i])])
                clocal[i] = new_c
        if method == "SCAFFOLD":
            for j, g in enumerate(cglobal):
                g.add_(sum(v[j] for v in dc) / len(clients))
        if method == "FedNova":
            tau_eff = float(np.dot(weights, counts))
            avg = {k: tau_eff * sum(float(w / n) * a[k] for w, n, a in zip(weights, counts, deltas)) for k in state}
        else:
            avg = {k: sum(float(w) * a[k] for w, a in zip(weights, deltas)) for k in state}
        model.load_state_dict({k: state[k] + avg[k] for k in state})

        if rd % check == 0 or rd == rounds:
            v = pauc(y[va], score_model(model, xx, va), 0.05)
            if v > best + 1e-5:
                best = v
                best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
        if rd >= min_round and wait >= patience:
            break

    model.load_state_dict(best_state)
    return (lambda ids: score_model(model, xx, ids)), None, dict(seconds=time.perf_counter() - t0)


def eval_scores(cs, cy, ts, ty, seed_tag):
    alpha = 0.01
    t_emp = emp_cut(cs[cy == 0], alpha)
    t_oracle = emp_cut(ts[ty == 0], alpha)
    locked_fpr, locked_rec, _, _ = eval_op(ty, ts, t_emp)
    _, oracle_rec, _, _ = eval_op(ty, ts, t_oracle)

    neg_s, pos_s = ts[ty == 0], ts[ty == 1]
    rng = make_rng("day3_ut", seed_tag)
    n_cal = min(2000, len(neg_s) // 2)
    n_pos_cal = max(1, int(round(n_cal * 0.01)))
    n_neg_cal = n_cal - n_pos_cal
    uq, ur = [], []
    for _ in range(5):
        cn = rng.choice(len(neg_s), size=n_neg_cal, replace=False)
        cp = rng.choice(len(pos_s), size=min(n_pos_cal, len(pos_s) // 2), replace=False)
        en = neg_s[~np.isin(np.arange(len(neg_s)), cn)]
        ep = pos_s[~np.isin(np.arange(len(pos_s)), cp)]
        u = np.concatenate([neg_s[cn], pos_s[cp]])
        tq = emp_cut(u, alpha)
        tr = wilks_cut(u, alpha, delta=0.05)
        uq.append(((en >= tq).mean(), (ep >= tq).mean(), float((en >= tq).mean() <= alpha)))
        ur.append(((en >= tr).mean(), (ep >= tr).mean(), float((en >= tr).mean() <= alpha)))
    uq, ur = np.array(uq), np.array(ur)

    return dict(
        auroc=pauc(ty, ts),
        pauc01=pauc(ty, ts, 0.01),
        pauc05=pauc(ty, ts, 0.05),
        ap=float(average_precision_score(ty, ts)),
        locked_fpr=locked_fpr,
        locked_recall=locked_rec,
        target_quantile_recall=oracle_rec,
        ut_quantile_fpr=float(uq[:, 0].mean()),
        ut_quantile_recall=float(uq[:, 1].mean()),
        ut_quantile_cov=float(uq[:, 2].mean()),
        ut_rat_fpr=float(ur[:, 0].mean()),
        ut_rat_recall=float(ur[:, 1].mean()),
        ut_rat_cov=float(ur[:, 2].mean()),
    )


def run_temporal(z: zipfile.ZipFile, work: Path, opts: dict, seeds):
    print("\n================ PART 1: CHRONOLOGICAL TEMPORAL HOLDOUT (NF-UNSW-NB15-v3) ================", flush=True)
    tc = np.load(io.BytesIO(z.read("temporal_cache.npz")), allow_pickle=False)
    X, y, d, blk = tc["x"].astype(np.float32), tc["y"].astype(np.int8), tc["d"], tc["block"]

    fit = np.flatnonzero(np.isin(blk, [0, 1]))
    stop = np.flatnonzero(blk == 2)
    cal = np.flatnonzero(blk == 3)
    test = np.flatnonzero(blk == 4)
    # Inner chronological validation for candidate selection: Block 0 -> Block 1
    in_tr = np.flatnonzero(blk == 0)
    in_va = np.flatnonzero(blk == 1)

    t_dir = work / "temporal"
    t_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tta_rows = []

    for s in seeds:
        clients_in = dirichlet_clients(in_tr, y, make_seed(s, "in"), K=3, alpha=0.5)
        clients_full = dirichlet_clients(fit, y, make_seed(s, "full"), K=3, alpha=0.5)
        for method in TEMPORAL_METHODS:
            key = f"{method}_s{s}"
            meta = t_dir / f"{key}.json"
            if meta.exists():
                rec = json.loads(meta.read_text())
                rows.append(rec["metrics"])
                if rec.get("tta_pair"):
                    tta_rows.append(rec["tta_pair"])
                continue

            cands = candidate_grid(method)[: opts["trials"]]
            scores_c = []
            for j, cfg in enumerate(cands):
                pred_c, _, _ = fit_method(X, y, d, in_tr, in_va, method, cfg, make_seed(s, "tmp_in", j), opts, clients_override=clients_in)
                scores_c.append(pauc(y[stop], pred_c(stop), 0.05))
            chosen = cands[int(np.argmax(scores_c))]

            pred_src, pred_tta, info = fit_method(X, y, d, fit, stop, method, chosen, make_seed(s, "tmp_fin"), opts, clients_override=clients_full)
            cs = pred_src(cal)
            tta_pair = None
            if method == "FedBN_TTA":
                ts_pre = pred_src(test)
                rng_tta = make_rng("tmp_tta", s)
                neg_t, pos_t = test[y[test] == 0], test[y[test] == 1]
                adapt_ids = np.sort(np.concatenate([
                    rng_tta.choice(neg_t, size=min(1980, len(neg_t) // 2), replace=False),
                    rng_tta.choice(pos_t, size=min(20, len(pos_t) // 2), replace=False),
                ]))
                ev_mask = ~np.isin(test, adapt_ids)
                ts_post = pred_tta(test, adapt_ids=adapt_ids)
                m_post = eval_scores(cs, y[cal], ts_post[ev_mask], y[test][ev_mask], f"tmp_{method}_{s}")
                m_pre = eval_scores(cs, y[cal], ts_pre[ev_mask], y[test][ev_mask], f"tmp_{method}_pre_{s}")
                tta_pair = dict(
                    seed=s,
                    pre_auroc=m_pre["auroc"], post_auroc=m_post["auroc"],
                    pre_locked_fpr=m_pre["locked_fpr"], post_locked_fpr=m_post["locked_fpr"],
                    pre_oracle_recall=m_pre["target_quantile_recall"], post_oracle_recall=m_post["target_quantile_recall"],
                    delta_oracle_recall=m_post["target_quantile_recall"] - m_pre["target_quantile_recall"],
                )
                metrics = dict(method=method, seed=s, **m_post)
            else:
                ts = pred_src(test)
                metrics = dict(method=method, seed=s, **eval_scores(cs, y[cal], ts, y[test], f"tmp_{method}_{s}"))

            save_json(meta, dict(config=chosen, info=info, metrics=metrics, tta_pair=tta_pair))
            rows.append(metrics)
            if tta_pair:
                tta_rows.append(tta_pair)
            print(f"TEMPORAL DONE | seed={s} | {method:18s} | AUROC={metrics['auroc']:.3f} | LockFPR={metrics['locked_fpr']*100:5.2f}% | TgtQRec={metrics['target_quantile_recall']*100:5.2f}% | UTqFPR={metrics['ut_quantile_fpr']*100:4.2f}% | {info['seconds']:.1f}s", flush=True)

    df_t = pd.DataFrame(rows)
    df_t.to_csv(work / "day3_temporal_runs.csv", index=False)
    df_tm = df_t.groupby("method").mean(numeric_only=True).reset_index()
    df_tm.to_csv(work / "day3_temporal_means.csv", index=False)
    if tta_rows:
        pd.DataFrame(tta_rows).to_csv(work / "day3_temporal_fedbn_tta.csv", index=False)
    print("\n=== TEMPORAL HOLDOUT MEANS (Block 4 Future Test) ===")
    disp = df_tm[["method", "auroc", "pauc01", "locked_fpr", "locked_recall", "target_quantile_recall", "ut_quantile_fpr", "ut_quantile_recall", "ut_rat_cov"]].copy()
    for c in disp.columns[3:]:
        disp[c] *= 100.0
    print(disp.round(2).to_string(index=False))


def run_ablation(z: zipfile.ZipFile, work: Path, opts: dict, domains, seeds):
    print("\n================ PART 2: MATCHED RETUNED FEATURE ABLATION (39 & 37 FEATURES) ================", flush=True)
    cache = np.load(io.BytesIO(z.read("nf_cache.npz")), allow_pickle=False)
    X_full, y, d = cache["x"].astype(np.float32), cache["y"].astype(np.int8), cache["d"]

    a_dir = work / "ablation"
    a_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for vname, drop_cols in ABLATION_VARIANTS.items():
        keep_idx = [i for i, c in enumerate(FEATURES) if c not in drop_cols]
        Xv = X_full[:, keep_idx]
        for held in domains:
            for s in seeds:
                fit, stop, cal, _, test = split_outer_nf(y, d, [held], s)
                src_domains = sorted(set(d[fit]))
                inner = [(h, fit[d[fit] != h], stop[d[stop] != h], np.r_[fit[d[fit] == h], stop[d[stop] == h]]) for h in src_domains]
                for method in ABLATION_METHODS:
                    key = f"{vname}_{held}_s{s}_{method}"
                    meta = a_dir / f"{key}.json"
                    if meta.exists():
                        rows.append(json.loads(meta.read_text())["metrics"])
                        continue

                    cands = candidate_grid(method)[: opts["trials"]]
                    cand_scores = []
                    for j, cfg in enumerate(cands):
                        in_p = []
                        for h, a_i, v_i, b_i in inner:
                            pred_c, _, _ = fit_method(Xv, y, d, a_i, v_i, method, cfg, make_seed(s, h, 0), opts)
                            in_p.append(pauc(y[b_i], pred_c(b_i), 0.05))
                        cand_scores.append(float(np.mean(in_p)))
                    chosen = cands[int(np.argmax(cand_scores))]

                    pred_src, _, info = fit_method(Xv, y, d, fit, stop, method, chosen, make_seed(s, "final"), opts)
                    cs = pred_src(cal)
                    ts = pred_src(test)
                    m = eval_scores(cs, y[cal], ts, y[test], key)
                    metrics = dict(variant=vname, n_features=len(keep_idx), held=held, seed=s, method=method, **m)
                    save_json(meta, dict(config=chosen, info=info, metrics=metrics))
                    rows.append(metrics)
                    print(f"ABLATION {vname:12s} | {held:18s} | s={s:4d} | {method:18s} | AUROC={m['auroc']:.3f} | LockFPR={m['locked_fpr']*100:5.1f}% | TgtQRec={m['target_quantile_recall']*100:5.1f}%", flush=True)

    df_a = pd.DataFrame(rows)
    df_a.to_csv(work / "day3_ablation_runs.csv", index=False)
    dom_m = df_a.groupby(["variant", "method", "held"]).mean(numeric_only=True).reset_index()
    meth_m = dom_m.groupby(["variant", "method"]).mean(numeric_only=True).reset_index()
    meth_m.to_csv(work / "day3_ablation_means.csv", index=False)
    print("\n=== FEATURE ABLATION MEANS (Domain-Averaged, % for rates) ===")
    disp = meth_m[["variant", "method", "auroc", "pauc01", "locked_fpr", "target_quantile_recall", "ut_quantile_fpr", "ut_quantile_recall"]].copy()
    for c in disp.columns[4:]:
        disp[c] *= 100.0
    print(disp.round(2).to_string(index=False))


def run_all(input_zip: Path, work: Path, run_part1: bool = True, run_part2: bool = True, smoke: bool = False):
    work.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opts = dict(
        device=device,
        smoke=smoke,
        rounds=10 if smoke else 150,
        min_round=5 if smoke else 30,
        local_steps=2 if smoke else 8,
        batch=256,
        check_every=5,
        patience=3 if smoke else 10,
        trials=2 if smoke else 4,
    )
    seeds = [0] if smoke else SEEDS
    domains = ["NF-BoT-IoT"] if smoke else DOMAINS
    print(f"Running FLIDS Day 3 on device={device} | smoke={smoke}", flush=True)

    with zipfile.ZipFile(input_zip) as z:
        if run_part1:
            run_temporal(z, work, opts, seeds)
        if run_part2:
            run_ablation(z, work, opts, domains, seeds)

    out_zip = work / "FLIDS_Day3_Temporal_Ablation_Results.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(work.glob("*.csv")):
            zf.write(p, arcname=p.name)
    print(f"\nExported summary archive: {out_zip} ({out_zip.stat().st_size / 1024:.1f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--work", default=Path("./day3_work"), type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_all(args.input, args.work, smoke=args.smoke)


if __name__ == "__main__":
    main()
