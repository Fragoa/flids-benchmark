"""FLIDS Day 2 Extension: FedLogReg, FedBN_TTA, FedAvg_Tail, and FedNova_Tail.

Follows the exact source-locked protocol of experiments/tuned/master.py:
- Exact same 93,810-row deduplicated NF-UQ-NIDS-v2 cache and split digests
- Exact same signed-log + source-fit StandardScaler preprocessing
- Exact same 4-candidate inner leave-one-source-out selection (max pAUC@5%)
- Exact same 150-round / min_round=30 / patience=10 / local_steps=8 / batch=256 budget
- Zero-leakage FedBN_TTA: adapts BatchNorm stats on a disjoint 2,000-flow unlabeled
  target stream at realistic pi=1% prevalence (zero target labels, zero gradients)
- Evaluates source-locked 1% FPR, retrospective 1% target-quantile recall,
  FedRAT, and zero-target-label unsupervised calibration (UT-Quantile, UT-RAT at pi=1%).

Usage:
    python experiments/day2/day2_methods.py --input /path/to/FLIDS_Master_V3_Fresh_Results.zip --work ./day2_work
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

import numpy as np
import pandas as pd
import torch
from scipy.stats import binom
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

SEEDS = [0, 42, 3407, 123, 2026]
DOMAINS = ["NF-UNSW-NB15", "NF-ToN-IoT", "NF-BoT-IoT", "NF-CSE-CIC-IDS2018"]
METHODS = ["FedLogReg", "FedBN_TTA", "FedAvg_Tail", "FedNova_Tail"]


def make_seed(*args) -> int:
    return int.from_bytes(hashlib.sha256(repr(args).encode()).digest()[:4], "little")


def make_rng(*args) -> np.random.Generator:
    return np.random.default_rng(make_seed(*args))


def signed_log(x: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(x, dtype=np.float64), -1e12, 1e12)
    return np.clip(np.sign(c) * np.log1p(np.abs(c)), -18.0, 18.0).astype(np.float32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def fedrat_cut(cal_scores, cal_y, cal_d, alpha=0.01, delta=0.05) -> float:
    groups = sorted(set(cal_d))
    cuts = [wilks_cut(cal_scores[(cal_d == g) & (cal_y == 0)], alpha, delta / len(groups)) for g in groups]
    return max(cuts)


def pauc(y: np.ndarray, s: np.ndarray, max_fpr=None) -> float:
    return float(roc_auc_score(y, s, max_fpr=max_fpr))


def eval_op(y: np.ndarray, s: np.ndarray, thr: float):
    n = s[y == 0]
    p = s[y == 1]
    fp = int((n >= thr).sum())
    tp = int((p >= thr).sum())
    return fp / len(n), tp / len(p), fp, tp


def dedup(x, y, d, f):
    keys = np.array([hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=16).hexdigest() for v in x])
    _, inv, _ = np.unique(keys, return_inverse=True, return_counts=True)
    bad = set()
    seen = {}
    for i, k in enumerate(inv):
        val = (str(d[i]), int(y[i]), str(f[i]))
        if k in seen:
            if seen[k][0] != val:
                bad.add(k)
        else:
            seen[k] = (val, i)
    return np.array([i for k, (_, i) in seen.items() if k not in bad], dtype=int)


def load_nf_cache(input_path: Path, work: Path):
    cached = work / "nf_cache.npz"
    if cached.exists():
        a = np.load(cached, allow_pickle=False)
        return a["x"], a["y"], a["d"], a["family"]

    if input_path.is_file() and input_path.suffix == ".npz":
        a = np.load(input_path, allow_pickle=False)
        x, y, d, f = a["x"], a["y"], a["d"], a["family"]
    else:
        with zipfile.ZipFile(input_path) as z:
            names = set(z.namelist())
            if "nf_cache.npz" in names:
                a = np.load(io.BytesIO(z.read("nf_cache.npz")), allow_pickle=False)
                x, y, d, f = a["x"], a["y"], a["d"], a["family"]
            elif "reproducibility/flids_exp02_cache.npz" in names:
                a = np.load(io.BytesIO(z.read("reproducibility/flids_exp02_cache.npz")), allow_pickle=False)
                x0, y0, d0, f0 = a["x"], a["y"], a["d"], a["attack"]
                keep = dedup(x0, y0, d0, f0)
                x, y, d, f = x0[keep], y0[keep], d0[keep], f0[keep]
            else:
                raise FileNotFoundError(
                    f"Neither nf_cache.npz nor reproducibility/flids_exp02_cache.npz found in {input_path}"
                )
    npwrite(cached, x=x, y=y, d=d, family=f)
    return x, y, d, f


def groupsplit(ids, y, s, frac):
    a, b = train_test_split(np.asarray(ids), test_size=frac, random_state=s, stratify=y[ids])
    return np.sort(a), np.sort(b)


def split_outer(y, d, held, s):
    """Exact same outer split as experiments/tuned/master.py."""
    src = np.flatnonzero(~np.isin(d, held))
    test = np.flatnonzero(np.isin(d, held))
    domains = sorted(set(d[src]))
    fit, stop, cal, st = [], [], [], []
    for dom in domains:
        ids = src[d[src] == dom]
        a, b = groupsplit(ids, y, make_seed(s, dom, "outer"), 0.30)
        c, e = groupsplit(b, y, make_seed(s, dom, "cal"), 2 / 3)
        g, h = groupsplit(e, y, make_seed(s, dom, "st"), 0.5)
        fit.extend(a)
        stop.extend(c)
        cal.extend(g)
        st.extend(h)
    parts = tuple(np.array(v, dtype=int) for v in (fit, stop, cal, st, test))
    assert len(np.unique(np.concatenate(parts))) == sum(map(len, parts))
    return parts


def candidate_grid(method: str):
    if method == "FedLogReg":
        return [
            dict(lr=0.01, wd=1e-2),
            dict(lr=0.01, wd=1e-3),
            dict(lr=0.03, wd=1e-3),
            dict(lr=0.03, wd=1e-4),
        ]
    if method == "FedBN_TTA":
        return [dict(lr=l) for l in (0.0003, 0.001, 0.003, 0.01)]
    if method == "FedAvg_Tail":
        return [
            dict(lr=0.001, lam=2.0, gamma=2.0),
            dict(lr=0.001, lam=5.0, gamma=2.0),
            dict(lr=0.003, lam=2.0, gamma=2.0),
            dict(lr=0.003, lam=5.0, gamma=2.0),
        ]
    if method == "FedNova_Tail":
        return [
            dict(lr=0.003, lam=2.0, gamma=2.0),
            dict(lr=0.003, lam=5.0, gamma=2.0),
            dict(lr=0.01, lam=2.0, gamma=2.0),
            dict(lr=0.01, lam=5.0, gamma=2.0),
        ]
    raise ValueError(f"Unknown method: {method}")


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
            nn.Linear(n_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def tail_weighted_bce(logits: torch.Tensor, targets: torch.Tensor, lam: float, gamma: float = 2.0) -> torch.Tensor:
    """Asymmetric negative-tail focal BCE: upweights high-scoring benign flows to suppress low-FPR FPs."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        neg_mask = (targets < 0.5).float()
        weights = 1.0 + neg_mask * (lam * (probs ** gamma))
    return (bce * weights).mean()


def score_model(net: nn.Module, xx: torch.Tensor, ids: np.ndarray) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for chunk in np.array_split(ids, max(1, math.ceil(len(ids) / 4096))):
            out.append(torch.sigmoid(net(xx[chunk])).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def adapt_bn_unsupervised(nets, xx: torch.Tensor, unlabeled_ids: np.ndarray, batch_size: int = 512):
    """Unsupervised AdaBN: recompute BN running stats from unlabeled target flows only (zero labels, zero grad)."""
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


def fit_method(raw, y, d, tr, va, method, cfg, s, opts):
    t0 = time.perf_counter()
    trans = signed_log(raw)
    sc = StandardScaler().fit(trans[tr])
    X = np.clip(sc.transform(trans).astype(np.float32), -20.0, 20.0)

    device = opts["device"]
    xx = torch.as_tensor(X, device=device)
    yy = torch.as_tensor(y, dtype=torch.float32, device=device)
    torch.manual_seed(s)
    r = make_rng(s, method)

    clients = [tr[d[tr] == dom] for dom in sorted(set(d[tr]))]
    rounds = opts["rounds"]
    local_steps = opts["local_steps"]
    check = opts["check_every"]
    patience = opts["patience"]
    min_round = opts["min_round"]
    B = opts["batch"]

    steps = 0
    comm = 0
    best = -np.inf
    best_state = None
    best_round = 0
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
                st = n.state_dict()
                st.update(state)
                n.load_state_dict(st)

        def pred_src(ids):
            return np.mean([score_model(n, xx, ids) for n in nets], axis=0)

        for rd in range(1, rounds + 1):
            sync()
            w = np.array([len(c) for c in clients], dtype=float)
            w /= w.sum()
            updates = []
            for i, ids in enumerate(clients):
                n = nets[i]
                n.train()
                opt = torch.optim.Adam(n.parameters(), lr=cfg["lr"])
                for _ in range(local_steps):
                    b = r.choice(ids, max(2, min(B, len(ids))), replace=len(ids) < 2)
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(n(xx[b]), yy[b])
                    loss.backward()
                    opt.step()
                    steps += 1
                updates.append({k: n.state_dict()[k].detach().clone() for k in shared})
                comm += 2 * sum(v.numel() * v.element_size() for v in state.values())
            state = {k: sum(float(a) * u[k] for a, u in zip(w, updates)) for k in shared}
            sync()
            if rd % check == 0 or rd == rounds:
                v = pauc(y[va], pred_src(va), 0.05)
                if v > best + 1e-5:
                    best = v
                    best_round = rd
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

        info = dict(
            seconds=time.perf_counter() - t0,
            steps=steps,
            bytes=comm,
            best_round=best_round,
            rounds_run=rd,
            source_stop_pauc05=best,
        )
        return pred_src, pred_tta, info

    def build_net():
        return LinearLogReg(X.shape[1]).to(device) if method == "FedLogReg" else MLP(X.shape[1]).to(device)

    model = build_net()
    mean_client_len = float(np.mean([len(c) for c in clients]))

    for rd in range(1, rounds + 1):
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        weights = np.array([len(c) for c in clients], dtype=float)
        weights /= weights.sum()
        deltas = []
        counts = []

        for ids in clients:
            local = build_net()
            local.load_state_dict(state)
            local.train()
            if method == "FedNova_Tail":
                opt = torch.optim.SGD(local.parameters(), lr=cfg["lr"])
                nsteps = max(1, min(4 * local_steps, round(local_steps * len(ids) / mean_client_len)))
            elif method == "FedLogReg":
                opt = torch.optim.Adam(local.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
                nsteps = local_steps
            else:
                opt = torch.optim.Adam(local.parameters(), lr=cfg["lr"])
                nsteps = local_steps

            for _ in range(nsteps):
                b = r.choice(ids, min(B, len(ids)), replace=False)
                opt.zero_grad()
                logits = local(xx[b])
                if method in ("FedAvg_Tail", "FedNova_Tail"):
                    loss = tail_weighted_bce(logits, yy[b], lam=cfg["lam"], gamma=cfg["gamma"])
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, yy[b])
                loss.backward()
                opt.step()
                steps += 1

            deltas.append({k: v.detach() - state[k] for k, v in local.state_dict().items()})
            counts.append(nsteps)
            comm += 2 * sum(v.numel() * v.element_size() for v in state.values())

        if method == "FedNova_Tail":
            tau_eff = float(np.dot(weights, counts))
            avg = {k: tau_eff * sum(float(w / n) * a[k] for w, n, a in zip(weights, counts, deltas)) for k in state}
        else:
            avg = {k: sum(float(w) * a[k] for w, a in zip(weights, deltas)) for k in state}
        model.load_state_dict({k: state[k] + avg[k] for k in state})

        if rd % check == 0 or rd == rounds:
            v = pauc(y[va], score_model(model, xx, va), 0.05)
            if v > best + 1e-5:
                best = v
                best_round = rd
                best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
        if rd >= min_round and wait >= patience:
            break

    model.load_state_dict(best_state)
    pred = lambda ids: score_model(model, xx, ids)
    info = dict(
        seconds=time.perf_counter() - t0,
        steps=steps,
        bytes=comm,
        best_round=best_round,
        rounds_run=rd,
        source_stop_pauc05=best,
    )
    return pred, None, info


def evaluate_bundle(folder: Path, held: str, s: int, method: str):
    a = np.load(folder / "scores.npz", allow_pickle=False)
    cy, cs, cd = a["cal_y"], a["cal_scores"], a["cal_d"]
    sy, ss = a["source_test_y"], a["source_test_scores"]
    ty_all, ts_all = a["target_y"], a["target_scores"]

    # For FedBN_TTA, exclude the 2,000 unlabeled adaptation flows from evaluation (zero leakage)
    if "tta_eval_mask" in a:
        ev_mask = a["tta_eval_mask"].astype(bool)
        ty, ts = ty_all[ev_mask], ts_all[ev_mask]
    else:
        ev_mask = np.ones(len(ty_all), dtype=bool)
        ty, ts = ty_all, ts_all

    alpha = 0.01
    t_emp = emp_cut(cs[cy == 0], alpha)
    t_fedrat = fedrat_cut(cs, cy, cd, alpha=alpha, delta=0.05)
    t_oracle = emp_cut(ts[ty == 0], alpha)

    locked_fpr, locked_rec, _, _ = eval_op(ty, ts, t_emp)
    fedrat_fpr, fedrat_rec, _, _ = eval_op(ty, ts, t_fedrat)
    _, oracle_rec, _, _ = eval_op(ty, ts, t_oracle)
    src_fpr, _, _, _ = eval_op(sy, ss, t_emp)

    # Zero-target-label unsupervised calibration at pi=1% (N_cal=2000, 5 disjoint draws)
    neg_s = ts[ty == 0]
    pos_s = ts[ty == 1]
    rng = make_rng("day2_ut", held, s, method)
    n_cal = min(2000, len(neg_s) // 2)
    n_pos_cal = max(1, int(round(n_cal * 0.01)))
    n_neg_cal = n_cal - n_pos_cal
    ut_q_res, ut_rat_res = [], []
    for _ in range(5):
        cn = rng.choice(len(neg_s), size=n_neg_cal, replace=False)
        cp = rng.choice(len(pos_s), size=min(n_pos_cal, len(pos_s) // 2), replace=False)
        en_mask = np.ones(len(neg_s), dtype=bool)
        en_mask[cn] = False
        ep_mask = np.ones(len(pos_s), dtype=bool)
        ep_mask[cp] = False
        u_stream = np.concatenate([neg_s[cn], pos_s[cp]])
        tq = emp_cut(u_stream, alpha)
        tr = wilks_cut(u_stream, alpha, delta=0.05)
        ev_n, ev_p = neg_s[en_mask], pos_s[ep_mask]
        ut_q_res.append(((ev_n >= tq).mean(), (ev_p >= tq).mean(), float((ev_n >= tq).mean() <= alpha)))
        ut_rat_res.append(((ev_n >= tr).mean(), (ev_p >= tr).mean(), float((ev_n >= tr).mean() <= alpha)))

    ut_q_arr = np.array(ut_q_res)
    ut_r_arr = np.array(ut_rat_res)

    row = dict(
        held=held,
        seed=s,
        method=method,
        auroc=pauc(ty, ts),
        pauc01=pauc(ty, ts, 0.01),
        pauc05=pauc(ty, ts, 0.05),
        ap=float(average_precision_score(ty, ts)),
        source_test_fpr=src_fpr,
        locked_fpr=locked_fpr,
        locked_recall=locked_rec,
        target_quantile_recall=oracle_rec,
        fedrat_fpr=fedrat_fpr,
        fedrat_recall=fedrat_rec,
        ut_quantile_fpr=float(ut_q_arr[:, 0].mean()),
        ut_quantile_recall=float(ut_q_arr[:, 1].mean()),
        ut_quantile_cov=float(ut_q_arr[:, 2].mean()),
        ut_rat_fpr=float(ut_r_arr[:, 0].mean()),
        ut_rat_recall=float(ut_r_arr[:, 1].mean()),
        ut_rat_cov=float(ut_r_arr[:, 2].mean()),
    )

    tta_pair = None
    if "target_scores_pre_tta" in a:
        ts_pre = a["target_scores_pre_tta"][ev_mask]
        t_oracle_pre = emp_cut(ts_pre[ty == 0], alpha)
        pre_fpr, pre_rec, _, _ = eval_op(ty, ts_pre, t_emp)
        _, pre_oracle_rec, _, _ = eval_op(ty, ts_pre, t_oracle_pre)
        tta_pair = dict(
            held=held,
            seed=s,
            pre_auroc=pauc(ty, ts_pre),
            post_auroc=row["auroc"],
            pre_pauc01=pauc(ty, ts_pre, 0.01),
            post_pauc01=row["pauc01"],
            pre_locked_fpr=pre_fpr,
            post_locked_fpr=locked_fpr,
            pre_locked_recall=pre_rec,
            post_locked_recall=locked_rec,
            pre_oracle_recall=pre_oracle_rec,
            post_oracle_recall=oracle_rec,
            delta_oracle_recall=oracle_rec - pre_oracle_rec,
            delta_locked_fpr=locked_fpr - pre_fpr,
        )
    return row, tta_pair


def run_all(input_path: Path, work: Path, methods=None, seeds=None, domains=None, smoke: bool = False):
    work.mkdir(parents=True, exist_ok=True)
    x, y, d, f = load_nf_cache(input_path, work)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opts = dict(
        device=device,
        rounds=10 if smoke else 150,
        min_round=5 if smoke else 30,
        local_steps=2 if smoke else 8,
        batch=256,
        check_every=5,
        patience=3 if smoke else 10,
        trials=2 if smoke else 4,
    )
    methods = methods or METHODS
    seeds = ([0] if smoke else SEEDS) if seeds is None else seeds
    domains = (["NF-BoT-IoT"] if smoke else DOMAINS) if domains is None else domains
    print(
        f"Running FLIDS Day 2 on device={device} | methods={methods} | domains={len(domains)} | seeds={seeds}",
        flush=True,
    )

    summary_rows = []
    tta_rows = []

    for held in domains:
        for s in seeds:
            fit, stop, cal, src_test, test = split_outer(y, d, [held], s)
            for method in methods:
                key = hashlib.sha256(repr(("day2_nf", held, s, method)).encode()).hexdigest()[:20]
                folder = work / "bundles" / key
                folder.mkdir(parents=True, exist_ok=True)
                meta = folder / "done.json"

                if not (meta.exists() and (folder / "scores.npz").exists()):
                    candidates = candidate_grid(method)[: opts["trials"]]
                    src_domains = sorted(set(d[fit]))
                    inner = [
                        (h, fit[d[fit] != h], stop[d[stop] != h], np.r_[fit[d[fit] == h], stop[d[stop] == h]])
                        for h in src_domains
                    ]
                    trial_rows = []
                    for j, cfg in enumerate(candidates):
                        for h, a_idx, v_idx, b_idx in inner:
                            tp = folder / f"trial_{j}_{hashlib.sha1(h.encode()).hexdigest()[:8]}.json"
                            if tp.exists():
                                tr_row = json.loads(tp.read_text())
                            else:
                                pred_fn, _, info = fit_method(x, y, d, a_idx, v_idx, method, cfg, make_seed(s, h, 0), opts)
                                val_scores = pred_fn(b_idx)
                                tr_row = dict(
                                    candidate=j,
                                    inner_holdout=h,
                                    pauc05=pauc(y[b_idx], val_scores, 0.05),
                                    pauc01=pauc(y[b_idx], val_scores, 0.01),
                                    auroc=pauc(y[b_idx], val_scores),
                                    config=cfg,
                                    **info,
                                )
                                save_json(tp, tr_row)
                            trial_rows.append(tr_row)

                    chosen_id = int(
                        pd.DataFrame(trial_rows)
                        .groupby("candidate")["pauc05"]
                        .mean()
                        .sort_values(ascending=False, kind="stable")
                        .index[0]
                    )
                    chosen = candidates[chosen_id]
                    pd.DataFrame(trial_rows).to_csv(folder / "tuning.csv", index=False)

                    pred_src, pred_tta, info = fit_method(x, y, d, fit, stop, method, chosen, make_seed(s, "final"), opts)
                    scores = {}
                    for role, ids in [("fit", fit), ("stop", stop), ("cal", cal), ("source_test", src_test)]:
                        scores[f"{role}_ids"] = ids
                        scores[f"{role}_y"] = y[ids]
                        scores[f"{role}_d"] = d[ids].astype(str)
                        scores[f"{role}_scores"] = pred_src(ids)

                    scores["target_ids"] = test
                    scores["target_y"] = y[test]
                    scores["target_d"] = d[test].astype(str)
                    scores["target_family"] = f[test].astype(str)

                    if method == "FedBN_TTA":
                        # Zero-leakage unsupervised AdaBN on a realistic 2,000-flow target stream at pi=1%
                        scores["target_scores_pre_tta"] = pred_src(test)
                        rng_tta = make_rng("day2_tta_stream", held, s)
                        neg_t = test[y[test] == 0]
                        pos_t = test[y[test] == 1]
                        adapt_neg = rng_tta.choice(neg_t, size=min(1980, len(neg_t) // 2), replace=False)
                        adapt_pos = rng_tta.choice(pos_t, size=min(20, len(pos_t) // 2), replace=False)
                        adapt_ids = np.sort(np.concatenate([adapt_neg, adapt_pos]))
                        eval_mask = ~np.isin(test, adapt_ids)
                        scores["tta_adapt_ids"] = adapt_ids
                        scores["tta_eval_mask"] = eval_mask
                        scores["target_scores"] = pred_tta(test, adapt_ids=adapt_ids)
                    else:
                        scores["target_scores"] = pred_src(test)

                    npwrite(folder / "scores.npz", **scores)
                    save_json(
                        meta,
                        dict(
                            held=held,
                            seed=s,
                            method=method,
                            config=chosen,
                            score_sha256=sha256_file(folder / "scores.npz"),
                            **info,
                        ),
                    )
                    print(f"DONE {held} | seed={s} | {method} | cfg={chosen} | {info['seconds']:.1f}s", flush=True)

                row, tta_pair = evaluate_bundle(folder, held, s, method)
                summary_rows.append(row)
                if tta_pair is not None:
                    tta_rows.append(tta_pair)

    df_op = pd.DataFrame(summary_rows)
    df_op.to_csv(work / "day2_operating_runs.csv", index=False)

    dom_means = df_op.groupby(["method", "held"]).mean(numeric_only=True).reset_index()
    method_means = dom_means.groupby("method").mean(numeric_only=True).reset_index()
    method_means.to_csv(work / "day2_method_means.csv", index=False)

    if tta_rows:
        pd.DataFrame(tta_rows).to_csv(work / "day2_fedbn_tta_paired.csv", index=False)

    cols = [
        "method", "auroc", "pauc01", "locked_fpr", "locked_recall",
        "target_quantile_recall", "fedrat_fpr", "ut_quantile_fpr", "ut_quantile_recall", "ut_rat_cov",
    ]
    disp = method_means[cols].copy()
    for c in cols[3:]:
        disp[c] *= 100.0
    print("\n=== DAY 2 METHOD MEANS (Domain-Averaged, % for rates) ===")
    print(disp.round(2).to_string(index=False))

    out_zip = work / "FLIDS_Day2_Methods_Results.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(work.rglob("*")):
            if p.is_file() and p != out_zip and p.name != "nf_cache.npz":
                zf.write(p, arcname=str(p.relative_to(work)))
    print(f"\nExported archive: {out_zip} ({out_zip.stat().st_size / (1024 * 1024):.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Path to FLIDS_Master_V3_Fresh_Results.zip, FLIDS_Exp09_Canonical_Input.zip, or nf_cache.npz")
    ap.add_argument("--work", default=Path("./day2_work"), type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_all(args.input, args.work, smoke=args.smoke)


if __name__ == "__main__":
    main()
