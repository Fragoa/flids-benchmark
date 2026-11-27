"""Day 1 analysis: Unsupervised target-stream calibration & BoT-IoT low-FPR tail attribution.

Usage from repository root:
    python analysis/manuscript/unsupervised_and_tail_audit.py /path/to/FLIDS_Master_V3_Fresh_Results.zip
"""
import io
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, ks_2samp

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


def emp_cut(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(scores.astype(np.float64))
    idx = len(s) - math.floor(alpha * len(s)) - 1
    idx = max(0, min(len(s) - 1, idx))
    return float(s[idx])


def wilks_cut(scores: np.ndarray, alpha: float, delta: float = 0.05) -> float:
    s = np.sort(scores.astype(np.float64))
    n = len(s)
    k = 1 + int(binom.ppf(1.0 - delta, n, 1.0 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def fedrat_cut(
    cal_scores: np.ndarray,
    cal_y: np.ndarray,
    cal_d: np.ndarray,
    alpha: float = 0.01,
    delta: float = 0.05,
) -> float:
    groups = np.unique(cal_d)
    J = len(groups)
    cuts = []
    for g in groups:
        sg = np.sort(cal_scores[(cal_d == g) & (cal_y == 0)].astype(np.float64))
        nj = len(sg)
        kj = 1 + int(binom.ppf(1.0 - delta / J, nj, 1.0 - alpha))
        cuts.append(float("inf") if kj > nj else float(sg[kj - 1]))
    return max(cuts)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python analysis/manuscript/unsupervised_and_tail_audit.py /path/to/FLIDS_Master_V3_Fresh_Results.zip"
        )
    zip_path = Path(sys.argv[1])
    out_dir = Path(__file__).resolve().parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        op = pd.read_csv(z.open("summary/operating.csv"))
        native = op.query(
            'track=="nf" and alpha==0.01 and status=="evaluated" and scenario=="native" and participation==1'
        ).copy()
        assert len(native) == 300, f"Expected 300 native runs, got {len(native)}"

        cache = np.load(io.BytesIO(z.read("nf_cache.npz")), allow_pickle=False)
        X_raw = cache["x"].astype(np.float64)
        y_all = cache["y"]
        d_all = cache["d"]
        assert X_raw.shape[1] == len(FEATURES)

        prevalences = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
        n_cal = 2000
        n_draws = 5
        alpha = 0.01

        rows_a = []
        tail_stats = []
        fam_rows = []
        bot_fedavg_top_masks = []
        bot_benign_ids_ref = None

        for run_idx, row in native.iterrows():
            a = np.load(io.BytesIO(z.read(f"bundles/{row.bundle}/scores.npz")), allow_pickle=False)
            ty = a["target_y"]
            ts = a["target_scores"].astype(np.float64)
            tids = a["target_ids"]
            tf = a["target_family"]
            cs = a["cal_scores"].astype(np.float64)
            cy = a["cal_y"]
            cd = a["cal_d"]

            neg_idx = np.where(ty == 0)[0]
            pos_idx = np.where(ty == 1)[0]
            neg_scores = ts[neg_idx]
            pos_scores = ts[pos_idx]

            t_emp = emp_cut(cs[cy == 0], alpha)
            t_fedrat = fedrat_cut(cs, cy, cd, alpha=alpha, delta=0.05)
            q99_tgt = emp_cut(neg_scores, alpha)

            tail_stats.append({
                "method": row.method,
                "held": row.held,
                "seed": int(row.seed),
                "src_cut": t_emp,
                "tgt_cut": q99_tgt,
                "shift_gap": q99_tgt - t_emp,
                "locked_fpr": float((neg_scores > t_emp).mean()),
                "locked_recall": float((pos_scores > t_emp).mean()),
                "oracle_recall": float((pos_scores > q99_tgt).mean()),
                "fedrat_fpr": float((neg_scores > t_fedrat).mean()),
                "fedrat_recall": float((pos_scores > t_fedrat).mean()),
                "benign_tie_at_q99": float((neg_scores == q99_tgt).mean()),
                "frac_pos_above_max_neg": float((pos_scores > neg_scores.max()).mean()),
            })

            for fam in np.unique(tf[ty == 1]):
                m = (ty == 1) & (tf == fam)
                if m.sum() >= 20:
                    fam_rows.append({
                        "held": row.held,
                        "family": fam,
                        "method": row.method,
                        "seed": int(row.seed),
                        "n": int(m.sum()),
                        "recall_at_1pct_target_fpr": float((ts[m] > q99_tgt).mean()),
                    })

            if row.method == "FedAvg" and row.held == "NF-BoT-IoT":
                b_ids = tids[neg_idx]
                if bot_benign_ids_ref is None:
                    bot_benign_ids_ref = b_ids.copy()
                order = np.argsort(-neg_scores, kind="stable")
                top_k = math.ceil(0.01 * len(neg_scores))
                m = np.zeros(len(neg_scores), dtype=bool)
                m[order[:top_k]] = True
                bot_fedavg_top_masks.append(m)

            rng = np.random.default_rng(100000 + int(run_idx))
            for pi in prevalences:
                n_pos_cal = int(round(n_cal * pi))
                n_neg_cal = n_cal - n_pos_cal
                metrics = {
                    "source_emp": [],
                    "fedrat": [],
                    "ut_quantile": [],
                    "ut_rat": [],
                    "ut_hybrid_fedrat": [],
                    "labeled_target_2000": [],
                }
                for _ in range(n_draws):
                    cal_neg_sel = rng.choice(len(neg_scores), size=n_neg_cal, replace=False)
                    eval_neg_mask = np.ones(len(neg_scores), dtype=bool)
                    eval_neg_mask[cal_neg_sel] = False
                    ev_neg = neg_scores[eval_neg_mask]

                    if n_pos_cal > 0:
                        cal_pos_sel = rng.choice(len(pos_scores), size=n_pos_cal, replace=False)
                        eval_pos_mask = np.ones(len(pos_scores), dtype=bool)
                        eval_pos_mask[cal_pos_sel] = False
                        ev_pos = pos_scores[eval_pos_mask]
                        u_stream = np.concatenate([neg_scores[cal_neg_sel], pos_scores[cal_pos_sel]])
                    else:
                        ev_pos = pos_scores
                        u_stream = neg_scores[cal_neg_sel]

                    t_ut_q = emp_cut(u_stream, alpha)
                    t_ut_rat = wilks_cut(u_stream, alpha, delta=0.05)
                    t_hyb = max(t_fedrat, t_ut_q)
                    t_lbl = emp_cut(neg_scores[cal_neg_sel], alpha)

                    for pol_name, thr in [
                        ("source_emp", t_emp),
                        ("fedrat", t_fedrat),
                        ("ut_quantile", t_ut_q),
                        ("ut_rat", t_ut_rat),
                        ("ut_hybrid_fedrat", t_hyb),
                        ("labeled_target_2000", t_lbl),
                    ]:
                        fp = int((ev_neg > thr).sum())
                        n_ev_neg = len(ev_neg)
                        fpr = fp / n_ev_neg
                        rec = float((ev_pos > thr).mean())
                        alert_rate = (1.0 - pi) * fpr + pi * rec
                        log_err = abs(math.log(((fp + 0.5) / (n_ev_neg + 1.0)) / alpha))
                        metrics[pol_name].append((fpr, rec, float(fpr <= alpha), alert_rate, log_err))

                for pol_name, vals in metrics.items():
                    arr = np.array(vals)
                    rows_a.append({
                        "method": row.method,
                        "held": row.held,
                        "seed": int(row.seed),
                        "pi": pi,
                        "policy": pol_name,
                        "fpr": arr[:, 0].mean(),
                        "recall": arr[:, 1].mean(),
                        "coverage": arr[:, 2].mean(),
                        "alert_rate": arr[:, 3].mean(),
                        "log_err": arr[:, 4].mean(),
                    })

        df_a = pd.DataFrame(rows_a)
        df_a.to_csv(out_dir / "unsupervised_calibration_300runs.csv", index=False)

        summary_a = (
            df_a.groupby(["pi", "policy"])[["fpr", "coverage", "recall", "alert_rate", "log_err"]]
            .mean()
            .reset_index()
        )
        summary_a.to_csv(out_dir / "unsupervised_calibration_summary.csv", index=False)

        by_method_pi01 = (
            df_a.query("pi == 0.01")
            .groupby(["method", "policy"])[["fpr", "coverage", "recall", "alert_rate", "log_err"]]
            .mean()
            .reset_index()
        )
        by_method_pi01.to_csv(out_dir / "unsupervised_calibration_by_method_pi01.csv", index=False)

        pd.DataFrame(tail_stats).to_csv(out_dir / "tail_diagnostics.csv", index=False)
        pd.DataFrame(fam_rows).to_csv(out_dir / "family_tail_recall.csv", index=False)

        # BoT-IoT top-1% benign feature attribution
        top_freq = np.stack(bot_fedavg_top_masks, axis=0).sum(axis=0)
        X_slog = np.clip(np.sign(X_raw) * np.log1p(np.abs(np.clip(X_raw, -1e12, 1e12))), -18, 18)
        bot_mask = d_all == "NF-BoT-IoT"
        sd_src = X_slog[~bot_mask].std(axis=0)
        mu_src = X_slog[~bot_mask].mean(axis=0)
        X_std = (X_slog - mu_src) / np.where(sd_src > 0, sd_src, 1.0)

        top_ids = bot_benign_ids_ref[top_freq >= 3]
        rest_ids = bot_benign_ids_ref[top_freq == 0]
        src_atk_mask = (~bot_mask) & (y_all == 1)
        src_ben_mask = (~bot_mask) & (y_all == 0)
        bot_atk_mask = bot_mask & (y_all == 1)

        feat_rows = []
        for fi, fname in enumerate(FEATURES):
            v_top = X_std[top_ids, fi]
            v_rest = X_std[rest_ids, fi]
            ks_stat, _ = ks_2samp(v_top, v_rest)
            feat_rows.append({
                "feature": fname,
                "ks_top_vs_rest": float(ks_stat),
                "std_mean_top_ben": float(v_top.mean()),
                "std_mean_rest_ben": float(v_rest.mean()),
                "std_mean_src_atk": float(X_std[src_atk_mask, fi].mean()),
                "std_mean_src_ben": float(X_std[src_ben_mask, fi].mean()),
                "std_mean_bot_atk": float(X_std[bot_atk_mask, fi].mean()),
                "raw_med_top_ben": float(np.median(X_raw[top_ids, fi])),
                "raw_med_rest_ben": float(np.median(X_raw[rest_ids, fi])),
                "raw_med_src_atk": float(np.median(X_raw[src_atk_mask, fi])),
                "raw_med_bot_atk": float(np.median(X_raw[bot_atk_mask, fi])),
            })
        df_feat = pd.DataFrame(feat_rows).sort_values("ks_top_vs_rest", ascending=False)
        df_feat.to_csv(out_dir / "bot_top1pct_feature_attribution.csv", index=False)

        print("Saved all Day 1 analysis tables to:", out_dir)
        print("Top 5 BoT-IoT features:", df_feat.head(5)[["feature", "ks_top_vs_rest"]].to_dict("records"))


if __name__ == "__main__":
    main()
