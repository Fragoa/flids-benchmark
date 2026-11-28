"""Generate publication-ready IEEEtran vector PDF figures for FLIDS Day 1 & Day 2 extensions.

Usage from repository root:
    python analysis/manuscript/plot_extensions.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "analysis" / "manuscript" / "tables"
FIGS = ROOT / "manuscript" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def make_pipeline_figure():
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.1)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec="#2c3e50", title_c="#1a252f"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=fc, edgecolor=ec, linewidth=1.0
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="top",
                fontsize=8.0, fontweight="bold", color=title_c)
        ax.text(x + w / 2, y + h - 0.50, body, ha="center", va="top",
                fontsize=6.9, color="#2c3e50", linespacing=1.28)

    def arrow(x1, y1, x2, y2, lbl=None, dy=0.10):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.1, mutation_scale=10)
        )
        if lbl:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, lbl, ha="center", va="bottom",
                    fontsize=6.6, color="#34495e", style="italic")

    # Stage 1: Source domains
    box(0.10, 0.35, 2.10, 2.40,
        "1. Source Domains (3/4)",
        "NF-UQ-NIDS-v2 (93,810 flows)\nFingerprint deduplicated\n\nPer-source strict split:\n• Fit (70%)  • Stop (10%)\n• Cal (10%)  • Test (10%)\nNo target data or labels",
        fc="#eef4fa", ec="#2b5c8f")

    # Stage 2: Training & Selection
    box(2.65, 0.35, 2.25, 2.40,
        "2. Source-Only Selection",
        "19 methods (7+4 FL, 7 Pool, 1 Loc)\n4 candidates / method\nInner LOSO max pAUC@5%\n\nNew extensions:\n• FedLogReg (linear FL)\n• FedBN_TTA (AdaBN)\n• FedAvg/Nova_Tail (BCE)",
        fc="#f4f1fa", ec="#5b4b8a")

    # Stage 3: Calibration branch (top = source-only, bottom = unlabeled target)
    box(5.35, 1.68, 2.15, 1.07,
        "3a. Source-Only Thresholds",
        "Empirical pooled / Conformal\nFedRAT (Wilks scalar / group)\nZero target samples used",
        fc="#fdf3e7", ec="#b96d1b")

    box(5.35, 0.35, 2.15, 1.12,
        "3b. Unlabeled Target Stream",
        "2,000 unlabeled flows (π ∈ [0,5%])\n• UT-Quantile / UT-RAT (Wilks)\n• Optional FedBN_TTA (AdaBN)\nZero target labels used",
        fc="#eaf7ee", ec="#247a45")

    # Stage 4: Disjoint Target Eval
    box(7.92, 0.35, 1.98, 2.40,
        "4. Disjoint Target Eval",
        "Held-out unseen dataset\n(1/4 LOSO × 5 seeds)\n\nJoint diagnostics:\n• Realized Target FPR\n• Budget Coverage (≤1%)\n• Attack Recall & Alerts\n• Retro. 1% Target Recall",
        fc="#fbeeee", ec="#a93226")

    arrow(2.26, 1.55, 2.59, 1.55)
    arrow(4.96, 2.10, 5.29, 2.18)
    arrow(4.96, 1.00, 5.29, 0.92)
    arrow(7.56, 2.18, 7.86, 2.00)
    arrow(7.56, 0.92, 7.86, 1.10)

    fig.tight_layout(pad=0.2)
    fig.savefig(FIGS / "protocol_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def make_unsupervised_calibration_figure():
    df = pd.read_csv(TABLES / "unsupervised_calibration_summary.csv")
    df["pi_pct"] = df["pi"] * 100.0

    policies = [
        ("source_emp", "Source Empirical", "#c0392b", "o", "--"),
        ("fedrat", "FedRAT (Source-Only)", "#d35400", "s", "-."),
        ("ut_quantile", "UT-Quantile (Unlabeled)", "#1f77b4", "^", "-"),
        ("ut_rat", "UT-RAT (Unlabeled Wilks)", "#27ae60", "D", "-"),
        ("labeled_target_2000", "Labeled Target-2000 (Oracle)", "#555555", "x", ":"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), sharex=True)

    # Panel A: Target FPR (%)
    ax = axes[0]
    for key, label, col, mk, ls in policies:
        sub = df[df["policy"] == key].sort_values("pi_pct")
        ax.plot(sub["pi_pct"], sub["fpr"] * 100.0, color=col, marker=mk, linestyle=ls,
                markersize=4.0, linewidth=1.35, label=label)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.9, alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 30)
    ax.set_ylabel("Mean Target FPR (%)")
    ax.set_xlabel("Unlabeled Stream Attack Prevalence π (%)")
    ax.set_title("(a) Target FPR (log scale)")
    ax.grid(True, which="major", linestyle=":", alpha=0.35)

    # Panel B: Target Budget Coverage (%)
    ax = axes[1]
    for key, label, col, mk, ls in policies:
        sub = df[df["policy"] == key].sort_values("pi_pct")
        ax.plot(sub["pi_pct"], sub["coverage"] * 100.0, color=col, marker=mk, linestyle=ls,
                markersize=4.0, linewidth=1.35)
    ax.axhline(95.0, color="#27ae60", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Target Runs ≤ 1% FPR (%)")
    ax.set_xlabel("Unlabeled Stream Attack Prevalence π (%)")
    ax.set_title("(b) Target Budget Coverage")
    ax.grid(True, linestyle=":", alpha=0.35)

    # Panel C: Attack Recall (%) — Contamination curve (zoom on target-calibrated policies + FedRAT)
    ax = axes[2]
    for key, label, col, mk, ls in policies[1:]:
        sub = df[df["policy"] == key].sort_values("pi_pct")
        ax.plot(sub["pi_pct"], sub["recall"] * 100.0, color=col, marker=mk, linestyle=ls,
                markersize=4.0, linewidth=1.35)
    ax.axvline(1.0, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(1.12, 13.3, "π = α = 1%", fontsize=6.8, color="#555555")
    ax.set_ylim(0, 17.0)
    ax.set_ylabel("Mean Attack Recall (%)")
    ax.set_xlabel("Unlabeled Stream Attack Prevalence π (%)")
    ax.set_title("(c) Recall & Contamination")
    ax.grid(True, linestyle=":", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09),
               ncol=3, frameon=False, columnspacing=1.0, handlelength=2.2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIGS / "unsupervised_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def make_day2_extensions_figure():
    df_tta = pd.read_csv(TABLES / "day2_fedbn_tta_paired.csv")
    df_feat = pd.read_csv(TABLES / "bot_top1pct_feature_attribution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))

    # Panel A: Paired FedBN Pre-TTA vs Post-TTA Target-Quantile Recall (%)
    ax = axes[0]
    g = df_tta.groupby("held")[["pre_oracle_recall", "post_oracle_recall"]].mean() * 100.0
    order = ["NF-BoT-IoT", "NF-CSE-CIC-IDS2018", "NF-ToN-IoT", "NF-UNSW-NB15"]
    short = ["BoT-IoT", "CSE-CIC-18", "ToN-IoT", "UNSW-NB15"]
    g = g.reindex(order)
    x = np.arange(len(order))
    w = 0.35
    b1 = ax.bar(x - w / 2, g["pre_oracle_recall"], width=w, label="FedBN (Source BN)", color="#7f8c8d", edgecolor="#2c3e50", lw=0.6)
    b2 = ax.bar(x + w / 2, g["post_oracle_recall"], width=w, label="FedBN_TTA (Unlabeled AdaBN)", color="#1f77b4", edgecolor="#1a252f", lw=0.6)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.45, f"{h:.1f}%", ha="center", va="bottom", fontsize=6.7)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylim(0, 22.0)
    ax.set_ylabel("Target-Quantile Recall @ 1% FPR (%)")
    ax.set_title("(a) Paired FedBN: Source BN vs. Unlabeled AdaBN")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    # Panel B: Top-8 BoT-IoT Features Separating Top-1% Benign Tail from Rest Benign
    ax = axes[1]
    top8 = df_feat.head(8).iloc[::-1]
    short_names = [
        f.replace("NUM_PKTS_", "PKTS_").replace("_BYTES", "B").replace("SRC_TO_DST_", "S2D_")
        for f in top8["feature"]
    ]
    y_pos = np.arange(len(top8))
    bars = ax.barh(y_pos, top8["ks_top_vs_rest"], color="#c0392b", alpha=0.85, edgecolor="#641e16", lw=0.6, height=0.62)
    for bar, val in zip(bars, top8["ks_top_vs_rest"]):
        ax.text(val - 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                ha="right", va="center", fontsize=6.8, color="white", fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=7.0)
    ax.set_xlim(0.80, 1.01)
    ax.set_xlabel("KS Statistic (Top-1% Benign vs. Rest Benign)")
    ax.set_title("(b) BoT-IoT Top-1% Benign Tail Feature Separation")
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)

    fig.tight_layout(pad=0.6)
    fig.savefig(FIGS / "day2_extensions.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_pipeline_figure()
    make_unsupervised_calibration_figure()
    make_day2_extensions_figure()
    print("Generated 3 new PDF figures in:", FIGS)
    for p in sorted(FIGS.glob("*.pdf")):
        print(" ", p.name, f"({p.stat().st_size / 1024:.1f} KB)")
