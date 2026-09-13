"""
Behavioral sanity checks: does Think-Aloud change risky-choice patterns?

We compare our Think-Aloud datasets (Experiment 1 and Experiment 2) against
original behavioral datasets without Think-Aloud:

- Experiment 1 (PT design, n=72): compare against Kahneman-style prospect theory
  aggregates in `data/prospect_theory_data.xlsx`.
  Matching is done by comparing the (probability, value) vectors for each option
  to avoid any problem-id coding mismatch.

- Experiment 2 (choice13k-sampled): compare our sampled data against the full
  Choice13k aggregates in `data/c13k_selections.csv`.
  Matching is done by extracting numeric IDs from our `problem_id` when possible.

Outputs:
- `results/behavior_replication/*.csv` and `*.json`
- `figures/behavior_replication/*.png`
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils.data_processing import load_datasets


OUT_DIR = Path("results/behavior_replication")
FIG_DIR = Path("figures/behavior_replication")


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _to_float_list(x) -> List[float]:
    """Parse either list-like objects or comma-separated strings into floats."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    out: List[float] = []
    for p in parts:
        if not p:
            continue
        if p.startswith("."):
            p = "0" + p
        out.append(float(p))
    return out


def _normalize_probs(probs: List[float]) -> List[float]:
    """Map probabilities to [0,1] floats, handling percentages."""
    if not probs:
        return []
    probs = [float(p) for p in probs]
    if any(p > 1.0 + 1e-9 for p in probs):
        probs = [p / 100.0 for p in probs]
    return probs


def _canonical_option_signature(
    probs: Iterable[float],
    vals: Iterable[float],
    prob_round: int = 4,
    val_round: int = 4,
) -> Tuple[Tuple[float, float], ...]:
    # Drop near-zero-probability outcomes and merge identical values.
    eps = 10 ** (-(prob_round + 1))
    merged: Dict[float, float] = {}
    for p, v in zip(probs, vals):
        p_f = float(p)
        if abs(p_f) <= eps:
            continue
        v_r = round(float(v), val_round)
        p_r = round(p_f, prob_round)
        merged[v_r] = merged.get(v_r, 0.0) + p_r

    # Renormalize if we're very close to 1 (handles rounding artifacts after merging)
    total = float(sum(merged.values()))
    if total > 0 and abs(total - 1.0) <= 0.02:
        for k in list(merged.keys()):
            merged[k] = merged[k] / total

    pv = [(round(p, prob_round), v) for v, p in merged.items()]
    pv.sort(key=lambda t: (t[0], t[1]))
    return tuple(pv)


@dataclass(frozen=True)
class ProblemSignature:
    opt_a: Tuple[Tuple[float, float], ...]
    opt_b: Tuple[Tuple[float, float], ...]

    def as_order_invariant(self):
        return tuple(sorted([self.opt_a, self.opt_b], key=lambda x: (len(x), x)))


def build_signature_from_pv_lists(p1, v1, p2, v2) -> ProblemSignature:
    p1 = _normalize_probs(_to_float_list(p1))
    v1 = _to_float_list(v1)
    p2 = _normalize_probs(_to_float_list(p2))
    v2 = _to_float_list(v2)
    return ProblemSignature(
        opt_a=_canonical_option_signature(p1, v1),
        opt_b=_canonical_option_signature(p2, v2),
    )


def _label_from_prop_b(prop_b: float, neutral_band: float) -> str:
    if prop_b is None or (isinstance(prop_b, float) and np.isnan(prop_b)):
        return "neutral"
    if abs(float(prop_b) - 0.5) <= neutral_band:
        return "neutral"
    return "B" if float(prop_b) > 0.5 else "A"


def _summarize_alignment(df: pd.DataFrame, neutral_band: float) -> Dict[str, float]:
    diffs = (df["prop_b_ours"] - df["prop_b_orig"]).astype(float).to_numpy()
    mae = float(np.mean(np.abs(diffs))) if len(diffs) else float("nan")
    mse = float(np.mean(diffs**2)) if len(diffs) else float("nan")
    rmse = float(math.sqrt(mse)) if mse == mse else float("nan")

    if len(df) >= 3:
        pear = stats.pearsonr(df["prop_b_orig"], df["prop_b_ours"])
        spear = stats.spearmanr(df["prop_b_orig"], df["prop_b_ours"])
        pear_r, pear_p = float(pear.statistic), float(pear.pvalue)
        sp_r, sp_p = float(spear.statistic), float(spear.pvalue)
    else:
        pear_r = pear_p = sp_r = sp_p = float("nan")

    labels_ours = df["prop_b_ours"].apply(lambda x: _label_from_prop_b(float(x), neutral_band))
    labels_orig = df["prop_b_orig"].apply(lambda x: _label_from_prop_b(float(x), neutral_band))
    agree = float(np.mean((labels_ours == labels_orig).to_numpy())) if len(df) else float("nan")

    if len(diffs) >= 2:
        t_res = stats.ttest_1samp(diffs, popmean=0.0)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
        diff_std = float(np.std(diffs, ddof=1))
    else:
        t_stat = t_p = diff_std = float("nan")

    return {
        "n_problems_matched": float(len(df)),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "pearson_r": pear_r,
        "pearson_p": pear_p,
        "spearman_r": sp_r,
        "spearman_p": sp_p,
        "majority_neutral_agreement": agree,
        "diff_mean": float(np.mean(diffs)) if len(diffs) else float("nan"),
        "diff_std": diff_std,
        "diff_t": t_stat,
        "diff_p": t_p,
    }


def _plot_scatter(df: pd.DataFrame, title: str, out_path: Path) -> None:
    x = df["prop_b_orig"].astype(float).to_numpy()
    y = df["prop_b_ours"].astype(float).to_numpy()
    plt.figure(figsize=(5.4, 5.0))
    plt.scatter(x, y, s=28, alpha=0.85, edgecolor="none")
    plt.plot([0, 1], [0, 1], color="#333333", linewidth=1.2, linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Original dataset: proportion choosing Option B")
    plt.ylabel("Think-Aloud dataset: proportion choosing Option B")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save PNG + high-resolution PDF for paper figures
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=450, bbox_inches="tight")
    plt.close()


def _plot_diff_hist(df: pd.DataFrame, title: str, out_path: Path) -> None:
    diffs = (df["prop_b_ours"] - df["prop_b_orig"]).astype(float).to_numpy()
    plt.figure(figsize=(6.0, 3.7))
    plt.hist(diffs, bins=18, color="#4c78a8", alpha=0.85)
    plt.axvline(0.0, color="#222222", linewidth=1.0)
    plt.xlabel("Difference in proportion B (Think-Aloud − Original)")
    plt.ylabel("Count of problems")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=450, bbox_inches="tight")
    plt.close()


def _plot_label_confusion(df: pd.DataFrame, neutral_band: float, title: str, out_path: Path) -> None:
    # If neutral_band == 0, we want a strict A/B confusion matrix (2x2).
    if float(neutral_band) == 0.0:
        labels = ["A", "B"]

        def _ab_label(prop_b: float) -> str:
            return "B" if float(prop_b) > 0.5 else "A"

        ours = df["prop_b_ours"].astype(float).apply(_ab_label)
        orig = df["prop_b_orig"].astype(float).apply(_ab_label)
    else:
        labels = ["A", "neutral", "B"]
        ours = df["prop_b_ours"].astype(float).apply(lambda x: _label_from_prop_b(float(x), neutral_band))
        orig = df["prop_b_orig"].astype(float).apply(lambda x: _label_from_prop_b(float(x), neutral_band))

    k = len(labels)
    mat = np.zeros((k, k), dtype=int)
    idx = {l: i for i, l in enumerate(labels)}
    for o, r in zip(orig.tolist(), ours.tolist()):
        mat[idx[o], idx[r]] += 1

    plt.figure(figsize=(4.2, 3.8) if k == 2 else (4.8, 4.2))
    plt.imshow(mat, cmap="Blues")
    plt.xticks(range(k), labels)
    plt.yticks(range(k), labels)
    for i in range(k):
        for j in range(k):
            plt.text(j, i, str(mat[i, j]), ha="center", va="center", color="#111111")
    plt.xlabel("Think-Aloud label")
    plt.ylabel("Original label")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=450, bbox_inches="tight")
    plt.close()


def _plot_wordcount_panel_with_zero_bar(
    out_prefix: Path,
    annotate_zero_bar: bool = False,
) -> None:
    """
    Horizontal 1×2 panel of Think-Aloud word-count histograms (Exp1/Exp2),
    each drawn on a single custom axis with two visual breaks:
    - a dedicated bar for word_count == 0
    - regular histogram bars for 1..175
    - a dedicated aggregated bar for word_count > 175

    This keeps the right tail compact (avoids large empty space) while still
    highlighting zero-word trials and preserving overall shape.
    """
    from utils.data_processing import load_datasets  # local import to avoid cycles

    # Load precomputed word counts (authoritative definition)
    _, ta1 = load_datasets("small")
    _, ta2 = load_datasets("large")
    wc1 = ta1["word_count"].to_numpy()
    wc2 = ta2["word_count"].to_numpy()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
    colors = ["#4c78a8", "#f58518"]

    label_fs = 16
    title_fs = 20
    tick_fs = 14

    cap_right = 175
    # Regular bins within 1..175
    bins = np.array([1, 21, 41, 61, 81, 101, 121, 141, 161, 176], dtype=float)
    gap_width_left = 16.0
    gap_width_right = 20.0
    zero_bar_width = 16.0
    tail_bar_width = 16.0

    def _xmap(x):
        x_arr = np.asarray(x, dtype=float)
        return np.where(x_arr >= 1, x_arr + gap_width_left, x_arr)

    for ax, wc, color, title in zip(
        axes, [wc1, wc2], colors, ["Experiment 1", "Experiment 2"]
    ):
        zero_mask = wc == 0
        zero_count = int(zero_mask.sum())
        wc_pos = wc[(wc >= 1) & (wc <= cap_right)]
        tail_count = int((wc > cap_right).sum())

        # Zero bar at x = 0.
        ax.bar(
            0.0,
            zero_count,
            width=zero_bar_width,
            color=color,
            edgecolor="white",
            alpha=0.85,
            zorder=3,
        )

        # Positive-count histogram (1..175), manually shifted right after left break.
        hist_counts, hist_edges = np.histogram(wc_pos, bins=bins)
        disp_left = _xmap(hist_edges[:-1])
        disp_width = np.diff(hist_edges)
        ax.bar(
            disp_left,
            hist_counts,
            width=disp_width,
            align="edge",
            color=color,
            edgecolor="white",
            alpha=0.8,
            zorder=2,
        )

        # Aggregate right tail (>175) into a dedicated final bar.
        cap_disp = float(_xmap(cap_right))
        tail_left = cap_disp + gap_width_right
        tail_center = tail_left + tail_bar_width / 2.0
        ax.bar(
            tail_center,
            tail_count,
            width=tail_bar_width,
            color=color,
            edgecolor="white",
            alpha=0.8,
            zorder=3,
        )

        # Highlight the visual break region between zero and positive bins.
        gap_left = zero_bar_width / 2.0
        gap_right = _xmap(1.0)
        ax.axvspan(gap_left, gap_right, color="white", zorder=4)
        ax.axvline(gap_left, color="#cccccc", linewidth=0.8, zorder=5)
        ax.axvline(gap_right, color="#cccccc", linewidth=0.8, zorder=5)

        # Break marks on the x-axis.
        trans = ax.get_xaxis_transform()
        x_break = (gap_left + gap_right) / 2.0
        dx = 1.8
        dy = 0.025
        for y0 in (-0.005, -0.04):
            ax.plot(
                [x_break - dx, x_break + dx],
                [y0 - dy, y0 + dy],
                transform=trans,
                color="#333333",
                linewidth=1.4,
                clip_on=False,
                zorder=6,
            )

        # Right-side break marks before the >175 aggregate bar.
        right_break_left = cap_disp
        right_break_right = tail_left
        ax.axvspan(right_break_left, right_break_right, color="white", zorder=4)
        ax.axvline(right_break_left, color="#cccccc", linewidth=0.8, zorder=5)
        ax.axvline(right_break_right, color="#cccccc", linewidth=0.8, zorder=5)
        trans = ax.get_xaxis_transform()
        x_break_r = (right_break_left + right_break_right) / 2.0
        for y0 in (-0.005, -0.04):
            ax.plot(
                [x_break_r - dx, x_break_r + dx],
                [y0 - dy, y0 + dy],
                transform=trans,
                color="#333333",
                linewidth=1.4,
                clip_on=False,
                zorder=6,
            )

        if annotate_zero_bar:
            # Zero-bar annotation as a compact callout to the right of the bar.
            ax.annotate(
                f"0 words: {zero_count} ({pct_zero:.1f}%)",
                xy=(0.0, zero_count),
                xytext=(10, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9.5,
                color="#222222",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="#d0d0d0",
                    linewidth=0.6,
                    alpha=0.96,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#8a8a8a",
                    linewidth=0.8,
                    shrinkA=0,
                    shrinkB=3,
                ),
                clip_on=False,
                zorder=7,
            )

        # Axis formatting for this panel
        ax.set_ylabel("Count", fontsize=label_fs)
        ax.set_title(title, fontsize=title_fs)
        ax.grid(False)
        ax.set_facecolor("white")
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.set_xlim(-zero_bar_width * 0.75, tail_center + tail_bar_width / 2.0 + 8.0)

        # Custom ticks: 0, regular interior ticks, and a final >175 bucket label.
        tick_values = [0, 25, 75, 125, 150]
        tick_positions = [0.0] + _xmap(np.array(tick_values[1:], dtype=float)).tolist()
        tick_positions.append(tail_center)
        tick_labels = [str(v) for v in tick_values] + [">175"]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Word count", fontsize=label_fs)

    fig.patch.set_facecolor("white")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_prefix.with_suffix(".pdf")
    png_path = out_prefix.with_suffix(".png")
    svg_path = out_prefix.with_suffix(".svg")
    plt.savefig(pdf_path, dpi=450, bbox_inches="tight")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()


def _plot_exp1_scatter_and_confusion_panel(
    exp1_df: pd.DataFrame,
    out_prefix: Path,
) -> None:
    """
    Two-panel figure for Exp1 replication:
    - Left: scatter (prop B original vs ours)
    - Right: strict A/B confusion matrix (no neutral class)
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    label_fs = 13
    title_fs = 14
    tick_fs = 12

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.4),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.16},
        constrained_layout=True,
    )
    ax_scatter, ax_conf = axes

    # --- Left panel: scatter ---
    ax_scatter.text(
        -0.12,
        1.10,
        "A",
        transform=ax_scatter.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
    )
    x = exp1_df["prop_b_orig"].astype(float).to_numpy()
    y = exp1_df["prop_b_ours"].astype(float).to_numpy()
    ax_scatter.scatter(x, y, s=36, alpha=0.85, edgecolor="none", color="#4c78a8")
    ax_scatter.plot([0, 1], [0, 1], color="#333333", linewidth=1.2, linestyle="--")
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0, 1)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("Original: proportion choosing Option B", fontsize=label_fs)
    ax_scatter.set_ylabel("Think-Aloud: proportion choosing Option B", fontsize=label_fs)
    ax_scatter.set_title("Problem-level choice rates", fontsize=title_fs)
    ax_scatter.tick_params(axis="both", labelsize=tick_fs)
    ax_scatter.set_facecolor("white")
    ax_scatter.grid(False)
    # Correlation annotation (Pearson)
    try:
        r, p = stats.pearsonr(x, y)
        r = float(r)
        p = float(p)
        p_txt = "< .001" if p < 0.001 else f"= {p:.3f}"
        ann = f"$r$ = {r:.2f}, $p$ {p_txt}"
        ax_scatter.text(
            0.03,
            0.97,
            ann,
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.95),
        )
    except Exception:
        pass

    # --- Right panel: 2x2 confusion (A/B only) ---
    ax_conf.text(
        -0.12,
        1.10,
        "B",
        transform=ax_conf.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
    )
    def _ab_label(prop_b: float) -> str:
        return "B" if float(prop_b) > 0.5 else "A"

    labels = ["A", "B"]
    ours = exp1_df["prop_b_ours"].astype(float).apply(_ab_label).tolist()
    orig = exp1_df["prop_b_orig"].astype(float).apply(_ab_label).tolist()
    mat = np.zeros((2, 2), dtype=int)
    idx = {"A": 0, "B": 1}
    for o, r in zip(orig, ours):
        mat[idx[o], idx[r]] += 1

    ax_conf.imshow(mat, cmap="Blues")
    ax_conf.set_xticks([0, 1])
    ax_conf.set_yticks([0, 1])
    ax_conf.set_xticklabels(labels, fontsize=tick_fs)
    ax_conf.set_yticklabels(labels, fontsize=tick_fs)
    for i in range(2):
        for j in range(2):
            ax_conf.text(j, i, str(mat[i, j]), ha="center", va="center", color="#111111", fontsize=12)
    ax_conf.set_xlabel("Think-Aloud label", fontsize=label_fs)
    ax_conf.set_ylabel("Original label", fontsize=label_fs)
    ax_conf.set_title("Majority label agreement", fontsize=title_fs)
    ax_conf.set_facecolor("white")
    ax_conf.grid(False)

    fig.patch.set_facecolor("white")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_prefix.with_suffix(".pdf")
    png_path = out_prefix.with_suffix(".png")
    svg_path = out_prefix.with_suffix(".svg")
    plt.savefig(pdf_path, dpi=450, bbox_inches="tight")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()


def compare_exp1_vs_pt(pt_xlsx_path: Path, min_n: int, neutral_band: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    decisions_df, _ = load_datasets("small")
    ours = decisions_df.dropna(subset=["choice"]).copy()
    ours["choice"] = ours["choice"].astype(int)

    sig_by_pid: Dict[str, ProblemSignature] = {}
    for pid, grp in ours.groupby("problem_id"):
        row = grp.iloc[0]
        sig_by_pid[str(pid)] = build_signature_from_pv_lists(row["p1"], row["v1"], row["p2"], row["v2"])

    ours_props = (
        ours.groupby("problem_id", as_index=False)
        .agg(n_ours=("choice", "size"), prop_b_ours=("choice", "mean"))
    )
    ours_props = ours_props[ours_props["n_ours"] >= min_n].copy()
    ours_props["sig"] = ours_props["problem_id"].astype(str).map(lambda pid: sig_by_pid.get(str(pid)))

    pt_df = pd.read_excel(pt_xlsx_path, sheet_name=0)

    def _pt_option_sig(prob_col, val_col) -> Tuple[Tuple[float, float], ...]:
        probs = _normalize_probs(_to_float_list(prob_col))
        vals = _to_float_list(val_col)
        # Some PT rows omit the remaining probability mass; our exp1 encoding
        # often includes it explicitly as a 0 outcome (e.g. 0.33,0.66 plus 0.01 at 0).
        s = float(np.sum(probs)) if probs else 0.0
        if probs and (len(probs) == len(vals)) and (s < 0.9995):
            missing = 1.0 - s
            # PT often omits the implicit 0-outcome entirely (missing mass can be large).
            if 0.0005 <= missing <= 1.0:
                probs = probs + [missing]
                vals = vals + [0.0]
        # If rounding makes it slightly >1, renormalize lightly.
        s2 = float(np.sum(probs)) if probs else 0.0
        if probs and abs(s2 - 1.0) <= 0.02 and s2 > 0:
            probs = [p / s2 for p in probs]
        return _canonical_option_signature(probs, vals)
    pt_rows = []
    for _, r in pt_df.iterrows():
        sig = ProblemSignature(
            opt_a=_pt_option_sig(r.get("Probability A"), r.get("Value A")),
            opt_b=_pt_option_sig(r.get("Probability B"), r.get("Value B")),
        )
        pt_rows.append(
            {
                "pt_problem": int(r.get("Unnamed: 0")) if not pd.isna(r.get("Unnamed: 0")) else None,
                "n_orig": float(r.get("Reported N")) if not pd.isna(r.get("Reported N")) else np.nan,
                "prop_b_orig": float(r.get("Reported Proportion B")) if not pd.isna(r.get("Reported Proportion B")) else np.nan,
                "sig": sig,
            }
        )
    pt_sig_df = pd.DataFrame(pt_rows).dropna(subset=["prop_b_orig"])

    inv_to_pt: Dict[Tuple, dict] = {}
    for row in pt_sig_df.to_dict(orient="records"):
        inv_to_pt[row["sig"].as_order_invariant()] = row

    matched = []
    for row in ours_props.to_dict(orient="records"):
        sig: ProblemSignature = row["sig"]
        if sig is None:
            continue
        pt_row = inv_to_pt.get(sig.as_order_invariant())
        if not pt_row:
            continue

        prop_b_ours = float(row["prop_b_ours"])
        swapped = False
        # Align our option2 to PT option B; if swapped, flip.
        if sig.opt_a == pt_row["sig"].opt_b and sig.opt_b == pt_row["sig"].opt_a:
            swapped = True
            prop_b_ours = 1.0 - prop_b_ours

        matched.append(
            {
                "dataset": "exp1_vs_pt",
                "problem_id": row["problem_id"],
                "pt_problem": pt_row["pt_problem"],
                "n_ours": int(row["n_ours"]),
                "n_orig": int(pt_row["n_orig"]) if pt_row["n_orig"] == pt_row["n_orig"] else None,
                "prop_b_ours": prop_b_ours,
                "prop_b_orig": float(pt_row["prop_b_orig"]),
                "swapped_options": swapped,
            }
        )

    out = pd.DataFrame(matched)
    if not out.empty:
        out["diff"] = out["prop_b_ours"] - out["prop_b_orig"]
        out["abs_diff"] = out["diff"].abs()
        out["label_ours"] = out["prop_b_ours"].apply(lambda x: _label_from_prop_b(float(x), neutral_band))
        out["label_orig"] = out["prop_b_orig"].apply(lambda x: _label_from_prop_b(float(x), neutral_band))
        out["agree_label"] = out["label_ours"] == out["label_orig"]
        summary = _summarize_alignment(out, neutral_band)
    else:
        summary = {}
    return out, summary


def compare_exp2_vs_choice13k(
    c13k_csv_path: Path,
    c13k_problems_json_path: Path,
    min_n: int,
    neutral_band: float,
    require_no_feedback: bool = False,
    require_no_ambiguity: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    decisions_df, _ = load_datasets("large")
    ours = decisions_df.dropna(subset=["choice"]).copy()
    ours["choice"] = ours["choice"].astype(int)
    ours["problem_num"] = ours["problem_id"].astype(str).str.extract(r"problem([0-9]+)", expand=False).astype(float)
    ours = ours.dropna(subset=["problem_num"]).copy()
    ours["problem_num"] = ours["problem_num"].astype(int)

    ours_props = (
        ours.groupby("problem_num", as_index=False)
        .agg(n_ours=("choice", "size"), prop_b_ours=("choice", "mean"))
    )
    ours_props = ours_props[ours_props["n_ours"] >= min_n].copy()

    c13k = pd.read_csv(c13k_csv_path)
    if "Problem" not in c13k.columns or "bRate" not in c13k.columns:
        raise ValueError("c13k_selections.csv missing required columns: Problem, bRate")

    # Optional filtering to align Choice13k conditions with our Think-Aloud task
    # (no feedback, no ambiguity).
    if require_no_feedback:
        if "Feedback" not in c13k.columns:
            raise ValueError("c13k_selections.csv missing required column: Feedback")
        c13k = c13k[c13k["Feedback"].astype(bool) == False]  # noqa: E712
    if require_no_ambiguity:
        if "Amb" not in c13k.columns:
            raise ValueError("c13k_selections.csv missing required column: Amb")
        c13k = c13k[c13k["Amb"].astype(bool) == False]  # noqa: E712
    orig = c13k[["Problem", "n", "bRate"]].copy()
    orig = orig.rename(columns={"Problem": "problem_num", "n": "n_orig", "bRate": "prop_b_orig"})
    orig["problem_num"] = orig["problem_num"].astype(int)
    orig = orig[orig["n_orig"] >= min_n].copy()

    merged = pd.merge(ours_props, orig, on="problem_num", how="inner")
    if merged.empty:
        return merged, {}

    # Stimuli sanity check using full problem definitions.
    problems = json.loads(c13k_problems_json_path.read_text(encoding="utf-8"))

    def _json_sig(problem_num: int):
        d = problems.get(str(problem_num))
        if not d:
            return None
        A = _canonical_option_signature([p for p, _ in d["A"]], [v for _, v in d["A"]])
        B = _canonical_option_signature([p for p, _ in d["B"]], [v for _, v in d["B"]])
        return A, B

    rep = ours.groupby("problem_num").head(1).set_index("problem_num")

    rows = []
    n_missing_stim = 0
    n_mismatch_stim = 0
    for r in merged.to_dict(orient="records"):
        pn = int(r["problem_num"])
        if pn not in rep.index:
            continue
        js = _json_sig(pn)
        if js is None:
            n_missing_stim += 1
            continue
        A_sig, B_sig = js
        sig_ours = build_signature_from_pv_lists(rep.loc[pn, "p1"], rep.loc[pn, "v1"], rep.loc[pn, "p2"], rep.loc[pn, "v2"])
        swapped = False
        if (sig_ours.opt_a, sig_ours.opt_b) == (A_sig, B_sig):
            swapped = False
        elif (sig_ours.opt_a, sig_ours.opt_b) == (B_sig, A_sig):
            swapped = True
        else:
            n_mismatch_stim += 1
            continue

        prop_b = float(r["prop_b_ours"])
        if swapped:
            prop_b = 1.0 - prop_b

        rows.append({**r, "prop_b_ours_raw": float(r["prop_b_ours"]), "prop_b_ours": prop_b, "swapped_options": swapped})

    out = pd.DataFrame(rows)
    diag = {
        "n_rows_initial_merge": int(len(merged)),
        "n_rows_after_stimuli_check": int(len(out)),
        "n_missing_stimuli_in_json": int(n_missing_stim),
        "n_stimuli_signature_mismatch": int(n_mismatch_stim),
        "stimuli_match_rate": float(len(out) / len(merged)) if len(merged) else float("nan"),
    }
    (OUT_DIR / "exp2_vs_choice13k_stimuli_check.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    if out.empty:
        return out, {}

    out["dataset"] = "exp2_vs_choice13k"
    out["diff"] = out["prop_b_ours"] - out["prop_b_orig"]
    out["abs_diff"] = out["diff"].abs()
    out["label_ours"] = out["prop_b_ours"].apply(lambda x: _label_from_prop_b(float(x), neutral_band))
    out["label_orig"] = out["prop_b_orig"].apply(lambda x: _label_from_prop_b(float(x), neutral_band))
    out["agree_label"] = out["label_ours"] == out["label_orig"]
    summary = _summarize_alignment(out, neutral_band)
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Think-Aloud behavior to original datasets.")
    parser.add_argument("--pt_xlsx", type=str, default="data/prospect_theory_data.xlsx")
    parser.add_argument("--c13k_csv", type=str, default="data/c13k_selections.csv")
    parser.add_argument("--c13k_problems_json", type=str, default="data/c13k_problems.json")
    parser.add_argument("--min_n", type=int, default=5)
    parser.add_argument("--neutral_band", type=float, default=0.01)
    parser.add_argument(
        "--exp2_require_no_feedback",
        action="store_true",
        help="Filter Choice13k baseline to Feedback==False before matching (Exp2 only).",
    )
    parser.add_argument(
        "--exp2_require_no_ambiguity",
        action="store_true",
        help="Filter Choice13k baseline to Amb==False before matching (Exp2 only).",
    )
    parser.add_argument(
        "--diagnostic_curve",
        action="store_true",
        help="If set, run a sweep over min_n and save diagnostic curves for Exp2.",
    )
    parser.add_argument(
        "--wordcount_plot_only",
        action="store_true",
        help="Only generate the Think-Aloud word-count histogram (no other analyses).",
    )
    parser.add_argument("--sweep_min_n_start", type=int, default=3)
    parser.add_argument("--sweep_min_n_end", type=int, default=9)
    args = parser.parse_args()

    _ensure_dirs()

    if args.wordcount_plot_only:
        out_prefix = FIG_DIR / "think_aloud_wordcount_zerosep_no_annotations"
        _plot_wordcount_panel_with_zero_bar(out_prefix, annotate_zero_bar=False)
        print(f"Saved word-count histogram to {out_prefix.with_suffix('.pdf')}")
        return

    exp1_df, exp1_summary = compare_exp1_vs_pt(Path(args.pt_xlsx), args.min_n, args.neutral_band)
    exp1_df.to_csv(OUT_DIR / "exp1_vs_pt_matched.csv", index=False)
    (OUT_DIR / "exp1_vs_pt_summary.json").write_text(json.dumps(exp1_summary, indent=2), encoding="utf-8")

    if not exp1_df.empty:
        _plot_scatter(exp1_df, "Experiment 1 vs Prospect Theory (problem-level prop B)", FIG_DIR / "exp1_scatter_propB.png")
        _plot_diff_hist(exp1_df, "Experiment 1 − Prospect Theory (prop B differences)", FIG_DIR / "exp1_hist_diff.png")
        # For the paper figure, use strict A/B confusion (no neutral class).
        _plot_label_confusion(exp1_df, 0.0, "Exp1 vs PT label agreement (A/B)", FIG_DIR / "exp1_confusion_labels.png")
        _plot_exp1_scatter_and_confusion_panel(
            exp1_df,
            FIG_DIR / "exp1_panel_scatter_confusion",
        )

    exp2_df, exp2_summary = compare_exp2_vs_choice13k(
        Path(args.c13k_csv),
        Path(args.c13k_problems_json),
        args.min_n,
        args.neutral_band,
        require_no_feedback=bool(args.exp2_require_no_feedback),
        require_no_ambiguity=bool(args.exp2_require_no_ambiguity),
    )
    exp2_df.to_csv(OUT_DIR / "exp2_vs_choice13k_matched.csv", index=False)
    (OUT_DIR / "exp2_vs_choice13k_summary.json").write_text(json.dumps(exp2_summary, indent=2), encoding="utf-8")

    if not exp2_df.empty:
        _plot_scatter(exp2_df, "Experiment 2 vs Choice13k (problem-level prop B)", FIG_DIR / "exp2_scatter_propB.png")
        _plot_diff_hist(exp2_df, "Experiment 2 − Choice13k (prop B differences)", FIG_DIR / "exp2_hist_diff.png")
        _plot_label_confusion(exp2_df, args.neutral_band, "Exp2 vs Choice13k label agreement (A/neutral/B)", FIG_DIR / "exp2_confusion_labels.png")

    def _print_summary(name: str, summ: Dict[str, float]) -> None:
        if not summ:
            print(f"{name}: no matched problems.")
            return
        print(f"\n=== {name} ===")
        print(f"Matched problems: {int(summ['n_problems_matched'])}")
        print(f"MAE: {summ['mae']:.4f} | RMSE: {summ['rmse']:.4f}")
        print(f"Pearson r: {summ['pearson_r']:.3f} (p={summ['pearson_p']:.2g})")
        print(f"Spearman r: {summ['spearman_r']:.3f} (p={summ['spearman_p']:.2g})")
        print(f"Label agreement (A/neutral/B): {summ['majority_neutral_agreement']:.3f}")
        print(f"Mean diff (ours-orig): {summ['diff_mean']:.4f} | t={summ['diff_t']:.3f} p={summ['diff_p']:.2g}")

    _print_summary("Experiment 1 vs Prospect Theory", exp1_summary)
    _print_summary("Experiment 2 vs Choice13k", exp2_summary)

    # Optional: diagnostic sweep for Exp2 over sample-size thresholds, with and without neutral class
    if args.diagnostic_curve:
        rows = []
        for min_n in range(int(args.sweep_min_n_start), int(args.sweep_min_n_end) + 1):
            for nb in [float(args.neutral_band), 0.0]:
                df2, summ2 = compare_exp2_vs_choice13k(
                    Path(args.c13k_csv),
                    Path(args.c13k_problems_json),
                    min_n,
                    nb,
                    require_no_feedback=bool(args.exp2_require_no_feedback),
                    require_no_ambiguity=bool(args.exp2_require_no_ambiguity),
                )
                rows.append(
                    {
                        "min_n": min_n,
                        "neutral_band": nb,
                        "n_problems_matched": int(summ2.get("n_problems_matched", 0)) if summ2 else 0,
                        "mae": summ2.get("mae", float("nan")) if summ2 else float("nan"),
                        "rmse": summ2.get("rmse", float("nan")) if summ2 else float("nan"),
                        "pearson_r": summ2.get("pearson_r", float("nan")) if summ2 else float("nan"),
                        "spearman_r": summ2.get("spearman_r", float("nan")) if summ2 else float("nan"),
                        "label_agreement": summ2.get("majority_neutral_agreement", float("nan")) if summ2 else float("nan"),
                        "diff_mean": summ2.get("diff_mean", float("nan")) if summ2 else float("nan"),
                        "diff_p": summ2.get("diff_p", float("nan")) if summ2 else float("nan"),
                    }
                )

        curve_df = pd.DataFrame(rows).sort_values(["neutral_band", "min_n"]).reset_index(drop=True)
        curve_csv = OUT_DIR / "exp2_diagnostic_curve.csv"
        curve_df.to_csv(curve_csv, index=False)

        def _plot_metric(metric: str, ylabel: str, filename: str) -> None:
            plt.figure(figsize=(6.3, 3.9))
            for nb, grp in curve_df.groupby("neutral_band"):
                xs = grp["min_n"].to_numpy()
                ys = grp[metric].to_numpy()
                label = f"neutral_band={nb:g}"
                plt.plot(xs, ys, marker="o", linewidth=1.6, label=label)
            plt.xlabel("Per-problem sample size threshold (min_n)")
            plt.ylabel(ylabel)
            plt.title(f"Experiment 2 diagnostics: {metric} vs min_n")
            plt.grid(True, alpha=0.25)
            plt.legend(frameon=True, fontsize=9)
            plt.tight_layout()
            out_path = FIG_DIR / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=220)
            plt.close()

        _plot_metric("n_problems_matched", "Matched problems", "exp2_curve_matched_n.png")
        _plot_metric("mae", "MAE |prop(B)|", "exp2_curve_mae.png")
        _plot_metric("rmse", "RMSE |prop(B)|", "exp2_curve_rmse.png")
        _plot_metric("pearson_r", "Pearson r", "exp2_curve_pearson_r.png")
        _plot_metric("label_agreement", "Agreement (A/neutral/B or A/B)", "exp2_curve_agreement.png")

        print(f"\nSaved Exp2 diagnostic sweep to {curve_csv} and plots to {FIG_DIR}")


if __name__ == "__main__":
    main()

