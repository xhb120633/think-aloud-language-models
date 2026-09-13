"""
Control-mode comparison figure.

This repository has an "action_removal" control condition for Exp1 (LLM
prediction) where we mask superficial action claims from think-aloud and
re-run prediction.

Your request: integrate the small-size action_removal results for
LLaMA-3.1-70B into the existing control-mode comparison figure:
  figures/control_comparison/control_mode_comparison.pdf
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import FIGURES_DIR

LLAMA70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
RESULTS_DIR = Path("results/exp1_llm_prediction")


@dataclass(frozen=True)
class ModeBar:
    name: str
    participant_df: pd.DataFrame  # columns: sub_id, acc, like


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def _load_llama70b_mode_df(
    data_size: str,
    mode_key: str,
    action_removal_results_filename_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load per-participant metrics for one mode from the Exp1 prediction JSON.

    mode_key in: {'human', 'permutation', 'base', 'action_removal'}
    """
    # Backwards compatible default behavior: if a per-run override is not passed,
    # we fall back to known filenames / standard naming.
    model_safe = _sanitize_model_name(LLAMA70B)
    # File naming conventions used by exp1_llm_prediction.py
    if mode_key == "permutation":
        fname = f"{model_safe}_{data_size}_permuted_results.json"
    else:
        fname = f"{model_safe}_{data_size}_{mode_key}_results.json"

    # NOTE: for action_removal, we support multiple filename variants:
    # - default: *_action_removal_results.json
    # - suffixed: *_action_removal_results_*.json / *_action_removal_results-*.json
    # We also allow callers to override the exact filename explicitly.
    if mode_key == "action_removal":
        if action_removal_results_filename_override:
            override_path = RESULTS_DIR / action_removal_results_filename_override
            if override_path.exists():
                path = override_path
            else:
                path = RESULTS_DIR / fname
        else:
            # Try exact standard filename first.
            path = RESULTS_DIR / fname
            if not path.exists():
                # Then try any suffixed variant for this data_size.
                # Examples:
                #   ..._small_action_removal_results_LLaMA.json
                #   ..._large_action_removal_results-GPT-5.4-nano.json
                pattern = f"{model_safe}_{data_size}_{mode_key}_results*.json"
                candidates = sorted(RESULTS_DIR.glob(pattern))
                if candidates:
                    # Prefer suffix styles we used before; otherwise fallback to first.
                    preferred = [p for p in candidates if p.name.endswith("_LLaMA.json")]
                    if preferred:
                        path = preferred[0]
                    else:
                        path = candidates[0]
    else:
        path = RESULTS_DIR / fname
    if not path.exists():
        # Fallback: combined file with all modes, filtered by row['mode']
        combined = RESULTS_DIR / f"{model_safe}_{data_size}_results.json"
        if not combined.exists():
            return pd.DataFrame(columns=["sub_id", "acc", "like"])
        path = combined

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame(columns=["sub_id", "acc", "like"])

    rows = data.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["sub_id", "acc", "like"])

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sub_id", "acc", "like"])

    if "mode" in df.columns:
        target = mode_key
        if mode_key == "permutation":
            target = "permutation"
        df = df[df["mode"].astype(str).eq(target)]

    required = {"sub_id", "extracted_choice", "actual_choice"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["sub_id", "acc", "like"])

    df = df.copy()
    df["acc"] = (df["extracted_choice"] == df["actual_choice"]).astype(float)

    if {"probability_option_a", "probability_option_b", "actual_choice"}.issubset(df.columns):
        df["like"] = df.apply(
            lambda x: x["probability_option_a"] if int(x["actual_choice"]) == 0 else x["probability_option_b"],
            axis=1,
        ).astype(float)
    else:
        df["like"] = np.nan

    if "sub_id" not in df.columns:
        return pd.DataFrame(columns=["sub_id", "acc", "like"])

    per_sub = (
        df.groupby("sub_id", as_index=False)[["acc", "like"]]
        .mean()
        .rename(columns={"sub_id": "sub_id"})
    )
    return per_sub


def _summarize_across_participants(participant_df: pd.DataFrame) -> Tuple[float, float]:
    """Return (mean, sem) for acc/like across participants."""
    if participant_df is None or participant_df.empty:
        return float("nan"), float("nan")
    vals = participant_df["acc" if "acc" in participant_df.columns else "like"].astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, sem


def _bh_adjust(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR adjustment preserving original order."""
    if not pvals:
        return []
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj_ranked = ranked * n / (np.arange(1, n + 1))
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
    adj_ranked = np.clip(adj_ranked, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj_ranked
    return out.tolist()


def _paired_tests_original_vs_controls(
    data_size: str,
    action_removal_results_filename_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run paired tests (per participant) for Original vs each control condition:
    Permuted, Masked(action_removal), Baseline.
    Applies BH/FDR correction within each experiment and metric.
    """
    cond_map = {
        "original": "human",
        "permuted": "permutation",
        "masked": "action_removal",
        "baseline": "base",
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for label, key in cond_map.items():
        dfs[label] = _load_llama70b_mode_df(
            data_size,
            key,
            action_removal_results_filename_override=action_removal_results_filename_override,
        )

    orig = dfs["original"]
    if orig.empty:
        return pd.DataFrame()

    rows: List[Dict[str, float | str | int]] = []
    controls = ["permuted", "masked", "baseline"]
    metrics = [("acc", "accuracy"), ("like", "likelihood")]
    for ctrl in controls:
        bdf = dfs[ctrl]
        if bdf.empty:
            continue
        merged = pd.merge(
            orig[["sub_id", "acc", "like"]],
            bdf[["sub_id", "acc", "like"]],
            on="sub_id",
            suffixes=("_orig", "_ctrl"),
            how="inner",
        ).replace([np.inf, -np.inf], np.nan)
        for m_col, m_label in metrics:
            sub = merged[["sub_id", f"{m_col}_orig", f"{m_col}_ctrl"]].dropna()
            n = len(sub)
            if n < 2:
                continue
            a = sub[f"{m_col}_orig"].astype(float).to_numpy()
            b = sub[f"{m_col}_ctrl"].astype(float).to_numpy()
            t_stat, p_val = ttest_rel(a, b, nan_policy="omit")
            diff = a - b
            sd = np.std(diff, ddof=1)
            d = float(np.mean(diff) / sd) if sd > 0 else float("nan")
            rows.append(
                {
                    "experiment": "Experiment 1" if data_size == "small" else "Experiment 2",
                    "data_size": data_size,
                    "comparison": f"original_vs_{ctrl}",
                    "metric": m_label,
                    "n": int(n),
                    "t": float(t_stat),
                    "df": int(n - 1),
                    "p": float(p_val),
                    "cohen_d": d,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # BH/FDR correction within each experiment and metric across the 3 comparisons.
    out["p_adj_fdr_bh"] = np.nan
    for (exp_name, metric), idx in out.groupby(["experiment", "metric"]).groups.items():
        pvals = out.loc[idx, "p"].astype(float).tolist()
        out.loc[idx, "p_adj_fdr_bh"] = _bh_adjust(pvals)
    return out


def create_control_mode_comparison_figure(
    data_size_small: str = "small",
    data_size_large: str = "large",
    output_suffix: str = "_action_removal_LLaMA",
    action_removal_results_filename_small: Optional[str] = None,
    action_removal_results_filename_large: Optional[str] = None,
) -> None:
    """
    Create `figures/control_comparison/control_mode_comparison.pdf` with 2x2 panels:

    - row 0: Experiment 1 (small)
    - row 1: Experiment 2 (large)
    - col 0: Accuracy
    - col 1: Likelihood

    Bars:
    - Exp1: base, permuted, original, action_removal (4 bars)
    - Exp2: base, permuted, original (3 bars)
    """
    def _mean_sem(df: pd.DataFrame, metric: str) -> Tuple[float, float]:
        if df is None or df.empty or metric not in df.columns:
            return float("nan"), float("nan")
        vals = df[metric].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan")
        mean = float(np.nanmean(vals))
        sem = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return mean, sem

    # Experiment panel definitions
    exp_panels = [
        (data_size_small, "Experiment 1", [
            ("human", "Original", None),
            ("permutation", "Permuted (context mismatch)", "//"),
            ("action_removal", "Masked (choice removed)", "oo"),
            ("base", "Baseline (task only)", "xx"),
        ]),
        (data_size_large, "Experiment 2", [
            ("human", "Original", None),
            ("permutation", "Permuted (context mismatch)", "//"),
            ("action_removal", "Masked (choice removed)", "oo"),
            ("base", "Baseline (task only)", "xx"),
        ]),
    ]

    metrics = [("acc", "Accuracy"), ("like", "Likelihood")]
    bar_color = "#ff7f0e"

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for row_i, (ds, exp_label, conds) in enumerate(exp_panels):
        for col_i, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[row_i, col_i]
            x = np.arange(len(conds))
            width = 0.9

            for i, (key, label, hatch) in enumerate(conds):
                per_size_override = (
                    action_removal_results_filename_small
                    if ds == data_size_small
                    else action_removal_results_filename_large
                )
                df = _load_llama70b_mode_df(
                    ds,
                    key,
                    action_removal_results_filename_override=per_size_override,
                )
                mean, sem = _mean_sem(df, metric_key)
                if not np.isfinite(mean):
                    continue
                yerr = sem * 1.96 if np.isfinite(sem) else 0.0
                ax.bar(
                    x[i],
                    mean,
                    width=width,
                    color=bar_color,
                    edgecolor="white",
                    linewidth=0.5,
                    yerr=yerr,
                    capsize=3,
                    hatch=hatch if hatch is not None else None,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([c[1] for c in conds], fontsize=7.5, rotation=20, ha="right")
            ax.set_ylim(0.0, 1.05)
            ax.grid(False)
            ax.set_facecolor("white")
            ax.set_ylabel(metric_title)
            ax.set_title(f"{exp_label} - {metric_title}", fontsize=10)

    out_dir = FIGURES_DIR / "control_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"control_mode_comparison{output_suffix}.pdf"
    fig.savefig(out_path, dpi=450, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    # Save paired t-tests with BH/FDR correction for each experiment.
    stats_small = _paired_tests_original_vs_controls(
        data_size_small,
        action_removal_results_filename_override=action_removal_results_filename_small,
    )
    stats_large = _paired_tests_original_vs_controls(
        data_size_large,
        action_removal_results_filename_override=action_removal_results_filename_large,
    )
    stats_df = pd.concat([stats_small, stats_large], ignore_index=True)
    stats_out = Path("results/statistical_tests")
    stats_out.mkdir(parents=True, exist_ok=True)
    stats_path = stats_out / f"control_mode_comparison_ttests{output_suffix}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved control t-tests with FDR to: {stats_path}")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action_removal_results_filename_small",
        type=str,
        default=None,
        help="Override small action_removal results filename under results/exp1_llm_prediction/",
    )
    parser.add_argument(
        "--action_removal_results_filename_large",
        type=str,
        default=None,
        help="Override large action_removal results filename under results/exp1_llm_prediction/",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_action_removal_LLaMA",
        help="Suffix used in output PDF filename.",
    )
    args = parser.parse_args()

    create_control_mode_comparison_figure(
        data_size_small="small",
        data_size_large="large",
        output_suffix=args.output_suffix,
        action_removal_results_filename_small=args.action_removal_results_filename_small,
        action_removal_results_filename_large=args.action_removal_results_filename_large,
    )
    print(
        f"control_mode_comparison{args.output_suffix}.pdf generated under figures/control_comparison/"
    )

if __name__ == "__main__":
    main() 