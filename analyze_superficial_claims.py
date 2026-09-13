"""
Compute superficial action-claim prevalence and word share from extraction CSVs.

Given an extraction CSV, we compute:
1) Prevalence: proportion of trials with `has_superficial_claim == True`.
2) Word share: among those trials, the average fraction of transcript words
   covered by `superficial_only_text` (word_count(split(superficial_only_text)) /
   word_count(full think-aloud)).
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pandas as pd


def _word_count_from_text_series(series: pd.Series) -> np.ndarray:
    """
    Compute approximate word count by splitting on whitespace.
    """

    def _count_one(x: object) -> float:
        s = "" if x is None else str(x)
        s = s.strip()
        if not s:
            return 0.0
        return float(len([t for t in s.split() if t]))

    return np.array([_count_one(x) for x in series.tolist()], dtype=float)


def analyze_extraction(
    extraction_path: str,
    exp_label: str,
) -> Dict[str, float]:
    print(f"\nLoading extraction CSV for {exp_label}: {extraction_path}")
    ext_df = pd.read_csv(extraction_path)

    required = {"has_superficial_claim", "word_count", "superficial_only_text"}
    missing = required - set(ext_df.columns)
    if missing:
        raise ValueError(f"{exp_label}: missing required columns: {sorted(missing)}")

    n_trials = int(len(ext_df))
    has_claim = ext_df["has_superficial_claim"].astype(bool).to_numpy()
    n_with_claim = int(has_claim.sum())
    prop_with_claim = n_with_claim / n_trials if n_trials > 0 else float("nan")

    # Word share among trials that actually have a claim + a nonzero transcript length.
    overall_word_count = ext_df["word_count"].astype(float).to_numpy()
    extracted_word_count = _word_count_from_text_series(ext_df["superficial_only_text"])

    valid = has_claim & np.isfinite(overall_word_count) & (overall_word_count > 0)
    if valid.any():
        word_share = float(np.mean(extracted_word_count[valid] / overall_word_count[valid]))
    else:
        word_share = float("nan")

    print(f"{exp_label}: trials={n_trials}")
    print(f"{exp_label}: with_superficial_claim={n_with_claim} ({prop_with_claim*100:.2f}%)")
    print(f"{exp_label}: avg_extracted_word_share={word_share*100:.2f}%")

    return {
        "prop_with_claim": float(prop_with_claim),
        "avg_word_share": float(word_share),
        "n_trials": float(n_trials),
        "n_with_claim": float(n_with_claim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze superficial action-claim extraction prevalence and word share."
    )
    parser.add_argument(
        "--extraction_csv_small",
        type=str,
        default="results/superficial_claims_openai_0_extraction_small_gpt-5.4-nano.csv",
        help="Experiment 1 extraction CSV (small).",
    )
    parser.add_argument(
        "--extraction_csv_large",
        type=str,
        default="results/superficial_claims_openai_0_extraction_large_gpt-5.4-nano.csv",
        help="Experiment 2 extraction CSV (large).",
    )
    args = parser.parse_args()

    res_small = analyze_extraction(args.extraction_csv_small, exp_label="Experiment 1")
    res_large = analyze_extraction(args.extraction_csv_large, exp_label="Experiment 2")

    print("\n=== Summary ===")
    print(
        f"Experiment 1: {res_small['prop_with_claim']*100:.2f}% trials with claims; "
        f"avg extracted word share={res_small['avg_word_share']*100:.2f}%"
    )
    print(
        f"Experiment 2: {res_large['prop_with_claim']*100:.2f}% trials with claims; "
        f"avg extracted word share={res_large['avg_word_share']*100:.2f}%"
    )


if __name__ == "__main__":
    main()

