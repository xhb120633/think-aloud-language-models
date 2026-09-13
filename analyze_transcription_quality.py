"""
Transcription Quality Analysis

This lightweight script loads paired think-aloud texts (Whisper transcribed vs
human edited) and produces two vertically stacked histograms:
  1) Word match proportion (fraction of human words found in Whisper text)
  2) Cosine similarity between texts (TF–IDF based)

Usage examples:
  python analyze_transcription_quality.py \
      --input data/behavioral_text_quality_data.csv \
      --human_col corrected_transcription \
      --whisper_col original_transcription \
      --output_dir figures/transcription_quality

If the input contains precomputed columns named 'word_match' and/or
'cos_similarity', those values will be used when possible; otherwise the script
computes them on the fly.
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Lightweight cleanup; keep it simple and dependency-free
    return " ".join(s.strip().lower().split())


def tokenize_words(s: str) -> list[str]:
    s = normalize_text(s)
    # Split on non-alphabetic characters; keep simple for robustness
    return [t for t in re_split_nonword(s) if t]


def re_split_nonword(s: str) -> list[str]:
    # Local small helper to avoid importing 're' everywhere
    import re as _re
    return _re.split(r"[^a-z0-9']+", s)


def compute_word_match(human_series: pd.Series, whisper_series: pd.Series) -> np.ndarray:
    """Return per-row fraction of human words present in Whisper text.

    Definition matches the legacy script: |human_words ∩ whisper_words| / |human_words|.
    When human has zero tokens:
      - If whisper also empty → 1.0 (perfect match on blank)
      - Else → 0.0
    """
    out = np.zeros(len(human_series), dtype=float)
    for i, (h, w) in enumerate(zip(human_series, whisper_series)):
        human_tokens = set(tokenize_words(h))
        whisper_tokens = set(tokenize_words(w))
        if len(human_tokens) == 0:
            out[i] = 1.0 if len(whisper_tokens) == 0 else 0.0
            continue
        inter = human_tokens.intersection(whisper_tokens)
        out[i] = len(inter) / float(len(human_tokens))
    return out


def compute_tfidf_cosine(human_series: pd.Series, whisper_series: pd.Series) -> np.ndarray:
    """Row-wise cosine similarity using a corpus-level TF–IDF.

    We fit one vectorizer on all texts (human + whisper) for stable features,
    then compute per-row cosine similarity.
    """
    texts = [normalize_text(t) for t in pd.concat([human_series, whisper_series], ignore_index=True)]
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    n = len(human_series)
    X_h = X[:n]
    X_w = X[n:]

    # Row-wise cosine between corresponding rows
    sims = np.zeros(n, dtype=float)
    # Efficient block computation using elementwise multiply summed rows
    # but for clarity use pairwise cosine on stacked pairs chunk-wise
    for i in range(n):
        a = X_h[i]
        b = X_w[i]
        denom = (a.multiply(a).sum() ** 0.5) * (b.multiply(b).sum() ** 0.5)
        if denom == 0:
            sims[i] = 1.0 if a.nnz == 0 and b.nnz == 0 else 0.0
            continue
        sims[i] = float(a.multiply(b).sum() / denom)
    return sims


def load_or_extract_columns(df: pd.DataFrame, human_col: str, whisper_col: str) -> Tuple[pd.Series, pd.Series]:
    if human_col not in df.columns or whisper_col not in df.columns:
        raise ValueError(f"Missing required columns: '{human_col}' and/or '{whisper_col}' in input file")
    return df[human_col].fillna(""), df[whisper_col].fillna("")


def read_dataframe_flexible(path: str) -> pd.DataFrame:
    """Read CSV with robust encoding fallbacks; also supports NPY upstream.

    Tries UTF-8, UTF-8-SIG, cp1252, latin-1 in order. Raises the last error if
    all fail.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001 - intentionally broad for fallback
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise ValueError(f"Failed to read file: {path}")


def make_histograms(word_match: np.ndarray, cos_sims: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Match analyze_cot_embeddings_similarity.py style: generous width, white bg, no titles
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # Word match histogram (top)
    ax1 = axes[0]
    ax1.hist(word_match, bins=np.linspace(0, 1, 30), color="#1f77b4", alpha=0.7, edgecolor="white", label="word match")
    ax1.set_ylabel("Count", fontsize=12)
    ax1.tick_params(axis='both', labelsize=11)
    ax1.set_facecolor('white')
    ax1.grid(False)
    ax1.legend(prop={'size': 11})

    # Cosine similarity histogram (bottom)
    ax2 = axes[1]
    ax2.hist(cos_sims, bins=np.linspace(0, 1, 30), color="#ff7f0e", alpha=0.7, edgecolor="white", label="cosine similarity")
    ax2.set_xlabel("Cosine similarity", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.tick_params(axis='both', labelsize=11)
    ax2.set_facecolor('white')
    ax2.grid(False)
    ax2.legend(prop={'size': 11})

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcription quality histograms (word match and cosine similarity)")
    parser.add_argument("--input", type=str, default="data/behavioral_text_quality_data.csv", help="Path to input CSV/NPY")
    parser.add_argument("--human_col", type=str, default="corrected_transcription", help="Column with human-edited text")
    parser.add_argument("--whisper_col", type=str, default="original_transcription", help="Column with Whisper-transcribed text")
    parser.add_argument("--output_dir", type=str, default="figures/transcription_quality", help="Directory to save the figure")
    args = parser.parse_args()

    # Load input (CSV/NPY)
    if args.input.lower().endswith(".npy"):
        arr = np.load(args.input, allow_pickle=True)
        df = pd.DataFrame(arr)
        # Try common headers if present; otherwise require user-specified indices
        # Users can convert NPY to CSV first if needed
    else:
        df = read_dataframe_flexible(args.input)

    # Extract text columns
    human_text, whisper_text = load_or_extract_columns(df, args.human_col, args.whisper_col)

    # Prefer precomputed metrics if available; else compute
    if "word_match" in df.columns and pd.api.types.is_numeric_dtype(df["word_match"]):
        word_match = df["word_match"].astype(float).to_numpy()
    else:
        word_match = compute_word_match(human_text, whisper_text)

    # Prefer precomputed cosine similarity if provided; fallback to TF–IDF
    used_precomputed = False
    if "cos_similarity" in df.columns:
        cos_col = pd.to_numeric(df["cos_similarity"], errors="coerce")
        if cos_col.notna().any():
            cos_sims = cos_col.astype(float).to_numpy()
            used_precomputed = True
        else:
            cos_sims = compute_tfidf_cosine(human_text, whisper_text)
    else:
        cos_sims = compute_tfidf_cosine(human_text, whisper_text)

    # Simple diagnostics to explain low values
    num_h_blank = int((human_text.fillna("") == "").sum())
    num_w_blank = int((whisper_text.fillna("") == "").sum())
    print(f"Loaded {len(df)} trials | human blanks: {num_h_blank} | whisper blanks: {num_w_blank} | cosine source: {'precomputed' if used_precomputed else 'tfidf'}")

    avg_word_match = float(np.nanmean(word_match)) if len(word_match) else float("nan")
    avg_cos_sim = float(np.nanmean(cos_sims)) if len(cos_sims) else float("nan")
    print(f"Mean word_match: {avg_word_match:.4f}")
    print(f"Mean cos_similarity: {avg_cos_sim:.4f}")

    # Plot
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output_dir, f"hist_wordmatch_cosine_{base}.pdf")
    make_histograms(word_match, cos_sims, out_path)


if __name__ == "__main__":
    main()


