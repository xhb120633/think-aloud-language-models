#!/usr/bin/env python3
"""
Analyze semantic similarity between model-generated chain-of-thought (CoT) and
human think-aloud embeddings.

This script:
- Loads embeddings and metadata saved by `embed_cot_data.py` from `results/embeddings/` by default
- Matches entries by (participant_id, trial_id)
- Computes cosine similarity between human and model embeddings for each matched pair
- Aggregates similarities per dataset variant (e.g., zeroshot, fewshot_N, permuted_N for any available N)
- Visualizes mean similarity with 95% CI error bars for each dataset and draws curves across all available N values
- Saves summary CSVs and plots per dataset size (small, large)

Usage examples:
    python analyze_cot_embeddings_similarity.py --data_sizes small
    python analyze_cot_embeddings_similarity.py --data_sizes both
    python analyze_cot_embeddings_similarity.py --embedding_dir results/embeddings --output_dir figures/embeddings_similarity

Notes:
- File naming conventions are those produced by `embed_cot_data.py`:
  - human_<size>_embeddings.pt + metadata json
  - model_zeroshot_<size>_embeddings.pt + metadata json
  - model_fewshot_<size>_<N>examples_embeddings.pt + metadata json
  - model_permuted_<size>_<N>examples_embeddings.pt + metadata json
- Example counts (N) are determined from filenames; all present counts are analyzed.
- If some datasets are missing (files not present), they are skipped with a warning.
"""

import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import to_rgba, to_hex
from scipy.stats import spearmanr


# -------------------------------
# Data structures
# -------------------------------

@dataclass
class EmbeddingSet:
    name: str  # e.g., "human", "model_zeroshot", "model_fewshot_1", "model_permuted_3"
    data_size: str  # "small" | "large"
    embeddings: np.ndarray  # shape: (num_items, dim)
    metadata: List[Dict]

# Exclude specific example counts globally from visualization and analysis
EXCLUDED_NUM_EXAMPLES = {15}


# -------------------------------
# Utilities
# -------------------------------

def load_pt_and_metadata(base_path: Path) -> Tuple[np.ndarray, List[Dict]]:
    """Load an embeddings tensor (.pt) and its paired metadata (.json).

    The saving script names files as:
      - `..._embeddings.pt` and `..._metadata.json`

    We receive `base_path` without extension (e.g., `..._embeddings`). This
    function tries both `<base>.json` and the canonical `<base> with
    _embeddings-> _metadata`.json`.
    """
    pt_path = base_path.with_suffix(".pt")

    # Try two possible metadata naming conventions
    # 1) Same base name with .json (legacy expectation)
    json_path = base_path.with_suffix(".json")
    # 2) Replace trailing `_embeddings` with `_metadata` (actual naming from saver)
    json_path_alt = base_path.parent / (base_path.name.replace("_embeddings", "_metadata") + ".json")

    if not pt_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {pt_path}")
    if not json_path.exists():
        if json_path_alt.exists():
            json_path = json_path_alt
        else:
            raise FileNotFoundError(
                f"Missing metadata file. Tried: {json_path} and {json_path_alt}"
            )

    tensor = torch.load(pt_path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        # Allow for lists/other structures saved with torch.save
        arr = np.array(tensor)

    with open(json_path, "r", encoding="utf-8") as f:
        meta_obj = json.load(f)

    # Metadata JSON produced by embed script wraps list under key "metadata"
    # { data_source, data_size, num_embeddings, embedding_dim, metadata: [...] }
    metadata = meta_obj.get("metadata", [])

    if len(metadata) != arr.shape[0]:
        print(
            f"⚠️  Warning: metadata length ({len(metadata)}) != embeddings rows ({arr.shape[0]}) for {pt_path.name}"
        )

    return arr, metadata


def parse_dataset_info_from_filename(stem: str) -> Tuple[str, str, Optional[int]]:
    """Parse dataset info from file stem (without extension).

    Examples:
      - human_small_embeddings -> ("human", "small", None)
      - model_zeroshot_large_embeddings -> ("model_zeroshot", "large", None)
      - model_fewshot_small_3examples_embeddings -> ("model_fewshot", "small", 3)
      - model_permuted_large_5examples_embeddings -> ("model_permuted", "large", 5)
    """
    # Normalize to ensure consistent parsing
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unrecognized filename pattern: {stem}")

    # Identify data source
    if parts[0] == "human":
        data_source = "human"
        data_size = parts[1]
        num_examples = None
    elif parts[0] == "model":
        # e.g., model_zeroshot_small_embeddings
        if len(parts) < 3:
            raise ValueError(f"Unrecognized model filename pattern: {stem}")
        data_source = f"model_{parts[1]}"  # zeroshot | fewshot | permuted
        data_size = parts[2]
        num_examples = None
        # Find any token like *_<N>examples_* anywhere in the stem
        m = re.search(r"_(\d+)examples(?:_|$)", stem)
        if m:
            num_examples = int(m.group(1))
    else:
        raise ValueError(f"Unknown data source in filename: {stem}")

    return data_source, data_size, num_examples


def build_dataset_label(data_source: str, num_examples: Optional[int]) -> str:
    """Create a concise label for plotting/summary.

    - human -> "human"
    - model_zeroshot -> "zeroshot"
    - model_fewshot + N -> f"fewshot_{N}"
    - model_permuted + N -> f"permuted_{N}"
    """
    if data_source == "human":
        return "human"
    if data_source == "model_zeroshot":
        return "zeroshot"
    if data_source == "model_fewshot":
        return f"fewshot_{num_examples}" if num_examples is not None else "fewshot"
    if data_source == "model_permuted":
        return f"permuted_{num_examples}" if num_examples is not None else "permuted"
    return data_source


def find_available_embedding_sets(embedding_dir: Path, target_sizes: List[str]) -> Dict[str, List[EmbeddingSet]]:
    """Scan directory for available embedding files and load them.

    Returns a dict mapping data_size -> list[EmbeddingSet].
    """
    results: Dict[str, List[EmbeddingSet]] = {size: [] for size in target_sizes}

    for pt_path in sorted(embedding_dir.glob("*_embeddings.pt")):
        stem = pt_path.stem  # without .pt
        try:
            data_source, data_size, num_examples = parse_dataset_info_from_filename(stem)
        except Exception as e:
            print(f"Skipping {pt_path.name}: {e}")
            continue

        if data_size not in target_sizes:
            continue

        base = pt_path.with_suffix("")  # remove extension
        try:
            arr, meta = load_pt_and_metadata(base)
        except Exception as e:
            print(f"Skipping {pt_path.name} due to load error: {e}")
            continue

        label = build_dataset_label(data_source, num_examples)
        results[data_size].append(EmbeddingSet(name=label, data_size=data_size, embeddings=arr, metadata=meta))

    # Ensure human is present for each requested size
    for size in target_sizes:
        has_human = any(ds.name == "human" for ds in results[size])
        if not has_human:
            print(f"⚠️  Warning: No human embeddings found for data_size={size}. Comparisons will be empty.")

    return results


def build_human_lookup(human_set: EmbeddingSet) -> Dict[Tuple[str, str], np.ndarray]:
    """Build a mapping from (participant_id, trial_id) -> embedding vector for humans."""
    lookup: Dict[Tuple[str, str], np.ndarray] = {}
    for emb, meta in zip(human_set.embeddings, human_set.metadata):
        pid = str(meta.get("participant_id", ""))
        tid = str(meta.get("trial_id", ""))
        if not pid or not tid:
            continue
        key = (pid, tid)
        # In case of duplicates, last one wins; should not typically happen
        lookup[key] = emb
    return lookup


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = vec_a.astype(np.float64)
    b = vec_b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def batch_rowwise_cosine(
    human_matrix: np.ndarray,
    model_matrix: np.ndarray,
    use_gpu: bool = True,
    batch_size: int = 8192,
) -> List[float]:
    """Compute row-wise cosine similarity between two same-shaped matrices.

    Uses torch with optional CUDA and processes in batches to limit memory use.
    """
    if human_matrix.shape != model_matrix.shape:
        raise ValueError(
            f"Shape mismatch for batch cosine: {human_matrix.shape} vs {model_matrix.shape}"
        )

    if human_matrix.size == 0:
        return []

    device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")

    sims: List[float] = []
    n = human_matrix.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        h = torch.from_numpy(human_matrix[start:end]).to(device=device, dtype=torch.float32)
        m = torch.from_numpy(model_matrix[start:end]).to(device=device, dtype=torch.float32)
        # Normalize rows
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        m = torch.nn.functional.normalize(m, p=2, dim=1)
        # Row-wise dot product
        batch_sims = (h * m).sum(dim=1)
        sims.extend(batch_sims.detach().cpu().tolist())
    return sims


def aggregate_similarities(
    human_set: EmbeddingSet,
    model_set: EmbeddingSet,
    use_gpu: bool = True,
    sim_batch_size: int = 8192,
) -> List[float]:
    """Compute cosine similarities for matched (participant_id, trial_id) pairs.

    Uses batched torch operations (optionally on GPU) for speed.
    """
    human_lookup = build_human_lookup(human_set)

    human_rows: List[np.ndarray] = []
    model_rows: List[np.ndarray] = []
    missing = 0

    for emb, meta in zip(model_set.embeddings, model_set.metadata):
        pid = str(meta.get("participant_id", ""))
        tid = str(meta.get("trial_id", ""))
        if not pid or not tid:
            continue
        key = (pid, tid)
        human_emb = human_lookup.get(key)
        if human_emb is None:
            missing += 1
            continue
        human_rows.append(human_emb)
        model_rows.append(emb)

    if missing > 0:
        print(f"  Note: {missing} entries in '{model_set.name}' had no human match and were skipped.")

    if not human_rows:
        return []

    human_matrix = np.vstack(human_rows)
    model_matrix = np.vstack(model_rows)
    return batch_rowwise_cosine(human_matrix, model_matrix, use_gpu=use_gpu, batch_size=sim_batch_size)


def aggregate_similarity_map(
    human_set: EmbeddingSet,
    model_set: EmbeddingSet,
    use_gpu: bool = True,
    sim_batch_size: int = 8192,
) -> Dict[Tuple[str, str], float]:
    """Compute a mapping from (participant_id, trial_id) -> cosine similarity.

    If multiple samples exist for the same (participant_id, trial_id), the
    similarities are averaged so results are directly comparable to runs
    with a single sample per example number.
    """
    human_lookup = build_human_lookup(human_set)

    keys: List[Tuple[str, str]] = []
    human_rows: List[np.ndarray] = []
    model_rows: List[np.ndarray] = []

    for emb, meta in zip(model_set.embeddings, model_set.metadata):
        pid = str(meta.get("participant_id", ""))
        tid = str(meta.get("trial_id", ""))
        if not pid or not tid:
            continue
        key = (pid, tid)
        human_emb = human_lookup.get(key)
        if human_emb is None:
            continue
        keys.append(key)
        human_rows.append(human_emb)
        model_rows.append(emb)

    if not keys:
        return {}

    human_matrix = np.vstack(human_rows)
    model_matrix = np.vstack(model_rows)
    sims = batch_rowwise_cosine(human_matrix, model_matrix, use_gpu=use_gpu, batch_size=sim_batch_size)

    # Average over duplicate keys (multiple samples for same participant × trial)
    from collections import defaultdict
    key_to_sims: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for k, s in zip(keys, sims):
        key_to_sims[k].append(float(s))

    return {k: float(np.mean(v)) if len(v) > 0 else float('nan') for k, v in key_to_sims.items()}


# -------------------------------
# Participant-level aggregation helpers
# -------------------------------

def pairmap_to_participant_means(pairmap: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    """Aggregate (participant_id, trial_id) -> sim into participant_id -> mean(sim).

    Uses a straight average over trials per participant.
    """
    from collections import defaultdict
    per_participant: Dict[str, List[float]] = defaultdict(list)
    for (pid, _tid), sim in pairmap.items():
        per_participant[str(pid)].append(float(sim))
    return {pid: float(np.mean(vals)) for pid, vals in per_participant.items() if len(vals) > 0}

def mean_ci_95_from_participants(part_means: Dict[str, float]) -> Tuple[float, float, int]:
    """Compute mean and 95% CI half-width across participants.

    Returns (mean, ci_half_width, n_participants).
    """
    if not part_means:
        return float('nan'), float('nan'), 0
    arr = np.array(list(part_means.values()), dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci = 1.96 * sem if n > 1 else 0.0
    return mean, ci, int(n)


def _normal_cdf(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def paired_t_test(d1: List[float], d2: List[float]) -> Tuple[float, int, float, str]:
    """Paired two-sided t-test. Returns (t_value, df, p_value, note).

    Tries scipy.stats.ttest_rel if available; otherwise uses normal approximation.
    """
    import math as _math
    n = len(d1)
    if n == 0:
        return float("nan"), 0, float("nan"), "no_pairs"

    diffs = np.array(d2, dtype=np.float64) - np.array(d1, dtype=np.float64)
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd_diff > 0:
        se = sd_diff / _math.sqrt(n)
        t_val = mean_diff / se
        df = n - 1
        try:
            from scipy import stats  # type: ignore
            p_val = float(stats.t.sf(abs(t_val), df) * 2.0)
            note = "paired_t_scipy"
        except Exception:
            # Normal approximation fallback
            p_val = float(2.0 * (1.0 - _normal_cdf(abs(t_val))))
            note = "paired_t_normal_approx"
        return t_val, df, p_val, note
    else:
        return float("nan"), n - 1, float("nan"), "insufficient_variance"


def cohens_dz(d1: List[float], d2: List[float]) -> float:
    diffs = np.array(d2, dtype=np.float64) - np.array(d1, dtype=np.float64)
    n = diffs.size
    if n <= 1:
        return float("nan")
    sd = float(diffs.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(diffs.mean() / sd)


def fdr_bh(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction. NaN p-values stay NaN.

    NOTE: We process in reverse rank order (largest p first) to enforce the
    standard BH monotonicity constraint. This ensures adjusted p-values are
    non-decreasing with rank and never smaller than the smallest scaled value.
    """
    m = len(pvals)
    indexed = [(i, p) for i, p in enumerate(pvals)]
    # Exclude NaNs from ordering
    valid = [(i, p) for i, p in indexed if not (isinstance(p, float) and np.isnan(p))]
    invalid = [i for i, p in indexed if isinstance(p, float) and np.isnan(p)]
    # Sort valid by raw p-value ascending
    valid.sort(key=lambda x: x[1])
    adj = [np.nan] * m
    # Compute adjusted values in reverse order (largest p to smallest)
    prev = float("inf")
    for rank, (i, p) in enumerate(reversed(valid), start=1):
        # For reversed list, rank runs from largest to smallest;
        # BH uses k = m, m-1, ..., 1
        k = m - rank + 1
        val = p * m / k
        if val < prev:
            prev = val
        adj[i] = min(prev, 1.0)
    # NaNs remain NaN
    for i in invalid:
        adj[i] = np.nan
    return adj


def mean_ci_95(values: List[float]) -> Tuple[float, float]:
    """Compute mean and 95% CI half-width using normal approximation."""
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else float("nan")
    half_width = 1.96 * se if n > 1 else 0.0
    return mean, half_width


def order_labels(labels: List[str]) -> List[str]:
    """Return labels in a logical plotting order."""
    def sort_key(lbl: str) -> Tuple[int, int]:
        if lbl == "human":
            return (0, 0)
        if lbl == "zeroshot":
            return (1, 0)
        if lbl.startswith("fewshot_"):
            try:
                n = int(lbl.split("_")[1])
            except Exception:
                n = 0
            return (2, n)
        if lbl.startswith("permuted_"):
            try:
                n = int(lbl.split("_")[1])
            except Exception:
                n = 0
            return (3, n)
        return (9, 0)

    return sorted(labels, key=sort_key)


def plot_bar_with_ci(size: str, label_to_stats: Dict[str, Tuple[float, float]], output_dir: Path) -> None:
    labels_ordered = order_labels(list(label_to_stats.keys()))
    means = [label_to_stats[l][0] for l in labels_ordered]
    ci_hw = [label_to_stats[l][1] for l in labels_ordered]

    x = np.arange(len(labels_ordered))
    width = 0.6

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, means, width=width, color="#4C78A8", alpha=0.85, yerr=ci_hw, capsize=6)
    plt.xticks(x, labels_ordered, rotation=0)
    plt.ylabel("Cosine similarity", fontsize=12)
    # Dynamic y-limits for better contrast
    if means:
        low = np.nanmin(np.array(means) - np.array(ci_hw))
        high = np.nanmax(np.array(means) + np.array(ci_hw))
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = 0.0, 1.0
        if high - low < 1e-6:
            low -= 0.02
            high += 0.02
        pad = max(0.01, 0.08 * (high - low))
        plt.ylim(max(0.0, low - pad), min(1.0, high + pad))
    else:
        plt.ylim(0, 1)
    # Three-line title with formal experiment label
    exp_label = "Experiment 1 (PT design)" if size == "small" else "Experiment 2 (choice13k-sampled)"
    # Title intentionally omitted for flexibility
    # Match in_context style: no grid, white background
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=11)
    plt.gcf().patch.set_facecolor('white')

    for rect, m in zip(bars, means):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=11)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"similarity_{size}.pdf"
    plt.tight_layout()
    plt.savefig(out_path, dpi=450)
    plt.close()
    print(f"Saved plot: {out_path}")


def save_summary_csv(size: str, label_to_stats: Dict[str, Tuple[float, float]], counts: Dict[str, int], output_dir: Path) -> None:
    rows = []
    for label in order_labels(list(label_to_stats.keys())):
        mean, ci_hw = label_to_stats[label]
        n = counts.get(label, 0)
        rows.append({
            "data_size": size,
            "dataset": label,
            "n_participants": n,
            "mean_cosine_similarity": mean,
            "ci95_half_width": ci_hw,
            "ci95_lower": (mean - ci_hw) if not math.isnan(mean) and not math.isnan(ci_hw) else float("nan"),
            "ci95_upper": (mean + ci_hw) if not math.isnan(mean) and not math.isnan(ci_hw) else float("nan"),
        })

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"similarity_summary_{size}.csv"

    try:
        import pandas as pd  # optional dependency
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
    except Exception:
        # Fallback: write CSV manually
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved summary CSV: {out_csv}")


# -------------------------------
# New visualizations (point+error, histograms)
# -------------------------------

def compute_mean_ci(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci = 1.96 * se if n > 1 else 0.0
    return mean, ci


def plot_similarity_curves(size: str, label_to_pairmap: Dict[str, Dict[Tuple[str, str], float]], output_dir: Path) -> None:
    """Plot fewshot vs permuted with 95% CI; zeroshot as baseline line.

    Updated to aggregate at participant-level (mean over trials), CI over participants.
    """
    # Collect available N values
    fewshot_ns = []
    perm_ns = []
    for lbl in label_to_pairmap.keys():
        if lbl.startswith("fewshot_"):
            m = re.match(r"fewshot_(\d+)$", lbl)
            if m:
                fewshot_ns.append(int(m.group(1)))
        if lbl.startswith("permuted_"):
            m = re.match(r"permuted_(\d+)$", lbl)
            if m:
                perm_ns.append(int(m.group(1)))
    fewshot_ns = sorted({n for n in set(fewshot_ns) if n not in EXCLUDED_NUM_EXAMPLES})
    perm_ns = sorted({n for n in set(perm_ns) if n not in EXCLUDED_NUM_EXAMPLES})
    xs_all = sorted({n for n in (set(fewshot_ns) | set(perm_ns)) if n not in EXCLUDED_NUM_EXAMPLES})

    # Build participant-level maps for each label
    label_to_partmeans: Dict[str, Dict[str, float]] = {
        lbl: pairmap_to_participant_means(pm) for lbl, pm in label_to_pairmap.items()
    }

    # Stats per N (participant-level)
    fs_x, fs_means, fs_cis = [], [], []
    for n in fewshot_ns:
        pm = label_to_partmeans.get(f"fewshot_{n}", {})
        m, c, _ = mean_ci_95_from_participants(pm)
        fs_x.append(n)
        fs_means.append(m)
        fs_cis.append(c)

    pm_x, pm_means, pm_cis = [], [], []
    for n in perm_ns:
        pmv = label_to_partmeans.get(f"permuted_{n}", {})
        m, c, _ = mean_ci_95_from_participants(pmv)
        pm_x.append(n)
        pm_means.append(m)
        pm_cis.append(c)

    zero_pm = label_to_partmeans.get("zeroshot", {})
    zero_mean, zero_ci, _ = mean_ci_95_from_participants(zero_pm)

    plt.figure(figsize=(10, 6))

    # Fewshot: points with error bars (no connecting lines)
    if fs_x:
        plt.errorbar(fs_x, fs_means, yerr=fs_cis, fmt='o', linestyle='None', color="#1f77b4", capsize=5, label="fewshot")

    # Permuted: dashed line with shaded CI
    if pm_x:
        plt.plot(pm_x, pm_means, linestyle='--', color="#ff7f0e", label="permuted")
        upper = np.array(pm_means) + np.array(pm_cis)
        lower = np.array(pm_means) - np.array(pm_cis)
        plt.fill_between(pm_x, lower, upper, color="#ff7f0e", alpha=0.15)

    # Zeroshot baseline as horizontal dashed line with shaded CI (match in_context style)
    if not math.isnan(zero_mean):
        plt.axhline(y=zero_mean, color='#8B4513', linestyle='--', linewidth=1.5, label='zero-shot')
        if not math.isnan(zero_ci) and zero_ci > 0:
            x_min = xs_all[0] - 0.2 if xs_all else 0.8
            x_max = xs_all[-1] + 0.2 if xs_all else 5.2
            plt.fill_between([x_min, x_max], [zero_mean-zero_ci, zero_mean-zero_ci], [zero_mean+zero_ci, zero_mean+zero_ci], color='#8B4513', alpha=0.1)

    # Dynamic y-limits
    series_lows = []
    series_highs = []
    if fs_means:
        series_lows.append(np.nanmin(np.array(fs_means) - np.array(fs_cis)))
        series_highs.append(np.nanmax(np.array(fs_means) + np.array(fs_cis)))
    if pm_means:
        series_lows.append(np.nanmin(np.array(pm_means) - np.array(pm_cis)))
        series_highs.append(np.nanmax(np.array(pm_means) + np.array(pm_cis)))
    if not math.isnan(zero_mean):
        zlow = zero_mean - (zero_ci if not math.isnan(zero_ci) else 0.0)
        zhigh = zero_mean + (zero_ci if not math.isnan(zero_ci) else 0.0)
        series_lows.append(zlow)
        series_highs.append(zhigh)

    if series_lows and series_highs:
        y_min = float(np.min(series_lows))
        y_max = float(np.max(series_highs))
        if y_max - y_min < 1e-6:
            y_min -= 0.02
            y_max += 0.02
        pad = max(0.01, 0.08 * (y_max - y_min))
        y_min = max(0.0, y_min - pad)
        y_max = min(1.0, y_max + pad)
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(0, 1)

    if xs_all:
        plt.xticks(xs_all, xs_all)
    plt.xlabel("Number of examples", fontsize=12)
    plt.ylabel("Cosine similarity", fontsize=12)
    plt.gca().tick_params(axis='both', labelsize=11)
    exp_label = "Experiment 1 (PT design)" if size == "small" else "Experiment 2 (choice13k-sampled)"
    # Title intentionally omitted for flexibility
    plt.grid(False)
    plt.legend(prop={'size': 11})

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"curves_similarity_{size}.pdf"
    plt.tight_layout()
    plt.savefig(out_path, dpi=450)
    plt.close()
    print(f"Saved curve plot: {out_path}")


def plot_similarity_curves_combined(
    label_to_pairmap_small: Dict[str, Dict[Tuple[str, str], float]],
    label_to_pairmap_large: Dict[str, Dict[Tuple[str, str], float]],
    output_dir: Path,
) -> None:
    """Single-axes plot showing small and large datasets together.

    Style:
    - Fewshot: points with error bars (no connecting lines)
    - Permuted: dashed line with shaded 95% CI
    - Zeroshot baseline: horizontal dotted line with shaded 95% CI band
    - No grid, white background, tight dynamic y-limits
    """
    # Build participant-level means per label for both sizes
    def stats_for_size(label_to_pairmap: Dict[str, Dict[Tuple[str, str], float]]):
        lbl_to_pm = {lbl: pairmap_to_participant_means(pm) for lbl, pm in label_to_pairmap.items()}
        # Ns
        fs_ns = sorted({int(m.group(1)) for lbl in lbl_to_pm.keys() if (m := re.match(r"fewshot_(\d+)$", lbl))})
        pm_ns = sorted({int(m.group(1)) for lbl in lbl_to_pm.keys() if (m := re.match(r"permuted_(\d+)$", lbl))})
        # Fewshot arrays
        fs_x, fs_means, fs_cis = [], [], []
        for n in fs_ns:
            m, c, _ = mean_ci_95_from_participants(lbl_to_pm.get(f"fewshot_{n}", {}))
            fs_x.append(n); fs_means.append(m); fs_cis.append(c)
        # Permuted arrays
        pm_x, pm_means, pm_cis = [], [], []
        for n in pm_ns:
            m, c, _ = mean_ci_95_from_participants(lbl_to_pm.get(f"permuted_{n}", {}))
            pm_x.append(n); pm_means.append(m); pm_cis.append(c)
        # Zeroshot
        z_mean, z_ci, _ = mean_ci_95_from_participants(lbl_to_pm.get("zeroshot", {}))
        return fs_ns, (fs_x, fs_means, fs_cis), pm_ns, (pm_x, pm_means, pm_cis), (z_mean, z_ci)

    fs_ns_s, fs_s, pm_ns_s, pm_s, z_s = stats_for_size(label_to_pairmap_small)
    fs_ns_l, fs_l, pm_ns_l, pm_l, z_l = stats_for_size(label_to_pairmap_large)

    xs_all = sorted({n for n in (set(fs_ns_s) | set(pm_ns_s) | set(fs_ns_l) | set(pm_ns_l)) if n not in EXCLUDED_NUM_EXAMPLES})

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Colors per dataset
    color_small = "#1f77b4"  # blue
    color_large = "#2ca02c"  # green

    # Fewshot (points)
    if fs_s[0]:
        ax.errorbar(fs_s[0], fs_s[1], yerr=fs_s[2], fmt='o', linestyle='None', color=color_small, capsize=5, label="fewshot (small)")
    if fs_l[0]:
        ax.errorbar(fs_l[0], fs_l[1], yerr=fs_l[2], fmt='s', linestyle='None', color=color_large, capsize=5, label="fewshot (large)")

    # Permuted (dashed + shaded)
    if pm_s[0]:
        ax.plot(pm_s[0], pm_s[1], linestyle='--', color=color_small, label="permuted (small)")
        ax.fill_between(np.array(pm_s[0], dtype=float), np.array(pm_s[1]) - np.array(pm_s[2]), np.array(pm_s[1]) + np.array(pm_s[2]), color=color_small, alpha=0.15)
    if pm_l[0]:
        ax.plot(pm_l[0], pm_l[1], linestyle='--', color=color_large, label="permuted (large)")
        ax.fill_between(np.array(pm_l[0], dtype=float), np.array(pm_l[1]) - np.array(pm_l[2]), np.array(pm_l[1]) + np.array(pm_l[2]), color=color_large, alpha=0.15)

    # Zeroshot baselines with shaded CI (match in_context style)
    def add_z_line(z_mean: float, z_ci: float, label: str):
        if not np.isnan(z_mean):
            x_min = xs_all[0] - 0.2 if xs_all else 0.8
            x_max = xs_all[-1] + 0.2 if xs_all else 5.2
            ax.axhline(y=z_mean, color='#8B4513', linestyle='--', linewidth=1.5, label=label)
            if not np.isnan(z_ci) and z_ci > 0:
                ax.fill_between([x_min, x_max], [z_mean - z_ci, z_mean - z_ci], [z_mean + z_ci, z_mean + z_ci], color='#8B4513', alpha=0.10)

    add_z_line(z_s[0], z_s[1], 'zero-shot (small)')
    add_z_line(z_l[0], z_l[1], 'zero-shot (large)')

    # Dynamic y-limits
    series_lows, series_highs = [], []
    for means, cis in [(fs_s[1], fs_s[2]), (fs_l[1], fs_l[2]), (pm_s[1], pm_s[2]), (pm_l[1], pm_l[2])]:
        if means:
            means_a = np.array(means, dtype=float); cis_a = np.array(cis, dtype=float)
            series_lows.append(np.nanmin(means_a - cis_a))
            series_highs.append(np.nanmax(means_a + cis_a))
    for z_mean, z_ci in [z_s, z_l]:
        if not np.isnan(z_mean):
            series_lows.append(z_mean - (z_ci if not np.isnan(z_ci) else 0.0))
            series_highs.append(z_mean + (z_ci if not np.isnan(z_ci) else 0.0))
    if series_lows and series_highs:
        y_min = float(np.min(series_lows)); y_max = float(np.max(series_highs))
        if y_max - y_min < 1e-6:
            y_min -= 0.02; y_max += 0.02
        pad = max(0.01, 0.08 * (y_max - y_min))
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    else:
        ax.set_ylim(0, 1)

    if xs_all:
        ax.set_xticks(xs_all)
        ax.set_xlabel("Number of examples", fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    # Title intentionally omitted for flexibility
    ax.grid(False)
    ax.set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    ax.legend(loc='lower right', prop={'size': 11})

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "curves_similarity_combined.pdf"
    plt.tight_layout()
    plt.savefig(out_path, dpi=450)
    plt.close()
    print(f"Saved combined curve plot: {out_path}")


def plot_similarity_curves_two_panel(
    label_to_pairmap_small: Dict[str, Dict[Tuple[str, str], float]],
    label_to_pairmap_large: Dict[str, Dict[Tuple[str, str], float]],
    output_dir: Path,
) -> None:
    """Two side-by-side panels (small | large) for CoT similarity curves.

    Style matches in-context plots:
    - Fewshot: points with error bars (participant-level 95% CI)
    - Permuted: dashed line with shaded 95% CI
    - Zeroshot: dotted line with shaded 95% CI
    - No grid, white background, unified legend at bottom center
    - Harmonized y-limits across panels
    """
    # Helper to compute per-size stats
    def stats_for_size(label_to_pairmap: Dict[str, Dict[Tuple[str, str], float]]):
        lbl_to_pm = {lbl: pairmap_to_participant_means(pm) for lbl, pm in label_to_pairmap.items()}
        fs_ns = sorted({int(m.group(1)) for lbl in lbl_to_pm.keys() if (m := re.match(r"fewshot_(\d+)$", lbl))})
        pm_ns = sorted({int(m.group(1)) for lbl in lbl_to_pm.keys() if (m := re.match(r"permuted_(\d+)$", lbl))})
        # Fewshot arrays
        fs_x, fs_means, fs_cis = [], [], []
        for n in fs_ns:
            m, c, _ = mean_ci_95_from_participants(lbl_to_pm.get(f"fewshot_{n}", {}))
            fs_x.append(n); fs_means.append(m); fs_cis.append(c)
        # Permuted arrays
        pm_x, pm_means, pm_cis = [], [], []
        for n in pm_ns:
            m, c, _ = mean_ci_95_from_participants(lbl_to_pm.get(f"permuted_{n}", {}))
            pm_x.append(n); pm_means.append(m); pm_cis.append(c)
        # Zeroshot
        z_mean, z_ci, _ = mean_ci_95_from_participants(lbl_to_pm.get("zeroshot", {}))
        return (fs_x, fs_means, fs_cis), (pm_x, pm_means, pm_cis), (z_mean, z_ci)

    small_stats = stats_for_size(label_to_pairmap_small)
    large_stats = stats_for_size(label_to_pairmap_large)

    # Build a shared x-axis across panels for equal width/scales
    def collect_xs(stats):
        (fs_x, _fm, _fc), (pm_x, _pm, _pc), _z = stats
        return set(fs_x) | set(pm_x)
    xs_union = sorted({n for n in (collect_xs(small_stats) | collect_xs(large_stats)) if n not in EXCLUDED_NUM_EXAMPLES})

    # Shared figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)

    def draw_panel(ax, stats, dataset_label: str, xs_all: list[int]):
        (fs_x, fs_means, fs_cis), (pm_x, pm_means, pm_cis), (z_mean, z_ci) = stats
        # Fewshot points
        if fs_x:
            color_fs = "#1f77b4"
            ax.errorbar(
                fs_x, fs_means, yerr=fs_cis,
                fmt='o', linestyle='None',
                color=color_fs, markerfacecolor=color_fs, markeredgecolor=color_fs,
                ecolor=to_hex(to_rgba(color_fs, 1.0)),
                capsize=4, markersize=6.5, elinewidth=1.6, linewidth=1.6,
                label="fewshot"
            )
        # Permuted dashed + shaded
        if pm_x:
            color_pm = "#ff7f0e"
            ax.plot(pm_x, pm_means, linestyle='--', color=color_pm, linewidth=1.6, label="permuted")
            ax.fill_between(np.array(pm_x, dtype=float), np.array(pm_means) - np.array(pm_cis), np.array(pm_means) + np.array(pm_cis), color=color_pm, alpha=0.12, linewidth=0)
            # Right-margin annotation similar to zero-shot label
            try:
                ax.text(1.01, float(pm_means[-1]), 'permuted', color=color_pm, alpha=0.7, fontsize=8, ha='left', va='center', transform=ax.get_yaxis_transform())
            except Exception:
                pass
        # Zeroshot baseline (match in_context style)
        if not np.isnan(z_mean):
            x_min = (xs_all[0] - 0.3) if xs_all else 0.7
            x_max = (xs_all[-1] + 0.3) if xs_all else 5.3
            ax.axhline(y=z_mean, color='#8B4513', linestyle='--', linewidth=1.5, label='zero-shot')
            if not np.isnan(z_ci) and z_ci > 0:
                ax.fill_between([x_min, x_max], [z_mean - z_ci, z_mean - z_ci], [z_mean + z_ci, z_mean + z_ci], color='#8B4513', alpha=0.10, linewidth=0)
            # Right-margin annotation like analyze_in_context
            try:
                ax.text(1.01, z_mean, 'zero-shot', color='#8B4513', alpha=0.7, fontsize=8, ha='left', va='center', transform=ax.get_yaxis_transform())
            except Exception:
                pass
        # Axis labels and style
        if xs_all:
            ax.set_xticks(xs_all)
            ax.set_xlim(xs_all[0]-0.3, xs_all[-1]+0.3)
        ax.set_xlabel("# of In-Context Examples", fontsize=11)
        ax.tick_params(axis='both', labelsize=11, pad=3)
        # Three-line panel title per request
        # Formal dataset naming aligned with in_context plots
        exp_label = "Experiment 1 (PT design)" if dataset_label == "small" else "Experiment 2 (choice13k-sampled)"
        title_lines = [
            "Within-individual in-context learning",
            "CoT vs. human Think-Aloud similarity",
            exp_label,
        ]
        # Panel title intentionally omitted for flexibility
        ax.set_facecolor('white')
        ax.grid(False)

    draw_panel(axes[0], small_stats, "small", xs_union)
    draw_panel(axes[1], large_stats, "large", xs_union)

    # Harmonize y-limits across panels using data
    def collect_bounds(stats):
        (fs_x, fs_means, fs_cis), (pm_x, pm_means, pm_cis), (z_mean, z_ci) = stats
        lows, highs = [], []
        if fs_means:
            a = np.array(fs_means); c = np.array(fs_cis)
            lows.append(np.nanmin(a - c)); highs.append(np.nanmax(a + c))
        if pm_means:
            a = np.array(pm_means); c = np.array(pm_cis)
            lows.append(np.nanmin(a - c)); highs.append(np.nanmax(a + c))
        if not np.isnan(z_mean):
            lows.append(z_mean - (z_ci if not np.isnan(z_ci) else 0.0))
            highs.append(z_mean + (z_ci if not np.isnan(z_ci) else 0.0))
        if lows and highs:
            y_min = float(np.min(lows)); y_max = float(np.max(highs))
            if y_max - y_min < 1e-6:
                y_min -= 0.02; y_max += 0.02
            pad = max(0.01, 0.08 * (y_max - y_min))
            return max(0.0, y_min - pad), min(1.0, y_max + pad)
        return 0.0, 1.0

    y0 = collect_bounds(small_stats)
    y1 = collect_bounds(large_stats)
    y_min = min(y0[0], y1[0]); y_max = max(y0[1], y1[1])
    for ax in axes:
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Cosine similarity")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Unified legend at bottom center; ensure equal panel area with consistent margins
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18, wspace=0.06)
    # Unified legend at bottom center
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    by_label = {l: h for h, l in zip(handles, labels)}
    leg = fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=min(4, len(by_label)), frameon=True, prop={'size':10})
    if leg:
        leg.get_frame().set_alpha(0.85)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "curves_similarity_two_panel.pdf"
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, dpi=450)
    plt.close()
    print(f"Saved two-panel curve plot: {out_path}")


def plot_similarity_histograms(size: str, label_to_pairmap: Dict[str, Dict[Tuple[str, str], float]], output_dir: Path, max_ns_per_plot: int = 5) -> None:
    """Plot overlaid histograms of distributions.

    - Fewshot (top-N counts) with zeroshot in one figure
    - Permuted (top-N counts) with zeroshot in another figure
    """
    bins = np.linspace(0, 1, 30)

    # Collect available Ns
    fs_ns = sorted({int(m.group(1)) for lbl in label_to_pairmap.keys() if (m := re.match(r"fewshot_(\d+)$", lbl)) and int(m.group(1)) not in EXCLUDED_NUM_EXAMPLES})
    pm_ns = sorted({int(m.group(1)) for lbl in label_to_pairmap.keys() if (m := re.match(r"permuted_(\d+)$", lbl)) and int(m.group(1)) not in EXCLUDED_NUM_EXAMPLES})

    # Limit to at most max_ns_per_plot for readability
    fs_ns_plot = fs_ns[:max_ns_per_plot]
    pm_ns_plot = pm_ns[:max_ns_per_plot]

    # Fewshot + zeroshot
    plt.figure(figsize=(10, 6))
    palette = plt.get_cmap('tab10')
    for idx, n in enumerate(fs_ns_plot):
        vals = list(label_to_pairmap.get(f"fewshot_{n}", {}).values())
        if vals:
            plt.hist(vals, bins=bins, density=True, alpha=0.35, label=f"fewshot_{n}", color=palette(idx % 10))
    zvals = list(label_to_pairmap.get("zeroshot", {}).values())
    if zvals:
        plt.hist(zvals, bins=bins, density=True, alpha=0.35, label="zeroshot", color="#8B4513")
    plt.xlabel("Cosine similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.gca().tick_params(axis='both', labelsize=11)
    title_suffix = " (truncated)" if len(fs_ns) > len(fs_ns_plot) else ""
    # Title intentionally omitted for flexibility
    plt.legend(prop={'size': 11})
    plt.tight_layout()
    out1 = output_dir / f"hist_fewshot_vs_zeroshot_{size}.pdf"
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Saved histogram: {out1}")

    # Permuted + zeroshot
    plt.figure(figsize=(10, 6))
    for idx, n in enumerate(pm_ns_plot):
        vals = list(label_to_pairmap.get(f"permuted_{n}", {}).values())
        if vals:
            plt.hist(vals, bins=bins, density=True, alpha=0.35, label=f"permuted_{n}", color=palette(idx % 10))
    if zvals:
        plt.hist(zvals, bins=bins, density=True, alpha=0.35, label="zeroshot", color="#8B4513")
    plt.xlabel("Cosine similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.gca().tick_params(axis='both', labelsize=11)
    title_suffix = " (truncated)" if len(pm_ns) > len(pm_ns_plot) else ""
    # Title intentionally omitted for flexibility
    plt.legend(prop={'size': 11})
    plt.tight_layout()
    out2 = output_dir / f"hist_permuted_vs_zeroshot_{size}.pdf"
    plt.savefig(out2, dpi=450)
    plt.close()
    print(f"Saved histogram: {out2}")


# -------------------------------
# Spearman correlations
# -------------------------------

def compute_spearman_correlations_for_size(
    size: str,
    label_to_pairmap: Dict[str, Dict[Tuple[str, str], float]],
    output_stats_dir: Path,
) -> None:
    """Compute Spearman correlation between number of examples and participant-level
    mean cosine similarity, separately for fewshot and permuted.

    Saves CSV to results/statistical_tests/spearman_cot_similarity_{size}.csv
    """
    # Build participant-level maps
    label_to_partmeans: Dict[str, Dict[str, float]] = {
        lbl: pairmap_to_participant_means(pm) for lbl, pm in label_to_pairmap.items()
    }

    rows: List[Dict] = []
    allowed_ns = {1, 2, 3, 5, 8, 10}
    for series in ["fewshot", "permuted"]:
        ns: List[int] = []
        means: List[float] = []
        for lbl, pm in label_to_partmeans.items():
            m = re.match(rf"{series}_(\d+)$", lbl)
            if not m:
                continue
            n_ex = int(m.group(1))
            if n_ex in EXCLUDED_NUM_EXAMPLES:
                continue
            if n_ex not in allowed_ns:
                continue
            mean, _ci, _n = mean_ci_95_from_participants(pm)
            if not np.isnan(mean):
                ns.append(n_ex)
                means.append(mean)
        if len(ns) > 1 and len(ns) == len(means):
            # Sort by N for reporting stability
            order = np.argsort(ns)
            ns_sorted = [ns[i] for i in order]
            means_sorted = [means[i] for i in order]
            rho, p = spearmanr(ns_sorted, means_sorted)
            rows.append({
                "data_size": size,
                "series": series,
                "n_points": len(ns_sorted),
                "spearman_rho": float(rho) if rho is not None else float("nan"),
                "p_value": float(p) if p is not None else float("nan"),
            })

    # Note: no cross-series correlation here (only per-series vs N)

    # Save
    output_stats_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_stats_dir / f"spearman_cot_similarity_{size}.csv"
    try:
        import pandas as pd  # optional
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    except Exception:
        import csv
        if rows:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            # Create an empty file with headers
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["data_size","series","n_points","spearman_rho","p_value"])
                writer.writeheader()
    print(f"Saved Spearman correlations: {out_csv}")

# -------------------------------
# Main
# -------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze similarity between human and model CoT embeddings.")
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="results/embeddings",
        help="Directory containing *_embeddings.pt and *_metadata.json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/embeddings_similarity",
        help="Output directory for plots and summary CSVs",
    )
    parser.add_argument(
        "--data_sizes",
        type=str,
        choices=["small", "large", "both"],
        default="both",
        help="Which data sizes to analyze",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU (CUDA) for batched cosine similarity if available",
    )
    parser.add_argument(
        "--sim_batch_size",
        type=int,
        default=2048,
        help="Batch size for cosine similarity computation",
    )

    args = parser.parse_args()

    target_sizes = ["small", "large"] if args.data_sizes == "both" else [args.data_sizes]

    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)

    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_dir}")

    print("Scanning for available embedding sets ...")
    size_to_sets = find_available_embedding_sets(embedding_dir, target_sizes)
    size_to_label_pairmaps: Dict[str, Dict[str, Dict[Tuple[str, str], float]]] = {}

    for size in target_sizes:
        sets = size_to_sets.get(size, [])
        human_sets = [s for s in sets if s.name == "human"]
        if not human_sets:
            print(f"❌ No human set for size '{size}'. Skipping plot.")
            continue
        human = human_sets[0]

        # Compute similarities for all model sets for this size
        label_to_values: Dict[str, List[float]] = {}
        label_to_pairmap: Dict[str, Dict[Tuple[str, str], float]] = {}
        for ds in sets:
            if ds.name == "human":
                continue
            sims = aggregate_similarities(
                human,
                ds,
                use_gpu=args.use_gpu,
                sim_batch_size=args.sim_batch_size,
            )
            if not sims:
                print(f"⚠️  No matched pairs for dataset '{ds.name}' (size={size})")
            label_to_values[ds.name] = sims
            label_to_pairmap[ds.name] = aggregate_similarity_map(
                human,
                ds,
                use_gpu=args.use_gpu,
                sim_batch_size=args.sim_batch_size,
            )

        size_to_label_pairmaps[size] = label_to_pairmap

        if not label_to_values:
            print(f"⚠️  No model datasets found for size '{size}'. Skipping plot.")
            continue

        # Compute participant-level stats for summary bar
        label_to_stats: Dict[str, Tuple[float, float]] = {}
        counts: Dict[str, int] = {}
        for label, pm in {lbl: pairmap_to_participant_means(pm) for lbl, pm in label_to_pairmap.items()}.items():
            mean, ci_hw, n_part = mean_ci_95_from_participants(pm)
            label_to_stats[label] = (mean, ci_hw)
            counts[label] = n_part

        plot_bar_with_ci(size, label_to_stats, output_dir)
        save_summary_csv(size, label_to_stats, counts, output_dir)

        # New: curves (points + error) and histograms using participant-level CI
        if "zeroshot" in label_to_pairmap or any(k.startswith("fewshot_") for k in label_to_pairmap.keys()) or any(k.startswith("permuted_") for k in label_to_pairmap.keys()):
            plot_similarity_curves(size, label_to_pairmap, output_dir)
            plot_similarity_histograms(size, label_to_pairmap, output_dir)
            # Spearman correlations (participant-level means vs N)
            stats_dir = Path(output_dir).parent / "results" / "statistical_tests"
            compute_spearman_correlations_for_size(size, label_to_pairmap, stats_dir)
        else:
            print("Not enough datasets to draw curves/histograms.")

        # Paired t-tests versus zeroshot baseline for this size (participant-level means)
        if "zeroshot" in label_to_pairmap:
            baseline_pm = pairmap_to_participant_means(label_to_pairmap["zeroshot"])
            test_labels = [lbl for lbl in label_to_pairmap.keys() if lbl != "zeroshot"]
            t_rows = []
            for lbl in order_labels(test_labels):
                comp_pm = pairmap_to_participant_means(label_to_pairmap[lbl])
                common_ids = list(set(baseline_pm.keys()) & set(comp_pm.keys()))
                base_vals = [baseline_pm[i] for i in common_ids]
                comp_vals = [comp_pm[i] for i in common_ids]
                t_val, df, p_val, note = paired_t_test(base_vals, comp_vals)
                d_z = cohens_dz(base_vals, comp_vals)
                t_rows.append({
                    "data_size": size,
                    "comparison": f"{lbl} vs zeroshot",
                    "n_pairs": len(common_ids),
                    "t_value": t_val,
                    "df": df,
                    "p_value": p_val,
                    "cohens_d_z": d_z,
                    "method": note,
                })

            # Adjust p-values via BH-FDR per "curve": each dataset label (e.g., fewshot_* vs zeroshot)
            for dataset_label in {row["comparison"].split(" vs ")[0] for row in t_rows}:
                mask_rows = [r for r in t_rows if r["comparison"].startswith(f"{dataset_label} vs ")]
                if not mask_rows:
                    continue
                pvals = [r["p_value"] for r in mask_rows]
                adj = fdr_bh(pvals)
                for r, a in zip(mask_rows, adj):
                    r["p_value_adj_bh"] = a

            # Save to CSV
            import pandas as pd  # optional
            out_dir = output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            t_csv = out_dir / f"ttest_vs_zeroshot_{size}.csv"
            try:
                pd.DataFrame(t_rows).to_csv(t_csv, index=False)
            except Exception:
                import csv
                with open(t_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(t_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(t_rows)
            print(f"Saved t-test results: {t_csv}")
        else:
            print("No zeroshot dataset available; skipping t-tests.")

        # New: Paired t-tests Fewshot_N vs Permuted_N for matching N (participant-level means)
        # Build maps of N -> pair maps
        fs_ns = sorted({int(m.group(1)) for lbl in label_to_pairmap.keys() if (m := re.match(r"fewshot_(\d+)$", lbl))})
        pm_ns = sorted({int(m.group(1)) for lbl in label_to_pairmap.keys() if (m := re.match(r"permuted_(\d+)$", lbl))})
        shared_ns = sorted(set(fs_ns) & set(pm_ns))
        if shared_ns:
            tp_rows = []
            for n in shared_ns:
                fs_pm = pairmap_to_participant_means(label_to_pairmap.get(f"fewshot_{n}", {}))
                pm_pm = pairmap_to_participant_means(label_to_pairmap.get(f"permuted_{n}", {}))
                # Align participant IDs
                common_ids = list(set(fs_pm.keys()) & set(pm_pm.keys()))
                fs_vals = [fs_pm[i] for i in common_ids]
                pm_vals = [pm_pm[i] for i in common_ids]
                t_val, df, p_val, note = paired_t_test(fs_vals, pm_vals)
                d_z = cohens_dz(fs_vals, pm_vals)
                tp_rows.append({
                    "data_size": size,
                    "comparison": f"fewshot_{n} vs permuted_{n}",
                    "n": n,
                    "n_pairs": len(common_ids),
                    "t_value": t_val,
                    "df": df,
                    "p_value": p_val,
                    "cohens_d_z": d_z,
                    "method": note,
                })

            # FDR correction across the N-comparisons
            pvals2 = [r["p_value"] for r in tp_rows]
            adj2 = fdr_bh(pvals2)
            for r, a in zip(tp_rows, adj2):
                r["p_value_adj_bh"] = a

            # Save to separate CSV
            try:
                import pandas as pd  # optional
                out_dir = output_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                t_csv2 = out_dir / f"ttest_fewshot_vs_permuted_{size}.csv"
                pd.DataFrame(tp_rows).to_csv(t_csv2, index=False)
            except Exception:
                import csv
                out_dir = output_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                t_csv2 = out_dir / f"ttest_fewshot_vs_permuted_{size}.csv"
                with open(t_csv2, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(tp_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(tp_rows)
            print(f"Saved fewshot vs permuted t-test results: {t_csv2}")
        else:
            print("No shared N between fewshot and permuted; skipping FS vs PM t-tests.")

    # Combined small+large single-axes plot if both present
    try:
        if "small" in size_to_label_pairmaps and "large" in size_to_label_pairmaps:
            plot_similarity_curves_combined(size_to_label_pairmaps["small"], size_to_label_pairmaps["large"], output_dir)
            plot_similarity_curves_two_panel(size_to_label_pairmaps["small"], size_to_label_pairmaps["large"], output_dir)
            # Also export combined four Spearman rows (small/large × fewshot/permuted)
            stats_dir_both = Path(output_dir).parent / "results" / "statistical_tests"
            try:
                # Build rows for each dataset
                rows_combined: List[Dict] = []
                def per_size_rows(size: str, label_to_pairmap: Dict[str, Dict[Tuple[str, str], float]]):
                    lbl_to_pm = {lbl: pairmap_to_participant_means(pm) for lbl, pm in label_to_pairmap.items()}
                    for series in ["fewshot", "permuted"]:
                        ns, means = [], []
                        allowed_ns = {1, 2, 3, 5, 8, 10}
                        for lbl, pm in lbl_to_pm.items():
                            m = re.match(rf"{series}_(\d+)$", lbl)
                            if not m:
                                continue
                            n_ex = int(m.group(1))
                            if n_ex in EXCLUDED_NUM_EXAMPLES:
                                continue
                            if n_ex not in allowed_ns:
                                continue
                            mean, _ci, _n = mean_ci_95_from_participants(pm)
                            if not np.isnan(mean):
                                ns.append(n_ex)
                                means.append(mean)
                        if len(ns) > 1 and len(ns) == len(means):
                            order = np.argsort(ns)
                            ns_sorted = [ns[i] for i in order]
                            means_sorted = [means[i] for i in order]
                            rho, p = spearmanr(ns_sorted, means_sorted)
                            rows_combined.append({
                                "data_size": size,
                                "series": series,
                                "n_points": len(ns_sorted),
                                "spearman_rho": float(rho) if rho is not None else float("nan"),
                                "p_value": float(p) if p is not None else float("nan"),
                            })
                per_size_rows("small", size_to_label_pairmaps["small"])
                per_size_rows("large", size_to_label_pairmaps["large"])
                stats_dir_both.mkdir(parents=True, exist_ok=True)
                out_csv_both = stats_dir_both / "spearman_cot_similarity_both.csv"
                try:
                    import pandas as pd  # optional
                    pd.DataFrame(rows_combined).to_csv(out_csv_both, index=False)
                except Exception:
                    import csv
                    if rows_combined:
                        with open(out_csv_both, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=list(rows_combined[0].keys()))
                            writer.writeheader()
                            writer.writerows(rows_combined)
                    else:
                        with open(out_csv_both, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=["data_size","series","n_points","spearman_rho","p_value"])
                            writer.writeheader()
                print(f"Saved combined Spearman correlations: {out_csv_both}")
            except Exception as e2:
                print(f"Warning: Failed to export combined Spearman correlations: {e2}")
    except Exception as e:
        print(f"Warning: Failed to render combined/two-panel plots: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
