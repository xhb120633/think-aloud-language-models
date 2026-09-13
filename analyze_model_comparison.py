"""
Analysis of model comparison results across different model types and datasets.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import math
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
import colorsys
from matplotlib.colors import to_rgba, to_hex
import re
import argparse
from scipy.stats import ttest_rel

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import PLOT_STYLE

# Add these mappings and helpers at the top after imports
MODEL_ORDER = [
    # Cognitive models
    'EV_Model', 'Best_PT_Model',
    # Neural network models
    'ValueBasedModel', 'ContextDependentModel',
    # LLM predictions - Llama
    'meta-llama/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    # LLM predictions - Qwen
    'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-72B-Instruct',
    # LLM predictions - GPT
    'gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1', 'gpt-4o',
    # Human
    'Human'
]

MODEL_NAME_MAP = {
    'EV_Model': 'EV',
    'Best_PT_Model': 'PT',
    'ValueBasedModel': 'ValueBased',
    'ContextDependentModel': 'ContextDep',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'meta-llama/Meta-Llama-3.1-70B-Instruct': 'Llama-3.1-70B',
    'Qwen/Qwen2.5-3B-Instruct': 'Qwen-2.5-3B',
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen-2.5-7B',
    'Qwen/Qwen2.5-14B-Instruct': 'Qwen-2.5-14B',
    'Qwen/Qwen2.5-32B-Instruct': 'Qwen-2.5-32B',
    'Qwen/Qwen2.5-72B-Instruct': 'Qwen-2.5-72B',
    'gpt-4.1-nano': 'GPT-4.1-nano',
    'gpt-4.1-mini': 'GPT-4.1-mini',
    'gpt-4.1': 'GPT-4.1',
    'gpt-4o': 'GPT-4o',
    'Human': 'Human Prediction'
}

def darken_color(color, factor=0.3):
    """Darken a hex color by a factor."""
    # Convert hex to RGB
    color = color.lstrip('#')
    r, g, b = tuple(int(color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Convert to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Darken the value
    v = max(0, v - factor)
    
    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    
    # Convert back to hex
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

# Update BASE_COLORS for distinct cognitive/neural models
BASE_COLORS = {
    'EV': '#1f77b4',         # Blue
    'PT': '#17becf',         # Teal
    'ValueBased': '#2ca02c', # Green
    'ContextDep': '#bcbd22', # Olive
    'llama': '#ff7f0e',      # Orange
    'qwen': '#9467bd',       # Purple
    'gpt': '#d62728',        # Red
    'human': '#222222',      # Dark gray
    'other': '#888888'       # Fallback
}

# Update color map for gpt-4.1 series and human
GPT_GREY = '#888888'
GPT_GREY_LIGHT = '#bbbbbb'
GPT_GREY_LIGHTER = '#dddddd'
HUMAN_BLACK = '#000000'

MODEL_COLOR_MAP = {
    # Cognitive models
    'EV': BASE_COLORS['EV'],
    'PT': BASE_COLORS['PT'],
    'ValueBased': BASE_COLORS['ValueBased'],
    'ContextDep': BASE_COLORS['ContextDep'],
    # Neural network models
    'PT-NN': BASE_COLORS['PT'],
    # LLM predictions - Llama
    'Llama-3-8B': BASE_COLORS['llama'],
    'Llama-3-70B': darken_color(BASE_COLORS['llama'], 0.3),
    # LLM predictions - Qwen
    'Qwen-2.5-3B': BASE_COLORS['qwen'],
    'Qwen-2.5-7B': darken_color(BASE_COLORS['qwen'], 0.2),
    'Qwen-2.5-14B': darken_color(BASE_COLORS['qwen'], 0.4),
    'Qwen-2.5-32B': darken_color(BASE_COLORS['qwen'], 0.6),
    'Qwen-2.5-72B': darken_color(BASE_COLORS['qwen'], 0.8),
    # LLM predictions - GPT (greys)
    'gpt-4.1-nano': GPT_GREY_LIGHTER,
    'gpt-4.1-mini': GPT_GREY_LIGHT,
    'gpt-4.1': GPT_GREY,
    'gpt-4o': '#d62728',  # Red, but will be excluded for small dataset
    # Human
    'Human': HUMAN_BLACK
}

# -------------------------------
# Control comparison for LLaMA-3.1-70B (original vs permuted vs base)
# -------------------------------

# Verbose logging for permutation-only fast path
PERM_LOG = False

def _find_llama70b_file(
    results_dir: str,
    data_size: str,
    mode_key: str,
    action_removal_llama_filename: bool = False,
    action_removal_results_filename_override: str | None = None,
) -> str | None:
    """Return path to the LLaMA-70B results file for a given dataset size and mode.
    mode_key in { 'human', 'permutation', 'base' }.
    For permutation, saved suffix is '_permuted_results.json'. For others, it's '_{mode}_results.json'.
    """
    suffix = 'permuted' if mode_key == 'permutation' else mode_key
    if mode_key == 'action_removal' and action_removal_llama_filename:
        alt_fname = f'meta-llama_Meta-Llama-3.1-70B-Instruct_{data_size}_{mode_key}_results_LLaMA.json'
        alt_path = os.path.join(results_dir, alt_fname)
        if os.path.exists(alt_path):
            return alt_path

    if mode_key == 'action_removal' and action_removal_results_filename_override:
        # The override filename is dataset-specific (typically includes `_small_` or `_large_`).
        # Only apply it when it matches the requested data_size.
        if f"_{data_size}_" in action_removal_results_filename_override:
            override_path = os.path.join(results_dir, action_removal_results_filename_override)
            if os.path.exists(override_path):
                return override_path

    fname = f'meta-llama_Meta-Llama-3.1-70B-Instruct_{data_size}_{suffix}_results.json'
    path = os.path.join(results_dir, fname)
    if os.path.exists(path):
        return path
    if PERM_LOG:
        print(f"[perm] File not found for mode={mode_key}: {fname}")
    return None


def _load_llama70b_mode_participant_metrics(
    data_size: str,
    mode_key: str,
    action_removal_llama_filename: bool = False,
    action_removal_results_filename_override: str | None = None,
) -> pd.DataFrame:
    """Return per-participant metrics DataFrame (sub_id, acc, like) for a mode."""
    results_dir = "results/exp1_llm_prediction"
    path = _find_llama70b_file(
        results_dir,
        data_size,
        mode_key,
        action_removal_llama_filename=action_removal_llama_filename,
        action_removal_results_filename_override=action_removal_results_filename_override,
    )
    # Fallback to combined results file when per-mode file not found (normal pipeline saves combined file)
    if path is None:
        combined = os.path.join(
            results_dir,
            f"meta-llama_Meta-Llama-3.1-70B-Instruct_{data_size}_results.json"
        )
        path = combined if os.path.exists(combined) else None
        if PERM_LOG and path is None:
            print(f"[perm] Combined file also not found for size={data_size}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame(columns=['sub_id', 'acc', 'like'])

    rows = data.get('results', [])
    if not rows:
        return pd.DataFrame(columns=['sub_id', 'acc', 'like'])

    df = pd.DataFrame(rows)
    if 'mode' in df.columns:
        df = df[df['mode'].astype(str).eq('permutation' if mode_key == 'permutation' else mode_key)]
    if df.empty:
        return pd.DataFrame(columns=['sub_id', 'acc', 'like'])

    if {'extracted_choice', 'actual_choice'}.issubset(df.columns):
        df['acc'] = (df['extracted_choice'] == df['actual_choice']).astype(float)
    else:
        df['acc'] = np.nan
    if {'probability_option_a', 'probability_option_b', 'actual_choice'}.issubset(df.columns):
        df['like'] = df.apply(lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'], axis=1)
    else:
        df['like'] = np.nan

    sub_col = 'sub_id' if 'sub_id' in df.columns else 'participant_id' if 'participant_id' in df.columns else None
    if sub_col is None:
        return pd.DataFrame(columns=['sub_id', 'acc', 'like'])
    grp = df.groupby(sub_col)
    out = grp[['acc', 'like']].mean().reset_index().rename(columns={sub_col: 'sub_id'})
    return out

def _summarize_participant_metrics(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Return mean_acc, sem_acc, mean_like, sem_like computed across participants."""
    if df is None or df.empty:
        return float('nan'), float('nan'), float('nan'), float('nan')
    acc = df['acc'].astype(float)
    like = df['like'].astype(float)
    mean_acc = float(np.nanmean(acc)) if len(acc) else float('nan')
    sem_acc = float(np.nanstd(acc, ddof=1) / np.sqrt(np.sum(~np.isnan(acc)))) if acc.notna().sum() > 1 else 0.0
    mean_like = float(np.nanmean(like)) if len(like) else float('nan')
    sem_like = float(np.nanstd(like, ddof=1) / np.sqrt(np.sum(~np.isnan(like)))) if like.notna().sum() > 1 else 0.0
    return mean_acc, sem_acc, mean_like, sem_like


def plot_llama70b_control_grouped(
    data_size_list: List[str] = ['small', 'large'],
    output_suffix: str = "",
    action_removal_llama_filename: bool = False,
    action_removal_results_filename_override: str | None = None,
    action_removal_results_filename_small: str | None = None,
    action_removal_results_filename_large: str | None = None,
) -> None:
    """Create grouped narrow bar plots for LLaMA-3.1-70B controls.

    Conditions:
    - original (human)
    - permuted (action-permutation control)
    - base (zero-shot / base)
    - action_removal (superficial action claims masked out; may exist only for small)

    Two figures: accuracy and likelihood. Also saves paired t-tests with BH-adjusted p-values
    across the comparisons that are available.
    """
    if PERM_LOG:
        print(f"[perm] Starting grouped control plotting for sizes: {data_size_list}")
    # Colors: use llama base color
    llama_color = BASE_COLORS['llama']

    # Collect stats
    rows = []
    for size in data_size_list:
        # Load per-participant metrics for all conditions we may have
        df_human = _load_llama70b_mode_participant_metrics(size, 'human')
        df_perm = _load_llama70b_mode_participant_metrics(size, 'permutation')
        df_base = _load_llama70b_mode_participant_metrics(size, 'base')
        per_size_override = (
            action_removal_results_filename_small
            if size == 'small'
            else action_removal_results_filename_large
        )
        df_ar = _load_llama70b_mode_participant_metrics(
            size,
            'action_removal',
            action_removal_llama_filename=action_removal_llama_filename,
            action_removal_results_filename_override=(
                per_size_override or action_removal_results_filename_override
            ),
        )

        # Summaries for plotting
        m_acc, se_acc, m_like, se_like = _summarize_participant_metrics(df_human)
        rows.append({'data_size': size, 'condition': 'original', 'accuracy': m_acc, 'accuracy_sem': se_acc, 'likelihood': m_like, 'likelihood_sem': se_like})
        m_acc, se_acc, m_like, se_like = _summarize_participant_metrics(df_perm)
        rows.append({'data_size': size, 'condition': 'permuted', 'accuracy': m_acc, 'accuracy_sem': se_acc, 'likelihood': m_like, 'likelihood_sem': se_like})
        m_acc, se_acc, m_like, se_like = _summarize_participant_metrics(df_base)
        rows.append({'data_size': size, 'condition': 'base', 'accuracy': m_acc, 'accuracy_sem': se_acc, 'likelihood': m_like, 'likelihood_sem': se_like})
        m_acc, se_acc, m_like, se_like = _summarize_participant_metrics(df_ar)
        rows.append({'data_size': size, 'condition': 'action_removal', 'accuracy': m_acc, 'accuracy_sem': se_acc, 'likelihood': m_like, 'likelihood_sem': se_like})

        # Save paired t-tests original vs permuted, original vs base, original vs action_removal
        try:
            def paired_tests(a_df: pd.DataFrame, b_df: pd.DataFrame, label: str) -> Dict[str, float]:
                merged = pd.merge(a_df[['sub_id', 'acc', 'like']], b_df[['sub_id', 'acc', 'like']], on='sub_id', suffixes=('_a', '_b'))
                merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
                out = {}
                if not merged.empty:
                    # accuracy
                    t1, p1 = ttest_rel(merged['acc_a'].astype(float).values, merged['acc_b'].astype(float).values, nan_policy='omit')
                    d1 = float((merged['acc_a'] - merged['acc_b']).mean() / np.nanstd((merged['acc_a'] - merged['acc_b']), ddof=1)) if merged.shape[0] > 1 else float('nan')
                    # likelihood
                    t2, p2 = ttest_rel(merged['like_a'].astype(float).values, merged['like_b'].astype(float).values, nan_policy='omit')
                    d2 = float((merged['like_a'] - merged['like_b']).mean() / np.nanstd((merged['like_a'] - merged['like_b']), ddof=1)) if merged.shape[0] > 1 else float('nan')
                    out = {
                        'n': int(merged.shape[0]),
                        't_acc': float(t1), 'p_acc': float(p1), 'd_acc': d1,
                        't_like': float(t2), 'p_like': float(p2), 'd_like': d2,
                        'label': label
                    }
                return out

            res_perm = paired_tests(df_human, df_perm, 'original_vs_permuted')
            res_base = paired_tests(df_human, df_base, 'original_vs_base')
            res_ar = paired_tests(df_human, df_ar, 'original_vs_action_removal')

            # BH adjust across the set of comparisons that are available
            comps_acc = [res_perm.get('p_acc', np.nan), res_base.get('p_acc', np.nan), res_ar.get('p_acc', np.nan)]
            comps_like = [res_perm.get('p_like', np.nan), res_base.get('p_like', np.nan), res_ar.get('p_like', np.nan)]
            available_mask = [bool(res_perm), bool(res_base), bool(res_ar)]

            if all(available_mask[:2]):
                # accuracy BH
                pvals_acc = [p for p, ok in zip(comps_acc, available_mask) if ok and np.isfinite(p)]
                pvals_like = [p for p, ok in zip(comps_like, available_mask) if ok and np.isfinite(p)]
                if pvals_acc:
                    padj_acc = _bh_adjust(pvals_acc)
                else:
                    padj_acc = []
                if pvals_like:
                    padj_like = _bh_adjust(pvals_like)
                else:
                    padj_like = []

                # Map adjusted values back into res_* dicts (only for available comparisons)
                acc_iter = iter(padj_acc)
                like_iter = iter(padj_like)
                for r, ok in zip([res_perm, res_base, res_ar], available_mask):
                    if ok:
                        r['p_acc_adj'] = float(next(acc_iter))
                        r['p_like_adj'] = float(next(like_iter))

                # Save to CSV (only include available comparisons)
                out_rows = []
                for r in [res_perm, res_base, res_ar]:
                    if not r:
                        continue
                    out_rows.append({
                        'comparison': r['label'],
                        'metric': 'accuracy',
                        'n': r['n'],
                        't': r['t_acc'],
                        'df': r['n'] - 1,
                        'p': r['p_acc'],
                        'p_adj': r.get('p_acc_adj', r['p_acc']),
                        'cohen_d': r['d_acc'],
                    })
                    out_rows.append({
                        'comparison': r['label'],
                        'metric': 'likelihood',
                        'n': r['n'],
                        't': r['t_like'],
                        'df': r['n'] - 1,
                        'p': r['p_like'],
                        'p_adj': r.get('p_like_adj', r['p_like']),
                        'cohen_d': r['d_like'],
                    })
                out_df = pd.DataFrame(out_rows)
                os.makedirs('results/statistical_tests', exist_ok=True)
                out_path = os.path.join(
                    'results/statistical_tests',
                    f'control_comparison_{size}{output_suffix}.csv',
                )
                out_df.to_csv(out_path, index=False)
                if PERM_LOG:
                    print(f"[perm] Saved control t-tests to {out_path}")
        except Exception as e:
            if PERM_LOG:
                print(f"[perm] Failed t-tests for size={size}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("No LLaMA-3.1-70B control/original results found.")
        return
    if PERM_LOG:
        print(f"[perm] Aggregated rows:\n{df}")

    # Order x-axis as small, large; grouped bars original vs permuted
    sizes = ['small', 'large']
    conditions = ['original', 'permuted', 'action_removal', 'base']

    def _plot_metric(metric: str, ylabel: str, out_path: str):
        if PERM_LOG:
            # Reflect the actual PDF output path in logs
            print(f"[perm] Preparing to plot {metric}...")
        plt.rcParams.update({
            'font.size': 9,
            'axes.linewidth': 0.6,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        # Slightly wider canvas to avoid overlap between multiline experiment tick labels.
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        x = np.arange(len(sizes))
        width = 0.18  # four bars

        vals = {
            cond: [
                df[(df['data_size'] == sz) & (df['condition'] == cond)][metric].values[0]
                if not df[(df['data_size'] == sz) & (df['condition'] == cond)].empty
                else np.nan
                for sz in sizes
            ]
            for cond in conditions
        }
        errs = {
            cond: [
                df[(df['data_size'] == sz) & (df['condition'] == cond)][f'{metric}_sem'].values[0]
                if not df[(df['data_size'] == sz) & (df['condition'] == cond)].empty
                else np.nan
                for sz in sizes
            ]
            for cond in conditions
        }

        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        hatch_map = {
            'original': None,
            'permuted': '//',
            'action_removal': 'oo',
            'base': 'xx',
        }
        label_map = {
            'original': 'Original',
            'permuted': 'Permuted (context mismatch)',
            'action_removal': 'Masked (choice removed)',
            'base': 'Baseline (task only)',
        }

        for cond, off in zip(conditions, offsets):
            hatch = hatch_map.get(cond)
            if hatch is None:
                ax.bar(
                    x + off,
                    vals[cond],
                    width,
                    yerr=np.array(errs[cond]) * 1.96,
                    capsize=3,
                    color=llama_color,
                    edgecolor='white',
                    linewidth=0.5,
                    label=label_map.get(cond, cond),
                )
            else:
                ax.bar(
                    x + off,
                    vals[cond],
                    width,
                    yerr=np.array(errs[cond]) * 1.96,
                    capsize=3,
                    color=llama_color,
                    edgecolor='white',
                    linewidth=0.5,
                    hatch=hatch,
                    label=label_map.get(cond, cond),
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            ['Experiment 1\n(PT design)', 'Experiment 2\n(choice13k-sampled)'],
            fontsize=9,
            ha='center',
            linespacing=1.15,
        )
        ax.tick_params(axis='x', pad=5)
        ax.set_ylabel(ylabel)

        # Dynamic y-limits (include all three conditions)
        all_means = np.array(
            sum([list(vals[c]) for c in conditions], []),
            dtype=float,
        )
        all_errs = np.array(
            sum([list(errs[c]) for c in conditions], []),
            dtype=float,
        ) * 1.96
        finite_mask = np.isfinite(all_means)
        if finite_mask.any():
            mmin = np.nanmin(all_means[finite_mask] - np.nan_to_num(all_errs[finite_mask], nan=0.0))
            mmax = np.nanmax(all_means[finite_mask] + np.nan_to_num(all_errs[finite_mask], nan=0.0))
            pad = max(0.01, 0.08 * (mmax - mmin if mmax > mmin else 0.05))
            ax.set_ylim(max(0.0, mmin - pad), min(1.0, mmax + pad))
        else:
            ax.set_ylim(0.0, 1.0)

        # Place legend outside on the right for cleaner layout
        ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        ax.grid(False)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        # Extra bottom margin for 2-line x tick labels.
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.28)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Save high-quality vector PDF
        out_path_pdf = os.path.splitext(out_path)[0] + '.pdf'
        if PERM_LOG:
            print(f"[perm] Plotting {metric} to {out_path_pdf}")
        plt.savefig(out_path_pdf, dpi=450, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)

    # Save plots
    acc_png = 'figures/control_comparison/llama70b_cot_control_grouped_accuracy.png'
    like_png = 'figures/control_comparison/llama70b_cot_control_grouped_likelihood.png'
    acc_png = os.path.splitext(acc_png)[0] + output_suffix + '.png'
    like_png = os.path.splitext(like_png)[0] + output_suffix + '.png'
    _plot_metric('accuracy', 'Accuracy', acc_png)
    _plot_metric('likelihood', 'Likelihood', like_png)
    if PERM_LOG:
        print("[perm] Finished plotting control grouped figures.")


def get_model_family_and_size(display_name):
    # Heuristic: parse display name to get family and size
    if display_name == 'Expected Value':  # Updated from 'EV'
        return 'EV', 1
    if display_name == 'Prospect Theory':  # Updated from 'PT'
        return 'PT', 1
    if display_name == 'ValueBased':
        return 'ValueBased', 1
    if display_name == 'ContextDep':
        return 'ContextDep', 1
    if display_name.startswith('Llama-3'):
        return 'llama', int(display_name.split('-')[-1].replace('B',''))
    if display_name.startswith('Qwen-2.5-'):
        # e.g. Qwen-2.5-3B, Qwen-2.5-7B, Qwen-2.5-14B, Qwen-2.5-32B, Qwen-2.5-72B
        size_str = display_name.split('-')[-1].replace('B','')
        try:
            size = int(size_str)
        except ValueError:
            size = 1
        return 'qwen', size
    if display_name.startswith('gpt-4.1') or display_name.startswith('GPT-4.1'):
        if 'nano' in display_name:
            return 'gpt', 1
        if 'mini' in display_name:
            return 'gpt', 2
        return 'gpt', 3
    if display_name == 'gpt-4o':
        return 'gpt', 4
    if display_name == 'Human Prediction' or display_name == 'Human Think Aloud':
        return 'human', 1
    return 'other', 1

def get_family_color(family, size, max_size):
    base = BASE_COLORS.get(family, '#888888')
    rgba = list(to_rgba(base))
    # For larger models: increase brightness (closer to white)
    # For smaller models: increase transparency
    if max_size > 1:
        brightness = 0.5 + 0.5 * (size-1)/(max_size-1)  # 0.5 to 1.0
        alpha = 0.5 + 0.5 * (size-1)/(max_size-1)       # 0.5 to 1.0
    else:
        brightness = 1.0
        alpha = 1.0
    rgba[:3] = [brightness * c + (1-brightness) * 1.0 for c in rgba[:3]]
    rgba[3] = alpha
    return to_hex(rgba, keep_alpha=True)

# Update display names for PT and EV
DISPLAY_NAME_OVERRIDES = {
    'PT': 'Prospect Theory',
    'EV': 'Expected Value',
    'Human Prediction': 'Human Prediction',  # Keep full name for human
    'GPT-4.1': 'GPT-4.1',
    'GPT-4.1-mini': 'GPT-4.1-mini',
    'GPT-4.1-nano': 'GPT-4.1-nano',
    'GPT-4o': 'GPT-4o',
    # Add more if needed
}

def order_and_rename_models(df, data_size='small'):
    # Map model names and order using display names
    df['display_name'] = df['model'].map(MODEL_NAME_MAP).fillna(df['model'])
    # Apply display name overrides
    df['display_name'] = df['display_name'].replace(DISPLAY_NAME_OVERRIDES)
    
    # Create consistent ordering regardless of dataset size
    # This ensures same y-tick order across both plots
    def get_order(x, idx):
        order_list = [DISPLAY_NAME_OVERRIDES.get(MODEL_NAME_MAP.get(m, m), MODEL_NAME_MAP.get(m, m)) for m in MODEL_ORDER]
        if x in order_list:
            return order_list.index(x)
        else:
            return 999 + idx
    
    df = df.reset_index(drop=True)
    df['order'] = [get_order(m, i) for i, m in enumerate(df['display_name'])]
    df = df.sort_values('order')
    return df

def load_llm_results(data_size: str) -> Dict:
    """
    Load LLM results from the new format.
    
    Args:
        data_size: Size of dataset used ('small' or 'large')
        
    Returns:
        Dictionary with results for each LLM model
    """
    results_dir = "results/exp1_llm_prediction"
    llm_results = {}
    
    # Find all result files in the first hierarchy
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json') or 'checkpoint' in filename:
            continue
            
        try:
            with open(os.path.join(results_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Get dataset size from metadata
            file_data_size = data['metadata'].get('data_size')
            if file_data_size != data_size:
                continue
                
            # Get model name from metadata
            model_name = data['metadata'].get('model_name')
            if not model_name:
                continue
                
            # Store results
            llm_results[model_name] = {
                'metadata': data['metadata'],
                'results': data['results']
            }
            print(f"Loaded results for {model_name} ({data_size} dataset)")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return llm_results

def load_other_model_results(data_size: str) -> Tuple[Dict, Dict]:
    """
    Load neural net and cognitive model results, excluding any model with 'LogisticRegression' in its name.
    
    Args:
        data_size: Size of dataset used ('small' or 'large')
        
    Returns:
        Tuple of (symbolic_results, NN_results)
    """
    baseline_filepath = f'results/model_result/{data_size}_dataset/'
    
    # Load symbolic model results
    if data_size == 'small':
        with open(baseline_filepath + f'symbolic_model_detailed_results_{data_size}.json', 'r', encoding='utf-8') as file:
            symbolic_results = json.load(file)
        with open(baseline_filepath + f'NN_model_detailed_results_{data_size}.json', 'r', encoding='utf-8') as file:
            NN_results = json.load(file)
    else:  # large dataset
        with open(baseline_filepath + f'by_progress/symbolic_model_detailed_results_{data_size}.json', 'r', encoding='utf-8') as file:
            symbolic_results = json.load(file)
        with open(baseline_filepath + f'by_progress/NN_model_detailed_results_{data_size}.json', 'r', encoding='utf-8') as file:
            NN_results = json.load(file)
    
    # Exclude models with 'LogisticRegression' in their name (case-insensitive)
    symbolic_results = {k: v for k, v in symbolic_results.items() if 'logisticregression' not in k.lower()}
    NN_results = {k: v for k, v in NN_results.items() if 'logisticregression' not in k.lower()}
    
    return symbolic_results, NN_results

def load_human_results(data_size: str) -> Dict:
    """
    Load human prediction results and aggregate at participant level (small dataset only).

    Returns a list of records with per-participant mean accuracy and likelihood so that
    downstream error bars reflect individual-level variability.
    """
    if data_size != 'small':
        return {}

    # Load human data from Excel files
    folder_path = 'E:/RA_coding/Data Prediction'
    df_human = pd.concat([pd.read_excel(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith('.xlsx')])
    df_human['model'] = 'human'

    # Load behavioral data
    csv_file = 'data/behavioral_text_data.csv'
    df = pd.read_csv(csv_file)
    df.drop(columns=['0'], errors='ignore', inplace=True)
    df.columns = ['sub_id', 'choice', 'p1', 'v1', 'p2', 'v2', 'problem_id', 'rt', 'think_aloud', 'word_count']

    # Process lists in the DataFrame
    for column in ['p1', 'v1', 'p2', 'v2']:
        df[column] = df[column].apply(literal_eval)

    # Match Trial_ID with problem_id
    b = np.unique(df['problem_id'])
    df_human['Trial_ID'] = df_human['Trial_ID'].apply(literal_eval)
    df_human['problem_id'] = df_human['Trial_ID'].apply(lambda x: b[int(x[0])])
    # Use provided Sub_ID as participant identifier
    df_human['sub_id'] = df_human['Sub_ID']

    # Merge with behavioral data
    df_human = pd.merge(df_human, df, on=['sub_id', 'problem_id'], how='left')
    df_human['choice'] = df_human['choice'].astype(int)

    # Keep minimal necessary columns
    df_human = df_human[['sub_id', 'problem_id', 'choice', 'Prediction', 'Confidence', 'model']]

    # Encode Prediction column: Option A as 0, Option B as 1
    df_human['Prediction'] = df_human['Prediction'].apply(lambda x: 0 if x == 'Option A' else 1)

    # Trial-level accuracy
    df_human['trial_correct'] = (df_human['Prediction'] == df_human['choice']).astype(float)

    # Trial-level likelihood proxy from confidence (symmetric around 0.5)
    df_human['trial_like'] = np.where(
        df_human['Prediction'] == df_human['choice'],
        0.5 + df_human['Confidence'] / 200.0,
        0.5 - df_human['Confidence'] / 200.0
    )

    # Aggregate per participant across trials (original subject IDs for now)
    per_sub = df_human.groupby('sub_id', as_index=False).agg(
        test_accuracy=('trial_correct', 'mean'),
        test_likelihood=('trial_like', 'mean')
    )
    per_sub['model'] = 'Human'

    # Align participant IDs with the label-encoded IDs used in LLM results
    try:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        # Fit on the same IDs as used in exp1 processing
        le.fit(df['sub_id'])
        orig_to_enc = {orig: int(enc) for orig, enc in zip(le.classes_, le.transform(le.classes_))}
        per_sub['sub_id'] = per_sub['sub_id'].map(lambda x: orig_to_enc.get(x, x))
    except Exception:
        # If encoding fails, keep original IDs (t-tests may not align)
        pass

    return per_sub.to_dict('records')

def process_llm_results(llm_results: Dict) -> pd.DataFrame:
    """
    Process LLM results into a format matching other models.
    
    Args:
        llm_results: Dictionary of LLM results
        
    Returns:
        DataFrame with processed results
    """
    rows = []
    for model_name, data in llm_results.items():
        # Convert results to DataFrame
        df = pd.DataFrame(data['results'])
        
        # Only process human mode
        mode_df = df[df['mode'] == 'human'].copy()  # Create a copy to avoid SettingWithCopyWarning
        if len(mode_df) > 0:
            # Group by participant and calculate accuracy
            participant_accuracy = mode_df.groupby('sub_id', group_keys=False).apply(
                lambda x: (x['extracted_choice'] == x['actual_choice']).mean()
            )
            
            # Calculate likelihood for each trial
            mode_df.loc[:, 'likelihood'] = mode_df.apply(
                lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                axis=1
            )
            
            # Calculate mean likelihood per participant
            participant_likelihood = mode_df.groupby('sub_id', group_keys=False)['likelihood'].mean()
            
            rows.append({
                'model': model_name,
                'accuracy': participant_accuracy.mean(),
                'accuracy_sem': participant_accuracy.sem(),
                'likelihood': participant_likelihood.mean(),
                'likelihood_sem': participant_likelihood.sem()
            })
    
    return pd.DataFrame(rows)

def process_other_models(data_size: str) -> pd.DataFrame:
    """
    Process neural net and cognitive model results.
    
    Args:
        data_size: Size of dataset used
        
    Returns:
        DataFrame with processed results
    """
    # Load results
    symbolic_results, NN_results = load_other_model_results(data_size)
    rows = []
    for model_name, model_data in symbolic_results.items():
        if data_size == 'small':
            last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
            round_data = model_data[last_round]
            test_accuracies = []
            test_likelihoods = []
            for participant_data in round_data.values():
                test_accuracies.append(participant_data['test_accuracy'])
                test_likelihoods.append(math.exp(-participant_data['test_loss']))
            rows.append({
                'model': model_name,
                'accuracy': np.nanmean(test_accuracies),
                'accuracy_sem': np.nanstd(test_accuracies) / np.sqrt(np.sum(~np.isnan(test_accuracies))),
                'likelihood': np.nanmean(test_likelihoods),
                'likelihood_sem': np.nanstd(test_likelihoods) / np.sqrt(np.sum(~np.isnan(test_likelihoods)))
            })
        else:
            last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
            last_step = max(model_data[last_round].keys(), key=lambda x: int(x.split('_')[1]))
            step_data = model_data[last_round][last_step]
            test_accuracies = []
            test_likelihoods = []
            for sample_data in step_data.values():
                for participant_data in sample_data.values():
                    test_accuracies.append(participant_data['test_accuracy'])
                    test_likelihoods.append(math.exp(-participant_data['test_loss']))
            rows.append({
                'model': model_name,
                'accuracy': np.nanmean(test_accuracies),
                'accuracy_sem': np.nanstd(test_accuracies) / np.sqrt(np.sum(~np.isnan(test_accuracies))),
                'likelihood': np.nanmean(test_likelihoods),
                'likelihood_sem': np.nanstd(test_likelihoods) / np.sqrt(np.sum(~np.isnan(test_likelihoods)))
            })
    for model_name, model_data in NN_results.items():
        if data_size == 'small':
            # Participant-level aggregation at last round if predictions available
            last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
            metrics = model_data[last_round]
            preds = metrics.get('test_predictions', [])
            if isinstance(preds, list) and len(preds) > 0:
                dfp = pd.DataFrame(preds)
                if not dfp.empty and {'sub_id', 'predicted_choice', 'actual_choice'}.issubset(dfp.columns):
                    # Compute per-participant accuracy and likelihood (exp(-mean BCE loss) if available)
                    acc_by_sub = dfp.groupby('sub_id').apply(lambda x: (x['predicted_choice'] == x['actual_choice']).mean())
                    if 'loss' in dfp.columns:
                        like_by_sub = dfp.groupby('sub_id')['loss'].mean().apply(lambda v: math.exp(-v))
                    else:
                        like_by_sub = pd.Series([np.nan] * len(acc_by_sub), index=acc_by_sub.index)
                    rows.append({
                        'model': model_name,
                        'accuracy': float(acc_by_sub.mean()),
                        'accuracy_sem': float(acc_by_sub.std(ddof=1) / np.sqrt(len(acc_by_sub))) if len(acc_by_sub) > 1 else 0.0,
                        'likelihood': float(like_by_sub.mean()),
                        'likelihood_sem': float(like_by_sub.std(ddof=1) / np.sqrt(np.sum(~np.isnan(like_by_sub)))) if like_by_sub.notna().sum() > 1 else 0.0
                    })
                    continue
            # Fallback to aggregated metrics
            rows.append({
                'model': model_name,
                'accuracy': metrics.get('test_accuracy', np.nan),
                'accuracy_sem': metrics.get('test_accuracy_sem', 0),
                'likelihood': math.exp(-metrics['test_loss']) if 'test_loss' in metrics else np.nan,
                'likelihood_sem': metrics.get('test_likelihood_sem', 0)
            })
        else:
            # Large + by_progress: flatten participant-level results across all samples at last step
            last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
            last_step = max(model_data[last_round].keys(), key=lambda x: int(x.split('_')[1]))
            step_data = model_data[last_round][last_step]
            acc_entries = []
            like_entries = []
            for sample_metrics in step_data.values():
                preds = sample_metrics.get('test_predictions', [])
                if not preds:
                    continue
                dfp = pd.DataFrame(preds)
                if dfp.empty or not {'sub_id', 'predicted_choice', 'actual_choice'}.issubset(dfp.columns):
                    continue
                acc_by_sub = dfp.groupby('sub_id').apply(lambda x: (x['predicted_choice'] == x['actual_choice']).mean())
                acc_entries.extend(acc_by_sub.tolist())
                if 'loss' in dfp.columns:
                    like_by_sub = dfp.groupby('sub_id')['loss'].mean().apply(lambda v: math.exp(-v))
                    like_entries.extend(like_by_sub.tolist())
            rows.append({
                'model': model_name,
                'accuracy': float(np.nanmean(acc_entries)) if len(acc_entries) > 0 else np.nan,
                'accuracy_sem': float(np.nanstd(acc_entries, ddof=1) / np.sqrt(np.sum(~np.isnan(acc_entries)))) if len(acc_entries) > 1 else 0.0,
                'likelihood': float(np.nanmean(like_entries)) if len(like_entries) > 0 else np.nan,
                'likelihood_sem': float(np.nanstd(like_entries, ddof=1) / np.sqrt(np.sum(~np.isnan(like_entries)))) if len(like_entries) > 1 else 0.0
            })
    return pd.DataFrame(rows)

def process_human_results(human_data: List[Dict]) -> pd.DataFrame:
    """
    Process human prediction results.
    
    Args:
        human_data: List of dictionaries containing human results
        
    Returns:
        DataFrame with processed results
    """
    if not human_data:
        return pd.DataFrame()
    df = pd.DataFrame(human_data)
    # Expect rows to be per-participant; compute mean and SEM across participants for 95% CI in plotting
    return pd.DataFrame([{
        'model': 'Human',
        'accuracy': float(np.nanmean(df['test_accuracy'])),
        'accuracy_sem': float(np.nanstd(df['test_accuracy'], ddof=1) / np.sqrt(np.sum(~np.isnan(df['test_accuracy'])))) if df['test_accuracy'].notna().sum() > 1 else 0.0,
        'likelihood': float(np.nanmean(df['test_likelihood'])),
        'likelihood_sem': float(np.nanstd(df['test_likelihood'], ddof=1) / np.sqrt(np.sum(~np.isnan(df['test_likelihood'])))) if df['test_likelihood'].notna().sum() > 1 else 0.0
    }])

def add_human_bar_if_missing(results_df, data_size):
    # Add a blank/null human bar for large dataset if not present
    # Check using 'model' column since 'display_name' doesn't exist yet
    if data_size == 'large' and not (results_df['model'] == 'Human').any():
        human_row = {}
        for col in results_df.columns:
            if col == 'model':
                human_row[col] = 'Human'
            else:
                human_row[col] = np.nan
        results_df = pd.concat([results_df, pd.DataFrame([human_row])], ignore_index=True)
    return results_df

def create_comparison_plot(
    results_df: pd.DataFrame,
    metric: str,
    data_size: str,
    save_path: str,
    x_limits: Tuple[float, float] | None = None,
    ticks: List[float] | None = None,
) -> Tuple[float, float]:
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 8,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333'
    })
    
    # Add blank/null human bar for large dataset first
    results_df = add_human_bar_if_missing(results_df, data_size)
    
    # Order and rename models (now includes all available models)
    results_df = order_and_rename_models(results_df, data_size).copy()
    
    # Exclude gpt-4o from small dataset AFTER ordering to maintain consistent order
    if data_size == 'small':
        # Filter out GPT-4o (mapped from 'gpt-4o' → 'GPT-4o' → 'GPT-4o')
        results_df = results_df[results_df['display_name'] != 'GPT-4o'].reset_index(drop=True)
    
    # Color assignment: use display names and family/size logic
    families = []
    sizes = []
    for name in results_df['display_name']:
        fam, sz = get_model_family_and_size(name)
        families.append(fam)
        sizes.append(sz)
    # Find max size per family for brightness scaling
    max_size_per_family = {fam: max([sz for f, sz in zip(families, sizes) if f == fam], default=1) for fam in set(families)}
    # Use custom color map for gpt-4.1 and human
    colors = []
    for name in results_df['display_name']:
        if name == 'Human Prediction':
            colors.append(HUMAN_BLACK)  # Pure black for Human
        elif name == 'GPT-4.1':
            colors.append(GPT_GREY)  # Darkest grey for GPT-4.1
        elif name == 'GPT-4.1-mini':
            colors.append(GPT_GREY_LIGHT)  # Medium grey for GPT-4.1-mini
        elif name == 'GPT-4.1-nano':
            colors.append(GPT_GREY_LIGHTER)  # Lightest grey for GPT-4.1-nano
        elif name == 'GPT-4o':
            colors.append('#d62728')  # Red for GPT-4o (only in large dataset)
        elif name == 'Prospect Theory':
            colors.append(BASE_COLORS['PT'])  # Teal for PT
        elif name == 'Expected Value':
            colors.append(BASE_COLORS['EV'])  # Blue for EV
        else:
            # Use family-based coloring for other models
            fam, sz = get_model_family_and_size(name)
            color = get_family_color(fam, sz, max_size_per_family[fam])
            colors.append(color)
    
    # Handle both valid and invalid (null) values for consistent plotting
    # Keep all rows but distinguish between valid and invalid data
    results_df_reversed = results_df.iloc[::-1].reset_index(drop=True)
    colors_reversed = colors[::-1]
    
    # Separate valid and invalid data
    valid_mask = pd.notna(results_df_reversed[metric]) & np.isfinite(results_df_reversed[metric])
    
    # For valid data, calculate 95% confidence intervals
    ci_95 = pd.Series(index=results_df_reversed.index, dtype=float)
    ci_95[valid_mask] = results_df_reversed.loc[valid_mask, f'{metric}_sem'] * 1.96
    ci_95[~valid_mask] = 0  # No error bars for null data
    
    # Create figure with appropriate size for publication
    fig, ax = plt.subplots(figsize=(4, len(results_df_reversed) * 0.3 + 1))
    
    y_pos = np.arange(len(results_df_reversed))
    
    # Calculate x-axis limits
    if x_limits is not None:
        x_min, x_max = x_limits
        x_range = max(x_max - x_min, 1e-6)
    else:
        # Dynamic x-axis limits based on valid data only
        valid_values = results_df_reversed.loc[valid_mask, metric]
        if len(valid_values) > 0:
            min_val = valid_values.min()
            max_val = valid_values.max()
            x_range = max_val - min_val
            x_min = max(min_val - 0.02, min_val - x_range * 0.1)  # Start slightly below worst performance
            x_max = max_val + x_range * 0.05  # Small padding above best performance
        else:
            x_min, x_max = 0.0, 1.0
            x_range = 1.0
    
    # Create horizontal bars - use 0 for null values, actual values for valid data
    bar_values = results_df_reversed[metric].fillna(0)
    bar_errors = ci_95.fillna(0)
    
    bars = ax.barh(
        y_pos,
        bar_values,
        height=0.6,
        xerr=bar_errors,
        capsize=2,
        color=colors_reversed,
        edgecolor='white',
        linewidth=0.5,
        error_kw={'linewidth': 0.5, 'capthick': 0.5}
    )
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df_reversed['display_name'], fontsize=8)
    
    # Add value labels at the end of bars (only for valid data)
    for i, (bar, value, ci) in enumerate(zip(bars, results_df_reversed[metric], ci_95)):
        if pd.notna(value) and np.isfinite(value):
            ax.text(
                value + ci + x_range * 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{value:.3f}',
                ha='left',
                va='center',
                fontsize=7,
                color='#333333'
            )
    
    # Set labels (no title - user will add their own)
    ax.set_xlabel(metric.capitalize(), fontsize=9, color='#333333')
    
    # Set x-axis limits and aligned ticks
    ax.set_xlim(x_min, x_max)
    try:
        if ticks is not None and len(ticks) > 0:
            ax.set_xticks(ticks)
        else:
            num_ticks = 6
            ax.set_xticks(np.linspace(x_min, x_max, num=num_ticks))
    except Exception:
        pass
    
    # Remove grid and set clean background
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Invert y-axis to have best performance at top
    ax.invert_yaxis()
    
    # Fine-tune layout
    plt.tight_layout()
    
    # Save with high quality settings
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save high-quality vector PDF
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(save_path_pdf, dpi=450, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Reset matplotlib parameters to default
    plt.rcParams.update(plt.rcParamsDefault)
    return float(x_min), float(x_max)

def sanitize_filename(name):
    return re.sub(r'[^\w\-\.]+', '_', name)

def create_llm_mode_comparison_plots(llm_results: Dict, data_size: str):
    # For each LLM, plot accuracy for base, cot, human modes
    for model_name, data in llm_results.items():
        df = pd.DataFrame(data['results'])
        if 'mode' not in df.columns:
            continue
        mode_accs = []
        for mode in ['base', 'cot', 'human']:
            mode_df = df[df['mode'] == mode]
            if len(mode_df) == 0:
                continue
            acc = (mode_df['extracted_choice'] == mode_df['actual_choice']).mean()
            sem = (mode_df['extracted_choice'] == mode_df['actual_choice']).sem()
            mode_accs.append({'mode': mode, 'accuracy': acc, 'accuracy_sem': sem})
        if not mode_accs:
            continue
        
        plot_df = pd.DataFrame(mode_accs)
        
        # Set publication-quality style
        plt.rcParams.update({
            'font.size': 8,
            'axes.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333',
            'text.color': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333'
        })
        
        # Calculate 95% confidence intervals
        ci_95 = plot_df['accuracy_sem'] * 1.96
        
        # Calculate dynamic y-axis limits
        min_val = plot_df['accuracy'].min()
        max_val = plot_df['accuracy'].max()
        y_range = max_val - min_val
        y_min = max(min_val - 0.02, min_val - y_range * 0.1)
        y_max = max_val + y_range * 0.05
        
        fig, ax = plt.subplots(figsize=(3, 2.5))
        colors = ['#888888', '#4f9dd9', '#222222']  # base, cot, human
        
        bars = ax.bar(
            plot_df['mode'], 
            plot_df['accuracy'], 
            yerr=ci_95, 
            color=colors, 
            capsize=2,
            edgecolor='white',
            linewidth=0.5,
            error_kw={'linewidth': 0.5, 'capthick': 0.5}
        )
        
        # Add value labels on top of bars
        for bar, value, ci in zip(bars, plot_df['accuracy'], ci_95):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    value + ci + y_range * 0.01, 
                    f'{value:.3f}', 
                    ha='center', 
                    va='bottom',
                    fontsize=7,
                    color='#333333'
                )
        
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('Accuracy', fontsize=9, color='#333333')
        ax.set_title(f"{MODEL_NAME_MAP.get(model_name, model_name)} ({data_size.title()})", 
                     fontsize=10, color='#333333', pad=10)
        ax.grid(False)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        os.makedirs('figures/llm_mode_comparison', exist_ok=True)
        safe_name = sanitize_filename(MODEL_NAME_MAP.get(model_name, model_name))
        # Save high-quality vector PDF
        out_pdf = f'figures/llm_mode_comparison/{safe_name}_{data_size}_mode_comparison.pdf'
        plt.savefig(out_pdf, dpi=450, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Reset matplotlib parameters to default
        plt.rcParams.update(plt.rcParamsDefault)


def _bh_adjust(pvals: List[float]) -> List[float]:
    """Benjamini–Hochberg FDR correction.

    Steps:
    1) Sort p-values ascending, keep original indices.
    2) Compute q_i = p_(i) * m / i on the sorted list (i = 1..m).
    3) Take the cumulative minimum from the largest rank to the smallest.
    4) Map adjusted values back to the original order and clamp to [0, 1].
    """
    m = len(pvals)
    if m == 0:
        return []

    # Sort by p-value ascending, keeping original indices
    ranked = sorted([(p, i) for i, p in enumerate(pvals)], key=lambda x: x[0])

    # Compute BH scaled values in sorted order
    scaled: List[float] = [0.0] * m
    for rank, (p, _) in enumerate(ranked, start=1):
        scaled[rank - 1] = (p * m) / rank

    # Enforce monotonicity: cumulative min from the end
    cummin = 1.0
    adj_sorted: List[float] = [0.0] * m
    for i in range(m - 1, -1, -1):
        cummin = min(cummin, scaled[i])
        adj_sorted[i] = min(cummin, 1.0)

    # Map back to original order
    adjusted = [0.0] * m
    for (p, idx), adj_val in zip(ranked, adj_sorted):
        adjusted[idx] = adj_val

    return adjusted


def build_participant_metrics(data_size: str) -> Dict[str, pd.DataFrame]:
    """Return model -> DataFrame(sub_id, acc, like) for all models in this dataset."""
    symbolic_results, NN_results = load_other_model_results(data_size)
    model_to_df: Dict[str, pd.DataFrame] = {}

    # Cognitive models (participant-level present in JSON)
    for model_name, model_data in symbolic_results.items():
        try:
            if data_size == 'small':
                last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
                entries = []
                for pid, pdata in model_data[last_round].items():
                    acc = float(pdata['test_accuracy'])
                    like = float(math.exp(-pdata['test_loss'])) if 'test_loss' in pdata else float('nan')
                    entries.append({'sub_id': int(pid) if str(pid).isdigit() else pid, 'acc': acc, 'like': like})
                model_to_df[model_name] = pd.DataFrame(entries)
            else:
                last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
                last_step = max(model_data[last_round].keys(), key=lambda x: int(x.split('_')[1]))
                step_data = model_data[last_round][last_step]
                # Aggregate per participant across samples
                pid_to_vals: Dict[str, List[Tuple[float, float]]] = {}
                for sample_data in step_data.values():
                    for pid, pdata in sample_data.items():
                        acc = float(pdata['test_accuracy'])
                        like = float(math.exp(-pdata['test_loss'])) if 'test_loss' in pdata else float('nan')
                        pid_to_vals.setdefault(pid, []).append((acc, like))
                entries = []
                for pid, lst in pid_to_vals.items():
                    accs = [a for a, _ in lst]
                    likes = [b for _, b in lst]
                    entries.append({'sub_id': int(pid) if str(pid).isdigit() else pid,
                                    'acc': float(np.nanmean(accs)),
                                    'like': float(np.nanmean(likes))})
                model_to_df[model_name] = pd.DataFrame(entries)
        except Exception:
            continue

    # NN models
    for model_name, model_data in NN_results.items():
        try:
            if data_size == 'small':
                last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
                metrics = model_data[last_round]
                preds = metrics.get('test_predictions', [])
                if preds:
                    dfp = pd.DataFrame(preds)
                    if not dfp.empty:
                        acc_by_sub = dfp.groupby('sub_id').apply(lambda x: (x['predicted_choice'] == x['actual_choice']).mean())
                        like_by_sub = None
                        if 'loss' in dfp.columns:
                            like_by_sub = dfp.groupby('sub_id')['loss'].mean().apply(lambda v: math.exp(-v))
                        model_to_df[model_name] = pd.DataFrame({
                            'sub_id': acc_by_sub.index,
                            'acc': acc_by_sub.values,
                            'like': like_by_sub.values if like_by_sub is not None else np.full(len(acc_by_sub), np.nan)
                        })
            else:
                last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
                last_step = max(model_data[last_round].keys(), key=lambda x: int(x.split('_')[1]))
                step_data = model_data[last_round][last_step]
                # Aggregate per participant across samples using predictions if available
                acc_map: Dict[int, List[float]] = {}
                like_map: Dict[int, List[float]] = {}
                for sample_metrics in step_data.values():
                    preds = sample_metrics.get('test_predictions', [])
                    if not preds:
                        continue
                    dfp = pd.DataFrame(preds)
                    if dfp.empty:
                        continue
                    acc_by_sub = dfp.groupby('sub_id').apply(lambda x: (x['predicted_choice'] == x['actual_choice']).mean())
                    for sid, val in acc_by_sub.items():
                        acc_map.setdefault(int(sid), []).append(float(val))
                    if 'loss' in dfp.columns:
                        like_by_sub = dfp.groupby('sub_id')['loss'].mean().apply(lambda v: math.exp(-v))
                        for sid, val in like_by_sub.items():
                            like_map.setdefault(int(sid), []).append(float(val))
                entries = []
                for sid, accs in acc_map.items():
                    likes = like_map.get(sid, [])
                    entries.append({'sub_id': sid,
                                    'acc': float(np.nanmean(accs)) if accs else np.nan,
                                    'like': float(np.nanmean(likes)) if likes else np.nan})
                if entries:
                    model_to_df[model_name] = pd.DataFrame(entries)
        except Exception:
            continue

    # LLM models (from exp1_llm_prediction files)
    llm_results = load_llm_results(data_size)
    for model_name, data in llm_results.items():
        try:
            df = pd.DataFrame(data['results'])
            if 'mode' in df.columns:
                df = df[df['mode'] == 'human']
            if df.empty or 'sub_id' not in df.columns:
                continue
            df = df.copy()
            df['acc'] = (df['extracted_choice'] == df['actual_choice']).astype(float)
            if {'probability_option_a', 'probability_option_b', 'actual_choice'}.issubset(df.columns):
                df['like'] = df.apply(lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'], axis=1)
            else:
                df['like'] = np.nan
            acc_by_sub = df.groupby('sub_id')['acc'].mean()
            like_by_sub = df.groupby('sub_id')['like'].mean()
            model_to_df[model_name] = pd.DataFrame({
                'sub_id': acc_by_sub.index,
                'acc': acc_by_sub.values,
                'like': like_by_sub.values
            })
        except Exception:
            continue

    # Human (small dataset only): include per-participant metrics for t-tests
    if data_size == 'small':
        try:
            human_records = load_human_results(data_size)
            if isinstance(human_records, list) and len(human_records) > 0:
                hdf = pd.DataFrame(human_records)
                if {'sub_id', 'test_accuracy', 'test_likelihood'}.issubset(hdf.columns):
                    model_to_df['Human'] = pd.DataFrame({
                        'sub_id': hdf['sub_id'],
                        'acc': hdf['test_accuracy'].astype(float),
                        'like': hdf['test_likelihood'].astype(float)
                    })
        except Exception:
            pass

    return model_to_df


def compute_ttest_table(model_to_df: Dict[str, pd.DataFrame], target_model: str, metric: str) -> pd.DataFrame:
    """Paired t-tests: target_model vs each other model at participant level.
    metric in {'acc','like'}
    """
    rows = []
    target_df = model_to_df.get(target_model)
    if target_df is None or target_df.empty:
        return pd.DataFrame(columns=['compare_to', 'n', 't', 'df', 'p', 'p_adj', 'cohen_d'])
    for model_name, df in model_to_df.items():
        if model_name == target_model:
            continue
        merged = pd.merge(target_df[['sub_id', metric]].rename(columns={metric: 'target'}),
                          df[['sub_id', metric]].rename(columns={metric: 'other'}),
                          on='sub_id', how='inner')
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
        n = len(merged)
        if n < 2:
            continue
        a = merged['target'].astype(float).values
        b = merged['other'].astype(float).values
        # Paired t-test
        t_stat, p_val = ttest_rel(a, b, nan_policy='omit')
        dfree = n - 1
        diff = a - b
        sd_diff = np.nanstd(diff, ddof=1)
        cohen_d = float(np.nanmean(diff) / sd_diff) if sd_diff > 0 else float('nan')
        rows.append({'compare_to': model_name, 'n': int(n), 't': float(t_stat), 'df': int(dfree), 'p': float(p_val), 'cohen_d': cohen_d})
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    # BH adjust
    out_df = out_df.sort_values('p').reset_index(drop=True)
    out_df['p_adj'] = _bh_adjust(out_df['p'].tolist())
    return out_df

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze model comparison and optional LLaMA-70B control plots")
    parser.add_argument(
        "--permutation_only",
        action="store_true",
        help="Only plot LLaMA-3.1-70B original vs permuted grouped control figures (faster)"
    )
    parser.add_argument(
        "--data_sizes",
        nargs='+',
        choices=["small", "large"],
        default=["small", "large"],
        help="Dataset sizes to include (when using --permutation_only or for full analysis)"
    )
    parser.add_argument(
        "--action_removal_llama_plots",
        action="store_true",
        help="Also generate extra LLaMA action_removal grouped control figures using *_action_removal_results_LLaMA.json",
    )
    parser.add_argument(
        "--action_removal_results_filename",
        type=str,
        default=None,
        help=(
            "Override the exact action_removal results filename (under results/exp1_llm_prediction/) "
            "to generate an additional grouped control plot set."
        ),
    )
    parser.add_argument(
        "--action_removal_plot_suffix",
        type=str,
        default=None,
        help="Optional suffix for output filenames (e.g. _action_removal_GPT-5.4-nano).",
    )
    parser.add_argument(
        "--action_removal_results_filename_small",
        type=str,
        default=None,
        help="Optional small-dataset-specific action_removal filename override.",
    )
    parser.add_argument(
        "--action_removal_results_filename_large",
        type=str,
        default=None,
        help="Optional large-dataset-specific action_removal filename override.",
    )
    args = parser.parse_args()

    def _derive_suffix_from_filename(fname: str) -> str:
        base = os.path.basename(fname)
        # Expected: ..._action_removal_results-<token>.json
        token = base.split("action_removal_results")[-1]
        token = token.lstrip("_-")
        token = token.replace(".json", "")
        return f"_action_removal_{token}"

    if args.permutation_only:
        global PERM_LOG
        PERM_LOG = True
        print(f"[perm] Running in permutation-only mode for sizes: {args.data_sizes}")
        # Fast path: only control/permutation grouped plots for LLaMA-70B
        plot_llama70b_control_grouped(args.data_sizes)
        if args.action_removal_results_filename:
            suffix = args.action_removal_plot_suffix or _derive_suffix_from_filename(
                args.action_removal_results_filename
            )
            plot_llama70b_control_grouped(
                args.data_sizes,
                output_suffix=suffix,
                action_removal_results_filename_override=args.action_removal_results_filename,
                action_removal_results_filename_small=args.action_removal_results_filename_small,
                action_removal_results_filename_large=args.action_removal_results_filename_large,
            )
        elif args.action_removal_results_filename_small or args.action_removal_results_filename_large:
            suffix = args.action_removal_plot_suffix or "_action_removal_custom"
            plot_llama70b_control_grouped(
                args.data_sizes,
                output_suffix=suffix,
                action_removal_results_filename_small=args.action_removal_results_filename_small,
                action_removal_results_filename_large=args.action_removal_results_filename_large,
            )
        elif args.action_removal_llama_plots:
            plot_llama70b_control_grouped(
                args.data_sizes,
                output_suffix="_action_removal_LLaMA",
                action_removal_llama_filename=True,
            )
        return

    # Process results for requested datasets
    small_x_limits: dict[str, Tuple[float, float]] = {}
    default_ticks = [0.5, 0.6, 0.7, 0.8]
    padding = 0.02
    for data_size in args.data_sizes:
        print(f"\nProcessing {data_size} dataset...")
        
        # Load and process LLM results
        llm_results = load_llm_results(data_size)
        llm_df = process_llm_results(llm_results)
        
        # Load and process other models
        other_df = process_other_models(data_size)
        
        # Load and process human results for small dataset
        human_data = load_human_results(data_size)
        human_df = process_human_results(human_data)
        
        # Combine results
        results_df = pd.concat([other_df, llm_df, human_df], ignore_index=True)
        
        # Create plots
        # Fix ticks to nice values and pad limits slightly beyond shown ticks
        acc_ticks = default_ticks
        acc_limits_input = (acc_ticks[0] - padding, acc_ticks[-1] + padding)
        acc_limits = create_comparison_plot(
            results_df,
            'accuracy',
            data_size,
            f'figures/model_comparison_accuracy_{data_size}.png',
            x_limits=acc_limits_input,
            ticks=acc_ticks,
        )
        
        like_ticks = default_ticks
        like_limits_input = (like_ticks[0] - padding, like_ticks[-1] + padding)
        like_limits = create_comparison_plot(
            results_df,
            'likelihood',
            data_size,
            f'figures/model_comparison_likelihood_{data_size}.png',
            x_limits=like_limits_input,
            ticks=like_ticks,
        )
        
        # Save results to CSV
        results_df.to_csv(
            f'results/model_comparison_{data_size}.csv',
            index=False
        )
        
        print(f"Results saved for {data_size} dataset")

        # Cache small dataset x-limits for reuse on large (kept for completeness)
        if data_size == 'small':
            small_x_limits['accuracy'] = acc_limits
            small_x_limits['likelihood'] = like_limits

        # Participant-level t-tests: LLaMA-3.1-70B vs others (including Human for small dataset);
        # save per dataset, per metric
        try:
            llama_key = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
            ttest_dir = 'results/statistical_tests'
            os.makedirs(ttest_dir, exist_ok=True)

            model_to_df = build_participant_metrics(data_size)
            if llama_key in model_to_df:
                acc_table = compute_ttest_table(model_to_df, llama_key, metric='acc')
                acc_out = os.path.join(ttest_dir, f'ttests_{data_size}_accuracy.csv')
                acc_table.to_csv(acc_out, index=False)
                print(f"Saved t-tests (accuracy) to {acc_out}")

                like_table = compute_ttest_table(model_to_df, llama_key, metric='like')
                like_out = os.path.join(ttest_dir, f'ttests_{data_size}_likelihood.csv')
                like_table.to_csv(like_out, index=False)
                print(f"Saved t-tests (likelihood) to {like_out}")

                # Additionally, for small dataset, save explicit LLaMA vs Human comparisons
                if data_size == 'small':
                    try:
                        acc_h = acc_table[acc_table['compare_to'] == 'Human']
                        like_h = like_table[like_table['compare_to'] == 'Human']
                        if not acc_h.empty or not like_h.empty:
                            rows = []
                            if not acc_h.empty:
                                r = acc_h.iloc[0].to_dict()
                                r['metric'] = 'accuracy'
                                rows.append(r)
                            if not like_h.empty:
                                r = like_h.iloc[0].to_dict()
                                r['metric'] = 'likelihood'
                                rows.append(r)
                            if rows:
                                llama_vs_human_out = os.path.join(ttest_dir, 'ttests_small_llama70b_vs_human.csv')
                                pd.DataFrame(rows)[['metric','n','t','df','p','p_adj','cohen_d']].to_csv(llama_vs_human_out, index=False)
                                print(f"Saved LLaMA-70B vs Human t-tests to {llama_vs_human_out}")
                        else:
                            print("Note: No Human comparison row found in t-test tables for small dataset.")
                    except Exception as e2:
                        print(f"Failed to save LLaMA vs Human comparison: {e2}")
            else:
                print(f"Warning: LLaMA-3.1-70B key '{llama_key}' not found for participant-level tests in {data_size}.")
        except Exception as e:
            print(f"Participant-level t-tests failed for {data_size}: {e}")

        # Create LLM mode comparison plots
        create_llm_mode_comparison_plots(llm_results, data_size)

    # After the standard comparisons, create grouped control plots for LLaMA-3.1-70B
    plot_llama70b_control_grouped(args.data_sizes)
    if args.action_removal_results_filename:
        suffix = args.action_removal_plot_suffix or _derive_suffix_from_filename(
            args.action_removal_results_filename
        )
        plot_llama70b_control_grouped(
            args.data_sizes,
            output_suffix=suffix,
            action_removal_results_filename_override=args.action_removal_results_filename,
            action_removal_results_filename_small=args.action_removal_results_filename_small,
            action_removal_results_filename_large=args.action_removal_results_filename_large,
        )
    elif args.action_removal_results_filename_small or args.action_removal_results_filename_large:
        suffix = args.action_removal_plot_suffix or "_action_removal_custom"
        plot_llama70b_control_grouped(
            args.data_sizes,
            output_suffix=suffix,
            action_removal_results_filename_small=args.action_removal_results_filename_small,
            action_removal_results_filename_large=args.action_removal_results_filename_large,
        )
    elif args.action_removal_llama_plots:
        plot_llama70b_control_grouped(
            args.data_sizes,
            output_suffix="_action_removal_LLaMA",
            action_removal_llama_filename=True,
        )

if __name__ == "__main__":
    main() 