"""
Analysis for Experiment 2: In-Context Generalization
This script analyzes and visualizes the results from the in-context generalization experiment.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
import colorsys
from matplotlib.colors import to_rgba, to_hex
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import re
import math
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, ttest_rel
from scipy.stats import spearmanr
from scipy.stats import beta
import warnings
import argparse

from utils.data_processing import load_results
from utils.visualization import (
    plot_in_context_generalization,
    save_figure
)
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
    'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct',
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
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'Llama-3-8B',
    'meta-llama/Meta-Llama-3.1-70B-Instruct': 'Llama-3-70B',
    'Qwen/Qwen2.5-3B-Instruct': 'Qwen-2.5-3B',
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen-2.5-7B',
    'Qwen/Qwen2.5-14B-Instruct': 'Qwen-2.5-14B',
    'Qwen/Qwen2.5-32B-Instruct': 'Qwen-2.5-32B',
    'gpt-4.1-nano': 'gpt-4.1-nano',
    'gpt-4.1-mini': 'gpt-4.1-mini',
    'gpt-4.1': 'gpt-4.1',
    'gpt-4o': 'gpt-4o',
    'Human': 'Human Prediction'
}

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

# Type colors for the new analysis structure
TYPE_COLORS = {
    'think_aloud': '#1f77b4',  # Blue
    'choice': '#ff7f0e',       # Orange  
    'both': '#2ca02c'          # Green
}

# Exclude specific num_examples values globally from visualization and analysis
EXCLUDED_NUM_EXAMPLES = {15}

def load_in_context_results(data_size: str, control_mode: bool = False) -> Dict:
    """
    Load in-context learning results with new structure supporting multiple samples.
    Handles cases where both single-mode and 'all' mode files exist - prioritizes single-mode data.
    
    Args:
        data_size: Size of dataset used ('small' or 'large')
        control_mode: If True, load control experiment results
        
    Returns:
        Dictionary with results for each (num_examples, example_type) combination
    """
    results_dir = "results/exp2_in_context"
    
    # First pass: collect file information
    file_info = []
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json') or 'checkpoint' in filename:
            continue
            
        # Filter for control mode files
        is_control_file = '_control' in filename
        if control_mode != is_control_file:
            continue
            
        try:
            with open(os.path.join(results_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Get dataset size, number of examples, and example_type from metadata
            file_data_size = data['metadata'].get('dataset') or data['metadata'].get('data_size')
            num_examples = data['metadata'].get('num_examples')
            example_type = data['metadata'].get('example_type', 'think_aloud')
            n_samples = data['metadata'].get('n_samples', 1)
            file_control_mode = data['metadata'].get('control_mode', False)
            
            if file_data_size != data_size or num_examples is None:
                continue
                
            # Double-check control mode matches
            if control_mode != file_control_mode:
                continue
            
            # Check what modes are present in this file
            modes_in_file = set()
            for result in data['results']:
                modes_in_file.add(result['mode'])
            
            file_info.append({
                'filename': filename,
                'data': data,
                'num_examples': num_examples,
                'example_type': example_type,
                'n_samples': n_samples,
                'modes': modes_in_file,
                'control_mode': file_control_mode
            })
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    # Second pass: organize by configuration and detect conflicts
    config_files = {}  # (num_examples, example_type) -> list of file_info
    for info in file_info:
        key = (info['num_examples'], info['example_type'])
        if key not in config_files:
            config_files[key] = []
        config_files[key].append(info)
    
    # Third pass: load data with priority logic
    in_context_results = {}
    for key, files in config_files.items():
        num_examples, example_type = key
        
        if len(files) == 1:
            # Only one file, use it directly
            file_info = files[0]
            mode_suffix = "control" if control_mode else "original"
            print(f"Loading {mode_suffix} file: {file_info['filename']}")
            print(f"  - Dataset: {data_size}, Examples: {num_examples}, Type: {example_type}, Samples: {file_info['n_samples']}")
            print(f"  - Control mode: {file_info['control_mode']}")
            print(f"  - Modes: {sorted(file_info['modes'])}")
            print(f"  - Total results in file: {len(file_info['data']['results'])}")
            
            if key not in in_context_results:
                in_context_results[key] = []
            in_context_results[key].extend(file_info['data']['results'])
        else:
            # Multiple files - need to prioritize single-mode over all-mode
            mode_suffix = "control" if control_mode else "original"
            print(f"\nMultiple {mode_suffix} files found for {num_examples} examples, {example_type} type:")
            
            # Separate single-mode files from multi-mode files
            single_mode_files = [f for f in files if len(f['modes']) == 1]
            multi_mode_files = [f for f in files if len(f['modes']) > 1]
            
            # Build mode-to-file mapping, prioritizing single-mode files
            mode_to_file = {}
            
            # First, add multi-mode files (lower priority)
            for file_info in multi_mode_files:
                for mode in file_info['modes']:
                    if mode not in mode_to_file:
                        mode_to_file[mode] = file_info
                print(f"  - Multi-mode file: {file_info['filename']} (modes: {sorted(file_info['modes'])})")
            
            # Then, add single-mode files (higher priority - will overwrite multi-mode)
            for file_info in single_mode_files:
                for mode in file_info['modes']:
                    mode_to_file[mode] = file_info
                print(f"  - Single-mode file: {file_info['filename']} (mode: {list(file_info['modes'])[0]})")
            
            # Load results from the prioritized files
            if key not in in_context_results:
                in_context_results[key] = []
            
            # Track which files we've used and what results we've taken from them
            used_files = {}
            for mode, file_info in mode_to_file.items():
                # Get results for this specific mode from this file
                mode_results = [result for result in file_info['data']['results'] if result['mode'] == mode]
                
                if file_info['filename'] not in used_files:
                    used_files[file_info['filename']] = []
                used_files[file_info['filename']].extend(mode_results)
                
                in_context_results[key].extend(mode_results)
            
            # Print summary of what was used from each file
            for filename, results in used_files.items():
                modes_used = sorted(set(result['mode'] for result in results))
                print(f"  - Using {len(results)} results from {filename} (modes: {modes_used})")
            
            print(f"  - Total combined results: {len(in_context_results[key])}")
    
    return in_context_results

def load_control_comparison_data(data_size: str) -> Tuple[Dict, Dict]:
    """
    Load both original and control results for comparison.
    
    Args:
        data_size: Size of dataset used ('small' or 'large')
        
    Returns:
        Tuple of (original_results, control_results)
    """
    print(f"Loading original results for {data_size} dataset...")
    original_results = load_in_context_results(data_size, control_mode=False)
    
    print(f"Loading control results for {data_size} dataset...")
    control_results = load_in_context_results(data_size, control_mode=True)
    
    return original_results, control_results

def process_control_comparison_results(original_results: Dict, control_results: Dict) -> pd.DataFrame:
    """
    Process both original and control results for comparison.
    
    Args:
        original_results: Dictionary of original results
        control_results: Dictionary of control results
        
    Returns:
        DataFrame with processed results for both conditions
    """
    all_rows = []
    
    # Process original results
    for (num_examples, example_type), results in original_results.items():
        if int(num_examples) in EXCLUDED_NUM_EXAMPLES:
            continue
        df = pd.DataFrame(results)
        
        print(f"\nProcessing original {num_examples} examples, {example_type} type:")
        print(f"  Total trials: {len(df)}")
        
        # Check if sample_id exists (for multi-sample experiments)
        has_samples = 'sample_id' in df.columns
        if has_samples:
            unique_samples = sorted(df['sample_id'].unique())
            print(f"  Sample IDs found: {unique_samples}")
        else:
            print(f"  No sample_id found - treating as single sample")
        
        # Group by mode (only within_individual for control comparison)
        for mode in df['mode'].unique():
            if 'within_individual' not in mode:
                continue  # Only process within_individual for control comparison
            # Only base mode
            if not str(mode).startswith('base_') and str(mode) != 'base_within_individual':
                continue
                
            mode_df = df[df['mode'] == mode].copy()
            if 'num_examples' in mode_df.columns:
                mode_df = mode_df[~mode_df['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
            print(f"  Mode {mode}: {len(mode_df)} trials")
            
            if has_samples:
                # Multiple samples: compute participant-level means across samples, then CI across participants
                # Compute per (sub_id, sample_id) metrics
                tmp = mode_df.copy()
                tmp['acc'] = (tmp['extracted_choice'] == tmp['actual_choice']).astype(float)
                tmp['likelihood'] = tmp.apply(
                        lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                        axis=1
                    )
                acc_per_ps = tmp.groupby(['sub_id', 'sample_id'], as_index=False)['acc'].mean()
                like_per_ps = tmp.groupby(['sub_id', 'sample_id'], as_index=False)['likelihood'].mean()

                # Average across samples per participant
                acc_per_sub = acc_per_ps.groupby('sub_id', as_index=False)['acc'].mean()
                like_per_sub = like_per_ps.groupby('sub_id', as_index=False)['likelihood'].mean()

                # Aggregate across participants
                mean_accuracy = float(acc_per_sub['acc'].mean()) if not acc_per_sub.empty else float('nan')
                sem_accuracy = float(acc_per_sub['acc'].std(ddof=1) / np.sqrt(len(acc_per_sub))) if len(acc_per_sub) > 1 else 0.0
                mean_likelihood = float(like_per_sub['likelihood'].mean()) if not like_per_sub.empty else float('nan')
                sem_likelihood = float(like_per_sub['likelihood'].std(ddof=1) / np.sqrt(len(like_per_sub))) if len(like_per_sub) > 1 else 0.0

                print(f"    Participants: {len(acc_per_sub)}, Accuracy: {mean_accuracy:.4f}±{sem_accuracy:.4f}, Likelihood: {mean_likelihood:.4f}±{sem_likelihood:.4f}")
                
                all_rows.append({
                    'num_examples': num_examples,
                    'example_type': example_type,
                    'mode': mode,
                    'condition': 'original',
                    'accuracy': mean_accuracy,
                    'accuracy_sem': sem_accuracy,
                    'likelihood': mean_likelihood,
                    'likelihood_sem': sem_likelihood,
                    'participant_accuracies': acc_per_sub['acc'].tolist(),
                    'participant_likelihoods': like_per_sub['likelihood'].tolist(),
                    'n_samples': len(mode_df['sample_id'].unique())
                })
            else:
                # Single sample: use original logic
                participant_accuracy = mode_df.groupby('sub_id', group_keys=False)[['extracted_choice', 'actual_choice']].apply(
                    lambda x: (x['extracted_choice'] == x['actual_choice']).mean()
                )
                # Calculate likelihood for each trial
                mode_df['likelihood'] = mode_df.apply(
                    lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                    axis=1
                )
                # Calculate mean likelihood per participant
                participant_likelihood = mode_df.groupby('sub_id', group_keys=False)['likelihood'].mean()
                
                print(f"    Single sample - Accuracy: {participant_accuracy.mean():.4f}±{participant_accuracy.sem():.4f}, Likelihood: {participant_likelihood.mean():.4f}±{participant_likelihood.sem():.4f}")
                
                all_rows.append({
                    'num_examples': num_examples,
                    'example_type': example_type,
                    'mode': mode,
                    'condition': 'original',
                    'accuracy': participant_accuracy.mean(),
                    'accuracy_sem': participant_accuracy.sem(),
                    'likelihood': participant_likelihood.mean(),
                    'likelihood_sem': participant_likelihood.sem(),
                    'participant_accuracies': participant_accuracy.tolist(),
                    'participant_likelihoods': participant_likelihood.tolist(),
                    'n_samples': 1
                })
    
    # Process control results
    for (num_examples, example_type), results in control_results.items():
        if int(num_examples) in EXCLUDED_NUM_EXAMPLES:
            continue
        df = pd.DataFrame(results)
        
        print(f"\nProcessing control {num_examples} examples, {example_type} type:")
        print(f"  Total trials: {len(df)}")
        
        # Check if sample_id exists (for multi-sample experiments)
        has_samples = 'sample_id' in df.columns
        if has_samples:
            unique_samples = sorted(df['sample_id'].unique())
            print(f"  Sample IDs found: {unique_samples}")
        else:
            print(f"  No sample_id found - treating as single sample")
        
        # Group by mode (only within_individual for control comparison)
        for mode in df['mode'].unique():
            if 'within_individual' not in mode:
                continue  # Only process within_individual for control comparison
            # Only base mode
            if not str(mode).startswith('base_') and str(mode) != 'base_within_individual':
                continue
                
            mode_df = df[df['mode'] == mode].copy()
            if 'num_examples' in mode_df.columns:
                mode_df = mode_df[~mode_df['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
            print(f"  Mode {mode}: {len(mode_df)} trials")
            
            if has_samples:
                # Multiple samples: compute participant-level means across samples, then CI across participants
                tmp = mode_df.copy()
                tmp['acc'] = (tmp['extracted_choice'] == tmp['actual_choice']).astype(float)
                tmp['likelihood'] = tmp.apply(
                        lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                        axis=1
                    )
                acc_per_ps = tmp.groupby(['sub_id', 'sample_id'], as_index=False)['acc'].mean()
                like_per_ps = tmp.groupby(['sub_id', 'sample_id'], as_index=False)['likelihood'].mean()

                acc_per_sub = acc_per_ps.groupby('sub_id', as_index=False)['acc'].mean()
                like_per_sub = like_per_ps.groupby('sub_id', as_index=False)['likelihood'].mean()

                mean_accuracy = float(acc_per_sub['acc'].mean()) if not acc_per_sub.empty else float('nan')
                sem_accuracy = float(acc_per_sub['acc'].std(ddof=1) / np.sqrt(len(acc_per_sub))) if len(acc_per_sub) > 1 else 0.0
                mean_likelihood = float(like_per_sub['likelihood'].mean()) if not like_per_sub.empty else float('nan')
                sem_likelihood = float(like_per_sub['likelihood'].std(ddof=1) / np.sqrt(len(like_per_sub))) if len(like_per_sub) > 1 else 0.0

                print(f"    Participants: {len(acc_per_sub)}, Accuracy: {mean_accuracy:.4f}±{sem_accuracy:.4f}, Likelihood: {mean_likelihood:.4f}±{sem_likelihood:.4f}")
                
                all_rows.append({
                    'num_examples': num_examples,
                    'example_type': example_type,
                    'mode': mode,
                    'condition': 'control',
                    'accuracy': mean_accuracy,
                    'accuracy_sem': sem_accuracy,
                    'likelihood': mean_likelihood,
                    'likelihood_sem': sem_likelihood,
                    'participant_accuracies': acc_per_sub['acc'].tolist(),
                    'participant_likelihoods': like_per_sub['likelihood'].tolist(),
                    'n_samples': len(mode_df['sample_id'].unique())
                })
            else:
                # Single sample: use original logic
                participant_accuracy = mode_df.groupby('sub_id', group_keys=False)[['extracted_choice', 'actual_choice']].apply(
                    lambda x: (x['extracted_choice'] == x['actual_choice']).mean()
                )
                # Calculate likelihood for each trial
                mode_df['likelihood'] = mode_df.apply(
                    lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                    axis=1
                )
                # Calculate mean likelihood per participant
                participant_likelihood = mode_df.groupby('sub_id', group_keys=False)['likelihood'].mean()
                
                print(f"    Single sample - Accuracy: {participant_accuracy.mean():.4f}±{participant_accuracy.sem():.4f}, Likelihood: {participant_likelihood.mean():.4f}±{participant_likelihood.sem():.4f}")
                
                all_rows.append({
                    'num_examples': num_examples,
                    'example_type': example_type,
                    'mode': mode,
                    'condition': 'control',
                    'accuracy': participant_accuracy.mean(),
                    'accuracy_sem': participant_accuracy.sem(),
                    'likelihood': participant_likelihood.mean(),
                    'likelihood_sem': participant_likelihood.sem(),
                    'participant_accuracies': participant_accuracy.tolist(),
                    'participant_likelihoods': participant_likelihood.tolist(),
                    'n_samples': 1
                })
    
    return pd.DataFrame(all_rows)

def perform_control_statistical_tests(
    original_results: Dict,
    control_results: Dict,
    data_size: str
) -> pd.DataFrame:
    """
    Perform statistical tests comparing original vs control results.
    
    Args:
        original_results: Dictionary of original results
        control_results: Dictionary of control results
        data_size: Dataset size for labeling
        
    Returns:
        DataFrame with statistical test results
    """
    test_results = []
    
    # Process each configuration
    for (num_examples, example_type) in original_results.keys():
        if (num_examples, example_type) not in control_results or int(num_examples) in EXCLUDED_NUM_EXAMPLES:
            print(f"Warning: No control results found for {num_examples} examples, {example_type} type")
            continue
            
        orig_df = pd.DataFrame(original_results[(num_examples, example_type)])
        ctrl_df = pd.DataFrame(control_results[(num_examples, example_type)])
        
        # Process each mode (only within_individual for control comparison)
        for mode in orig_df['mode'].unique():
            if 'within_individual' not in mode:
                continue
            # Only base mode
            if not str(mode).startswith('base_') and str(mode) != 'base_within_individual':
                continue
                
            orig_mode_df = orig_df[orig_df['mode'] == mode].copy()
            ctrl_mode_df = ctrl_df[ctrl_df['mode'] == mode].copy()
            # Exclude rows with excluded num_examples
            for md in (orig_mode_df, ctrl_mode_df):
                if 'num_examples' in md.columns:
                    md.drop(md[md['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)].index, inplace=True)
            
            if len(orig_mode_df) == 0 or len(ctrl_mode_df) == 0:
                continue
            
            # Get participant-level data for both conditions
            for metric in ['accuracy', 'likelihood']:
                # Original condition
                if 'sample_id' in orig_mode_df.columns:
                    # Multi-sample: compute participant-level means across samples (not per-sample)
                    tmp = orig_mode_df.copy()
                    if metric == 'accuracy':
                        tmp['metric_val'] = (tmp['extracted_choice'] == tmp['actual_choice']).astype(float)
                    else:
                        tmp['metric_val'] = tmp.apply(
                                lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                                axis=1
                            )
                    # First average within (sub_id, sample_id), then average across samples for each participant
                    per_ps = tmp.groupby(['sub_id', 'sample_id'], as_index=False)['metric_val'].mean()
                    per_participant = per_ps.groupby('sub_id', as_index=False)['metric_val'].mean()
                    orig_participant_metrics = per_participant['metric_val'].astype(float).tolist()
                else:
                    # Single sample: aggregate at participant level
                    orig_participant_metrics = []
                    for sub_id, group in orig_mode_df.groupby('sub_id'):
                        if metric == 'accuracy':
                            value = (group['extracted_choice'] == group['actual_choice']).mean()
                        else:  # likelihood
                            group_copy = group.copy()
                            group_copy['likelihood'] = group_copy.apply(
                                lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                                axis=1
                            )
                            value = group_copy['likelihood'].mean()
                        orig_participant_metrics.append(value)
                
                # Control condition
                if 'sample_id' in ctrl_mode_df.columns:
                    # Multi-sample: compute participant-level means across samples (not per-sample)
                    tmp = ctrl_mode_df.copy()
                    if metric == 'accuracy':
                        tmp['metric_val'] = (tmp['extracted_choice'] == tmp['actual_choice']).astype(float)
                    else:
                        tmp['metric_val'] = tmp.apply(
                                lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                                axis=1
                            )
                    per_ps = tmp.groupby(['sub_id', 'sample_id'], as_index=False)['metric_val'].mean()
                    per_participant = per_ps.groupby('sub_id', as_index=False)['metric_val'].mean()
                    ctrl_participant_metrics = per_participant['metric_val'].astype(float).tolist()
                else:
                    # Single sample: aggregate at participant level
                    ctrl_participant_metrics = []
                    for sub_id, group in ctrl_mode_df.groupby('sub_id'):
                        if metric == 'accuracy':
                            value = (group['extracted_choice'] == group['actual_choice']).mean()
                        else:  # likelihood
                            group_copy = group.copy()
                            group_copy['likelihood'] = group_copy.apply(
                                lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                                axis=1
                            )
                            value = group_copy['likelihood'].mean()
                        ctrl_participant_metrics.append(value)
                
                # Convert to arrays and remove NaN values
                orig_data = np.array(orig_participant_metrics)
                ctrl_data = np.array(ctrl_participant_metrics)
                
                orig_data = orig_data[~np.isnan(orig_data)]
                ctrl_data = ctrl_data[~np.isnan(ctrl_data)]
                
                if len(orig_data) == 0 or len(ctrl_data) == 0:
                    continue
                
                # Participant-level n (for reference and CI)
                n_orig = int(len(orig_data))
                n_ctrl = int(len(ctrl_data))
                # Prefer paired t-test when arrays have same length (same participants)
                if n_orig == n_ctrl:
                    t_stat, p_value = ttest_rel(orig_data, ctrl_data)
                    df = n_orig - 1
                else:
                    t_stat, p_value = ttest_ind(orig_data, ctrl_data)
                    df = n_orig + n_ctrl - 2
                
                # Calculate effect size and CI
                cohens_d = calculate_cohens_d(orig_data, ctrl_data)
                ci_lower, ci_upper = calculate_effect_size_ci(orig_data, ctrl_data)
                
                test_results.append({
                    'data_size': data_size,
                    'num_examples': num_examples,
                    'example_type': example_type,
                    'mode': mode,
                    'metric': metric,
                    'original_mean': np.mean(orig_data),
                    'original_std': np.std(orig_data, ddof=1),
                    'original_n': n_orig,
                    'control_mean': np.mean(ctrl_data),
                    'control_std': np.std(ctrl_data, ddof=1),
                    'control_n': n_ctrl,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': df,
                    'cohens_d': cohens_d,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'significant': p_value < 0.05,
                    'analysis_level': 'participant'
                })
    
    df_out = pd.DataFrame(test_results)
    if len(df_out) > 0:
        # FDR (Benjamini-Hochberg) per family: per (example_type, metric) across num_examples
        df_out['p_adj'] = np.nan
        for (etype, m), grp in df_out.groupby(['example_type', 'metric']):
            idx = grp.index
            pvals = df_out.loc[idx, 'p_value'].astype(float).tolist()
            if pvals:
                adj = _bh_adjust_local(pvals)
                df_out.loc[idx, 'p_adj'] = adj
        df_out['significant_adj'] = df_out['p_adj'] < 0.05
    return df_out

def create_control_comparison_plot(
    comparison_df: pd.DataFrame,
    data_size: str,
    example_type: str,
    llm_mode: str,  # 'base' or 'cot'
    metric: str,  # 'accuracy' or 'likelihood'
    save_path: str
) -> None:
    """
    Create control comparison plot for a specific example type.
    Solid line for original, dashed line for control, with shaded error areas.
    Uses the same color from TYPE_COLORS for both lines.
    Also shows zero-shot baseline as a horizontal line.
    
    Args:
        comparison_df: DataFrame with both original and control results
        data_size: Dataset size ('small' or 'large')
        example_type: Type of examples ('think_aloud', 'choice', 'both')
        llm_mode: LLM mode ('base' or 'cot')
        metric: Metric to plot ('accuracy' or 'likelihood')
        save_path: Path to save the plot
    """
    plt.style.use(PLOT_STYLE)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data for this specific configuration
    mode_name = f"{llm_mode}_within_individual"
    plot_data = comparison_df[
        (comparison_df['example_type'] == example_type) & 
        (comparison_df['mode'] == mode_name)
    ]
    
    if plot_data.empty:
        print(f"No data found for {example_type}, {llm_mode} mode")
        plt.close()
        return
    
    # Get the color for this example type
    color = TYPE_COLORS[example_type]
    
    # Plot zero-shot baseline (load from LLM results)
    try:
        llm_results = load_llm_results(data_size)
        if llm_results:
            llm_df = process_llm_results(llm_results)
            baseline_data = llm_df[
                (llm_df['num_examples'] == 0) & 
                (llm_df['mode'] == llm_mode) & 
                (llm_df['example_type'] == 'baseline')
            ]
            if not baseline_data.empty:
                baseline_value = baseline_data[metric].values[0]
                ax.axhline(
                    y=baseline_value,
                    color='#8B4513',  # Brown color for baseline
                    linestyle=':',
                    label=f'{llm_mode} zero-shot baseline',
                    linewidth=2,
                    alpha=0.8
                )
    except Exception as e:
        print(f"Warning: Could not load baseline data: {e}")
    
    # Plot original results (solid line)
    orig_data = plot_data[plot_data['condition'] == 'original'].copy()
    if not orig_data.empty:
        orig_data = orig_data.sort_values('num_examples')
        xs = orig_data['num_examples'].values
        ys = orig_data[metric].values
        yerrs = orig_data[f'{metric}_sem'].values
        
        # Plot solid line with shaded error area
        ax.plot(xs, ys, '-', color=color, linewidth=2, label=f'Original ({example_type.replace("_", " ")})')
        ax.fill_between(xs, ys - yerrs, ys + yerrs, color=color, alpha=0.2)
    
    # Plot control results (dashed line)
    ctrl_data = plot_data[plot_data['condition'] == 'control'].copy()
    if not ctrl_data.empty:
        ctrl_data = ctrl_data.sort_values('num_examples')
        xs = ctrl_data['num_examples'].values
        ys = ctrl_data[metric].values
        yerrs = ctrl_data[f'{metric}_sem'].values
        
        # Plot dashed line with shaded error area
        ax.plot(xs, ys, '--', color=color, linewidth=2, label=f'Control ({example_type.replace("_", " ")})')
        ax.fill_between(xs, ys - yerrs, ys + yerrs, color=color, alpha=0.1)
    
    # Set axis properties
    all_num_examples = sorted(plot_data['num_examples'].unique())
    ax.set_xticks([0] + all_num_examples)  # Include 0 for baseline
    ax.set_xlim(-0.5, max(all_num_examples) + 0.5)
    
    ax.set_xlabel('Number of Examples')
    ax.set_ylabel(metric.capitalize())
    exp_label = 'Experiment 1 (PT design)' if data_size == 'small' else 'Experiment 2 (choice13k-sampled)'
    # Title intentionally omitted for flexibility
    # Dynamic y-limits based on plotted data to reduce blank space
    y_vals = []
    y_errs = []
    if 'baseline_value' in locals():
        y_vals.append(float(baseline_value))
        y_errs.append(0.0)
    if not orig_data.empty:
        y_vals.extend(orig_data[metric].astype(float).tolist())
        y_errs.extend(orig_data[f'{metric}_sem'].astype(float).tolist())
    if not ctrl_data.empty:
        y_vals.extend(ctrl_data[metric].astype(float).tolist())
        y_errs.extend(ctrl_data[f'{metric}_sem'].astype(float).tolist())
    if y_vals:
        y_vals_np = np.array(y_vals, dtype=float)
        y_errs_np = np.array(y_errs, dtype=float)
        lo = float(np.nanmin(y_vals_np - 1.96 * np.nan_to_num(y_errs_np)))
        hi = float(np.nanmax(y_vals_np + 1.96 * np.nan_to_num(y_errs_np)))
        pad = max(0.005, 0.05 * (hi - lo if hi > lo else 0.1))
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    
    # No grid / white background
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.legend()
    
    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    os.makedirs(out_dir, exist_ok=True)
    # Save high-quality vector PDF
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(save_path_pdf, dpi=450, bbox_inches='tight')
    # Also save to legacy filename (without context suffix) for compatibility
    # Only when callable context implies within_individual
    suffix = ''
    if 'noPT' in save_path:
        suffix = '_noPT'
    legacy_path = f"{out_dir}/two_panel_accuracy_{llm_mode}{suffix}.pdf"
    try:
        plt.savefig(legacy_path, dpi=450, bbox_inches='tight')
        print(f"Saved legacy two-panel figure to: {legacy_path}")
    except Exception:
        pass
    plt.close()


def create_three_panel_control_plot(
    comparison_df: pd.DataFrame,
    data_size: str,
    llm_mode: str,
    save_path: str,
    connect_lines: bool = False
):
    """Three-panel control figure per dataset and mode.
    Each panel compares original (dots with 95% CI) vs permuted (dashed line + shaded 95% CI)
    for one example_type: think_aloud, choice, both.
    """
    plt.style.use(PLOT_STYLE)
    example_types = ['think_aloud', 'choice', 'both']
    # Stack panels vertically to reduce horizontal whitespace
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.2), sharex=True, sharey=True)

    for ax, etype in zip(axes, example_types):
        mode_name = f"{llm_mode}_within_individual"
        plot_data = comparison_df[(comparison_df['example_type'] == etype) & (comparison_df['mode'] == mode_name)]
        if plot_data.empty:
            ax.set_visible(False)
            continue
        # Original (dots, 95% CI error bars, no connecting line)
        orig = plot_data[plot_data['condition'] == 'original'].copy().sort_values('num_examples')
        if not orig.empty:
            xs = orig['num_examples'].astype(int).values
            ys = orig['accuracy'].astype(float).values
            es = orig['accuracy_sem'].astype(float).values
            color = TYPE_COLORS.get(etype, '#1f77b4')
            ax.errorbar(xs, ys, yerr=1.96 * es, fmt='o',
                        markerfacecolor=color, markeredgecolor=color,
                        ecolor=to_hex(to_rgba(color, 1.0)), elinewidth=1.6,
                        capsize=4, color=color, markersize=6.5, linewidth=0.0,
                        label='original')
            if connect_lines:
                try:
                    ax.plot(xs, ys, '-', color=color, linewidth=1.2, alpha=0.35)
                except Exception:
                    pass
        # Permuted (dashed line with shaded 95% CI)
        ctrl = plot_data[plot_data['condition'] == 'control'].copy().sort_values('num_examples')
        if not ctrl.empty:
            xs2 = ctrl['num_examples'].astype(int).values
            ys2 = ctrl['accuracy'].astype(float).values
            es2 = ctrl['accuracy_sem'].astype(float).values
            color = TYPE_COLORS.get(etype, '#1f77b4')
            ax.plot(xs2, ys2, '--', color=color, linewidth=1.6, label='permuted (control)')
            ax.fill_between(xs2, ys2 - 1.96 * es2, ys2 + 1.96 * es2, color=color, alpha=0.12, linewidth=0)

        # Axes styling
        xs_all = np.unique(np.concatenate([orig['num_examples'].values if not orig.empty else [],
                                           ctrl['num_examples'].values if not ctrl.empty else []]))
        # Exclude EXCLUDED_NUM_EXAMPLES
        xs_all = sorted([int(x) for x in xs_all if int(x) not in EXCLUDED_NUM_EXAMPLES])
        ax.set_xticks(xs_all)
        ax.grid(False)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # Only bottom panel shows shared x-label; others no xlabel to save space
        if etype == example_types[-1]:
            ax.set_xlabel('# of In-Context Examples')
        else:
            ax.set_xlabel('')
        # Panel title intentionally omitted for flexibility

    # Shared y-label (no figure-level title; user will add later)
    fig.supylabel('Accuracy')

    # Bottom-center legend across figure
    # Tight margins with space for legend (no suptitle, so slightly lower top)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12, hspace=0.26)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    by_label = {l: h for h, l in zip(handles, labels)}
    leg = fig.legend(
        by_label.values(), by_label.keys(),
        loc='lower center', bbox_to_anchor=(0.5, 0.04),
        ncol=min(3, len(by_label)), frameon=True
    )
    if leg:
        leg.get_frame().set_alpha(0.85)

    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.92])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save high-quality vector PDF
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(save_path_pdf, dpi=450, bbox_inches='tight')
    plt.close()

def analyze_control_results(data_size: str = 'small', experiment: str = None):
    """
    Analyze control experiment results comparing original vs control within-individual learning.
    
    Args:
        data_size: Dataset size ('small' or 'large')
        experiment: Experiment label for filenames ('exp1', 'exp2', 'exp3'). If None, inferred from data_size (small->exp1, large->exp2).
    """
    if experiment is None:
        experiment = 'exp1' if data_size == 'small' else 'exp2'
    # Unique key so exp1/exp2/exp3 outputs never override each other
    exp_data_key = f"{experiment}_{data_size}"
    print(f"\n{'='*60}")
    print(f"CONTROL EXPERIMENT ANALYSIS - {experiment.upper()} ({data_size.upper()} DATASET)")
    print(f"{'='*60}")
    
    # Load both original and control results
    original_results, control_results = load_control_comparison_data(data_size)
    
    # Process results for comparison
    comparison_df = process_control_comparison_results(original_results, control_results)
    
    # Perform statistical tests
    print(f"\nPerforming statistical tests for control comparison...")
    test_results_df = perform_control_statistical_tests(
        original_results, control_results, data_size
    )
    
    # Save statistical test results (explicit experiment name: no override between exp1/exp2/exp3)
    os.makedirs('results/statistical_tests', exist_ok=True)
    control_csv_path = f'results/statistical_tests/control_comparison_{exp_data_key}.csv'
    test_results_df.to_csv(control_csv_path, index=False)
    print(f"Saved control comparison statistical test results to: {control_csv_path}")
    
    # Three-panel control plots (per experiment and data_size)
    for mode in ['base']:
        save_path = f'figures/control_comparison/{exp_data_key}/three_panel_{mode}_accuracy.png'
        create_three_panel_control_plot(
                    comparison_df,
                    data_size,
                    mode,
                    save_path
                )
    
    print(f"\nControl comparison analysis completed for {experiment} ({data_size} dataset)")
    print(f"Generated control comparison plots: 1 three-panel figure")
    
    # Statistical test summary
    print(f"\nStatistical Test Summary for Control Comparison ({experiment}, {data_size} dataset):")
    print(f"  Total comparisons: {len(test_results_df)}")
    print(f"  Significant (raw p < 0.05): {len(test_results_df[test_results_df['significant']])}")
    if 'significant_adj' in test_results_df.columns:
        print(f"  Significant (FDR-adjusted p_adj < 0.05): {len(test_results_df[test_results_df['significant_adj']])}")
    
    # Show significant results (using FDR-adjusted if available)
    significant_results = test_results_df[test_results_df.get('significant_adj', test_results_df['significant'])]
    if len(significant_results) > 0:
        print(f"\nSignificant differences (FDR-adjusted):")
        for _, row in significant_results.iterrows():
            direction = "Original > Control" if row['original_mean'] > row['control_mean'] else "Control > Original"
            print(f"  {row['num_examples']} examples, {row['example_type']}, {row['mode']}, {row['metric']}: {direction}")
            print(f"    Original: {row['original_mean']:.3f}±{row['original_std']:.3f}, Control: {row['control_mean']:.3f}±{row['control_std']:.3f}")
            p_adj_str = f", p_adj: {row['p_adj']:.3f}" if 'p_adj' in row and pd.notna(row.get('p_adj')) else ""
            print(f"    Cohen's d: {row['cohens_d']:.3f}, p: {row['p_value']:.3f}{p_adj_str}")
    else:
        print(f"  No significant differences found between original and control conditions (at FDR 0.05).")
    
    # Show effect sizes
    print(f"\nEffect Size Summary (Cohen's d):")
    for metric in ['accuracy', 'likelihood']:
        metric_results = test_results_df[test_results_df['metric'] == metric]
        if len(metric_results) > 0:
            mean_effect = metric_results['cohens_d'].mean()
            print(f"  {metric.capitalize()}: Mean effect size = {mean_effect:.3f}")
            
            # Categorize effect sizes
            small_effects = len(metric_results[metric_results['cohens_d'].abs() < 0.2])
            medium_effects = len(metric_results[(metric_results['cohens_d'].abs() >= 0.2) & (metric_results['cohens_d'].abs() < 0.5)])
            large_effects = len(metric_results[metric_results['cohens_d'].abs() >= 0.5])
            
            print(f"    Small effects (|d| < 0.2): {small_effects}")
            print(f"    Medium effects (0.2 ≤ |d| < 0.5): {medium_effects}")
            print(f"    Large effects (|d| ≥ 0.5): {large_effects}")
    
    return comparison_df, test_results_df

def load_pt_model_results(data_size: str) -> Dict:
    """
    Load PT model results from exp2_in_context_pt directory.
    
    Args:
        data_size: Size of dataset used ('small' or 'large')
        
    Returns:
        Dictionary with PT results for each number of examples
    """
    results_dir = "results/exp2_in_context_pt"
    pt_results = {}
    
    # Find all PT result files for the specified dataset size
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
            
        # Parse filename: PT_{data_size}_within_individual_{num_examples}examples_20samples_results.json
        if not filename.startswith(f'PT_{data_size}_within_individual_'):
            continue
            
        try:
            with open(os.path.join(results_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Get metadata
            metadata = data.get('metadata', {})
            file_data_size = metadata.get('dataset')
            num_examples = metadata.get('num_examples')
            context_type = metadata.get('context_type')
            
            if (file_data_size != data_size or 
                context_type != 'within_individual' or 
                num_examples is None):
                continue
                
            print(f"Loading PT results from: {filename}")
            print(f"  - Dataset: {file_data_size}, Examples: {num_examples}")
            print(f"  - Total results: {len(data.get('results', []))}")
            
            # Store results
            pt_results[num_examples] = data['results']
            
        except Exception as e:
            print(f"Error loading PT file {filename}: {e}")
    
    return pt_results

def process_pt_model_results(pt_results: Dict) -> pd.DataFrame:
    """
    Process PT model results into a DataFrame for plotting.
    
    Args:
        pt_results: Dictionary of PT results by number of examples
        
    Returns:
        DataFrame with processed PT results
    """
    rows = []
    
    for num_examples, results in pt_results.items():
        if not results:
            continue
            
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(results)
        
        # Calculate mean accuracy per participant
        participant_accuracy = df.groupby('sub_id')['test_accuracy'].mean()

        # For likelihood, use geometric mean across entries (equivalently, arithmetic mean of NLL)
        # Prefer using test_loss (NLL) if available; otherwise aggregate via log of likelihoods
        if 'test_loss' in df.columns:
            # Geometric mean likelihood = exp(-mean(NLL)) at participant level
            participant_likelihood = np.exp(-df.groupby('sub_id')['test_loss'].mean())
        elif 'test_likelihood' in df.columns:
            # Geometric mean across likelihoods with numerical stability
            participant_likelihood = df.groupby('sub_id')['test_likelihood'].apply(
                lambda s: float(np.exp(np.mean(np.log(np.clip(np.asarray(s, dtype=float), 1e-12, 1.0)))))
            )
        else:
            # Fallback: create empty series aligned to participants
            participant_likelihood = pd.Series(index=participant_accuracy.index, dtype=float)

        rows.append({
            'num_examples': num_examples,
            'model_type': 'PT',
            'accuracy': participant_accuracy.mean(),
            'accuracy_sem': participant_accuracy.sem(),
            'likelihood': participant_likelihood.mean(),
            'likelihood_sem': participant_likelihood.sem(),
            'participant_accuracies': participant_accuracy.tolist(),
            'participant_likelihoods': participant_likelihood.tolist()
        })
    
    return pd.DataFrame(rows)

def load_llm_results(data_size: str) -> Dict:
    """
    Load LLM results from experiment 1 (zero-shot baselines).
    Now loads LLaMA3.1-70B results to match the in-context learning model.
    
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
            file_data_size = data['metadata'].get('dataset') or data['metadata'].get('data_size')
            if file_data_size != data_size:
                continue
                
            # Get model name from metadata
            model_name = data['metadata'].get('model_name')
            # Load LLaMA3.1-70B results to match the in-context learning model
            if not model_name or model_name != 'meta-llama/Meta-Llama-3.1-70B-Instruct':
                continue
                
            print(f"Loading baseline results from: {filename}")
            print(f"  - Model: {model_name}, Dataset: {file_data_size}")
                
            # Store results
            llm_results[model_name] = {
                'metadata': data['metadata'],
                'results': data['results']
            }
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return llm_results

def load_other_model_results(data_size: str) -> Tuple[Dict, Dict]:
    """
    Load neural net and cognitive model results.
    
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

def process_in_context_results(in_context_results: Dict) -> pd.DataFrame:
    """
    Process in-context learning results into a DataFrame, handling multiple samples.
    
    Args:
        in_context_results: Dictionary of results for each (num_examples, example_type) combination
        
    Returns:
        DataFrame with processed results aggregated across samples
    """
    rows = []
    for (num_examples, example_type), results in in_context_results.items():
        if int(num_examples) in EXCLUDED_NUM_EXAMPLES:
            continue
        df = pd.DataFrame(results)
        
        print(f"\nProcessing {num_examples} examples, {example_type} type:")
        print(f"  Total trials: {len(df)}")
        
        # Check if sample_id exists (for multi-sample experiments)
        has_samples = 'sample_id' in df.columns
        if has_samples:
            unique_samples = sorted(df['sample_id'].unique())
            print(f"  Sample IDs found: {unique_samples}")
        else:
            print(f"  No sample_id found - treating as single sample")
        
        # Group by mode
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode].copy()
            # Exclude rows with num_examples in EXCLUDED_NUM_EXAMPLES
            if 'num_examples' in mode_df.columns:
                mode_df = mode_df[~mode_df['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
            print(f"  Mode {mode}: {len(mode_df)} trials")
            
            if has_samples:
                # Multiple samples: aggregate across samples
                # Get unique sample IDs for this specific mode
                mode_samples = sorted(mode_df['sample_id'].unique())
                print(f"    Sample IDs for this mode: {mode_samples}")
                
                sample_accuracies = []
                sample_likelihoods = []
                
                for sample_id in mode_samples:
                    sample_df = mode_df[mode_df['sample_id'] == sample_id].copy()
                    
                    if len(sample_df) == 0:
                        print(f"    Warning: No data found for sample {sample_id} in mode {mode}")
                        continue
                    
                    # Calculate accuracy per participant for this sample
                    participant_accuracy = sample_df.groupby('sub_id', group_keys=False)[['extracted_choice', 'actual_choice']].apply(
                        lambda x: (x['extracted_choice'] == x['actual_choice']).mean()
                    )
                    
                    # Calculate likelihood per trial, then per participant for this sample
                    sample_df['likelihood'] = sample_df.apply(
                        lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                        axis=1
                    )
                    participant_likelihood = sample_df.groupby('sub_id', group_keys=False)['likelihood'].mean()
                    
                    # Store sample-level aggregated metrics (only if we have valid data)
                    if len(participant_accuracy) > 0 and not participant_accuracy.isna().all():
                        sample_accuracies.append(participant_accuracy.mean())
                        sample_likelihoods.append(participant_likelihood.mean())
                
                if len(sample_accuracies) == 0:
                    print(f"    Warning: No valid samples found for mode {mode}")
                    continue
                
                # Aggregate across samples
                mean_accuracy = np.mean(sample_accuracies)
                sem_accuracy = np.std(sample_accuracies, ddof=1) / np.sqrt(len(sample_accuracies)) if len(sample_accuracies) > 1 else 0
                mean_likelihood = np.mean(sample_likelihoods)
                sem_likelihood = np.std(sample_likelihoods, ddof=1) / np.sqrt(len(sample_likelihoods)) if len(sample_likelihoods) > 1 else 0
                
                print(f"    Valid samples: {len(sample_accuracies)}, Accuracy: {mean_accuracy:.4f}±{sem_accuracy:.4f}, Likelihood: {mean_likelihood:.4f}±{sem_likelihood:.4f}")
                
                rows.append({
                    'num_examples': num_examples,
                    'example_type': example_type,
                    'mode': mode,
                    'accuracy': mean_accuracy,
                    'accuracy_sem': sem_accuracy,
                    'likelihood': mean_likelihood,
                    'likelihood_sem': sem_likelihood,
                    'sample_accuracies': sample_accuracies,
                    'sample_likelihoods': sample_likelihoods,
                    'n_samples': len(sample_accuracies)
                })
            else:
                # Single sample: use original logic
                participant_accuracy = mode_df.groupby('sub_id', group_keys=False)[['extracted_choice', 'actual_choice']].apply(
                    lambda x: (x['extracted_choice'] == x['actual_choice']).mean()
                )
                # Calculate likelihood for each trial
                mode_df['likelihood'] = mode_df.apply(
                    lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                    axis=1
                )
                # Calculate mean likelihood per participant
                participant_likelihood = mode_df.groupby('sub_id', group_keys=False)['likelihood'].mean()
                
                print(f"    Single sample - Accuracy: {participant_accuracy.mean():.4f}±{participant_accuracy.sem():.4f}, Likelihood: {participant_likelihood.mean():.4f}±{participant_likelihood.sem():.4f}")
                
                rows.append({
                    'num_examples': num_examples,
                    'example_type': example_type,
                    'mode': mode,
                    'accuracy': participant_accuracy.mean(),
                    'accuracy_sem': participant_accuracy.sem(),
                    'likelihood': participant_likelihood.mean(),
                    'likelihood_sem': participant_likelihood.sem(),
                    'participant_accuracies': participant_accuracy.tolist(),
                    'participant_likelihoods': participant_likelihood.tolist(),
                    'n_samples': 1
                })
    return pd.DataFrame(rows)

def process_llm_results(llm_results: Dict) -> pd.DataFrame:
    """
    Process LLM results into a DataFrame (zero-shot baselines).
    
    Args:
        llm_results: Dictionary of LLM results
        
    Returns:
        DataFrame with processed results
    """
    rows = []
    for model_name, data in llm_results.items():
        df = pd.DataFrame(data['results'])
        # Process base, cot, and human modes
        for mode in ['base', 'cot', 'human']:
            mode_df = df[df['mode'] == mode].copy()  # Use .copy() to avoid SettingWithCopyWarning
            if len(mode_df) > 0:
                # Group by participant and calculate accuracy
                participant_accuracy = mode_df.groupby('sub_id', group_keys=False)[['extracted_choice', 'actual_choice']].apply(
                    lambda x: (x['extracted_choice'] == x['actual_choice']).mean()
                )
                # Calculate likelihood for each trial
                mode_df['likelihood'] = mode_df.apply(
                    lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                    axis=1
                )
                # Calculate mean likelihood per participant
                participant_likelihood = mode_df.groupby('sub_id', group_keys=False)['likelihood'].mean()
                rows.append({
                    'num_examples': 0,  # Zero-shot
                    'example_type': 'baseline',  # Mark as baseline
                    'mode': mode,
                    'accuracy': participant_accuracy.mean(),
                    'accuracy_sem': participant_accuracy.sem(),
                    'likelihood': participant_likelihood.mean(),
                    'likelihood_sem': participant_likelihood.sem(),
                    'participant_accuracies': participant_accuracy.tolist(),
                    'participant_likelihoods': participant_likelihood.tolist()
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
    
    # Process PT model
    if 'Best_PT_Model' in symbolic_results:
        model_data = symbolic_results['Best_PT_Model']
        if data_size == 'small':
            last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
            round_data = model_data[last_round]
            test_accuracies = []
            test_likelihoods = []
            for participant_data in round_data.values():
                test_accuracies.append(participant_data['test_accuracy'])
                test_likelihoods.append(math.exp(-participant_data['test_loss']))
            rows.append({
                'model': 'PT',
                'accuracy': np.nanmean(test_accuracies),
                'accuracy_sem': np.nanstd(test_accuracies) / np.sqrt(np.sum(~np.isnan(test_accuracies))),
                'likelihood': np.nanmean(test_likelihoods),
                'likelihood_sem': np.nanstd(test_likelihoods) / np.sqrt(np.sum(~np.isnan(test_likelihoods)))
            })
        else:  # large dataset
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
                'model': 'PT',
                'accuracy': np.nanmean(test_accuracies),
                'accuracy_sem': np.nanstd(test_accuracies) / np.sqrt(np.sum(~np.isnan(test_accuracies))),
                'likelihood': np.nanmean(test_likelihoods),
                'likelihood_sem': np.nanstd(test_likelihoods) / np.sqrt(np.sum(~np.isnan(test_likelihoods)))
            })
    
    # Process ContextDependent model
    if 'ContextDependentModel' in NN_results:
        model_data = NN_results['ContextDependentModel']
        if data_size == 'small':
            last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
            metrics = model_data[last_round]
            rows.append({
                'model': 'ContextDep',
                'accuracy': metrics['test_accuracy'],
                'accuracy_sem': metrics.get('test_accuracy_sem', 0),
                'likelihood': math.exp(-metrics['test_loss']),
                'likelihood_sem': metrics.get('test_likelihood_sem', 0)
            })
        else:  # large dataset
            last_round = max(model_data.keys(), key=lambda x: int(x.split('_')[1]))
            last_step = max(model_data[last_round].keys(), key=lambda x: int(x.split('_')[1]))
            step_data = model_data[last_round][last_step]
            test_accuracies = []
            test_likelihoods = []
            for sample_metrics in step_data.values():
                test_accuracies.append(sample_metrics['test_accuracy'])
                test_likelihoods.append(math.exp(-sample_metrics['test_loss']))
            rows.append({
                'model': 'ContextDep',
                'accuracy': np.nanmean(test_accuracies),
                'accuracy_sem': np.nanstd(test_accuracies) / np.sqrt(np.sum(~np.isnan(test_accuracies))),
                'likelihood': np.nanmean(test_likelihoods),
                'likelihood_sem': np.nanstd(test_likelihoods) / np.sqrt(np.sum(~np.isnan(test_likelihoods)))
            })
    
    return pd.DataFrame(rows)

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    
    # Calculate means
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    # Calculate pooled standard deviation
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    return d

def calculate_effect_size_ci(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate 95% confidence interval for Cohen's d effect size.
    
    Args:
        group1: First group data
        group2: Second group data
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for CI
    """
    n1, n2 = len(group1), len(group2)
    d = calculate_cohens_d(group1, group2)
    
    # Calculate standard error of Cohen's d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d ** 2) / (2 * (n1 + n2)))
    
    # Calculate critical t-value
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Calculate confidence interval
    margin_error = t_crit * se_d
    ci_lower = d - margin_error
    ci_upper = d + margin_error
    
    return ci_lower, ci_upper

def load_participant_level_data(in_context_results: Dict, llm_results: Dict) -> Dict:
    """
    Load and aggregate data at participant level for proper statistical comparison.
    
    Returns:
        Dictionary with participant-level data for each condition
    """
    participant_data = {}
    
    # Process in-context results
    for (num_examples, example_type), results in in_context_results.items():
        df = pd.DataFrame(results)
        
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode].copy()
            
            if len(mode_df) == 0:
                continue
            
            # Calculate participant-level aggregates
            # For each participant, average across all trials and samples
            participant_accuracy = mode_df.groupby('sub_id', group_keys=False).apply(
                lambda x: (x['extracted_choice'] == x['actual_choice']).mean()
            )
            
            # Calculate likelihood per trial, then per participant
            mode_df['likelihood'] = mode_df.apply(
                lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                axis=1
            )
            participant_likelihood = mode_df.groupby('sub_id', group_keys=False)['likelihood'].mean()
            
            key = (num_examples, example_type, mode)
            participant_data[key] = {
                'accuracy': participant_accuracy.tolist(),
                'likelihood': participant_likelihood.tolist(),
                'n_participants': len(participant_accuracy),
                'sub_ids': participant_accuracy.index.tolist(),
            }
    
    # Process baseline results
    for model_name, data in llm_results.items():
        df = pd.DataFrame(data['results'])
        
        for mode in ['base', 'cot', 'human']:
            mode_df = df[df['mode'] == mode].copy()
            if len(mode_df) > 0:
                # Calculate participant-level aggregates
                participant_accuracy = mode_df.groupby('sub_id', group_keys=False).apply(
                    lambda x: (x['extracted_choice'] == x['actual_choice']).mean()
                )
                
                mode_df['likelihood'] = mode_df.apply(
                    lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                    axis=1
                )
                participant_likelihood = mode_df.groupby('sub_id', group_keys=False)['likelihood'].mean()
                
                key = (0, 'baseline', mode)
                participant_data[key] = {
                    'accuracy': participant_accuracy.tolist(),
                    'likelihood': participant_likelihood.tolist(),
                    'n_participants': len(participant_accuracy),
                    'sub_ids': participant_accuracy.index.tolist(),
                }
    
    return participant_data

def perform_participant_level_statistical_tests(
    in_context_results: Dict,
    llm_results: Dict,
    data_size: str
) -> pd.DataFrame:
    """
    Perform proper statistical tests using participant-level data.
    This handles the cross-validation structure correctly.
    
    Key design decisions:
    1. Aggregates at participant level for both conditions
    2. For within-individual: Tests if participant-specific learning improves performance
    3. For within-context: Tests if general pattern learning improves performance
    4. Both use same aggregation approach since we're testing participant-level improvements
    
    Args:
        in_context_results: Raw in-context results
        llm_results: Raw LLM baseline results  
        data_size: Dataset size for labeling
        
    Returns:
        DataFrame with statistical test results
    """
    # Load participant-level data
    participant_data = load_participant_level_data(in_context_results, llm_results)
    
    test_results = []
    
    # Get baseline data for each mode
    baseline_data = {}
    for mode in ['base', 'cot']:
        baseline_key = (0, 'baseline', mode)
        if baseline_key in participant_data:
            baseline_data[mode] = participant_data[baseline_key]
    
    # Compare each in-context condition against corresponding baseline
    for key, data in participant_data.items():
        num_examples, example_type, mode = key
        if int(num_examples) in EXCLUDED_NUM_EXAMPLES:
            continue
        
        if num_examples == 0:  # Skip baseline entries
            continue
            
        # Extract base mode from in-context mode (e.g., 'base_within_individual' -> 'base')
        base_mode = mode.split('_')[0]
        
        if base_mode not in baseline_data:
            print(f"Warning: No baseline found for mode {base_mode}")
            continue
        
        baseline = baseline_data[base_mode]
        
        # Perform tests for both accuracy and likelihood
        for metric in ['accuracy', 'likelihood']:
            in_context_data_array = np.array(data[metric])
            baseline_data_array = np.array(baseline[metric])
            
            # Remove any NaN values
            in_context_data_array = in_context_data_array[~np.isnan(in_context_data_array)]
            baseline_data_array = baseline_data_array[~np.isnan(baseline_data_array)]
            
            if len(in_context_data_array) == 0 or len(baseline_data_array) == 0:
                print(f"Warning: No valid data for {metric} comparison in {mode}")
                continue
            
            # Perform independent t-test (now properly comparing participant means)
            t_stat, p_value = ttest_ind(in_context_data_array, baseline_data_array)
            
            # Calculate effect size and CI
            cohens_d = calculate_cohens_d(in_context_data_array, baseline_data_array)
            ci_lower, ci_upper = calculate_effect_size_ci(in_context_data_array, baseline_data_array)
            
            # Calculate degrees of freedom
            df_test = len(in_context_data_array) + len(baseline_data_array) - 2
            
            test_results.append({
                'data_size': data_size,
                'num_examples': num_examples,
                'example_type': example_type,
                'mode': mode,
                'base_mode': base_mode,
                'metric': metric,
                'in_context_mean': np.mean(in_context_data_array),
                'in_context_std': np.std(in_context_data_array, ddof=1),
                'in_context_n': len(in_context_data_array),
                'baseline_mean': np.mean(baseline_data_array),
                'baseline_std': np.std(baseline_data_array, ddof=1),
                'baseline_n': len(baseline_data_array),
                't_statistic': t_stat,
                'p_value': p_value,
                'degrees_of_freedom': df_test,
                'cohens_d': cohens_d,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < 0.05,
                'analysis_level': 'participant'
            })
    
    return pd.DataFrame(test_results)

def perform_participant_sample_level_tests(
    in_context_results: Dict,
    llm_results: Dict,
    data_size: str
) -> pd.DataFrame:
    """
    Perform statistical tests at participant × sample level.
    This shows the effect of different sample sizes explicitly.
    
    Sample sizes:
    - Baseline: 72 participants × 1 sample = 72 data points
    - In-context: 72 participants × 20/40 samples = 1,440/2,880 data points
    
    Note: Samples within participants are not fully independent, so interpret with caution.
    This test primarily shows the effect of unequal sample sizes.
    
    Args:
        in_context_results: Raw in-context results
        llm_results: Raw LLM baseline results
        data_size: Dataset size for labeling
        
    Returns:
        DataFrame with statistical test results
    """
    # Load participant-level data but keep sample structure
    participant_sample_data = {}
    
    # Process in-context results - keep participant × sample level
    for (num_examples, example_type), results in in_context_results.items():
        df = pd.DataFrame(results)
        
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode].copy()
            
            if len(mode_df) == 0:
                continue
            
            # Check if we have sample_id (multi-sample) or not (single sample)
            has_samples = 'sample_id' in mode_df.columns
            
            if has_samples:
                # Group by participant AND sample - this gives us participant × sample level data
                participant_sample_accuracy = []
                participant_sample_likelihood = []
                
                for (sub_id, sample_id), group in mode_df.groupby(['sub_id', 'sample_id']):
                    # Calculate accuracy for this participant × sample combination
                    accuracy = (group['extracted_choice'] == group['actual_choice']).mean()
                    
                    # Calculate likelihood for this participant × sample combination
                    group_copy = group.copy()
                    group_copy['likelihood'] = group_copy.apply(
                        lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                        axis=1
                    )
                    likelihood = group_copy['likelihood'].mean()
                    
                    participant_sample_accuracy.append(accuracy)
                    participant_sample_likelihood.append(likelihood)
            else:
                # Single sample - group by participant only
                participant_sample_accuracy = []
                participant_sample_likelihood = []
                
                for sub_id, group in mode_df.groupby('sub_id'):
                    accuracy = (group['extracted_choice'] == group['actual_choice']).mean()
                    
                    group_copy = group.copy()
                    group_copy['likelihood'] = group_copy.apply(
                        lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                        axis=1
                    )
                    likelihood = group_copy['likelihood'].mean()
                    
                    participant_sample_accuracy.append(accuracy)
                    participant_sample_likelihood.append(likelihood)
            
            key = (num_examples, example_type, mode)
            participant_sample_data[key] = {
                'accuracy': participant_sample_accuracy,
                'likelihood': participant_sample_likelihood,
                'n_participant_samples': len(participant_sample_accuracy),
                'has_multiple_samples': has_samples
            }
    
    # Process baseline results - participant level only (1 sample per participant)
    for model_name, data in llm_results.items():
        df = pd.DataFrame(data['results'])
        
        for mode in ['base', 'cot', 'human']:
            mode_df = df[df['mode'] == mode].copy()
            if len(mode_df) > 0:
                participant_sample_accuracy = []
                participant_sample_likelihood = []
                
                for sub_id, group in mode_df.groupby('sub_id'):
                    accuracy = (group['extracted_choice'] == group['actual_choice']).mean()
                    
                    group_copy = group.copy()
                    group_copy['likelihood'] = group_copy.apply(
                        lambda x: x['probability_option_a'] if x['actual_choice'] == 0 else x['probability_option_b'],
                        axis=1
                    )
                    likelihood = group_copy['likelihood'].mean()
                    
                    participant_sample_accuracy.append(accuracy)
                    participant_sample_likelihood.append(likelihood)
                
                key = (0, 'baseline', mode)
                participant_sample_data[key] = {
                    'accuracy': participant_sample_accuracy,
                    'likelihood': participant_sample_likelihood,
                    'n_participant_samples': len(participant_sample_accuracy),
                    'has_multiple_samples': False
                }
    
    # Perform statistical tests
    test_results = []
    
    # Get baseline data for each mode
    baseline_data = {}
    for mode in ['base', 'cot']:
        baseline_key = (0, 'baseline', mode)
        if baseline_key in participant_sample_data:
            baseline_data[mode] = participant_sample_data[baseline_key]
    
    # Compare each in-context condition against corresponding baseline
    for key, data in participant_sample_data.items():
        num_examples, example_type, mode = key
        if int(num_examples) in EXCLUDED_NUM_EXAMPLES:
            continue
        
        if num_examples == 0:  # Skip baseline entries
            continue
            
        # Extract base mode from in-context mode
        base_mode = mode.split('_')[0]
        
        if base_mode not in baseline_data:
            print(f"Warning: No baseline found for mode {base_mode}")
            continue
        
        baseline = baseline_data[base_mode]
        
        # Perform tests for both accuracy and likelihood
        for metric in ['accuracy', 'likelihood']:
            in_context_data_array = np.array(data[metric])
            baseline_data_array = np.array(baseline[metric])
            
            # Remove any NaN values
            in_context_data_array = in_context_data_array[~np.isnan(in_context_data_array)]
            baseline_data_array = baseline_data_array[~np.isnan(baseline_data_array)]
            
            if len(in_context_data_array) == 0 or len(baseline_data_array) == 0:
                print(f"Warning: No valid data for {metric} comparison in {mode}")
                continue
            
            # Perform independent t-test with unequal sample sizes
            t_stat, p_value = ttest_ind(in_context_data_array, baseline_data_array)
            
            # Calculate effect size and CI
            cohens_d = calculate_cohens_d(in_context_data_array, baseline_data_array)
            ci_lower, ci_upper = calculate_effect_size_ci(in_context_data_array, baseline_data_array)
            
            # Calculate degrees of freedom
            df_test = len(in_context_data_array) + len(baseline_data_array) - 2
            
            # Calculate sample size ratio for reporting
            sample_size_ratio = len(in_context_data_array) / len(baseline_data_array)
            
            test_results.append({
                'data_size': data_size,
                'num_examples': num_examples,
                'example_type': example_type,
                'mode': mode,
                'base_mode': base_mode,
                'metric': metric,
                'in_context_mean': np.mean(in_context_data_array),
                'in_context_std': np.std(in_context_data_array, ddof=1),
                'in_context_n': len(in_context_data_array),
                'baseline_mean': np.mean(baseline_data_array),
                'baseline_std': np.std(baseline_data_array, ddof=1),
                'baseline_n': len(baseline_data_array),
                't_statistic': t_stat,
                'p_value': p_value,
                'degrees_of_freedom': df_test,
                'cohens_d': cohens_d,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < 0.05,
                'sample_size_ratio': sample_size_ratio,
                'analysis_level': 'participant_sample',
                'in_context_has_multiple_samples': data['has_multiple_samples']
            })
    
    return pd.DataFrame(test_results)

def perform_comprehensive_statistical_tests(
    in_context_results: Dict,
    llm_results: Dict,
    data_size: str
) -> pd.DataFrame:
    """
    Perform comprehensive statistical tests using multiple approaches appropriate for accuracy data.
    
    Includes:
    1. Parametric tests (t-test, paired t-test)
    2. Non-parametric tests (Mann-Whitney U, Wilcoxon)
    3. Transformed data tests (arcsine transformation)
    4. Effect sizes (Cohen's d, rank-biserial correlation)
    
    Args:
        in_context_results: Raw in-context results
        llm_results: Raw LLM baseline results
        data_size: Dataset size for labeling
        
    Returns:
        DataFrame with comprehensive statistical test results
    """
    # Load participant-level data
    participant_data = load_participant_level_data(in_context_results, llm_results)
    
    test_results = []
    
    # Get baseline data for each mode
    baseline_data = {}
    for mode in ['base', 'cot']:
        baseline_key = (0, 'baseline', mode)
        if baseline_key in participant_data:
            baseline_data[mode] = participant_data[baseline_key]
    
    # Compare each in-context condition against corresponding baseline
    for key, data in participant_data.items():
        num_examples, example_type, mode = key
        
        if num_examples == 0:  # Skip baseline entries
            continue
            
        # Extract base mode from in-context mode
        base_mode = mode.split('_')[0]
        
        if base_mode not in baseline_data:
            continue
        
        baseline = baseline_data[base_mode]
        
        # Perform comprehensive tests for both accuracy and likelihood
        for metric in ['accuracy', 'likelihood']:
            in_context_data_array = np.array(data[metric])
            baseline_data_array = np.array(baseline[metric])
            
            # Remove any NaN values
            in_context_data_array = in_context_data_array[~np.isnan(in_context_data_array)]
            baseline_data_array = baseline_data_array[~np.isnan(baseline_data_array)]
            
            if len(in_context_data_array) == 0 or len(baseline_data_array) == 0:
                continue
            
            # Basic descriptive statistics
            ic_mean, ic_std = np.mean(in_context_data_array), np.std(in_context_data_array, ddof=1)
            bl_mean, bl_std = np.mean(baseline_data_array), np.std(baseline_data_array, ddof=1)
            
            # 1. Independent t-test (parametric)
            t_stat, t_p = ttest_ind(in_context_data_array, baseline_data_array)
            cohens_d = calculate_cohens_d(in_context_data_array, baseline_data_array)
            ci_lower, ci_upper = calculate_effect_size_ci(in_context_data_array, baseline_data_array)
            
            # 2. Mann-Whitney U test (non-parametric alternative to t-test)
            mw_stat, mw_p = mannwhitneyu(in_context_data_array, baseline_data_array, alternative='two-sided')
            
            # Calculate rank-biserial correlation (non-parametric effect size)
            n1, n2 = len(in_context_data_array), len(baseline_data_array)
            rank_biserial = 1 - (2 * mw_stat) / (n1 * n2)
            
            # 3. Arcsine transformation for bounded data (only for accuracy)
            if metric == 'accuracy':
                # Apply arcsine square-root transformation
                ic_transformed = np.arcsin(np.sqrt(np.clip(in_context_data_array, 0.001, 0.999)))
                bl_transformed = np.arcsin(np.sqrt(np.clip(baseline_data_array, 0.001, 0.999)))
                
                # T-test on transformed data
                t_trans_stat, t_trans_p = ttest_ind(ic_transformed, bl_transformed)
                cohens_d_trans = calculate_cohens_d(ic_transformed, bl_transformed)
            else:
                t_trans_stat = t_trans_p = cohens_d_trans = np.nan
            
            # 4. Check normality assumptions
            def check_normality(data):
                if len(data) < 3:
                    return np.nan, "too_small"
                try:
                    _, p_val = stats.shapiro(data[:5000])  # Shapiro-Wilk test (limit to 5000 samples)
                    return p_val, "normal" if p_val > 0.05 else "non_normal"
                except:
                    return np.nan, "failed"
            
            ic_norm_p, ic_norm = check_normality(in_context_data_array)
            bl_norm_p, bl_norm = check_normality(baseline_data_array)
            
            # 5. Recommended test based on data characteristics
            if metric == 'accuracy':
                if (ic_mean < 0.1 or ic_mean > 0.9 or bl_mean < 0.1 or bl_mean > 0.9):
                    recommended = "non_parametric"  # Near ceiling/floor
                elif ic_norm == "non_normal" or bl_norm == "non_normal":
                    recommended = "non_parametric"  # Non-normal
                else:
                    recommended = "arcsine_transform"  # Bounded but reasonable
            else:  # likelihood
                if ic_norm == "non_normal" or bl_norm == "non_normal":
                    recommended = "non_parametric"
                else:
                    recommended = "parametric"
            
            # Store comprehensive results
            test_results.append({
                'data_size': data_size,
                'num_examples': num_examples,
                'example_type': example_type,
                'mode': mode,
                'base_mode': base_mode,
                'metric': metric,
                
                # Descriptive statistics
                'in_context_mean': ic_mean,
                'in_context_std': ic_std,
                'in_context_n': len(in_context_data_array),
                'baseline_mean': bl_mean,
                'baseline_std': bl_std,
                'baseline_n': len(baseline_data_array),
                
                # Parametric tests
                't_statistic': t_stat,
                't_p_value': t_p,
                'cohens_d': cohens_d,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                
                # Non-parametric tests
                'mannwhitney_u': mw_stat,
                'mannwhitney_p': mw_p,
                'rank_biserial_r': rank_biserial,
                
                # Transformed tests (accuracy only)
                't_transformed_stat': t_trans_stat,
                't_transformed_p': t_trans_p,
                'cohens_d_transformed': cohens_d_trans,
                
                # Assumption checks
                'ic_normality_p': ic_norm_p,
                'bl_normality_p': bl_norm_p,
                'ic_normal': ic_norm,
                'bl_normal': bl_norm,
                'recommended_test': recommended,
                
                # Significance flags
                't_significant': t_p < 0.05,
                'mw_significant': mw_p < 0.05,
                'trans_significant': t_trans_p < 0.05 if not np.isnan(t_trans_p) else False,
                
                'analysis_level': 'participant_comprehensive'
            })
    
    return pd.DataFrame(test_results)

def perform_statistical_tests(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform t-tests comparing in-context conditions against baseline conditions.
    Uses participant-level aggregation to handle cross-validation structure properly.
    
    For in-context learning:
    - Data structure: 72 participants × 19 trials × 20/40 cross-validation samples
    - Aggregation: Average across trials and samples for each participant
    
    For baseline:
    - Data structure: 72 participants × 19 trials × 1 sample  
    - Aggregation: Average across trials for each participant
    
    This ensures we're comparing participant-level means, respecting the experimental design.
    
    Args:
        results_df: DataFrame with all results including baselines
        
    Returns:
        DataFrame with statistical test results
    """
    test_results = []
    
    # Get baseline data for each mode
    baseline_data = {}
    for mode in ['base', 'cot']:
        baseline_row = results_df[
            (results_df['num_examples'] == 0) & 
            (results_df['mode'] == mode) & 
            (results_df['example_type'] == 'baseline')
        ]
        if not baseline_row.empty:
            baseline_data[mode] = baseline_row.iloc[0]
    
    # Compare each in-context condition against corresponding baseline
    in_context_rows = results_df[results_df['num_examples'] > 0]
    # Exclude rows with num_examples in EXCLUDED_NUM_EXAMPLES
    in_context_rows = in_context_rows[~in_context_rows['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
    
    for _, row in in_context_rows.iterrows():
        if int(row['num_examples']) in EXCLUDED_NUM_EXAMPLES:
            continue
        # Extract base mode from in-context mode (e.g., 'base_within_individual' -> 'base')
        base_mode = row['mode'].split('_')[0]
        
        if base_mode not in baseline_data:
            print(f"Warning: No baseline found for mode {base_mode}")
            continue
            
        baseline_row = baseline_data[base_mode]
        
        # Perform tests for both accuracy and likelihood
        for metric in ['accuracy', 'likelihood']:
            # Initialize variables to None
            in_context_data = None
            baseline_data_array = None
            
            try:
                # Get in-context data
                if 'n_samples' in row and row['n_samples'] > 1:
                    # Multi-sample: use sample-level data
                    field_name = f'sample_{metric}s'
                    
                    if field_name in row and row[field_name] is not None:
                        field_value = row[field_name]
                        if isinstance(field_value, list) and len(field_value) > 0:
                            in_context_data = np.array(field_value)
                        else:
                            print(f"Warning: Empty sample data for {metric} in {row['mode']}")
                            continue
                    else:
                        print(f"Warning: Field '{field_name}' not found in row with mode {row['mode']}")
                        continue
                else:
                    # Single sample: use participant-level data
                    field_name = f'participant_{metric}s'
                    
                    if field_name in row and row[field_name] is not None:
                        field_value = row[field_name]
                        if isinstance(field_value, list) and len(field_value) > 0:
                            in_context_data = np.array(field_value)
                        else:
                            print(f"Warning: Empty participant data for {metric} in {row['mode']}")
                            continue
                    else:
                        print(f"Warning: Field '{field_name}' not found in row with mode {row['mode']}")
                        continue
                
                # Get baseline data
                baseline_field = f'participant_{metric}s'
                if baseline_field in baseline_row and baseline_row[baseline_field] is not None:
                    baseline_field_value = baseline_row[baseline_field]
                    if isinstance(baseline_field_value, list) and len(baseline_field_value) > 0:
                        baseline_data_array = np.array(baseline_field_value)
                    else:
                        print(f"Warning: Empty baseline data for {metric} in {base_mode}")
                        continue
                else:
                    print(f"Warning: Field '{baseline_field}' not found in baseline for mode {base_mode}")
                    continue
                
            except Exception as e:
                print(f"Error processing {metric} for mode {row['mode']}: {e}")
                continue
            
            # Check if we have valid data arrays
            if in_context_data is None or baseline_data_array is None:
                print(f"Warning: Missing data arrays for {metric} comparison")
                continue
                
            if len(in_context_data) == 0 or len(baseline_data_array) == 0:
                print(f"Warning: Empty data arrays for {metric} comparison")
                continue
            
            # Remove any NaN values
            in_context_data = in_context_data[~np.isnan(in_context_data)]
            baseline_data_array = baseline_data_array[~np.isnan(baseline_data_array)]
            
            if len(in_context_data) == 0 or len(baseline_data_array) == 0:
                print(f"Warning: No valid data after removing NaNs for {metric} comparison")
                continue
            
            # Perform independent t-test
            t_stat, p_value = ttest_ind(in_context_data, baseline_data_array)
            
            # Calculate effect size and CI
            cohens_d = calculate_cohens_d(in_context_data, baseline_data_array)
            ci_lower, ci_upper = calculate_effect_size_ci(in_context_data, baseline_data_array)
            
            # Calculate degrees of freedom
            df = len(in_context_data) + len(baseline_data_array) - 2
            
            test_results.append({
                'data_size': 'small',  # This should be passed as parameter ideally
                'num_examples': row['num_examples'],
                'example_type': row['example_type'],
                'mode': row['mode'],
                'base_mode': base_mode,
                'metric': metric,
                'in_context_mean': np.mean(in_context_data),
                'in_context_std': np.std(in_context_data, ddof=1),
                'in_context_n': len(in_context_data),
                'baseline_mean': np.mean(baseline_data_array),
                'baseline_std': np.std(baseline_data_array, ddof=1),
                'baseline_n': len(baseline_data_array),
                't_statistic': t_stat,
                'p_value': p_value,
                'degrees_of_freedom': df,
                'cohens_d': cohens_d,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < 0.05
            })
    
    return pd.DataFrame(test_results)

def create_learning_curve_plot(
    results_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    data_size: str,
    llm_mode: str,  # 'base' or 'cot'
    context_type: str,  # 'within_individual' or 'within_context'
    metric: str,  # 'accuracy' or 'likelihood'
    save_path: str,
    include_pt: bool = True,
    connect_lines: bool = False
) -> None:
    """
    Create learning curve plot grouped by example types for a specific LLM mode.
    Shows dots for in-context learning, zero-shot baseline, and PT model progression (within_individual only).
    """
    plt.style.use(PLOT_STYLE)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot zero-shot baseline for this mode as a dashed line
    zero_shot = results_df[(results_df['num_examples'] == 0) & 
                          (results_df['mode'] == llm_mode) & 
                          (results_df['example_type'] == 'baseline')]
    if not zero_shot.empty:
        baseline_value = zero_shot[metric].values[0]
        ax.axhline(
            y=baseline_value,
            color='#8B4513',  # Brown color (different from human baseline)
            linestyle='--',
            label=f'{llm_mode} zero-shot',
            linewidth=2
        )
    
    # Define dodge offsets for the three example types
    type_offsets = {
        'think_aloud': -0.2,
        'choice': 0.0,
        'both': 0.2
    }
    
    # Plot in-context results for each example type (dots only, no lines, dodged)
    for example_type in ['think_aloud', 'choice', 'both']:
        in_context_mode = f"{llm_mode}_{context_type}"
        type_data = results_df[(results_df['mode'] == in_context_mode) & 
                               (results_df['example_type'] == example_type)]
        
        if not type_data.empty:
            # Exclude EXCLUDED_NUM_EXAMPLES
            type_data = type_data[~type_data['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
            xs = type_data['num_examples'].values + type_offsets[example_type]  # Add offset
            ys = type_data[metric].values
            yerrs = type_data[f'{metric}_sem'].values
            ax.errorbar(
                xs, ys, yerr=1.96 * yerrs,
                fmt='o',  # dots only
                color=TYPE_COLORS[example_type], 
                capsize=5,
                markersize=6,
                linewidth=0,
                label=example_type.replace('_', ' ')
            )
            if connect_lines:
                try:
                    # connect using un-offset integer xs to avoid zig-zag perception
                    xs_line = type_data['num_examples'].values
                    ax.plot(xs_line, ys, '-', color=TYPE_COLORS[example_type], linewidth=1.2, alpha=0.35, zorder=1)
                except Exception:
                    pass
    
    # Add PT model line ONLY for within_individual context (optional)
    if include_pt and context_type == 'within_individual':
        try:
            # Load and process PT model results
            pt_results = load_pt_model_results(data_size)
            if pt_results:
                pt_df = process_pt_model_results(pt_results)
                
                if not pt_df.empty:
                    # Sort by number of examples for proper line connection
                    pt_df = pt_df.sort_values('num_examples')
                    
                    xs = pt_df['num_examples'].values
                    ys = pt_df[metric].values
                    yerrs = pt_df[f'{metric}_sem'].values
                    
                    # Calculate 95% confidence intervals
                    ci_95 = yerrs * 1.96
                    
                    # Plot PT model line with same color as existing PT model
                    pt_color = BASE_COLORS.get('PT', '#17becf')  # Teal color for PT
                    ax.plot(
                        xs, ys,
                        '--',  # dashed line
                        color=pt_color,
                        linewidth=2,
                        label='PT model',
                        alpha=0.8
                    )
                    
                    # Add 95% confidence interval shaded area
                    ax.fill_between(
                        xs, ys - ci_95, ys + ci_95,
                        color=pt_color,
                        alpha=0.2,
                        linewidth=0
                    )
        except Exception as e:
            print(f"Warning: Could not load PT model results: {e}")
    
    # Get all unique num_examples for x-axis (excluding 0)
    all_num_examples = sorted([x for x in results_df['num_examples'].unique() if x > 0 and int(x) not in EXCLUDED_NUM_EXAMPLES])
    if all_num_examples:
        ax.set_xticks([0] + all_num_examples)
        ax.set_xlim(-0.5, max(all_num_examples) + 0.5)
    else:
        # Fallback if no in-context examples found
        ax.set_xticks([0])
        ax.set_xlim(-0.5, 1)
    
    ax.set_xlabel('Number of Examples')
    ax.set_ylabel(metric.capitalize())
    exp_label = 'Experiment 1 (PT design)' if data_size == 'small' else 'Experiment 2 (choice13k-sampled)'
    # Title intentionally omitted for flexibility
    # Dynamic y-limits using plotted data (all example types) and baseline
    mode_key = f"{llm_mode}_{context_type}"
    df_mode = results_df[(results_df['mode'] == mode_key) & (results_df['num_examples'] > 0)]
    y_vals = []
    y_errs = []
    if 'baseline_value' in locals():
        y_vals.append(float(baseline_value))
        y_errs.append(0.0)
    if not df_mode.empty:
        y_vals.extend(df_mode[metric].astype(float).tolist())
        y_errs.extend(df_mode[f'{metric}_sem'].astype(float).tolist())
    if y_vals:
        y_vals_np = np.array(y_vals, dtype=float)
        y_errs_np = np.array(y_errs, dtype=float)
        lo = float(np.nanmin(y_vals_np - 1.96 * np.nan_to_num(y_errs_np)))
        hi = float(np.nanmax(y_vals_np + 1.96 * np.nan_to_num(y_errs_np)))
        pad = max(0.005, 0.05 * (hi - lo if hi > lo else 0.1))
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    # No grid - make sure grid is turned off
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save high-quality vector PDF
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(save_path_pdf, dpi=450, bbox_inches='tight')
    plt.close()

def create_effect_size_plot(
    test_results_df: pd.DataFrame,
    data_size: str,
    llm_mode: str,  # 'base' or 'cot'
    context_type: str,  # 'within_individual' or 'within_context'
    metric: str,  # 'accuracy' or 'likelihood'
    save_path: str
) -> None:
    """
    Create effect size plot showing Cohen's d with 95% CI for each condition.
    Similar structure to learning curves but Y-axis shows effect size.
    """
    plt.style.use(PLOT_STYLE)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter test results for this specific configuration
    in_context_mode = f"{llm_mode}_{context_type}"
    mode_results = test_results_df[
        (test_results_df['mode'] == in_context_mode) & 
        (test_results_df['metric'] == metric)
    ]
    # Exclude EXCLUDED_NUM_EXAMPLES
    mode_results = mode_results[~mode_results['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
    
    if mode_results.empty:
        print(f"No test results found for {llm_mode}_{context_type}, {metric}")
        plt.close()
        return
    
    # Plot effect sizes for each example type (lines only, no jitter, high transparency)
    for example_type in ['think_aloud', 'choice', 'both']:
        type_data = mode_results[mode_results['example_type'] == example_type]
        
        if not type_data.empty:
            # Sort by num_examples to ensure proper line connection
            type_data = type_data.sort_values('num_examples')
            
            xs = type_data['num_examples'].values  # No offset - all lines at same x positions
            ys = type_data['cohens_d'].values
            ci_lower = type_data['ci_lower'].values
            ci_upper = type_data['ci_upper'].values
            
            # Plot the main line (no dots, just lines)
            ax.plot(
                xs, ys,
                '-',  # lines only, no dots
                color=TYPE_COLORS[example_type], 
                linewidth=2,
                alpha=0.5,  # More transparency for overlapping lines
                label=example_type.replace('_', ' ')
            )
            
            # Add shaded area for 95% confidence interval with higher transparency
            ax.fill_between(
                xs, ci_lower, ci_upper,
                color=TYPE_COLORS[example_type],
                alpha=0.1,  # Even higher transparency for overlapping areas
                linewidth=0
            )
    
    # Add reference line at 0 (no effect)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add reference lines for small (0.2), medium (0.5), and large (0.8) effect sizes
    all_num_examples = sorted([int(x) for x in mode_results['num_examples'].unique() if int(x) not in EXCLUDED_NUM_EXAMPLES])
    x_max = max(all_num_examples) + 0.5
    
    for effect_size, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
        ax.axhline(y=effect_size, color='lightgray', linestyle=':', alpha=0.7, linewidth=1)
        ax.axhline(y=-effect_size, color='lightgray', linestyle=':', alpha=0.7, linewidth=1)
        # Add labels on the right side
        ax.text(x_max + 0.1, effect_size, f'{label} (+)', ha='left', va='center', 
                fontsize=8, alpha=0.7, color='gray')
        ax.text(x_max + 0.1, -effect_size, f'{label} (-)', ha='left', va='center', 
                fontsize=8, alpha=0.7, color='gray')
    
    # Set x-axis
    ax.set_xticks(all_num_examples)
    ax.set_xlim(min(all_num_examples) - 0.5, x_max)
    
    ax.set_xlabel('Number of Examples')
    ax.set_ylabel(f"Cohen's d Effect Size ({metric.capitalize()})")
    exp_label = 'Experiment 1 (PT design)' if data_size == 'small' else 'Experiment 2 (choice13k-sampled)'
    # Title intentionally omitted for flexibility
    
    # Set y-axis limits to show effect sizes clearly
    y_max = max(abs(mode_results['ci_upper'].max()), abs(mode_results['ci_lower'].min()))
    ax.set_ylim(-y_max * 1.1, y_max * 1.1)
    
    # No grid
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save high-quality vector PDF
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(save_path_pdf, dpi=450, bbox_inches='tight')
    plt.close()

def create_mode_comparison_plot(
    results_df: pd.DataFrame,
    data_size: str,
    example_type: str,
    num_examples: int,
    save_path: str
) -> None:
    """
    Create plot comparing base vs cot modes for a specific example type and number of examples.
    Use grouped (dodged) bars for each mode, no grid.
    """
    plt.style.use(PLOT_STYLE)
    # Filter results for this combination
    plot_data = []
    # Get within_individual results
    for mode in ['base', 'cot']:
        mode_name = f"{mode}_within_individual"
        mode_df = results_df[(results_df['num_examples'] == num_examples) & 
                            (results_df['mode'] == mode_name) & 
                            (results_df['example_type'] == example_type)]
        if not mode_df.empty:
            plot_data.append({
                'mode': f"{mode}_individual",
                'accuracy': mode_df['accuracy'].values[0],
                'accuracy_sem': mode_df['accuracy_sem'].values[0],
                'likelihood': mode_df['likelihood'].values[0],
                'likelihood_sem': mode_df['likelihood_sem'].values[0]
            })
    # Get within_context results (only for small dataset)
    if data_size == 'small':
        for mode in ['base', 'cot']:
            mode_name = f"{mode}_within_context"
            mode_df = results_df[(results_df['num_examples'] == num_examples) & 
                                (results_df['mode'] == mode_name) & 
                                (results_df['example_type'] == example_type)]
            if not mode_df.empty:
                plot_data.append({
                    'mode': f"{mode}_context",
                    'accuracy': mode_df['accuracy'].values[0],
                    'accuracy_sem': mode_df['accuracy_sem'].values[0],
                    'likelihood': mode_df['likelihood'].values[0],
                    'likelihood_sem': mode_df['likelihood_sem'].values[0]
                })
    if not plot_data:
        print(f"No data found for {data_size}, {example_type}, {num_examples} examples")
        return
    plot_df = pd.DataFrame(plot_data)
    # Use x = np.arange(n_groups) and plot bars at those positions
    n_groups = len(plot_df)
    x = np.arange(n_groups)
    bar_width = 0.35  # Smaller bar width for dodging
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot accuracy (grouped bars)
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        color = '#1f77b4' if 'base' in row['mode'] else '#ff7f0e'
        ax1.bar(x[i] - bar_width/2, row['accuracy'], bar_width, yerr=1.96 * row['accuracy_sem'], capsize=5, color=color, label=row['mode'])
        ax1.text(x[i] - bar_width/2, row['accuracy'] + 0.01, f'{row["accuracy"]:.3f}', ha='center', va='bottom', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df['mode'], rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    # Title intentionally omitted for flexibility
    # Dynamic y-limits for accuracy panel
    acc_vals = plot_df['accuracy'].astype(float).values
    acc_errs = plot_df['accuracy_sem'].astype(float).values
    if acc_vals.size:
        acc_lo = float(np.nanmin(acc_vals - 1.96 * np.nan_to_num(acc_errs)))
        acc_hi = float(np.nanmax(acc_vals + 1.96 * np.nan_to_num(acc_errs)))
        acc_pad = max(0.005, 0.05 * (acc_hi - acc_lo if acc_hi > acc_lo else 0.1))
        ax1.set_ylim(max(0.0, acc_lo - acc_pad), min(1.0, acc_hi + acc_pad))
    # No grid
    ax1.grid(False)
    # Plot likelihood (grouped bars)
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        color = '#1f77b4' if 'base' in row['mode'] else '#ff7f0e'
        ax2.bar(x[i] - bar_width/2, row['likelihood'], bar_width, yerr=1.96 * row['likelihood_sem'], capsize=5, color=color, label=row['mode'])
        ax2.text(x[i] - bar_width/2, row['likelihood'] + 0.01, f'{row["likelihood"]:.3f}', ha='center', va='bottom', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_df['mode'], rotation=45, ha='right')
    ax2.set_ylabel('Likelihood')
    # Title intentionally omitted for flexibility
    # Dynamic y-limits for likelihood panel
    like_vals = plot_df['likelihood'].astype(float).values
    like_errs = plot_df['likelihood_sem'].astype(float).values
    if like_vals.size:
        like_lo = float(np.nanmin(like_vals - 1.96 * np.nan_to_num(like_errs)))
        like_hi = float(np.nanmax(like_vals + 1.96 * np.nan_to_num(like_errs)))
        like_pad = max(0.005, 0.05 * (like_hi - like_lo if like_hi > like_lo else 0.1))
        ax2.set_ylim(max(0.0, like_lo - like_pad), min(1.0, like_hi + like_pad))
    # No grid
    ax2.grid(False)
    exp_label = 'Experiment 1 (PT design)' if data_size == 'small' else 'Experiment 2 (choice13k-sampled)'
    # Figure title intentionally omitted for flexibility
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    fig.patch.set_facecolor('white')
    # Only show legend once
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save high-quality vector PDF
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(save_path_pdf, dpi=450, bbox_inches='tight')
    plt.close()

def create_two_panel_accuracy_plot(
    results_small: pd.DataFrame,
    results_large: pd.DataFrame,
    llm_small: pd.DataFrame,
    llm_large: pd.DataFrame,
    llm_mode: str,
    context_type: str,
    include_pt: bool,
    save_path: str,
) -> None:
    """Four-panel figure for within-individual learning (square layout).

    Top row: Accuracy (small | large)
    Bottom row: Δ Accuracy vs. zero-shot baseline (small | large)
    Maintains the same width but roughly doubles height for paper layout.
    """
    plt.style.use(PLOT_STYLE)

    # Determine all available N across both datasets for this mode/context
    mode_key = f"{llm_mode}_{context_type}"
    def collect_ns(df: pd.DataFrame) -> set:
        try:
            ns = set(int(n) for n in df[df['mode'] == mode_key]['num_examples'].unique() if int(n) > 0 and int(n) not in EXCLUDED_NUM_EXAMPLES)
        except Exception:
            ns = set()
        return ns
    xs_union = sorted(collect_ns(results_small) | collect_ns(results_large))
    if not xs_union:
        xs_union = [1, 2, 3, 5, 10]

    # 2x2 grid: use smaller panels so fonts appear larger relative to axes
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.8))

    def _plot_dataset(ax, data_df: pd.DataFrame, llm_df: pd.DataFrame, title: str, data_size_label: str, panel_label: str):
        # Baseline from zeroshot for the given mode
        zero_row = llm_df[(llm_df['num_examples'] == 0) & (llm_df['mode'] == llm_mode) & (llm_df['example_type'] == 'baseline')]
        if not zero_row.empty:
            base_y = float(zero_row['accuracy'].values[0])
            # Participant-level 95% CI band for zero-shot
            if 'participant_accuracies' in zero_row.columns:
                bl_parts = np.asarray(zero_row.iloc[0]['participant_accuracies'], dtype=float)
                bl_parts = bl_parts[np.isfinite(bl_parts)]
                if bl_parts.size > 1:
                    sem = float(np.nanstd(bl_parts, ddof=1) / np.sqrt(bl_parts.size))
                else:
                    sem = 0.0
                bl_ci = 1.96 * sem
                x_min, x_max = min(xs_union)-0.3, max(xs_union)+0.3
                ax.fill_between([x_min, x_max], [base_y - bl_ci, base_y - bl_ci], [base_y + bl_ci, base_y + bl_ci],
                                color="#8B4513", alpha=0.10, linewidth=0)
            ax.axhline(y=base_y, color='#8B4513', linestyle='--', linewidth=1.5, alpha=0.7, label='zero-shot')
            # Annotate baseline at right margin
            ax.text(1.01, base_y, 'zero-shot', color='#8B4513', alpha=0.7, fontsize=8, ha='left', va='center', transform=ax.get_yaxis_transform())

        # Plot three conditions with error bars
        markers = {'think_aloud': 'o', 'choice': 's', 'both': '^'}
        # Colorblind-safe: TA (blue), Choice (orange), Both (green)
        colors = {'think_aloud': '#1f77b4', 'choice': '#ff7f0e', 'both': '#2ca02c'}
        order = ['think_aloud', 'choice', 'both']
        for etype in order:
            df_t = data_df[(data_df['mode'] == mode_key) & (data_df['example_type'] == etype)]
            if df_t.empty:
                continue
            # Map by num_examples for consistent xs
            by_n = {int(n): (float(a), float(a_sem)) for n, a, a_sem in zip(df_t['num_examples'], df_t['accuracy'], df_t['accuracy_sem']) if int(n) not in EXCLUDED_NUM_EXAMPLES}
            xs_plot, ys_plot, es_plot = [], [], []
            for n in xs_union:
                if n in by_n:
                    xs_plot.append(n)
                    y, e = by_n[n]
                    ys_plot.append(y)
                    es_plot.append(e)
            if xs_plot:
                ax.errorbar(
                    xs_plot,
                    ys_plot,
                    yerr=1.96 * np.array(es_plot),
                    fmt=markers[etype],
                    markerfacecolor=colors[etype],
                    markeredgecolor=colors[etype],
                    ecolor=to_hex(to_rgba(colors[etype], 1.0)),
                    elinewidth=1.6,
                    capsize=4,
                    color=colors[etype],
                    markersize=6.5,
                    linewidth=1.6,
                    label=etype.replace('_', ' ').title()
                )

        # Optional PT overlay (within_individual only)
        if include_pt and context_type == 'within_individual':
            try:
                pt_results = load_pt_model_results(data_size_label)
                if pt_results:
                    pt_df = process_pt_model_results(pt_results)
                    if not pt_df.empty:
                        pt_df = pt_df.sort_values('num_examples')
                        xs = pt_df['num_examples'].values
                        ys = pt_df['accuracy'].values
                        yerrs = (pt_df['accuracy_sem'].values if 'accuracy_sem' in pt_df.columns else None)
                        ax.plot(xs, ys, '--', color=BASE_COLORS.get('PT', '#17becf'), linewidth=1.6, label='PT model', alpha=0.7)
                        if yerrs is not None:
                            ax.fill_between(xs, ys - 1.96*yerrs, ys + 1.96*yerrs, color=BASE_COLORS.get('PT', '#17becf'), alpha=0.15, linewidth=0)
            except Exception as e:
                print(f"Warning: PT overlay failed for {data_size_label}: {e}")

        # Panel titles (panel label intentionally omitted)
        # Three-line title with formal experiment label on its own line
        # Panel title intentionally omitted for flexibility
        ax.set_xticks(xs_union)
        ax.set_xlabel('# of In-Context Examples', fontsize=12)
        ax.set_xlim(min(xs_union)-0.3, max(xs_union)+0.3)
        ax.grid(False)
        ax.set_facecolor('white')
        # Make ticks more compact and increase font size
        ax.tick_params(axis='x', labelsize=11, pad=3)
        ax.tick_params(axis='y', labelsize=11)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    _plot_dataset(axes[0, 0], results_small, llm_small, 'Accuracy (small dataset)', 'small', 'A')
    _plot_dataset(axes[0, 1], results_large, llm_large, 'Accuracy (large dataset)', 'large', 'B')

    # Bottom row: external delta panels (Δ vs baseline)
    def _plot_delta_panel(ax, data_df: pd.DataFrame, llm_df: pd.DataFrame):
        try:
            zero_row = llm_df[(llm_df['num_examples'] == 0) & (llm_df['mode'] == llm_mode) & (llm_df['example_type'] == 'baseline')]
            if zero_row.empty:
                return
            baseline = float(zero_row['accuracy'].values[0])
            markers = {'think_aloud': 'o', 'choice': 's', 'both': '^'}
            colors = {'think_aloud': '#1f77b4', 'choice': '#ff7f0e', 'both': '#2ca02c'}
            for etype in ['think_aloud', 'choice', 'both']:
                df_t = data_df[(data_df['mode'] == mode_key) & (data_df['example_type'] == etype)]
                if df_t.empty:
                    continue
                by_n = {int(n): (float(a), float(a_sem)) for n, a, a_sem in zip(df_t['num_examples'], df_t['accuracy'], df_t['accuracy_sem']) if int(n) not in EXCLUDED_NUM_EXAMPLES}
                xs_plot, deltas, errs = [], [], []
                for n in xs_union:
                    if n in by_n:
                        val, e = by_n[n]
                        xs_plot.append(n)
                        deltas.append(val - baseline)
                        errs.append(e)
                if xs_plot:
                    ax.errorbar(xs_plot, deltas, yerr=1.96 * np.array(errs), fmt=markers[etype], color=colors[etype],
                                markersize=5.0, linewidth=1.0, capsize=3, label=etype.replace('_', ' ').title())
            ax.axhline(y=0.0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.6)
            ax.set_xticks(xs_union)
            ax.set_xlabel('# of In-Context Examples')
            ax.grid(False)
            ax.set_facecolor('white')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        except Exception:
            pass

    _plot_delta_panel(axes[1, 0], results_small, llm_small)
    _plot_delta_panel(axes[1, 1], results_large, llm_large)

    # Y labels per row
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_ylabel('Δ Accuracy vs baseline', fontsize=12)

    # Harmonize y-limits across the top accuracy panels only
    top_axes = [axes[0, 0], axes[0, 1]]
    all_y = []
    for ax in top_axes:
        ymin, ymax = ax.get_ylim()
        all_y.append((ymin, ymax))
    if all_y:
        ymin = min(y for y, _ in all_y)
        ymax = max(y for _, y in all_y)
        for ax in top_axes:
            ax.set_ylim(ymin, ymax)

    # Build a unified legend using all four axes, optionally removing PT for no-PT plots
    handles_labels = []
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        h, l = ax.get_legend_handles_labels()
        handles_labels.extend(list(zip(h, l)))
    if not include_pt:
        pairs = [(h, l) for h, l in handles_labels if l != 'PT model']
    else:
        pairs = handles_labels
    by_label = {l: h for h, l in pairs}
    # Legend at bottom center of the figure, single row, translucent frame
    # Make room for legend
    fig.subplots_adjust(bottom=0.18)
    leg = fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.02),
                     ncol=min(5, len(by_label)), frameon=True, prop={'size': 11})
    if leg:
        leg.get_frame().set_alpha(0.85)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save high-quality vector PDF
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(save_path_pdf, dpi=450, bbox_inches='tight')
    plt.close()


def _bh_adjust_local(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    ranked = sorted([(p, i) for i, p in enumerate(pvals)], key=lambda x: x[0])
    scaled = [(p * m) / (rank + 1) for rank, (p, _) in enumerate(ranked)]
    adj = [0.0] * m
    cummin = 1.0
    for i in range(m - 1, -1, -1):
        cummin = min(cummin, scaled[i])
        pval = min(cummin, 1.0)
        _, idx = ranked[i]
        adj[idx] = pval
    return adj


def create_single_panel_accuracy_plot(
    results_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    data_size: str,
    llm_mode: str,
    context_type: str,
    save_path: str,
    connect_lines: bool = False
):
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Baseline with shaded 95% CI
    zero_row = llm_df[(llm_df['num_examples'] == 0) & (llm_df['mode'] == llm_mode) & (llm_df['example_type'] == 'baseline')]
    base_y = None
    if not zero_row.empty:
        base_y = float(zero_row['accuracy'].values[0])
        if 'participant_accuracies' in zero_row.columns:
            bl_parts = np.asarray(zero_row.iloc[0]['participant_accuracies'], dtype=float)
            bl_parts = bl_parts[np.isfinite(bl_parts)]
            sem = float(np.nanstd(bl_parts, ddof=1) / np.sqrt(bl_parts.size)) if bl_parts.size > 1 else 0.0
            bl_ci = 1.96 * sem
        else:
            bl_ci = 0.0
    # Collect x range first (needed for band)
    mode_key = f"{llm_mode}_{context_type}"
    used = results_df[(results_df['mode'] == mode_key) & (results_df['num_examples'] > 0)]
    if not used.empty:
        used = used[~used['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
    xs_union = sorted(used['num_examples'].unique()) if not used.empty else []
    if base_y is not None and xs_union:
        x_min, x_max = min(xs_union) - 0.3, max(xs_union) + 0.3
        ax.fill_between([x_min, x_max], [base_y - bl_ci, base_y - bl_ci], [base_y + bl_ci, base_y + bl_ci],
                        color="#8B4513", alpha=0.10, linewidth=0)
        ax.axhline(y=base_y, color='#8B4513', linestyle='--', linewidth=1.5, alpha=0.7, label='zero-shot')

    markers = {'think_aloud': 'o', 'choice': 's', 'both': '^'}
    colors = {'think_aloud': '#1f77b4', 'choice': '#ff7f0e', 'both': '#2ca02c'}
    for etype in ['think_aloud', 'choice', 'both']:
        df_t = results_df[(results_df['mode'] == mode_key) & (results_df['example_type'] == etype) & (results_df['num_examples'] > 0)]
        if not df_t.empty:
            df_t = df_t[~df_t['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
        if df_t.empty:
            continue
        xs = df_t['num_examples'].astype(int).values
        ys = df_t['accuracy'].astype(float).values
        es = df_t['accuracy_sem'].astype(float).values
        ax.errorbar(xs, ys, yerr=1.96 * es, fmt=markers[etype], markerfacecolor=colors[etype], markeredgecolor=colors[etype],
                    ecolor=to_hex(to_rgba(colors[etype], 1.0)), elinewidth=1.6, capsize=4, color=colors[etype], markersize=6.5, linewidth=1.6,
                    label=etype.replace('_', ' ').title())
        if connect_lines:
            try:
                ax.plot(xs, ys, '-', color=colors[etype], linewidth=1.2, alpha=0.35)
            except Exception:
                pass

    # Title intentionally omitted for flexibility
    ax.set_xticks(xs_union)
    ax.set_xlabel('# of In-Context Examples')
    ax.set_ylabel('Accuracy')
    ax.grid(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Bottom-center legend
    fig.subplots_adjust(bottom=0.22)
    leg = fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), frameon=True, ncol=4)
    if leg:
        leg.get_frame().set_alpha(0.85)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save high-quality vector PDF
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(save_path_pdf, dpi=450, bbox_inches='tight')
    plt.close()
def export_two_panel_within_individual_stats_from_raw(
    data_sizes: list[str] = ['small', 'large'],
    llm_mode: str = 'base'
):
    for data_size in data_sizes:
        try:
            in_context_results = load_in_context_results(data_size)
            llm_results = load_llm_results(data_size)
            participant_data = load_participant_level_data(in_context_results, llm_results)
            baseline_key = (0, 'baseline', llm_mode)
            if baseline_key not in participant_data:
                print(f"No baseline found for {data_size} {llm_mode}")
                continue
            baseline_arr = np.asarray(participant_data[baseline_key]['accuracy'], dtype=float)
            baseline_ids = set(participant_data[baseline_key].get('sub_ids', []))
            rows: list[dict] = []
            for (n, example_type, mode), data in participant_data.items():
                if int(n) in EXCLUDED_NUM_EXAMPLES:
                    continue
                if mode != f"{llm_mode}_within_individual" or n == 0:
                    continue
                arr = np.asarray(data['accuracy'], dtype=float)
                # Align participants: intersect subject IDs between in-context and baseline
                sub_ids = data.get('sub_ids', [])
                if sub_ids and baseline_ids:
                    # Build aligned arrays if possible
                    try:
                        align_ids = [sid for sid in sub_ids if sid in baseline_ids]
                        if align_ids:
                            # Map indices
                            idx_ic = [sub_ids.index(sid) for sid in align_ids]
                            idx_bl = [list(baseline_ids).index(sid) if isinstance(baseline_ids, list) else None for sid in align_ids]
                            # If baseline_ids is a set, skip reindexing; fall back to original arrays
                    except Exception:
                        pass
                if arr.size == 0 or baseline_arr.size == 0:
                    continue
                def mean_ci95(a: np.ndarray) -> tuple[float, float, float, int]:
                    a = a[np.isfinite(a)]
                    n_val = int(a.size)
                    if n_val == 0:
                        return float('nan'), float('nan'), float('nan'), 0
                    mean = float(np.nanmean(a))
                    sem = float(np.nanstd(a, ddof=1) / np.sqrt(n_val)) if n_val > 1 else 0.0
                    ci = 1.96 * sem
                    return mean, mean - ci, mean + ci, n_val
                mean_ic, ci_l_ic, ci_u_ic, n_ic = mean_ci95(arr)
                mean_bl, ci_l_bl, ci_u_bl, n_bl = mean_ci95(baseline_arr)
                # Use paired test if we have the same N and (implicitly) same participants; else independent
                if arr.size == baseline_arr.size:
                    t_stat, p_val = ttest_rel(arr, baseline_arr, nan_policy='omit')
                    d = float(np.nanmean(arr - baseline_arr) / (np.nanstd(arr - baseline_arr, ddof=1) if arr.size > 1 else np.nan))
                    dfree = int(arr.size - 1)
                else:
                    t_stat, p_val = ttest_ind(arr, baseline_arr, nan_policy='omit')
                    d = calculate_cohens_d(arr, baseline_arr)
                    dfree = int((~np.isnan(arr)).sum() + (~np.isnan(baseline_arr)).sum() - 2)
                rows.append({
                    'dataset': data_size,
                    'mode': llm_mode,
                    'example_type': example_type,
                    'num_examples': n,
                    'mean_in_context': mean_ic,
                    'ci95_lower_in_context': ci_l_ic,
                    'ci95_upper_in_context': ci_u_ic,
                    'mean_zero_shot': mean_bl,
                    'ci95_lower_zero_shot': ci_l_bl,
                    'ci95_upper_zero_shot': ci_u_bl,
                    'n_in_context': n_ic,
                    'n_zero_shot': n_bl,
                    't': float(t_stat),
                    'df': dfree,
                    'p': float(p_val),
                    'cohen_d': float(d),
                })
            if not rows:
                print(f"No rows to export for {data_size}")
                continue
            df_out = pd.DataFrame(rows)
            # BH adjust FDR per "curve": for each example_type separately across num_examples
            df_out['p_adj'] = np.nan
            for etype in df_out['example_type'].unique():
                mask = df_out['example_type'] == etype
                pvals = df_out.loc[mask, 'p'].astype(float).tolist()
                if not pvals:
                    continue
                adj = _bh_adjust_local(pvals)
                df_out.loc[mask, 'p_adj'] = adj
            out_path = f'results/statistical_tests/two_panel_within_individual_{data_size}_{llm_mode}.xlsx'
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
                    df_out.to_excel(writer, sheet_name='stats', index=False)
                print(f"Saved two-panel stats to: {out_path}")
            except Exception:
                csv_path = out_path.replace('.xlsx', '.csv')
                df_out.to_csv(csv_path, index=False)
                print(f"Saved two-panel stats to: {csv_path}")
        except Exception as e:
            print(f"Failed to export two-panel stats for {data_size}: {e}")


def export_within_context_stats_from_raw(
    data_sizes: list[str] = ['small'],
    llm_mode: str = 'base'
):
    for data_size in data_sizes:
        try:
            in_context_results = load_in_context_results(data_size)
            llm_results = load_llm_results(data_size)
            participant_data = load_participant_level_data(in_context_results, llm_results)
            baseline_key = (0, 'baseline', llm_mode)
            if baseline_key not in participant_data:
                print(f"No baseline found for {data_size} {llm_mode}")
                continue
            baseline_arr = np.asarray(participant_data[baseline_key]['accuracy'], dtype=float)
            baseline_ids = set(participant_data[baseline_key].get('sub_ids', []))
            rows: list[dict] = []
            for (n, example_type, mode), data in participant_data.items():
                if int(n) in EXCLUDED_NUM_EXAMPLES:
                    continue
                if mode != f"{llm_mode}_within_context" or n == 0:
                    continue
                arr = np.asarray(data['accuracy'], dtype=float)
                sub_ids = data.get('sub_ids', [])
                # Paired t-test when sizes match (same participants)
                if arr.size == baseline_arr.size:
                    t_stat, p_val = ttest_rel(arr, baseline_arr, nan_policy='omit')
                    d = float(np.nanmean(arr - baseline_arr) / (np.nanstd(arr - baseline_arr, ddof=1) if arr.size > 1 else np.nan))
                    dfree = int(arr.size - 1)
                else:
                    t_stat, p_val = ttest_ind(arr, baseline_arr, nan_policy='omit')
                    d = calculate_cohens_d(arr, baseline_arr)
                    dfree = int((~np.isnan(arr)).sum() + (~np.isnan(baseline_arr)).sum() - 2)
                # CI helper
                def mean_ci95(a: np.ndarray) -> tuple[float, float, float, int]:
                    a = a[np.isfinite(a)]
                    n_val = int(a.size)
                    if n_val == 0:
                        return float('nan'), float('nan'), float('nan'), 0
                    mean = float(np.nanmean(a))
                    sem = float(np.nanstd(a, ddof=1) / np.sqrt(n_val)) if n_val > 1 else 0.0
                    ci = 1.96 * sem
                    return mean, mean - ci, mean + ci, n_val
                mean_ic, ci_l_ic, ci_u_ic, n_ic = mean_ci95(arr)
                mean_bl, ci_l_bl, ci_u_bl, n_bl = mean_ci95(baseline_arr)
                rows.append({
                    'dataset': data_size,
                    'mode': llm_mode,
                    'example_type': example_type,
                    'num_examples': n,
                    'mean_in_context': mean_ic,
                    'ci95_lower_in_context': ci_l_ic,
                    'ci95_upper_in_context': ci_u_ic,
                    'mean_zero_shot': mean_bl,
                    'ci95_lower_zero_shot': ci_l_bl,
                    'ci95_upper_zero_shot': ci_u_bl,
                    'n_in_context': n_ic,
                    'n_zero_shot': n_bl,
                    't': float(t_stat),
                    'df': dfree,
                    'p': float(p_val),
                    'cohen_d': float(d),
                })
            if not rows:
                print(f"No within-context rows to export for {data_size}")
                continue
            df_out = pd.DataFrame(rows)
            # BH adjust FDR per "curve": for each example_type separately across num_examples
            df_out['p_adj'] = np.nan
            for etype in df_out['example_type'].unique():
                mask = df_out['example_type'] == etype
                pvals = df_out.loc[mask, 'p'].astype(float).tolist()
                if not pvals:
                    continue
                adj = _bh_adjust_local(pvals)
                df_out.loc[mask, 'p_adj'] = adj
            out_path = f'results/statistical_tests/within_context_{data_size}_{llm_mode}.xlsx'
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
                    df_out.to_excel(writer, sheet_name='stats', index=False)
                print(f"Saved within-context stats to: {out_path}")
            except Exception:
                csv_path = out_path.replace('.xlsx', '.csv')
                df_out.to_csv(csv_path, index=False)
                print(f"Saved within-context stats to: {csv_path}")
        except Exception as e:
            print(f"Failed to export within-context stats for {data_size}: {e}")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze in-context learning results")
    parser.add_argument("--mode", type=str, default="standard", 
                       choices=["standard", "analyze_control_results"],
                       help="Analysis mode: 'standard' for regular analysis, 'analyze_control_results' for control experiment analysis")
    parser.add_argument("--data_size", type=str, default="small",
                       choices=["small", "large"],
                       help="Dataset size to analyze")
    parser.add_argument("--experiment", type=str, default=None,
                       choices=["exp1", "exp2", "exp3"],
                       help="Experiment label for control comparison outputs (exp1/exp2/exp3). If not set, inferred: small->exp1, large->exp2.")
    
    args = parser.parse_args()
    
    if args.mode == "analyze_control_results":
        # Run control analysis for one or both datasets (default: both so exp1 and exp2 don't override)
        print("Running control experiment analysis...")
        for data_size in ["small", "large"]:
            analyze_control_results(data_size=data_size, experiment=args.experiment)
        return
    
    # Process results for both datasets (original standard analysis)
    for data_size in ['small', 'large']:
        print(f"\nProcessing {data_size} dataset...")
        
        # Load and process in-context learning results
        in_context_results = load_in_context_results(data_size)
        in_context_df = process_in_context_results(in_context_results)
        
        # Load and process LLM results (zero-shot baselines)
        llm_results = load_llm_results(data_size)
        llm_df = process_llm_results(llm_results)
        
        # Load and process baseline models
        baseline_df = process_other_models(data_size)
        
        # Combine results
        results_df = pd.concat([llm_df, in_context_df], ignore_index=True)
        
        # Ensure num_examples is int
        results_df['num_examples'] = results_df['num_examples'].astype(int)
        
        print(f"\nLoaded results for {data_size} dataset:")
        print(f"  - Unique num_examples: {sorted(results_df['num_examples'].unique())}")
        print(f"  - Unique example_types: {sorted(results_df['example_type'].unique())}")
        print(f"  - Unique modes: {sorted(results_df['mode'].unique())}")
        
        # Show sample information for multi-sample experiments
        if 'n_samples' in results_df.columns:
            multi_sample_data = results_df[results_df['n_samples'] > 1]
        else:
            multi_sample_data = pd.DataFrame()  # Empty dataframe if no n_samples column
        if len(multi_sample_data) > 0:
            print(f"  - Multi-sample experiments found:")
            for _, row in multi_sample_data.iterrows():
                print(f"    * {row['num_examples']} examples, {row['example_type']} type, {row['mode']} mode: {row['n_samples']} samples")
        
        # Check for non-consecutive num_examples
        all_num_examples = sorted([x for x in results_df['num_examples'].unique() if x > 0])
        if len(all_num_examples) > 1:
            expected_consecutive = list(range(min(all_num_examples), max(all_num_examples) + 1))
            missing_values = [x for x in expected_consecutive if x not in all_num_examples]
            if missing_values:
                print(f"  - Note: Non-consecutive num_examples detected. Missing: {missing_values}")
            else:
                print(f"  - Consecutive num_examples from {min(all_num_examples)} to {max(all_num_examples)}")
        
        # Cache per-size outputs for two-panel figures
        if 'combined_cache' not in locals():
            combined_cache = {}
        combined_cache[data_size] = {
            'in_context_df': in_context_df.copy(),
            'llm_df': llm_df.copy(),
        }
        
        # Perform proper participant-level statistical tests
        print(f"\nPerforming participant-level statistical tests for {data_size} dataset...")
        print("Note: Using participant-level aggregation to handle cross-validation structure correctly.")
        test_results_df = perform_participant_level_statistical_tests(
            in_context_results, llm_results, data_size
        )
        
        # Perform comprehensive statistical tests (addressing accuracy data issues)
        print(f"\nPerforming comprehensive statistical tests for {data_size} dataset...")
        print("Note: Includes parametric, non-parametric, and transformed tests appropriate for accuracy data.")
        comprehensive_test_results_df = perform_comprehensive_statistical_tests(
            in_context_results, llm_results, data_size
        )
        
        # Perform participant × sample level tests  
        print(f"\nPerforming participant × sample level statistical tests for {data_size} dataset...")
        print("Note: Shows effect of different sample sizes (72×1 vs 72×20/40).")
        print("Warning: Samples within participants are not fully independent.")
        participant_sample_test_results_df = perform_participant_sample_level_tests(
            in_context_results, llm_results, data_size
        )
        
        # Save statistical test results to CSV
        os.makedirs('results/statistical_tests', exist_ok=True)
        
        csv_path = f'results/statistical_tests/participant_level_tests_{data_size}.csv'
        test_results_df.to_csv(csv_path, index=False)
        print(f"Saved participant-level statistical test results to: {csv_path}")
        
        comprehensive_csv_path = f'results/statistical_tests/comprehensive_tests_{data_size}.csv'
        comprehensive_test_results_df.to_csv(comprehensive_csv_path, index=False)
        print(f"Saved comprehensive statistical test results to: {comprehensive_csv_path}")
        
        participant_sample_csv_path = f'results/statistical_tests/participant_sample_level_tests_{data_size}.csv'
        participant_sample_test_results_df.to_csv(participant_sample_csv_path, index=False)
        print(f"Saved participant × sample level statistical test results to: {participant_sample_csv_path}")
        
        # Also perform the original tests for comparison (but note the issues)
        print(f"\nPerforming original statistical tests for {data_size} dataset (for comparison only)...")
        print("Warning: These tests don't properly handle cross-validation structure - use participant-level tests instead.")
        original_test_results_df = perform_statistical_tests(results_df)
        original_test_results_df['data_size'] = data_size
        original_csv_path = f'results/statistical_tests/original_tests_{data_size}.csv'
        original_test_results_df.to_csv(original_csv_path, index=False)
        print(f"Saved original statistical test results to: {original_csv_path}")
        
        # Create learning curve plots for each mode and context type
        modes = ['base', 'cot']
        context_types = ['within_individual', 'within_context'] if data_size == 'small' else ['within_individual']
        metrics = ['accuracy', 'likelihood']
        
        for mode in modes:
            for context_type in context_types:
                for metric in metrics:
                    # With PT overlay
                    save_path = f'figures/in_context/learning_curves/{mode}/learning_curve_{mode}_{context_type}_{metric}_{data_size}.png'
                    create_learning_curve_plot(
                        results_df,
                        baseline_df,
                        data_size,
                        mode,
                        context_type,
                        metric,
                        save_path,
                        include_pt=True
                    )
                    # Without PT overlay (unified suffix)
                    save_path_no_pt = f'figures/in_context/learning_curves/{mode}/learning_curve_{mode}_{context_type}_{metric}_{data_size}_noPT.png'
                    create_learning_curve_plot(
                        results_df,
                        baseline_df,
                        data_size,
                        mode,
                        context_type,
                        metric,
                        save_path_no_pt,
                        include_pt=False
                    )
        
        # Create effect size plots for each mode and context type (using participant × sample level data)
        for mode in modes:
            for context_type in context_types:
                for metric in metrics:
                    save_path = f'figures/in_context/effect_sizes/{mode}/effect_size_{mode}_{context_type}_{metric}_{data_size}.png'
                    create_effect_size_plot(
                        participant_sample_test_results_df,
                        data_size,
                        mode,
                        context_type,
                        metric,
                        save_path
                    )
        
        # Create mode comparison plots for each example type and number of examples
        example_types = ['think_aloud', 'choice', 'both']
        # Get unique num_examples (excluding 0 which is baseline)
        unique_num_examples = sorted([x for x in results_df['num_examples'].unique() if x > 0])
        
        for example_type in example_types:
            for num_examples in unique_num_examples:
                save_path = f'figures/in_context/mode_comparisons/{data_size}/{example_type}/mode_comparison_{data_size}_{example_type}_{num_examples}_examples.png'
                create_mode_comparison_plot(
                    results_df,
                    data_size,
                    example_type,
                    num_examples,
                    save_path
                )
        
        # Export two-panel within_individual stats (participant-level CI and t-tests vs zero-shot) using raw results
        export_two_panel_within_individual_stats_from_raw([data_size], llm_mode=modes[0])
        # Export within-context stats (small dataset only)
        if data_size == 'small':
            export_within_context_stats_from_raw([data_size], llm_mode=modes[0])

        # Single-panel plots mirroring style
        try:
            # Within-individual (single panel per dataset)
            sp_out = f'figures/in_context/learning_curves/{modes[0]}/single_panel_accuracy_{modes[0]}_within_individual_{data_size}.png'
            create_single_panel_accuracy_plot(results_df, llm_df, data_size, modes[0], 'within_individual', sp_out)
            # Within-context for small dataset only if present
            if data_size == 'small':
                sp_out_ctx = f'figures/in_context/learning_curves/{modes[0]}/single_panel_accuracy_{modes[0]}_within_context_{data_size}.png'
                create_single_panel_accuracy_plot(results_df, llm_df, data_size, modes[0], 'within_context', sp_out_ctx)
        except Exception as e:
            print(f"Warning: Failed to create single panel plots for {data_size}: {e}")

        # Spearman correlations: number of examples vs accuracy for within_individual and within_context, by example type
        try:
            corr_rows = []
            for ctx in context_types:
                for etype in ['think_aloud', 'choice', 'both']:
                    mode_key = f"{modes[0]}_{ctx}"
                    df_ctx = results_df[(results_df['mode'] == mode_key) & (results_df['example_type'] == etype) & (results_df['num_examples'] > 0)]
                    if not df_ctx.empty:
                        df_ctx = df_ctx[~df_ctx['num_examples'].astype(int).isin(EXCLUDED_NUM_EXAMPLES)]
                    if df_ctx.empty:
                        continue
                    # Use participant-level means already aggregated per configuration
                    xs = df_ctx['num_examples'].astype(int).values
                    ys = df_ctx['accuracy'].astype(float).values
                    if len(np.unique(xs)) > 1 and len(xs) == len(ys):
                        rho, p = spearmanr(xs, ys)
                        corr_rows.append({'dataset': data_size, 'context': ctx, 'example_type': etype, 'rho': float(rho), 'p': float(p)})
            if corr_rows:
                corr_df = pd.DataFrame(corr_rows)
                corr_out = f'results/statistical_tests/spearman_two_panel_{data_size}.csv'
                os.makedirs(os.path.dirname(corr_out), exist_ok=True)
                corr_df.to_csv(corr_out, index=False)
                print(f"Saved Spearman correlations to: {corr_out}")
        except Exception as e:
            print(f"Warning: Failed Spearman correlation export for {data_size}: {e}")
        
        print(f"Analysis completed for {data_size} dataset")
        print(f"Generated learning curve plots: {len(modes) * len(context_types) * len(metrics)}")
        print(f"Generated effect size plots: {len(modes) * len(context_types) * len(metrics)}")
        print(f"Generated mode comparison plots: {len(example_types) * len(unique_num_examples)}")
        
        # Statistical test summary
        print(f"\nStatistical Test Summary for {data_size} dataset:")
        print(f"  Participant-level tests: {len(test_results_df)} comparisons")
        print(f"    Significant (p < 0.05): {len(test_results_df[test_results_df['significant']])}")
        print(f"  Comprehensive tests: {len(comprehensive_test_results_df)} comparisons")
        print(f"    T-test significant: {len(comprehensive_test_results_df[comprehensive_test_results_df['t_significant']])}")
        print(f"    Mann-Whitney significant: {len(comprehensive_test_results_df[comprehensive_test_results_df['mw_significant']])}")
        print(f"    Transformed test significant: {len(comprehensive_test_results_df[comprehensive_test_results_df['trans_significant']])}")
        print(f"  Participant × sample level tests: {len(participant_sample_test_results_df)} comparisons")
        print(f"    Significant (p < 0.05): {len(participant_sample_test_results_df[participant_sample_test_results_df['significant']])}")
        print(f"  Original tests: {len(original_test_results_df)} comparisons")
        print(f"    Significant (p < 0.05): {len(original_test_results_df[original_test_results_df['significant']])}")
        
        # Statistical recommendations
        if len(comprehensive_test_results_df) > 0:
            print(f"\nStatistical Recommendations for {data_size} dataset:")
            recs = comprehensive_test_results_df['recommended_test'].value_counts()
            for test_type, count in recs.items():
                print(f"  {test_type}: {count} comparisons")
            
            # Show normality violations
            accuracy_tests = comprehensive_test_results_df[comprehensive_test_results_df['metric'] == 'accuracy']
            if len(accuracy_tests) > 0:
                non_normal = len(accuracy_tests[(accuracy_tests['ic_normal'] == 'non_normal') | 
                                              (accuracy_tests['bl_normal'] == 'non_normal')])
                print(f"  Accuracy data normality violations: {non_normal}/{len(accuracy_tests)} comparisons")
        
        # Show sample size differences in participant × sample tests
        if len(participant_sample_test_results_df) > 0:
            print(f"\nSample size ratios (in-context/baseline) for {data_size} dataset:")
            unique_ratios = participant_sample_test_results_df['sample_size_ratio'].unique()
            for ratio in sorted(unique_ratios):
                count = len(participant_sample_test_results_df[participant_sample_test_results_df['sample_size_ratio'] == ratio])
                print(f"  Ratio {ratio:.1f}:1 - {count} comparisons")
        
        # Show a few example comparisons from participant-level tests
        if len(test_results_df) > 0:
            print(f"\nExample participant-level test results for {data_size} dataset:")
            for _, row in test_results_df.head(3).iterrows():
                sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"  {row['mode']}, {row['example_type']}, {row['num_examples']} ex, {row['metric']}: ")
                print(f"    Cohen's d = {row['cohens_d']:.3f}, p = {row['p_value']:.3f}{sig_marker}, n = {row['in_context_n']} vs {row['baseline_n']}")
        
        # Show comparison between analysis levels for a few examples
        if len(test_results_df) > 0 and len(participant_sample_test_results_df) > 0:
            print(f"\nComparison of analysis levels for {data_size} dataset (first few examples):")
            for i in range(min(3, len(test_results_df))):
                participant_row = test_results_df.iloc[i]
                # Find matching participant × sample row
                matching_rows = participant_sample_test_results_df[
                    (participant_sample_test_results_df['mode'] == participant_row['mode']) &
                    (participant_sample_test_results_df['example_type'] == participant_row['example_type']) &
                    (participant_sample_test_results_df['num_examples'] == participant_row['num_examples']) &
                    (participant_sample_test_results_df['metric'] == participant_row['metric'])
                ]
                if len(matching_rows) > 0:
                    ps_row = matching_rows.iloc[0]
                    print(f"  {participant_row['mode']}, {participant_row['example_type']}, {participant_row['num_examples']} ex, {participant_row['metric']}:")
                    print(f"    Participant-level:        d={participant_row['cohens_d']:.3f}, p={participant_row['p_value']:.3f}, n={participant_row['in_context_n']}vs{participant_row['baseline_n']}")
                    print(f"    Participant×sample-level: d={ps_row['cohens_d']:.3f}, p={ps_row['p_value']:.3f}, n={ps_row['in_context_n']}vs{ps_row['baseline_n']}")
        
        # Show statistical test comparison for accuracy data
        if len(comprehensive_test_results_df) > 0:
            accuracy_comprehensive = comprehensive_test_results_df[comprehensive_test_results_df['metric'] == 'accuracy']
            if len(accuracy_comprehensive) > 0:
                print(f"\nStatistical Test Comparison for Accuracy Data ({data_size} dataset):")
                print("Note: Different tests may give different results for bounded accuracy data.")
                for i, row in accuracy_comprehensive.head(3).iterrows():
                    print(f"  {row['mode']}, {row['example_type']}, {row['num_examples']} ex:")
                    print(f"    T-test:           p={row['t_p_value']:.3f}, d={row['cohens_d']:.3f} {'*' if row['t_significant'] else ''}")
                    print(f"    Mann-Whitney U:   p={row['mannwhitney_p']:.3f}, r={row['rank_biserial_r']:.3f} {'*' if row['mw_significant'] else ''}")
                    if not np.isnan(row['t_transformed_p']):
                        print(f"    Arcsine-transform: p={row['t_transformed_p']:.3f}, d={row['cohens_d_transformed']:.3f} {'*' if row['trans_significant'] else ''}")
                    print(f"    Recommended: {row['recommended_test']}")
                    print()
                
                # Statistical advice
                print("Statistical Advice:")
                print("- For accuracy near 0.5: t-tests may be acceptable")
                print("- For accuracy near 0 or 1: use non-parametric tests")
                print("- For bounded accuracy (0.1-0.9): consider arcsine transformation")
                print("- Mann-Whitney U test is generally safer for accuracy data")
                print("- Check 'recommended_test' column in comprehensive_tests CSV for guidance")

    # After both sizes processed, produce two-panel Accuracy figures (with and without PT)
    try:
        small_cache = combined_cache.get('small') if 'combined_cache' in locals() else None
        large_cache = combined_cache.get('large') if 'combined_cache' in locals() else None
        if small_cache and large_cache:
            for mode in ['base', 'cot']:
                ctx = 'within_individual'
                for include_pt in [True, False]:
                    suffix = '' if include_pt else '_noPT'
                    out_path = f"figures/in_context/learning_curves/{mode}/two_panel_accuracy_{mode}_{ctx}{suffix}.png"
                    create_two_panel_accuracy_plot(
                        small_cache['in_context_df'],
                        large_cache['in_context_df'],
                        small_cache['llm_df'],
                        large_cache['llm_df'],
                        mode,
                        ctx,
                        include_pt,
                        out_path,
                    )
    except Exception as e:
        print(f"Warning: Failed to generate two-panel accuracy figures: {e}")

if __name__ == "__main__":
    main() 