"""
Visualization utilities for the think-aloud analysis project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .config import PLOT_STYLE, COLORS, FIGURES_DIR

def set_plot_style():
    """Set the plot style for all visualizations."""
    plt.style.use('seaborn')
    plt.rcParams.update(PLOT_STYLE)

def save_figure(fig: plt.Figure, filename: str):
    """
    Save a figure to disk.
    
    Args:
        fig: Matplotlib figure
        filename: Name of the file to save
    """
    fig.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close(fig)

def plot_model_comparison(
    results: Dict[str, float],
    title: str = "Model Comparison",
    ylabel: str = "Accuracy"
) -> plt.Figure:
    """
    Create a bar plot comparing model performance.
    
    Args:
        results: Dictionary of model names and their performance
        title: Plot title
        ylabel: Y-axis label
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    models = list(results.keys())
    scores = list(results.values())
    
    bars = ax.bar(
        models,
        scores,
        color=[COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"]]
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )
    
    # Customize plot
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_in_context_generalization(
    results: Dict[str, List[float]],
    xlabel: str = "Number of Shots",
    ylabel: str = "Accuracy"
) -> plt.Figure:
    """
    Create a line plot showing in-context generalization performance.
    
    Args:
        results: Dictionary of model names and their performance across shots
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines for each model
    for i, (model, scores) in enumerate(results.items()):
        ax.plot(
            range(1, len(scores) + 1),
            scores,
            marker='o',
            label=model,
            color=list(COLORS.values())[i]
        )
    
    # Customize plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def plot_sentence_contribution(
    contributions: Dict[str, List[float]],
    sentences: List[str],
    title: str = "Sentence Contribution Analysis"
) -> plt.Figure:
    """
    Create a heatmap showing sentence contributions.
    
    Args:
        contributions: Dictionary of model names and their sentence contributions
        sentences: List of sentences
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    # Create DataFrame for heatmap
    df = pd.DataFrame(contributions, index=sentences)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        df,
        annot=True,
        cmap='YlOrRd',
        center=0,
        ax=ax
    )
    
    # Customize plot
    ax.set_title(title)
    ax.set_xlabel("Models")
    ax.set_ylabel("Sentences")
    
    return fig

def plot_control_experiments(
    results: Dict[str, Dict[str, float]],
    title: str = "Control Experiments"
) -> plt.Figure:
    """
    Create a grouped bar plot for control experiments.
    
    Args:
        results: Dictionary of experiment names and their results
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    # Prepare data
    experiments = list(results.keys())
    models = list(results[experiments[0]].keys())
    
    x = np.arange(len(experiments))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bars
    for i, model in enumerate(models):
        values = [results[exp][model] for exp in experiments]
        ax.bar(
            x + i * width,
            values,
            width,
            label=model,
            color=list(COLORS.values())[i]
        )
    
    # Customize plot
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(experiments)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig 