"""
Configuration settings for the think-aloud analysis project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create necessary directories
for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data paths
SMALL_DATA_PATH = DATA_DIR / "small_dataset"
LARGE_DATA_PATH = DATA_DIR / "large_dataset"

# LLM Experiment Modes
LLM_MODES = {
    "base": "Direct generation without think-aloud",
    "cot": "Chain-of-thought with self-generated think-aloud",
    "human": "Human think-aloud based prediction"
}

# LLM Inference Configuration
LLM_CONFIG = {
    "temperature": 0.0,  # Deterministic outputs
    "max_tokens": 1000,  # Sufficient for think-aloud and choice
    "top_p": 1.0,       # No nucleus sampling
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None        # No specific stop sequences
}

# Model settings
MODEL_CONFIGS = {
    "ev": {
        "name": "Expected Value",
        "params": {}
    },
    "pt": {
        "name": "Prospect Theory",
        "params": {
            "alpha": 0.88,  # risk attitude
            "lambda": 2.25,  # loss aversion
            "gamma": 0.65   # probability weighting
        }
    },
    "neural": {
        "name": "Neural Network",
        "params": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1
        }
    },
    "llm": {
        "name": "LLM",
        "params": {
            "model": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 100
        }
    }
}

# Available models for experiments
AVAILABLE_MODELS = {
    "huggingface": [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "t5-base"
    ],
    "openai": [
        "gpt-3.5-turbo",
        "gpt-4",
        "text-davinci-003"
    ]
}

# Data size configurations
DATA_SIZE_CONFIGS = {
    "small": {
        "max_samples": 1000,
        "train_ratio": 0.9,
        "test_ratio": 0.1,
        "random_seed": 42
    },
    "large": {
        "max_samples": 10000,
        "train_ratio": 0.9,
        "test_ratio": 0.1,
        "random_seed": 42
    }
}

# Visualization settings
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3
}

# Color palette (Nature Human Behaviour style)
COLORS = {
    "primary": "#1f77b4",    # blue
    "secondary": "#ff7f0e",  # orange
    "tertiary": "#2ca02c",   # green
    "quaternary": "#d62728", # red
    "quinary": "#9467bd",    # purple
    "gray": "#7f7f7f"
}

# Experiment settings
EXPERIMENT_CONFIGS = {
    "model_comparison": {
        "num_folds": 5,
        "test_size": 0.1,
        "random_state": 42
    },
    "in_context": {
        "num_shots": [1, 3, 5],
        "num_examples": 10
    },
    "sentence_contribution": {
        "window_size": 3,
        "num_permutations": 100
    },
    "control": {
        "num_ablation_runs": 5,
        "shuffle_seeds": [42, 43, 44, 45, 46]
    }
} 