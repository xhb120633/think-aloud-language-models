"""
Cognitive model fitting for risky decision-making experiment.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    DATA_PATHS,
    MODEL_CONFIGS,
    DATA_SIZE_CONFIGS
)
from utils.data_processing import (
    load_datasets,
    split_data_by_participant,
    get_participant_data,
    save_results
)
from utils.metrics import compute_metrics
from models.cognitive_models import ExpectedValueModel, ProspectTheoryModel

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run cognitive model fitting")
    parser.add_argument(
        "--data_size",
        type=str,
        default="large",
        choices=["small", "large", "all"],
        help="Size of dataset to use"
    )
    return parser.parse_args()

def run_cognitive_modeling_for_participant(
    participant_data: Dict,
    splits: Dict[str, np.ndarray],
    model_name: str,
    model: object
) -> Dict:
    """
    Run cognitive model fitting for a single participant.
    
    Args:
        participant_data: Dictionary of participant's data
        splits: Dictionary of train/test splits
        model_name: Name of the model
        model: Model instance
        
    Returns:
        Dictionary of results for this participant
    """
    results = []
    
    # Process each split
    for train_idx, test_idx in zip(splits['train_idx'], splits['test_idx']):
        # Fit model if applicable
        if model_name == "prospect_theory":
            fitted_params = model.fit(
                participant_data["probabilities"][train_idx],
                participant_data["outcomes"][train_idx],
                participant_data["choices"][train_idx]
            )
            split_results = {"params": fitted_params}
        else:
            split_results = {}
        
        # Make predictions
        predictions = model.predict(
            participant_data["probabilities"][test_idx],
            participant_data["outcomes"][test_idx]
        )
        
        # Compute metrics
        split_results["metrics"] = compute_metrics(
            predictions,
            participant_data["choices"][test_idx]
        )
        
        results.append(split_results)
    
    return results

def run_cognitive_modeling(
    data_size: str = "large"
) -> Dict:
    """
    Run cognitive model fitting.
    
    Args:
        data_size: Size of dataset to use
        
    Returns:
        Dictionary of results
    """
    if data_size == "all":
        data_sizes = ["small", "large"]
    else:
        data_sizes = [data_size]
    
    all_results = {}
    for ds in data_sizes:
        print(f"\nProcessing {ds} dataset...")
        
        # Load data
        decisions_df, _ = load_datasets(ds)
        
        # Get data splits
        splits = split_data_by_participant(decisions_df, ds)
        
        # Initialize models
        models = {
            "expected_value": ExpectedValueModel(),
            "prospect_theory": ProspectTheoryModel()
        }
        
        # Process each participant
        participant_results = {}
        for sub_id in decisions_df["sub_id"].unique():
            print(f"\nProcessing participant {sub_id}...")
            
            # Get participant data
            participant_data = get_participant_data(
                decisions_df,
                None,  # No think-aloud data needed for cognitive models
                sub_id,
                "cognitive"
            )
            
            # Run modeling for each model
            participant_results[sub_id] = {}
            for model_name, model in models.items():
                participant_results[sub_id][model_name] = run_cognitive_modeling_for_participant(
                    participant_data,
                    splits[sub_id],
                    model_name,
                    model
                )
        
        all_results[ds] = participant_results
    
    return all_results

def main():
    """Main entry point."""
    args = parse_args()
    
    # Run experiment
    results = run_cognitive_modeling(data_size=args.data_size)
    
    # Save results
    save_results(
        results,
        "cognitive_modeling",
        "all_models",
        args.data_size
    )

if __name__ == "__main__":
    main() 