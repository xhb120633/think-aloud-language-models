"""
Experiment 2: In-Context Prospect Theory Fitting
This experiment parallels the LLM in-context learning but uses Prospect Theory model fitting.
Instead of LLM predictions with examples, we fit PT models on the training examples and test on test data.

The core idea: Match everything exactly from LLM in-context learning, where those examples 
become the training data we sample to fit the PT model. For each number of examples, we use 
the same sampling rule (random seed) to select training data, then fit PT models and test 
on the same test trials.

Key Features:
- Exact same data splits and sampling as exp2_in_context.py
- Fit PT models on training examples (with multiple random initializations)
- Test fitted models on the same test trials as LLM experiments
- Support for both within_individual and within_context learning paradigms
- Parallel processing for multiple starting point fitting
- Same result structure and saving format as LLM experiments

Usage:
    Normal: python exp2_in_context_pt.py --context_type within_individual --num_examples 2
    Control: python exp2_in_context_pt.py --context_type within_individual --control_mode --num_examples 2
    Multiple samples: python exp2_in_context_pt.py --context_type within_individual --n_samples 5 --num_examples 2
"""

import os
import json
import logging
import inspect
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import random
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import warnings

# Suppress numerical warnings during optimization
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero encountered.*')

from models.cognitive_models import ProspectTheoryModel, ExpectedValueModel
from utils.data_processing import (
    load_datasets,
    preprocess_think_aloud,
    get_participant_data
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def prepare_within_individual_examples(
    participant_data: Dict,
    participant_id: int,
    test_trial_index: int,
    num_examples: int,
    data_size: str = "small",
    random_state: int = 42,
    control_mode: bool = False,
    all_participants_data: Dict = None,
    decisions_df: pd.DataFrame = None
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Prepare training examples for within-individual PT fitting.
    Uses the exact same logic as exp2_in_context.py for consistency.
    
    Args:
        participant_data: Dictionary containing participant's data
        participant_id: ID of the participant (integer from LabelEncoder)
        test_trial_index: Index of the test trial
        num_examples: Number of training examples to prepare
        data_size: Size of dataset ("small" or "large")
        random_state: Random seed for reproducibility
        control_mode: If True, sample from other participants' data instead of same participant
        all_participants_data: Dictionary of all participants' data (required for control_mode)
        
    Returns:
        Tuple of (list of training example dictionaries, list of example trial IDs, list of example participant IDs)
    """
    if control_mode and all_participants_data is None:
        raise ValueError("all_participants_data must be provided when control_mode=True")
    
    if data_size == "small":
        # For small dataset, use leave-one-out (original implementation)
        train_indices = [i for i in range(len(participant_data["problem_id"])) if i != test_trial_index]
        train_indices.sort()
        
        # Set random seed based on participant ID and test trial
        combined_seed = hash(f"{participant_id}_{test_trial_index}") % (2**32) + random_state
        random.seed(combined_seed)
        
        # Randomly sample from the fixed training set
        selected_indices = random.sample(train_indices, min(num_examples, len(train_indices)))
    else:
        # For large dataset, use 90-10 split like cognitive/neural models
        all_indices = np.arange(len(participant_data["problem_id"]))
        
        # Create train/test split using the same 90-10 ratio
        train_indices, test_indices = train_test_split(
            all_indices,
            train_size=0.9,
            test_size=0.1,
            random_state=random_state
        )
        
        # If the test trial is not in the test set, we need to swap it
        if test_trial_index not in test_indices:
            # Find a trial in test_indices to swap with
            swap_idx = test_indices[0]
            # Remove test_trial_index from train_indices and add swap_idx
            train_indices = np.concatenate([train_indices[train_indices != test_trial_index], [swap_idx]])
            # Remove swap_idx from test_indices and add test_trial_index
            test_indices = np.concatenate([test_indices[test_indices != swap_idx], [test_trial_index]])
        
        # Randomly sample from the training set
        random.seed(random_state)
        selected_indices = random.sample(train_indices.tolist(), min(num_examples, len(train_indices)))
    
    # Prepare examples
    examples = []
    example_trial_ids = []
    example_participant_ids = []
    
    if not control_mode:
        # Original logic: use same participant's data
        # Get original p1, v1, p2, v2 vectors from decisions_df for secure processing
        for idx in selected_indices:
            trial_id = participant_data["problem_id"][idx]
            choice = participant_data["choices"][idx]
            
            # Find the original row in decisions_df to get p1, v1, p2, v2
            if decisions_df is not None:
                original_row = decisions_df[
                    (decisions_df['sub_id'] == participant_id) & 
                    (decisions_df['problem_id'] == trial_id)
                ].iloc[0]
                
                p1 = original_row['p1']
                v1 = [v / 1000.0 for v in original_row['v1']]  # Scale down values
                p2 = original_row['p2'] 
                v2 = [v / 1000.0 for v in original_row['v2']]  # Scale down values
            else:
                # Fallback to processed data (less secure)
                p1 = [participant_data["probabilities"][idx][0]]
                v1 = [participant_data["outcomes"][idx][0] / 1000.0]
                p2 = [participant_data["probabilities"][idx][1]]
                v2 = [participant_data["outcomes"][idx][1] / 1000.0]
            
            example = {
                "trial_id": trial_id,
                "p1": p1,
                "v1": v1,
                "p2": p2,
                "v2": v2,
                "choice": choice
            }
            examples.append(example)
            example_trial_ids.append(str(trial_id))
            example_participant_ids.append(str(participant_id))
    else:
        # Control mode: use other participants' data for the same trials
        if data_size == "small":
            # For small dataset: keep same trial IDs but sample from other participants
            for idx in selected_indices:
                trial_id = participant_data["problem_id"][idx]
                # Find other participants who have this trial
                available_participants = []
                for other_participant_id, other_participant_data in all_participants_data.items():
                    if other_participant_id != participant_id and trial_id in other_participant_data["problem_id"]:
                        other_trial_index = np.where(other_participant_data["problem_id"] == trial_id)[0][0]
                        available_participants.append((other_participant_id, other_participant_data, other_trial_index))
                
                if available_participants:
                    # Randomly select one other participant for this trial
                    selected_participant_id, selected_participant_data, selected_trial_index = random.choice(available_participants)
                    choice = selected_participant_data["choices"][selected_trial_index]
                    
                    # Get original p1, v1, p2, v2 vectors for control participant
                    if decisions_df is not None:
                        original_row = decisions_df[
                            (decisions_df['sub_id'] == selected_participant_id) & 
                            (decisions_df['problem_id'] == trial_id)
                        ].iloc[0]
                        
                        p1 = original_row['p1']
                        v1 = [v / 1000.0 for v in original_row['v1']]  # Scale down values
                        p2 = original_row['p2'] 
                        v2 = [v / 1000.0 for v in original_row['v2']]  # Scale down values
                    else:
                        # Fallback to processed data
                        p1 = [selected_participant_data["probabilities"][selected_trial_index][0]]
                        v1 = [selected_participant_data["outcomes"][selected_trial_index][0] / 1000.0]
                        p2 = [selected_participant_data["probabilities"][selected_trial_index][1]]
                        v2 = [selected_participant_data["outcomes"][selected_trial_index][1] / 1000.0]
                    
                    example = {
                        "trial_id": trial_id,
                        "p1": p1,
                        "v1": v1,
                        "p2": p2,
                        "v2": v2,
                        "choice": choice
                    }
                    examples.append(example)
                    example_trial_ids.append(str(trial_id))
                    example_participant_ids.append(str(selected_participant_id))
        else:
            # For large dataset: keep same trial IDs but sample from other participants
            # Also ensure test context is not in the example contexts
            test_trial_id = participant_data["problem_id"][test_trial_index]
            
            for idx in selected_indices:
                trial_id = participant_data["problem_id"][idx]
                
                # Skip if this trial has the same context as the test trial
                if trial_id == test_trial_id:
                    continue
                
                # Find other participants who have this trial
                available_participants = []
                for other_participant_id, other_participant_data in all_participants_data.items():
                    if other_participant_id != participant_id and trial_id in other_participant_data["problem_id"]:
                        other_trial_index = np.where(other_participant_data["problem_id"] == trial_id)[0][0]
                        available_participants.append((other_participant_id, other_participant_data, other_trial_index))
                
                if available_participants:
                    # Randomly select one other participant for this trial
                    selected_participant_id, selected_participant_data, selected_trial_index = random.choice(available_participants)
                    
                    example = {
                        "trial_id": trial_id,
                        "probabilities": selected_participant_data["probabilities"][selected_trial_index],
                        "outcomes": selected_participant_data["outcomes"][selected_trial_index],
                        "choice": selected_participant_data["choices"][selected_trial_index]
                    }
                    examples.append(example)
                    example_trial_ids.append(str(trial_id))
                    example_participant_ids.append(str(selected_participant_id))
    
    return examples, example_trial_ids, example_participant_ids

def prepare_within_context_examples(
    all_participants_data: Dict,
    test_participant_id: str,
    test_trial_id: str,
    num_examples: int,
    random_state: int = 2024,
    decisions_df: pd.DataFrame = None
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Prepare training examples for within-context PT fitting by finding trials with the same ID
    from other participants.
    
    Args:
        all_participants_data: Dictionary containing all participants' data
        test_participant_id: ID of the test participant
        test_trial_id: ID of the test trial
        num_examples: Number of training examples to prepare
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (list of training example dictionaries, list of example trial IDs, list of example participant IDs)
    """
    # Find participants who have the same trial ID
    available_examples = []
    for participant_id, participant_data in all_participants_data.items():
        if participant_id == test_participant_id:
            continue
            
        # Find the index of the trial in this participant's data
        trial_ids = participant_data["problem_id"]
        if test_trial_id in trial_ids:
            trial_index = np.where(trial_ids == test_trial_id)[0][0]
            available_examples.append({
                "participant_id": participant_id,
                "trial_index": trial_index,
                "participant_data": participant_data
            })
    
    if len(available_examples) == 0:
        return [], [], []
    
    # Randomly select examples
    random.seed(random_state)
    selected_examples = random.sample(available_examples, min(num_examples, len(available_examples)))
    
    # Format examples
    examples = []
    example_trial_ids = []
    example_participant_ids = []
    
    for example in selected_examples:
        participant_id = example["participant_id"]
        trial_index = example["trial_index"]
        participant_data = example["participant_data"]
        choice = participant_data["choices"][trial_index]
        
        # Get original p1, v1, p2, v2 vectors from decisions_df
        if decisions_df is not None:
            original_row = decisions_df[
                (decisions_df['sub_id'] == participant_id) & 
                (decisions_df['problem_id'] == test_trial_id)
            ].iloc[0]
            
            p1 = original_row['p1']
            v1 = [v / 1000.0 for v in original_row['v1']]  # Scale down values
            p2 = original_row['p2'] 
            v2 = [v / 1000.0 for v in original_row['v2']]  # Scale down values
        else:
            # Fallback to processed data
            p1 = [participant_data["probabilities"][trial_index][0]]
            v1 = [participant_data["outcomes"][trial_index][0] / 1000.0]
            p2 = [participant_data["probabilities"][trial_index][1]]
            v2 = [participant_data["outcomes"][trial_index][1] / 1000.0]
        
        example_dict = {
            "trial_id": test_trial_id,
            "p1": p1,
            "v1": v1,
            "p2": p2,
            "v2": v2,
            "choice": choice,
            "participant_id": participant_id
        }
        examples.append(example_dict)
        example_trial_ids.append(str(test_trial_id))
        example_participant_ids.append(str(participant_id))
    
    return examples, example_trial_ids, example_participant_ids


def fit_pt_model_with_retries(
    training_examples: List[Dict],
    model_type: str = "PT",
    n_repeat: int = 50,
    num_cores: Optional[int] = None
) -> Tuple[Dict, float, float]:
    """
    Fit Prospect Theory model on training examples with multiple random initializations.
    Uses the original model's internal parallelization for efficiency.
    
    Args:
        training_examples: List of training example dictionaries
        model_type: Type of model ("PT" for Prospect Theory, "EV" for Expected Value)
        n_repeat: Number of random initializations for fitting
        num_cores: Number of CPU cores to use for internal model parallelization
        
    Returns:
        Tuple of (fitted_parameters, training_accuracy, neg_log_likelihood)
    """
    if len(training_examples) == 0:
        return {}, 0.0, np.inf
    
    # Convert training examples to the format expected by cognitive models
    # Now using secure p1, v1, p2, v2 format directly from original data
    p1_list = []
    v1_list = []
    p2_list = []
    v2_list = []
    choices_list = []
    
    for example in training_examples:
        # Convert percentages to probabilities before model fitting
        p1_converted = [p / 100.0 for p in example["p1"]]  # Convert percentage to probability
        p2_converted = [p / 100.0 for p in example["p2"]]  # Convert percentage to probability
        
        p1_list.append(p1_converted)
        v1_list.append(example["v1"])
        p2_list.append(p2_converted)
        v2_list.append(example["v2"])
        choices_list.append([1 - example["choice"]])  # Convert to [0] or [1] format expected by models
    
    # Use original model fitting approach with optimized internal parallelization
    if model_type == "PT":
        model = ProspectTheoryModel()
    else:  # EV
        model = ExpectedValueModel()
    
    try:
        # Use original model fitting with internal parallelization
        fit_result = model.fit(
            p1=p1_list,
            v1=v1_list,
            p2=p2_list,
            v2=v2_list,
            choices=choices_list,
            n_repeat=n_repeat,
            num_cores=num_cores if num_cores else 1
        )
        
        # Calculate training accuracy
        training_accuracy = calculate_model_accuracy(model, training_examples)
        
        return fit_result['best_params'], training_accuracy, fit_result['neg_log_likelihood']
        
    except Exception as e:
        logger.error(f"Error fitting {model_type} model: {e}")
        logger.error(f"Model type: {type(model)}")
        logger.error(f"Model location: {model.__class__.__module__}")
        return {}, 0.0, np.inf

def calculate_model_accuracy(model, examples: List[Dict]) -> float:
    """Calculate accuracy of fitted model on given examples."""
    if len(examples) == 0:
        return 0.0
    
    correct = 0
    total = len(examples)
    
    for example in examples:
        # Use the secure p1, v1, p2, v2 format directly
        p1 = example["p1"]
        v1 = example["v1"]
        p2 = example["p2"]
        v2 = example["v2"]
        actual_choice = example["choice"]
        
        # Convert to numpy arrays with proper shape for model prediction
        p1_test = np.array([p1])
        v1_test = np.array([v1])
        p2_test = np.array([p2])
        v2_test = np.array([v2])
        
        # Make prediction
        if hasattr(model, 'predict_choices'):
            prediction = model.predict_choices(p1_test, v1_test, p2_test, v2_test)[0]
        else:
            prediction = model.predict(p1_test, v1_test, p2_test, v2_test)[0] > 0.5
            prediction = int(prediction)
        
        if prediction == actual_choice:
            correct += 1
    
    return correct / total

def calculate_model_likelihood(model, examples: List[Dict]) -> float:
    """Calculate average likelihood of fitted model on given examples."""
    if len(examples) == 0:
        return 0.5
    
    total_likelihood = 0.0
    valid_examples = 0
    
    for example in examples:
        # Use the secure p1, v1, p2, v2 format directly
        p1 = example["p1"]
        v1 = example["v1"]
        p2 = example["p2"]
        v2 = example["v2"]
        actual_choice = example["choice"]
        
        # Convert to numpy arrays with proper shape for model prediction
        p1_test = np.array([p1])
        v1_test = np.array([v1])
        p2_test = np.array([p2])
        v2_test = np.array([v2])
        
        # Get choice probabilities
        choice_prob = model.predict(p1_test, v1_test, p2_test, v2_test)[0]
        
        # Calculate likelihood of actual choice
        if actual_choice == 0:
            likelihood = 1 - choice_prob
        else:
            likelihood = choice_prob
        
        total_likelihood += likelihood
        valid_examples += 1
    
    return total_likelihood / valid_examples if valid_examples > 0 else 0.5

def process_participant_trial_pt(
    participant_id: int,
    participant_data: Dict,
    test_trial_index: int,
    num_examples: int,
    data_size: str = "small",
    sample_i: int = 0,
    sample_seed: int = 2025,
    control_mode: bool = False,
    all_participants_data: Dict = None,
    model_type: str = "PT",
    n_repeat: int = 50,
    decisions_df: pd.DataFrame = None,
    num_cores: int = None
) -> Dict:
    """Process a single participant's trial with PT model fitting."""
    # Get test data using original p1, v1, p2, v2 vectors
    test_choice = participant_data["choices"][test_trial_index]
    test_trial_id = participant_data["problem_id"][test_trial_index]
    
    # Get original p1, v1, p2, v2 for test trial
    if decisions_df is not None:
        test_row = decisions_df[
            (decisions_df['sub_id'] == participant_id) & 
            (decisions_df['problem_id'] == test_trial_id)
        ].iloc[0]
        test_p1 = test_row['p1']
        test_v1 = [v / 1000.0 for v in test_row['v1']]  # Scale down values
        test_p2 = test_row['p2']
        test_v2 = [v / 1000.0 for v in test_row['v2']]  # Scale down values
    else:
        # Fallback to processed data
        test_p1 = [participant_data["probabilities"][test_trial_index][0]]
        test_v1 = [participant_data["outcomes"][test_trial_index][0] / 1000.0]
        test_p2 = [participant_data["probabilities"][test_trial_index][1]]
        test_v2 = [participant_data["outcomes"][test_trial_index][1] / 1000.0]
    
    # Prepare training examples with sample-specific seed
    training_examples, example_trial_ids, example_participant_ids = prepare_within_individual_examples(
        participant_data,
        participant_id,
        test_trial_index,
        num_examples,
        data_size=data_size,
        random_state=sample_seed,
        control_mode=control_mode,
        all_participants_data=all_participants_data,
        decisions_df=decisions_df
    )
    
    # Fit PT model on training examples
    fitted_params, training_accuracy, neg_log_likelihood = fit_pt_model_with_retries(
        training_examples,
        model_type=model_type,
        n_repeat=n_repeat,
        num_cores=num_cores  # Use internal model parallelization
    )
    
    # Test fitted model on test trial
    if fitted_params:
        # Create model with fitted parameters
        if model_type == "PT":
            test_model = ProspectTheoryModel(fitted_params)
        else:
            test_model = ExpectedValueModel(fitted_params)
        
        # Convert test probabilities from percentages to decimals before prediction
        test_p1_converted = [p / 100.0 for p in test_p1]  # Convert percentage to probability
        test_p2_converted = [p / 100.0 for p in test_p2]  # Convert percentage to probability
        
        # Make prediction on test trial using converted probabilities
        p1_test = np.array([test_p1_converted])
        v1_test = np.array([test_v1])
        p2_test = np.array([test_p2_converted])
        v2_test = np.array([test_v2])
        
        if hasattr(test_model, 'predict_choices'):
            predicted_choice = test_model.predict_choices(p1_test, v1_test, p2_test, v2_test)[0]
        else:
            choice_prob = test_model.predict(p1_test, v1_test, p2_test, v2_test)[0]
            predicted_choice = int(choice_prob > 0.5)
        
        # Calculate test likelihood
        choice_prob = test_model.predict(p1_test, v1_test, p2_test, v2_test)[0]
        
        if test_choice == 0:
            test_likelihood = 1 - choice_prob
        else:
            test_likelihood = choice_prob
            
        test_accuracy = 1.0 if predicted_choice == test_choice else 0.0
        
    else:
        # Fallback if fitting failed
        predicted_choice = None
        test_likelihood = 0.5
        test_accuracy = 0.0
    
    # Store result in same format as LLM experiment
    result = {
        "sub_id": str(participant_id),
        "trial_id": participant_data["problem_id"][test_trial_index],
        "model_type": model_type,
        "num_examples": num_examples,
        "sample_id": sample_i,
        "sample_seed": sample_seed,
        "control_mode": control_mode,
        "actual_choice": test_choice,
        "predicted_choice": predicted_choice,
        "test_accuracy": test_accuracy,
        "test_likelihood": test_likelihood,
        "training_accuracy": training_accuracy,
        "neg_log_likelihood": neg_log_likelihood,
        "fitted_params": fitted_params,
        "num_training_examples": len(training_examples),
        "example_trial_ids": example_trial_ids,
        "example_participant_ids": example_participant_ids,
        "pa": [float(p) / 100.0 for p in test_p1] if test_p1 else [],  # Complete probability vector for Option A
        "va": [float(v) for v in test_v1] if test_v1 else [],  # Complete value vector for Option A
        "pb": [float(p) / 100.0 for p in test_p2] if test_p2 else [],  # Complete probability vector for Option B
        "vb": [float(v) for v in test_v2] if test_v2 else []   # Complete value vector for Option B
    }
    return result

def process_trial_participant_pt(
    trial_id: str,
    test_participant_id: int,
    participants_data: Dict,
    num_examples: int,
    sample_i: int = 0,
    sample_seed: int = 2025,
    model_type: str = "PT",
    n_repeat: int = 50,
    decisions_df: pd.DataFrame = None,
    num_cores: int = None
) -> Dict:
    """Process a single trial for a participant in within-context mode with PT fitting."""
    # Get test participant's data for this trial
    test_participant_data = participants_data[test_participant_id]
    test_trial_index = np.where(test_participant_data["problem_id"] == trial_id)[0][0]
    
    # Get test data using original p1, v1, p2, v2 vectors
    test_choice = test_participant_data["choices"][test_trial_index]
    
    # Get original p1, v1, p2, v2 for test trial
    if decisions_df is not None:
        test_row = decisions_df[
            (decisions_df['sub_id'] == test_participant_id) & 
            (decisions_df['problem_id'] == trial_id)
        ].iloc[0]
        test_p1 = test_row['p1']
        test_v1 = [v / 1000.0 for v in test_row['v1']]  # Scale down values
        test_p2 = test_row['p2']
        test_v2 = [v / 1000.0 for v in test_row['v2']]  # Scale down values
    else:
        # Fallback to processed data
        test_p1 = [test_participant_data["probabilities"][test_trial_index][0]]
        test_v1 = [test_participant_data["outcomes"][test_trial_index][0] / 1000.0]
        test_p2 = [test_participant_data["probabilities"][test_trial_index][1]]
        test_v2 = [test_participant_data["outcomes"][test_trial_index][1] / 1000.0]
    
    # Prepare training examples from other participants
    training_examples, example_trial_ids, example_participant_ids = prepare_within_context_examples(
        participants_data,
        test_participant_id,
        trial_id,
        num_examples,
        random_state=sample_seed,
        decisions_df=decisions_df
    )
    
    # Fit PT model on training examples
    fitted_params, training_accuracy, neg_log_likelihood = fit_pt_model_with_retries(
        training_examples,
        model_type=model_type,
        n_repeat=n_repeat,
        num_cores=num_cores  # Use internal model parallelization
    )
    
    # Test fitted model on test trial
    if fitted_params:
        # Create model with fitted parameters
        if model_type == "PT":
            test_model = ProspectTheoryModel(fitted_params)
        else:
            test_model = ExpectedValueModel(fitted_params)
        
        # Convert test probabilities from percentages to decimals before prediction
        test_p1_converted = [p / 100.0 for p in test_p1]  # Convert percentage to probability
        test_p2_converted = [p / 100.0 for p in test_p2]  # Convert percentage to probability
        
        # Make prediction on test trial using converted probabilities
        p1_test = np.array([test_p1_converted])
        v1_test = np.array([test_v1])
        p2_test = np.array([test_p2_converted])
        v2_test = np.array([test_v2])
        
        if hasattr(test_model, 'predict_choices'):
            predicted_choice = test_model.predict_choices(p1_test, v1_test, p2_test, v2_test)[0]
        else:
            choice_prob = test_model.predict(p1_test, v1_test, p2_test, v2_test)[0]
            predicted_choice = int(choice_prob > 0.5)
        
        # Calculate test likelihood
        choice_prob = test_model.predict(p1_test, v1_test, p2_test, v2_test)[0]
        
        if test_choice == 0:
            test_likelihood = 1 - choice_prob
        else:
            test_likelihood = choice_prob
            
        test_accuracy = 1.0 if predicted_choice == test_choice else 0.0
        
    else:
        # Fallback if fitting failed
        predicted_choice = None
        test_likelihood = 0.5
        test_accuracy = 0.0
    
    # Store result
    result = {
        "sub_id": str(test_participant_id),
        "trial_id": trial_id,
        "model_type": model_type,
        "num_examples": num_examples,
        "sample_id": sample_i,
        "sample_seed": sample_seed,
        "actual_choice": test_choice,
        "predicted_choice": predicted_choice,
        "test_accuracy": test_accuracy,
        "test_likelihood": test_likelihood,
        "training_accuracy": training_accuracy,
        "neg_log_likelihood": neg_log_likelihood,
        "fitted_params": fitted_params,
        "num_training_examples": len(training_examples),
        "example_trial_ids": example_trial_ids,
        "example_participant_ids": example_participant_ids,
        "pa": [float(p) / 100.0 for p in test_p1] if test_p1 else [],  # Complete probability vector for Option A
        "va": [float(v) for v in test_v1] if test_v1 else [],  # Complete value vector for Option A
        "pb": [float(p) / 100.0 for p in test_p2] if test_p2 else [],  # Complete probability vector for Option B
        "vb": [float(v) for v in test_v2] if test_v2 else []   # Complete value vector for Option B
    }
    return result

def process_complete_trial_pt(
    trial_info: Dict,
    participants_data: Dict,
    num_examples: int,
    data_size: str,
    control_mode: bool = False,
    model_type: str = "PT",
    n_repeat: int = 50,
    n_samples: int = 20,
    decisions_df: pd.DataFrame = None,
    num_cores_per_trial: int = 1
) -> List[Dict]:
    """
    Process a complete trial (all samples) for trial-level parallelization.
    Each process handles one trial with all its samples sequentially.
    
    Args:
        trial_info: Dictionary containing trial information
        participants_data: All participants data
        num_examples: Number of examples to use
        data_size: Dataset size
        control_mode: If True, sample from other participants' data
        model_type: Type of model to fit ("PT" or "EV")
        n_repeat: Number of random initializations for fitting
        n_samples: Number of samples per trial
        decisions_df: Original decisions DataFrame
        num_cores_per_trial: Number of cores each trial can use for PT fitting
        
    Returns:
        List of result dictionaries for all samples of this trial
    """
    trial_results = []
    
    try:
        for sample_i in range(n_samples):
            sample_seed = 2025 + sample_i
            
            if trial_info['context_type'] == "within_context":
                # Within-context processing
                result = process_trial_participant_pt(
                    trial_info['trial_id'],
                    trial_info['participant_id'],
                    participants_data,
                    num_examples,
                    sample_i,
                    sample_seed,
                    model_type,
                    n_repeat,
                    decisions_df,
                    num_cores_per_trial
                )
            else:
                # Within-individual processing
                result = process_participant_trial_pt(
                    trial_info['participant_id'],
                    trial_info['participant_data'],
                    trial_info['test_trial_index'],
                    num_examples,
                    data_size,
                    sample_i,
                    sample_seed,
                    control_mode,
                    participants_data,
                    model_type,
                    n_repeat,
                    decisions_df,
                    num_cores_per_trial
                )
            trial_results.append(result)
            
    except Exception as e:
        logger.error(f"Error processing complete trial: {e}")
        # Add error results for remaining samples
        for remaining_sample_i in range(len(trial_results), n_samples):
            error_result = create_error_result_pt(
                {**trial_info, 'sample_i': remaining_sample_i}, 
                str(e), 
                control_mode, 
                model_type
            )
            trial_results.append(error_result)
    
    return trial_results

def process_batch_group_pt(
    batch_group: List[Dict],
    participants_data: Dict,
    num_examples: int,
    data_size: str,
    control_mode: bool = False,
    model_type: str = "PT",
    n_repeat: int = 50,
    num_workers: int = 4,
    decisions_df: pd.DataFrame = None
) -> List[Dict]:
    """
    Process a batch group (multiple samples of the same trial) through parallel PT fitting.
    All tasks in the group are for the same trial but with different example samplings.
    Back to original sample-level parallelization for best performance.
    
    Args:
        batch_group: List of task dictionaries for the same trial
        participants_data: All participants data
        num_examples: Number of examples to use
        data_size: Dataset size
        control_mode: If True, sample from other participants' data instead of same participant
        model_type: Type of model to fit ("PT" or "EV")
        n_repeat: Number of random initializations for fitting
        num_workers: Number of parallel workers for this batch
        
    Returns:
        List of result dictionaries
    """
    if not batch_group:
        return []
    
    # Process each task in the batch group in parallel (original approach)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for task in batch_group:
            if "within_context" in task.get('context_type', ''):
                # Within-context processing
                future = executor.submit(
                    process_complete_trial_pt,
                    task,
                    participants_data,
                    num_examples,
                    data_size,
                    control_mode,
                    model_type,
                    n_repeat,
                    task['n_samples'],
                    decisions_df,
                    1  # Use single core per process to avoid nested parallelism
                )
            else:
                # Within-individual processing
                future = executor.submit(
                    process_complete_trial_pt,
                    task,
                    participants_data,
                    num_examples,
                    data_size,
                    control_mode,
                    model_type,
                    n_repeat,
                    task['n_samples'],
                    decisions_df,
                    1  # Use single core per process to avoid nested parallelism
                )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                group_results = future.result()
                results.extend(group_results)
            except Exception as e:
                logger.error(f"Error processing PT task: {e}")
                # Add error result
                task = batch_group[len(results)]  # Get corresponding task
                error_result = create_error_result_pt(task, str(e), control_mode, model_type)
                results.append(error_result)
    
    return results

def create_error_result_pt(task: Dict, error_message: str, control_mode: bool = False, model_type: str = "PT") -> Dict:
    """Create an error result for failed PT processing."""
    return {
        "sub_id": str(task.get('participant_id', 'unknown')),
        "trial_id": task.get('trial_id', 'unknown'),
        "model_type": model_type,
        "num_examples": task.get('num_examples', 0),
        "sample_id": task.get('sample_i', 0),
        "sample_seed": task.get('sample_seed', 0),
        "control_mode": control_mode,
        "actual_choice": None,
        "predicted_choice": None,
        "test_accuracy": 0.0,
        "test_likelihood": 0.5,
        "training_accuracy": 0.0,
        "neg_log_likelihood": np.inf,
        "fitted_params": {},
        "num_training_examples": 0,
        "example_trial_ids": [],
        "example_participant_ids": [],
        "error": error_message,
        "pa": [],  # Empty vector for error case
        "va": [],  # Empty vector for error case
        "pb": [],  # Empty vector for error case
        "vb": []   # Empty vector for error case
    }

def run_in_context_pt_experiment(
    data_size: str = "small",
    context_type: str = "within_individual",
    num_examples: int = 2,
    model_type: str = "PT",
    output_dir: str = "results/exp2_in_context_pt",
    checkpoint_interval: int = 100,
    random_state: int = 42,
    n_samples: int = 1,
    control_mode: bool = False,
    n_repeat: int = 50,
    num_workers: int = 4
) -> None:
    """
    Run the in-context Prospect Theory fitting experiment.
    
    Args:
        data_size: Size of dataset ("small" or "large")
        context_type: Context learning type:
            - "within_individual": Fit on same participant's other trials
            - "within_context": Fit on other participants with same trial
        num_examples: Number of training examples to use for PT fitting
        model_type: Type of model to fit ("PT" for Prospect Theory, "EV" for Expected Value)
        output_dir: Directory to save results
        checkpoint_interval: Number of trials to process before saving checkpoint
        random_state: Random seed for reproducibility
        n_samples: Number of different example samplings per test trial (default: 1)
        control_mode: If True, use other participants' data instead of same participant for within_individual learning
        n_repeat: Number of random initializations for PT fitting (default: 50)
        num_workers: Number of parallel workers for PT fitting (default: 4)
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # For large dataset, only within_individual context is supported
    if data_size == "large" and context_type == "within_context":
        print("⚠️  Large dataset detected. Only within_individual context is supported.")
        context_type = "within_individual"
    
    # Control mode is only relevant for within_individual context
    if control_mode and context_type == "within_context":
        print("⚠️  Control mode is only relevant for within_individual context. Setting control_mode=False.")
        control_mode = False
    
    # Load and preprocess data
    print(f"\nLoading {data_size} dataset...")
    decisions_df, think_aloud_df = load_datasets(data_size)
    think_aloud_df = preprocess_think_aloud(think_aloud_df)
    
    # Store the original decisions_df for secure p1, v1, p2, v2 access
    original_decisions_df = decisions_df.copy()
    
    # Get all participant IDs
    participant_ids = decisions_df["sub_id"].unique().tolist()
    print(f"Found {len(participant_ids)} participants")
    
    # Prepare participants data
    participants_data = {}
    for participant_id in tqdm(participant_ids, desc="Loading participant data"):
        participants_data[participant_id] = get_participant_data(
            decisions_df,
            think_aloud_df,
            participant_id,
            "llm"  # Use same data processing as LLM experiments
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint directory
    control_suffix = "_control" if control_mode else ""
    checkpoint_dir = os.path.join(
        output_dir, 
        "checkpoints", 
        f"{model_type}_{data_size}_{context_type}_{num_examples}examples_{n_samples}samples{control_suffix}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nExperiment Configuration:")
    print(f"  Model Type: {model_type}")
    print(f"  Context Type: {context_type}")
    print(f"  Number of Examples: {num_examples}")
    print(f"  Number of Samples: {n_samples}")
    print(f"  Control Mode: {control_mode}")
    print(f"  Random Initializations per Fit: {n_repeat}")
    print(f"  Parallel Workers: {num_workers}")
    print(f"  Total combinations: {n_samples} sample(s) per trial")
    
    # Prepare all trials for optimal parallel processing at trial level
    print(f"Organizing tasks for parallel PT fitting...")
    
    # Create all trial tasks upfront
    all_trial_tasks = []
    
    if context_type == "within_individual":
        # Collect all trials for within_individual
        for participant_id in participant_ids:
            participant_data = participants_data[participant_id]
            
            if data_size == "large":
                # For large dataset, use 90-10 split - only process test trials
                all_indices = np.arange(len(participant_data["problem_id"]))
                train_indices, test_indices = train_test_split(
                    all_indices,
                    train_size=0.9,
                    test_size=0.1,
                    random_state=random_state
                )
                test_trial_indices = test_indices.tolist()
            else:
                # For small dataset, use leave-one-out - each trial is tested once
                test_trial_indices = list(range(len(participant_data["choices"])))
            
            for test_trial_index in test_trial_indices:
                trial_task = {
                    'context_type': context_type,
                    'participant_id': participant_id,
                    'participant_data': participant_data,
                    'test_trial_index': test_trial_index,
                    'n_samples': n_samples
                }
                all_trial_tasks.append(trial_task)
    
    else:  # within_context
        # For within_context, organize by trial_id and participant combinations
        unique_trial_ids = set()
        for participant_data in participants_data.values():
            unique_trial_ids.update(participant_data["problem_id"])
        
        for trial_id in unique_trial_ids:
            # Find participants who have this trial
            participants_with_trial = []
            for participant_id, participant_data in participants_data.items():
                if trial_id in participant_data["problem_id"]:
                    participants_with_trial.append(participant_id)
            
            # For each participant with this trial, create test tasks
            for test_participant_id in participants_with_trial:
                trial_task = {
                    'context_type': context_type,
                    'trial_id': trial_id,
                    'participant_id': test_participant_id,
                    'n_samples': n_samples
                }
                all_trial_tasks.append(trial_task)
    
    print(f"Created {len(all_trial_tasks)} trial tasks")
    print(f"Total samples to process: {len(all_trial_tasks) * n_samples}")
    
    # Calculate cores per trial for optimal resource usage
    cores_per_trial = max(1, num_workers // min(num_workers, len(all_trial_tasks)))
    effective_workers = min(num_workers, len(all_trial_tasks))
    
    print(f"Using {effective_workers} parallel workers, {cores_per_trial} cores per trial")
    
    # Process all trials with dynamic work distribution
    print(f"Processing {len(all_trial_tasks)} trials with dynamic PT fitting...")
    
    all_results = []
    completed_trials = 0
    
    # Use submit/as_completed pattern for dynamic work distribution
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        # Submit all trial tasks immediately
        future_to_trial = {}
        for trial_task in all_trial_tasks:
            future = executor.submit(
                process_complete_trial_pt,
                trial_task,
                participants_data,
                num_examples,
                data_size,
                control_mode,
                model_type,
                n_repeat,
                n_samples,
                original_decisions_df,
                cores_per_trial
            )
            future_to_trial[future] = trial_task
        
        # Process results as they complete (dynamic load balancing)
        from concurrent.futures import as_completed
        
        with tqdm(total=len(all_trial_tasks), desc="Completing trials") as pbar:
            for future in as_completed(future_to_trial):
                trial_task = future_to_trial[future]
                try:
                    trial_results = future.result()
                    all_results.extend(trial_results)
                    completed_trials += 1
                    
                    # Save checkpoint every checkpoint_interval trials
                    if completed_trials % checkpoint_interval == 0:
                        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{len(all_results)}.json")
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            # Convert NumPy types to native Python types for JSON serialization
                            checkpoint_data = convert_numpy_types({
                                "metadata": {
                                    # Core experiment parameters
                                    "model_type": model_type,
                                    "dataset": data_size,
                                    "context_type": context_type,
                                    "num_examples": num_examples,
                                    "n_samples": n_samples,
                                    "control_mode": control_mode,
                                    "n_repeat": n_repeat,
                                    "num_workers": num_workers,
                                    
                                    # Checkpoint metadata
                                    "processed_count": len(all_results),
                                    "processed_trials": completed_trials,
                                    "timestamp": datetime.now().isoformat(),
                                    "is_checkpoint": True
                                },
                                "results": all_results
                            })
                            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                        print(f"Saved checkpoint with {len(all_results)} results to {checkpoint_file}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'completed': completed_trials,
                        'results': len(all_results),
                        'participant': trial_task['participant_id'],
                        'trial': trial_task.get('test_trial_index', trial_task.get('trial_id', 'unknown'))
                    })
                    
                except Exception as e:
                    trial_id = trial_task.get('test_trial_index', trial_task.get('trial_id', 'unknown'))
                    print(f"Error processing trial {trial_task['participant_id']}-{trial_id}: {str(e)}")
                    pbar.update(1)
    
    # Save final results
    output_file = os.path.join(
        output_dir, 
        f"{model_type}_{data_size}_{context_type}_{num_examples}examples_{n_samples}samples{control_suffix}_results.json"
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert NumPy types to native Python types for JSON serialization
        serializable_data = convert_numpy_types({
            "metadata": {
                # Core experiment parameters
                "model_type": model_type,
                "dataset": data_size,
                "context_type": context_type,
                "num_examples": num_examples,
                "n_samples": n_samples,
                "control_mode": control_mode,
                "n_repeat": n_repeat,
                "num_workers": num_workers,
                
                # Execution metadata
                "timestamp": datetime.now().isoformat(),
                "total_trials": len(all_results)
            },
            "results": all_results
        })
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to:")
    print(f"  {output_file}")
    print(f"Total results: {len(all_results)}")
    
    # Summary
    if len(all_results) > 0:
        print("\n" + "="*60)
        print("PROSPECT THEORY IN-CONTEXT EXPERIMENT COMPLETION SUMMARY")
        print("="*60)
        
        print("KEY PARAMETERS:")
        print(f"  Model Type: {model_type}")
        print(f"  Dataset: {data_size}")
        print(f"  Context Type: {context_type}")
        print(f"  Number of Examples: {num_examples}")
        print(f"  Number of Samples: {n_samples}")
        print(f"  Control Mode: {control_mode}")
        print(f"  Random Initializations per Fit: {n_repeat}")
        print(f"  Parallel Workers: {num_workers}")
        
        # Calculate performance metrics
        successful_results = [r for r in all_results if r.get('predicted_choice') is not None]
        if successful_results:
            avg_test_accuracy = np.mean([r['test_accuracy'] for r in successful_results])
            avg_training_accuracy = np.mean([r['training_accuracy'] for r in successful_results])
            avg_test_likelihood = np.mean([r['test_likelihood'] for r in successful_results])
            
            print(f"\nPERFORMANCE METRICS:")
            print(f"  Successful Fits: {len(successful_results)}/{len(all_results)} ({len(successful_results)/len(all_results)*100:.1f}%)")
            print(f"  Average Test Accuracy: {avg_test_accuracy:.3f}")
            print(f"  Average Training Accuracy: {avg_training_accuracy:.3f}")
            print(f"  Average Test Likelihood: {avg_test_likelihood:.3f}")
            
            # Show parameter statistics for PT models
            if model_type == "PT" and successful_results:
                param_stats = {}
                param_names = ['alpha', 'beta', 'gamma', 'lambda', 'tau']
                for param in param_names:
                    values = [r['fitted_params'].get(param, np.nan) for r in successful_results if r['fitted_params']]
                    values = [v for v in values if not np.isnan(v)]
                    if values:
                        param_stats[param] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                if param_stats:
                    print(f"\nFITTED PARAMETER STATISTICS:")
                    for param, stats in param_stats.items():
                        print(f"  {param}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        print("="*60)

def main():
    """
    Main function for in-context Prospect Theory fitting experiment.
    
    This experiment parallels the LLM in-context learning but uses PT model fitting.
    The core idea is to match everything exactly from LLM in-context learning, where 
    those examples become the training data we sample to fit the PT model.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run in-context Prospect Theory fitting experiment")
    parser.add_argument("--data_size", type=str, default="small",
                      choices=["small", "large"],
                      help="Size of dataset to use")
    parser.add_argument("--num_examples", type=int, default=1,
                      help="Number of training examples to use for PT fitting")
    parser.add_argument("--context_type", type=str, default="within_individual",
                      choices=["within_individual", "within_context"],
                      help="Context learning type")
    parser.add_argument("--model_type", type=str, default="PT",
                      choices=["PT", "EV"],
                      help="Type of model to fit (PT=Prospect Theory, EV=Expected Value)")
    parser.add_argument("--checkpoint_interval", type=int, default=999999,
                      help="Number of trials to process before saving checkpoint")
    parser.add_argument("--n_samples", type=int, default=20,
                      help="Number of different example samplings per test trial (default: 1)")
    parser.add_argument("--output_dir", type=str, default="./results/exp2_in_context_pt",
                      help="Directory to save results")
    parser.add_argument("--control_mode", action="store_true", default=False,
                      help="Run control experiment: use other participants' data instead of same participant for within_individual learning")
    parser.add_argument("--n_repeat", type=int, default=100,
                      help="Number of random initializations for PT fitting (default: 50)")
    parser.add_argument("--num_workers", type=int, default=31,
                      help="Number of parallel workers for PT fitting (default: 4)")
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("IN-CONTEXT PROSPECT THEORY FITTING EXPERIMENT")
    print("="*60)
    print(f"Model Type: {args.model_type}")
    print(f"Data Size: {args.data_size}")
    print(f"Context Type: {args.context_type}")
    print(f"Number of Examples: {args.num_examples}")
    print(f"Number of Samples: {args.n_samples}")
    print(f"Control Mode: {args.control_mode}")
    print(f"Random Initializations per Fit: {args.n_repeat}")
    print(f"Parallel Workers: {args.num_workers}")
    print("="*60)
    
    # Warn about control_mode with within_context
    if args.control_mode and args.context_type == "within_context":
        print("⚠️  WARNING: Control mode is designed for within_individual context only.")
        print("   Control mode will be ignored for within_context experiments.")
        print("="*60)
    
    # Warn about computational load for large n_samples
    if args.n_samples > 1:
        estimated_fits = args.n_samples * args.n_repeat
        print(f"⚠️  WARNING: Running {args.n_samples} samples with {args.n_repeat} initializations each will require")
        print(f"   {estimated_fits} PT model fits per trial (computationally intensive)")
        print(f"   Consider using fewer samples or initializations for faster processing")
        print("="*60)
    
    # Run experiment
    run_in_context_pt_experiment(
        data_size=args.data_size,
        context_type=args.context_type,
        num_examples=args.num_examples,
        model_type=args.model_type,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        random_state=42,
        n_samples=args.n_samples,
        control_mode=args.control_mode,
        n_repeat=args.n_repeat,
        num_workers=args.num_workers
    )
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main() 