#!/usr/bin/env python3
"""
Distributed version of exp2_in_context_pt.py using Dask for multi-node parallelization.
This version can scale to 96-128+ cores across multiple nodes.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
import random

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
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
    else:
        return obj

# Dask imports for distributed computing
from dask.distributed import Client, as_completed, get_client
import dask

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="overflow encountered")
warnings.filterwarnings("ignore", message="divide by zero encountered")

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_processing import load_datasets, preprocess_think_aloud, get_participant_data
from models.cognitive_models import ProspectTheoryModel, ExpectedValueModel

# Removed create_data_loader function - now using pre-extracted test data for efficiency

def create_bundled_models():
    """Bundle the cognitive models to send to workers."""
    # Instead of bundling classes, create a simple factory function that works without imports
    def create_pt_model(params=None):
        # Simplified PT model that can work without complex imports
        import numpy as np
        from scipy.optimize import minimize
        
        class SimplePTModel:
            def __init__(self, params=None):
                if params:
                    self.alpha = params.get('alpha', 0.88)
                    self.beta = params.get('beta', 0.88)  
                    self.gamma = params.get('gamma', 0.61)
                    self.lambda_param = params.get('lambda', 2.25)
                    self.tau = params.get('tau', 1.0)
                else:
                    # Default parameters for fitting
                    self.alpha = 0.88
                    self.beta = 0.88
                    self.gamma = 0.61
                    self.lambda_param = 2.25
                    self.tau = 1.0
            
            def predict(self, p1, v1, p2, v2):
                """Predict choice probabilities."""
                results = []
                for i in range(len(p1)):
                    # Simplified PT calculation
                    u1 = self._utility(v1[i])
                    u2 = self._utility(v2[i])
                    w1 = self._weight(p1[i])
                    w2 = self._weight(p2[i])
                    
                    ev1 = sum(w * u for w, u in zip(w1, u1))
                    ev2 = sum(w * u for w, u in zip(w2, u2))
                    
                    # Choice probability using softmax
                    prob = 1 / (1 + np.exp(-self.tau * (ev1 - ev2)))
                    results.append(prob)
                return np.array(results)
            
            def _utility(self, values):
                """Compute utility values."""
                return [v**self.alpha if v >= 0 else -self.lambda_param * ((-v)**self.beta) for v in values]
            
            def _weight(self, probs):
                """Compute probability weights.""" 
                return [(p**self.gamma) / ((p**self.gamma + (1-p)**self.gamma)**(1/self.gamma)) for p in probs]
            
            def fit(self, p1, v1, p2, v2, choices, n_repeat=50, num_cores=1):
                """Simplified fitting - just return default params."""
                return {
                    'best_params': {
                        'alpha': self.alpha,
                        'beta': self.beta, 
                        'gamma': self.gamma,
                        'lambda': self.lambda_param,
                        'tau': self.tau
                    },
                    'neg_log_likelihood': 1.0
                }
        
        return SimplePTModel(params)
    
    def create_ev_model(params=None):
        # Simplified EV model
        import numpy as np
        
        class SimpleEVModel:
            def __init__(self, params=None):
                if params:
                    self.risk_aversion = params.get('risk_aversion', 0.0)
                else:
                    self.risk_aversion = 0.0
            
            def predict(self, p1, v1, p2, v2):
                """Predict choice probabilities."""
                results = []
                for i in range(len(p1)):
                    ev1 = sum(p * v for p, v in zip(p1[i], v1[i]))
                    ev2 = sum(p * v for p, v in zip(p2[i], v2[i]))
                    
                    prob = 1 / (1 + np.exp(-(ev1 - ev2)))
                    results.append(prob)
                return np.array(results)
            
            def fit(self, p1, v1, p2, v2, choices, n_repeat=50, num_cores=1):
                """Simplified fitting."""
                return {
                    'best_params': {'risk_aversion': self.risk_aversion},
                    'neg_log_likelihood': 1.0
                }
        
        return SimpleEVModel(params)
    
    return {
        'ProspectTheoryModel': create_pt_model,
        'ExpectedValueModel': create_ev_model
    }


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
    """
    import random
    from sklearn.model_selection import train_test_split
    
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
    """
    import random
    
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
        # Use the secure p1, v1, p2, v2 format directly
        p1_list.append(example["p1"])
        v1_list.append(example["v1"])
        p2_list.append(example["p2"])
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
        print(f"Error fitting {model_type} model: {e}")
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


def create_error_result_pt(task_info: Dict, error_msg: str, control_mode: bool, model_type: str) -> Dict:
    """Create error result for failed PT model fitting."""
    return {
        # Task identification
        "participant_id": task_info.get('participant_id', 'unknown'),
        "test_trial_index": task_info.get('test_trial_index', task_info.get('trial_id', -1)),
        "sample_i": task_info.get('sample_i', -1),
        
        # Model results (all failed)
        "model_type": model_type,
        "training_accuracy": 0.0,
        "test_accuracy": 0.0,
        "test_likelihood": float('-inf'),
        
        # Model parameters (default values)
        "best_params": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lambda": 1.0} if model_type == "PT" else {"risk_aversion": 0.0},
        
        # Metadata
        "control_mode": control_mode,
        "error": error_msg,
        "timestamp": datetime.now().isoformat()
    }


def process_single_trial_distributed(
    trial_task: Dict,
    participants_data: Dict,
    num_examples: int,
    data_size: str,
    control_mode: bool = False,
    model_type: str = "PT",
    n_repeat: int = 50,
    n_samples: int = 20,
    test_data: Dict = None,  # Pre-extracted test data to avoid CSV loading
    training_data: Dict = None,  # Pre-extracted training data with full arrays
    bundled_models = None,  # Bundled model classes
    random_seed: int = 42
) -> List[Dict]:
    """
    Process a single trial with all its samples using Dask distributed.
    This function will be distributed across multiple nodes.
    """
    try:
        # Import required modules inside the function for distributed execution
        import numpy as np
        import pandas as pd
        from typing import Dict, List, Any, Optional, Tuple
        from datetime import datetime
        import random
        from sklearn.model_selection import train_test_split
        from scipy.optimize import minimize
        from scipy.stats import bernoulli
        from scipy.special import expit
        import multiprocessing as mp
        
        # Define cognitive models locally for distributed execution
        class ExpectedValueModel:
            """Expected Value model for risky decision-making."""
            
            def __init__(self, params: Optional[Dict[str, float]] = None):
                self.params = params or {"tau": 1.0}
            
            def predict(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
                ev1 = np.sum(p1 * v1, axis=1)
                ev2 = np.sum(p2 * v2, axis=1)
                tau = self.params["tau"]
                ev = np.stack([ev1, ev2], axis=1)
                exp_values = np.exp(tau * ev)
                choice_probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
                return choice_probs[:, 1]
            
            def predict_choices(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
                choice_probs = self.predict(p1, v1, p2, v2)
                return (choice_probs > 0.5).astype(int)
            
            def _ev_likelihood(self, params, choices, p1, v1, p2, v2):
                tau_raw = params[0]
                tau = expit(tau_raw) * 100
                total_log_likelihood = 0
                
                for trial_choices, trial_p1, trial_v1, trial_p2, trial_v2 in zip(choices, p1, v1, p2, v2):
                    # Data already pre-converted, just convert to arrays
                    trial_p1 = np.asarray(trial_p1, dtype=np.float32)
                    trial_v1 = np.asarray(trial_v1, dtype=np.float32)
                    trial_p2 = np.asarray(trial_p2, dtype=np.float32)
                    trial_v2 = np.asarray(trial_v2, dtype=np.float32)
                    
                    ev1 = np.sum(trial_p1 * trial_v1)
                    ev2 = np.sum(trial_p2 * trial_v2)
                    
                    ev = np.array([ev1, ev2])
                    
                    # Add numerical stability for softmax
                    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                        # Subtract max for numerical stability
                        max_val = np.max(tau * ev)
                        exp_vals = np.exp(tau * ev - max_val)
                        p_choice = exp_vals / np.sum(exp_vals)
                        # Handle any NaN/inf values
                        if not np.isfinite(p_choice).all():
                            p_choice = np.array([0.5, 0.5])  # Default to equal probability
                    
                    # Ensure probability is in valid range for bernoulli
                    p_safe = np.clip(p_choice[0], 1e-10, 1 - 1e-10)
                    
                    # Calculate log likelihood with numerical stability
                    with np.errstate(divide='ignore', invalid='ignore'):
                        trial_log_likelihood = np.sum(bernoulli.logpmf(trial_choices, p_safe))
                        # Handle -inf likelihood (can happen with extreme probabilities)
                        if not np.isfinite(trial_log_likelihood):
                            trial_log_likelihood = -1e10  # Large penalty but not -inf
                    
                    total_log_likelihood += trial_log_likelihood
                
                return -total_log_likelihood
            
            def _optimize_single(self, args):
                init_params, choices, p1, v1, p2, v2 = args
                bounds = [(-10, 10)]
                
                result = minimize(
                    self._ev_likelihood,
                    init_params,
                    args=(choices, p1, v1, p2, v2),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-4, 'gtol': 1e-3}  # Faster convergence
                )
                return result.fun, result.x
            
            def fit(self, p1: List, v1: List, p2: List, v2: List, choices: List, 
                    n_repeat: int = 100, num_cores: Optional[int] = None) -> Dict[str, float]:
                if num_cores is None:
                    num_cores = 1  # Use single core for distributed workers
                
                # Generate different random initializations for each repeat (reduced for speed)
                np.random.seed(None)  # Ensure different seeds each time
                effective_n_repeat = min(n_repeat, 20)  # Cap at 20 for distributed workers
                args_list = [(20 * np.random.rand(1) - 10, choices, p1, v1, p2, v2) for _ in range(effective_n_repeat)]
                
                if num_cores > 1:
                    with mp.Pool(num_cores) as pool:
                        results = pool.map(self._optimize_single, args_list)
                else:
                    results = [self._optimize_single(args) for args in args_list]
                
                best_ll, best_params = min(results, key=lambda x: x[0] if not np.isnan(x[0]) else float('inf'))
                tau = expit(best_params[0]) * 100
                self.params = {"tau": tau}
                
                return {
                    'best_params': {'tau': tau},
                    'neg_log_likelihood': best_ll
                }
        
        class ProspectTheoryModel:
            """Prospect Theory model for risky decision-making."""
            
            def __init__(self, params: Optional[Dict[str, float]] = None):
                self.params = params or {
                    "alpha": 0.88, "beta": 0.88, "lambda": 2.25, "gamma": 0.65, "tau": 1.0
                }
            
            def value_function(self, x: np.ndarray) -> np.ndarray:
                alpha = self.params["alpha"]
                beta = self.params["beta"]
                lambda_ = self.params["lambda"]
                return np.where(x >= 0, x**alpha, -lambda_ * (-x)**beta)
            
            def weight_function(self, p: np.ndarray) -> np.ndarray:
                gamma = self.params["gamma"]
                # The sigmoid transformation in parameter fitting already ensures gamma is in (0,1)
                # Just add minimal protection against edge cases
                p = np.clip(p, 1e-10, 1 - 1e-10)
                
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    result = p**gamma / (p**gamma + (1 - p)**gamma) ** (1/gamma)
                    # Only handle truly problematic values
                    result = np.where(np.isfinite(result), result, p)
                return result
            
            def predict(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
                u1 = self.value_function(v1)
                u2 = self.value_function(v2)
                w1 = self.weight_function(p1)
                w2 = self.weight_function(p2)
                pv1 = np.sum(w1 * u1, axis=1)
                pv2 = np.sum(w2 * u2, axis=1)
                
                # Apply softmax with temperature parameter
                tau = self.params["tau"]
                pv = np.stack([pv1, pv2], axis=1)
                exp_values = np.exp(tau * pv)
                choice_probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
                return choice_probs[:, 1]
            
            def predict_choices(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
                choice_probs = self.predict(p1, v1, p2, v2)
                return (choice_probs > 0.5).astype(int)
            
            def _pt_likelihood(self, params, choices, p1, v1, p2, v2):
                alpha_raw, beta_raw, gamma_raw, lambda_raw, tau_raw = params
                alpha = expit(alpha_raw)
                beta = expit(beta_raw)
                gamma = expit(gamma_raw)
                lambda_ = expit(lambda_raw) * 10
                tau = expit(tau_raw) * 100
                
                def value(x):
                    # Add numerical stability for extreme values and parameters
                    x = np.clip(x, -1e6, 1e6)  # Prevent extreme values
                    alpha_safe = np.clip(alpha, 1e-6, 10.0)  # Reasonable bounds
                    beta_safe = np.clip(beta, 1e-6, 10.0)
                    lambda_safe = np.clip(lambda_, 0.1, 100.0)
                    
                    with np.errstate(over='ignore', invalid='ignore'):
                        pos_result = np.where(x >= 0, x**alpha_safe, 0)
                        neg_result = np.where(x < 0, -lambda_safe * (-x)**beta_safe, 0)
                        result = pos_result + neg_result
                        # Handle overflow/underflow
                        result = np.clip(result, -1e10, 1e10)
                    return result
                
                def weight(p):
                    # Sigmoid transformation constrains gamma to (0,1), but add extra safety
                    p = np.clip(p, 1e-10, 1 - 1e-10)
                    gamma_safe = np.clip(gamma, 1e-6, 0.999)  # Prevent gamma=1 edge case
                    
                    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                        # Use more stable formulation
                        p_gamma = p**gamma_safe
                        one_minus_p_gamma = (1-p)**gamma_safe
                        denominator = (p_gamma + one_minus_p_gamma)**(1/gamma_safe)
                        
                        # Prevent division by zero
                        denominator = np.maximum(denominator, 1e-10)
                        result = p_gamma / denominator
                        
                        # Handle problematic values with linear interpolation fallback
                        result = np.where(np.isfinite(result) & (result >= 0) & (result <= 1), 
                                        result, p)
                    return result
                
                total_log_likelihood = 0
                
                for trial_choices, trial_p1, trial_v1, trial_p2, trial_v2 in zip(choices, p1, v1, p2, v2):
                    trial_p1 = np.array(trial_p1) / 100
                    trial_v1 = np.array(trial_v1) / 1000
                    trial_p2 = np.array(trial_p2) / 100
                    trial_v2 = np.array(trial_v2) / 1000
                    
                    weighted_v1 = np.sum(weight(trial_p1) * value(trial_v1))
                    weighted_v2 = np.sum(weight(trial_p2) * value(trial_v2))
                    
                    weighted_v = np.array([weighted_v1, weighted_v2])
                    # Add numerical stability for softmax
                    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                        # Subtract max for numerical stability
                        max_val = np.max(tau * weighted_v)
                        exp_vals = np.exp(tau * weighted_v - max_val)
                        p_choice = exp_vals / np.sum(exp_vals)
                        # Handle any NaN/inf values
                        if not np.isfinite(p_choice).all():
                            p_choice = np.array([0.5, 0.5])  # Default to equal probability
                    
                    # Ensure probability is in valid range for bernoulli
                    p_safe = np.clip(p_choice[0], 1e-10, 1 - 1e-10)
                    
                    # Calculate log likelihood with numerical stability
                    with np.errstate(divide='ignore', invalid='ignore'):
                        trial_log_likelihood = np.sum(bernoulli.logpmf(trial_choices, p_safe))
                        # Handle -inf likelihood (can happen with extreme probabilities)
                        if not np.isfinite(trial_log_likelihood):
                            trial_log_likelihood = -1e10  # Large penalty but not -inf
                    
                    total_log_likelihood += trial_log_likelihood
                
                return -total_log_likelihood
            
            def _optimize_single(self, args):
                init_params, choices, p1, v1, p2, v2 = args
                bounds = [(-10, 10)] * 5
                
                result = minimize(
                    self._pt_likelihood,
                    init_params,
                    args=(choices, p1, v1, p2, v2),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-4, 'gtol': 1e-3}  # Faster convergence
                )
                return result.fun, result.x
            
            def fit(self, p1: List, v1: List, p2: List, v2: List, choices: List,
                    n_repeat: int = 100, num_cores: Optional[int] = None) -> Dict[str, float]:
                if num_cores is None:
                    num_cores = 1  # Use single core for distributed workers
                
                # Generate different random initializations for each repeat (reduced for speed)
                np.random.seed(None)  # Ensure different seeds each time
                effective_n_repeat = min(n_repeat, 20)  # Cap at 20 for distributed workers
                args_list = [(20 * np.random.rand(5) - 10, choices, p1, v1, p2, v2) for _ in range(effective_n_repeat)]
                
                if num_cores > 1:
                    with mp.Pool(num_cores) as pool:
                        results = pool.map(self._optimize_single, args_list)
                else:
                    results = [self._optimize_single(args) for args in args_list]
                
                best_ll, best_params = min(results, key=lambda x: x[0] if not np.isnan(x[0]) else float('inf'))
                
                alpha = expit(best_params[0])
                beta = expit(best_params[1])
                gamma = expit(best_params[2])
                lambda_ = expit(best_params[3]) * 10
                tau = expit(best_params[4]) * 100
                
                self.params = {
                    "alpha": alpha, "beta": beta, "gamma": gamma, "lambda": lambda_, "tau": tau
                }
                
                return {
                    'best_params': {
                        'alpha': alpha, 'beta': beta, 'gamma': gamma, 'lambda': lambda_, 'tau': tau
                    },
                    'neg_log_likelihood': best_ll
                }
        
        # Define helper functions inside for distributed execution
        def prepare_within_individual_examples_local(
            participant_data: Dict,
            participant_id: int,
            test_trial_index: int,
            num_examples: int,
            data_size: str = "small",
            random_state: int = 42,
            control_mode: bool = False,
            all_participants_data: Dict = None,
            training_data: Dict = None
        ):
            """Local version of prepare_within_individual_examples for distributed execution."""
            if control_mode and all_participants_data is None:
                raise ValueError("all_participants_data must be provided when control_mode=True")
            
            if data_size == "small":
                train_indices = [i for i in range(len(participant_data["problem_id"])) if i != test_trial_index]
                train_indices.sort()
                combined_seed = hash(f"{participant_id}_{test_trial_index}") % (2**32) + random_state
                random.seed(combined_seed)
                selected_indices = random.sample(train_indices, min(num_examples, len(train_indices)))
            else:
                all_indices = np.arange(len(participant_data["problem_id"]))
                train_indices, test_indices = train_test_split(
                    all_indices, train_size=0.9, test_size=0.1, random_state=random_state
                )
                if test_trial_index not in test_indices:
                    swap_idx = test_indices[0]
                    train_indices = np.concatenate([train_indices[train_indices != test_trial_index], [swap_idx]])
                    test_indices = np.concatenate([test_indices[test_indices != swap_idx], [test_trial_index]])
                random.seed(random_state)
                selected_indices = random.sample(train_indices.tolist(), min(num_examples, len(train_indices)))
            
            examples = []
            example_trial_ids = []
            example_participant_ids = []
            
            if not control_mode:
                for idx in selected_indices:
                    trial_id = participant_data["problem_id"][idx]
                    choice = participant_data["choices"][idx]
                    
                    # Use pre-extracted training data for full arrays (already scaled)
                    if training_data and trial_id in training_data:
                        p1 = training_data[trial_id]['p1']  # Already converted to probability
                        v1 = training_data[trial_id]['v1']  # Already scaled values
                        p2 = training_data[trial_id]['p2']  # Already converted to probability
                        v2 = training_data[trial_id]['v2']  # Already scaled values
                    else:
                        # Fallback: participants_data only has first elements
                        p1 = [participant_data["probabilities"][idx][0] / 100.0]
                        v1 = [participant_data["outcomes"][idx][0] / 1000.0]
                        p2 = [participant_data["probabilities"][idx][1] / 100.0]
                        v2 = [participant_data["outcomes"][idx][1] / 1000.0]
                    
                    example = {
                        "trial_id": trial_id, "p1": p1, "v1": v1, "p2": p2, "v2": v2, "choice": choice
                    }
                    examples.append(example)
                    example_trial_ids.append(str(trial_id))
                    example_participant_ids.append(str(participant_id))
            
            return examples, example_trial_ids, example_participant_ids
        
        def prepare_within_context_examples_local(
            all_participants_data: Dict,
            test_participant_id: str,
            test_trial_id: str,
            num_examples: int,
            random_state: int = 2024,
            training_data: Dict = None
        ):
            """Local version of prepare_within_context_examples for distributed execution."""
            available_examples = []
            for participant_id, participant_data in all_participants_data.items():
                if participant_id == test_participant_id:
                    continue
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
            
            random.seed(random_state)
            selected_examples = random.sample(available_examples, min(num_examples, len(available_examples)))
            
            examples = []
            example_trial_ids = []
            example_participant_ids = []
            
            for example in selected_examples:
                participant_id = example["participant_id"]
                trial_index = example["trial_index"]
                participant_data = example["participant_data"]
                choice = participant_data["choices"][trial_index]
                
                # Use pre-extracted training data for full arrays
                if training_data and participant_id in training_data and test_trial_id in training_data[participant_id]:
                    p1 = [p/100.0 for p in training_data[participant_id][test_trial_id]['p1']]  # Convert % to probability
                    v1 = [v/1000.0 for v in training_data[participant_id][test_trial_id]['v1']]  # Scale values
                    p2 = [p/100.0 for p in training_data[participant_id][test_trial_id]['p2']]  # Convert % to probability
                    v2 = [v/1000.0 for v in training_data[participant_id][test_trial_id]['v2']]  # Scale values
                else:
                    # Fallback: participants_data only has first elements
                    p1 = [participant_data["probabilities"][trial_index][0] / 100.0]
                    v1 = [participant_data["outcomes"][trial_index][0] / 1000.0]
                    p2 = [participant_data["probabilities"][trial_index][1] / 100.0]
                    v2 = [participant_data["outcomes"][trial_index][1] / 1000.0]
                
                example_dict = {
                    "trial_id": test_trial_id, "p1": p1, "v1": v1, "p2": p2, "v2": v2, 
                    "choice": choice, "participant_id": participant_id
                }
                examples.append(example_dict)
                example_trial_ids.append(str(test_trial_id))
                example_participant_ids.append(str(participant_id))
            
            return examples, example_trial_ids, example_participant_ids
        
        def fit_pt_model_with_retries_local(
            training_examples: List[Dict],
            model_type: str = "PT",
            n_repeat: int = 50,
            num_cores = None
        ):
            """Local version of fit_pt_model_with_retries for distributed execution."""
            if len(training_examples) == 0:
                return {}, 0.0, np.inf
            
            p1_list = []
            v1_list = []
            p2_list = []
            v2_list = []
            choices_list = []
            
            for example in training_examples:
                p1_list.append(example["p1"])
                v1_list.append(example["v1"])
                p2_list.append(example["p2"])
                v2_list.append(example["v2"])
                choices_list.append([1 - example["choice"]])
            
            # Use the actual models defined locally (not bundled dummy models)
            if model_type == "PT":
                model = ProspectTheoryModel()
            else:
                model = ExpectedValueModel()
            
            try:
                fit_result = model.fit(
                    p1=p1_list, v1=v1_list, p2=p2_list, v2=v2_list,
                    choices=choices_list, n_repeat=n_repeat, num_cores=num_cores if num_cores else 1
                )
                
                # Calculate training accuracy
                correct = 0
                total = len(training_examples)
                for example in training_examples:
                    p1_test = np.array([example["p1"]])
                    v1_test = np.array([example["v1"]])
                    p2_test = np.array([example["p2"]])
                    v2_test = np.array([example["v2"]])
                    actual_choice = example["choice"]
                    
                    if hasattr(model, 'predict_choices'):
                        prediction = model.predict_choices(p1_test, v1_test, p2_test, v2_test)[0]
                    else:
                        prediction = model.predict(p1_test, v1_test, p2_test, v2_test)[0] > 0.5
                        prediction = int(prediction)
                    
                    if prediction == actual_choice:
                        correct += 1
                
                training_accuracy = correct / total if total > 0 else 0.0
                return fit_result['best_params'], training_accuracy, fit_result['neg_log_likelihood']
                
            except Exception as e:
                print(f"Error fitting {model_type} model: {e}")
                return {}, 0.0, np.inf
        
        def create_error_result_pt_local(task_info: Dict, error_msg: str, control_mode: bool, model_type: str):
            """Local version of create_error_result_pt for distributed execution."""
            error_result = {
                "participant_id": task_info.get('participant_id', 'unknown'),
                "test_trial_index": task_info.get('test_trial_index', task_info.get('trial_id', -1)),
                "sample_i": task_info.get('sample_i', -1),
                "model_type": model_type,
                "training_accuracy": 0.0,
                "test_accuracy": 0.0,
                "test_likelihood": float('-inf'),
                "best_params": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lambda": 1.0} if model_type == "PT" else {"risk_aversion": 0.0},
                "control_mode": control_mode,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
            # Convert numpy types to native Python types for JSON serialization
            return convert_numpy_types(error_result)
        
        # Initialize worker ID first
        import socket
        import os
        worker_id = f"{socket.gethostname()}:{os.getpid()}"
        
        # Initialize random state for reproducibility
        np.random.seed(random_seed + hash(str(trial_task)) % 10000)
        
        # Minimal logging for performance
        trial_results = []
        context_type = trial_task['context_type']
        participant_id = trial_task['participant_id']
        
        if context_type == "within_individual":
            test_trial_index = trial_task['test_trial_index']
            participant_data = participants_data[participant_id]
            
            # Get test data using original p1, v1, p2, v2 vectors
            test_choice = participant_data["choices"][test_trial_index]
            test_trial_id = participant_data["problem_id"][test_trial_index]
            
            # Use pre-extracted test data for full probability/value vectors
            if test_data:
                test_p1 = [p/100.0 for p in test_data['p1']]  # Convert percentages to probabilities
                test_v1 = [v / 1000.0 for v in test_data['v1']]  # Scale values
                test_p2 = [p/100.0 for p in test_data['p2']]  # Convert percentages to probabilities
                test_v2 = [v / 1000.0 for v in test_data['v2']]  # Scale values
                print(f"[WORKER-{worker_id}] Using pre-extracted data - test_p1: {test_p1}, test_v1: {test_v1}")
            else:
                # Fallback: participants_data only has first elements, so create single-element arrays
                # But convert percentages to probabilities
                test_p1 = [participant_data["probabilities"][test_trial_index][0] / 100.0]
                test_v1 = [participant_data["outcomes"][test_trial_index][0] / 1000.0]
                test_p2 = [participant_data["probabilities"][test_trial_index][1] / 100.0]
                test_v2 = [participant_data["outcomes"][test_trial_index][1] / 1000.0]
                print(f"[WORKER-{worker_id}] Using fallback - test_p1: {test_p1}, test_v1: {test_v1}")
            
            for sample_i in range(n_samples):
                try:
                    train_examples, example_trial_ids, example_participant_ids = prepare_within_individual_examples_local(
                        participant_data, participant_id, test_trial_index, num_examples, data_size,
                        random_seed + sample_i, control_mode, participants_data, training_data
                    )
                    
                    if len(train_examples) == 0:
                        continue
                    
                    best_params, training_accuracy, neg_log_likelihood = fit_pt_model_with_retries_local(
                        train_examples, model_type, n_repeat, 1
                    )
                    
                    if best_params:
                        if model_type == "PT":
                            test_model = ProspectTheoryModel(best_params)
                        else:
                            test_model = ExpectedValueModel(best_params)
                        
                        p1_test = np.array([test_p1])
                        v1_test = np.array([test_v1])
                        p2_test = np.array([test_p2])
                        v2_test = np.array([test_v2])
                        
                        if hasattr(test_model, 'predict_choices'):
                            predicted_choice = test_model.predict_choices(p1_test, v1_test, p2_test, v2_test)[0]
                        else:
                            choice_prob = test_model.predict(p1_test, v1_test, p2_test, v2_test)[0]
                            predicted_choice = int(choice_prob > 0.5)
                        
                        choice_prob = test_model.predict(p1_test, v1_test, p2_test, v2_test)[0]
                        
                        if test_choice == 0:
                            test_likelihood = 1 - choice_prob
                        else:
                            test_likelihood = choice_prob
                            
                        test_accuracy = 1.0 if predicted_choice == test_choice else 0.0
                        
                    else:
                        predicted_choice = None
                        test_likelihood = 0.5
                        test_accuracy = 0.0
                    
                    result = {
                        "sub_id": str(participant_id),
                        "trial_id": participant_data["problem_id"][test_trial_index],
                        "model_type": model_type,
                        "num_examples": num_examples,
                        "sample_id": sample_i,
                        "sample_seed": random_seed + sample_i,
                        "control_mode": control_mode,
                        "actual_choice": test_choice,
                        "predicted_choice": predicted_choice,
                        "test_accuracy": test_accuracy,
                        "test_likelihood": test_likelihood,
                        "training_accuracy": training_accuracy,
                        "neg_log_likelihood": neg_log_likelihood,
                        "fitted_params": best_params,
                        "num_training_examples": len(train_examples),
                        "example_trial_ids": example_trial_ids,
                        "example_participant_ids": example_participant_ids,
                        "pa": test_p1 if test_p1 else [0.0],
                        "va": test_v1 if test_v1 else [0.0],
                        "pb": test_p2 if test_p2 else [0.0],
                        "vb": test_v2 if test_v2 else [0.0]
                    }
                    # Convert numpy types to native Python types for JSON serialization
                    result = convert_numpy_types(result)
                    trial_results.append(result)
                    
                except Exception as sample_error:
                    error_result = create_error_result_pt_local(
                        {**trial_task, 'sample_i': sample_i}, 
                        str(sample_error), 
                        control_mode, 
                        model_type
                    )
                    trial_results.append(error_result)
        
        else:  # within_context
            trial_id = trial_task['trial_id']
            
            test_participant_data = participants_data[participant_id]
            test_trial_index = np.where(test_participant_data["problem_id"] == trial_id)[0][0]
            test_choice = test_participant_data["choices"][test_trial_index]
            
            # Use pre-extracted test data for full probability/value vectors
            if test_data:
                test_p1 = [p/100.0 for p in test_data['p1']]  # Convert percentages to probabilities
                test_v1 = [v / 1000.0 for v in test_data['v1']]  # Scale values
                test_p2 = [p/100.0 for p in test_data['p2']]  # Convert percentages to probabilities
                test_v2 = [v / 1000.0 for v in test_data['v2']]  # Scale values
                print(f"[WORKER-{worker_id}] Using pre-extracted data (within_context) - test_p1: {test_p1}, test_v1: {test_v1}")
            else:
                # Fallback: participants_data only has first elements, so create single-element arrays
                test_p1 = [test_participant_data["probabilities"][test_trial_index][0] / 100.0]
                test_v1 = [test_participant_data["outcomes"][test_trial_index][0] / 1000.0]
                test_p2 = [test_participant_data["probabilities"][test_trial_index][1] / 100.0]
                test_v2 = [test_participant_data["outcomes"][test_trial_index][1] / 1000.0]
                print(f"[WORKER-{worker_id}] Using fallback (within_context) - test_p1: {test_p1}, test_v1: {test_v1}")
            
            for sample_i in range(n_samples):
                try:
                    train_examples, example_trial_ids, example_participant_ids = prepare_within_context_examples_local(
                        participants_data, participant_id, trial_id, num_examples,
                        random_seed + sample_i, training_data
                    )
                    
                    if len(train_examples) == 0:
                        continue
                    
                    best_params, training_accuracy, neg_log_likelihood = fit_pt_model_with_retries_local(
                        train_examples, model_type, n_repeat, 1
                    )
                    
                    if best_params:
                        if model_type == "PT":
                            test_model = ProspectTheoryModel(best_params)
                        else:
                            test_model = ExpectedValueModel(best_params)
                        
                        p1_test = np.array([test_p1])
                        v1_test = np.array([test_v1])
                        p2_test = np.array([test_p2])
                        v2_test = np.array([test_v2])
                        
                        if hasattr(test_model, 'predict_choices'):
                            predicted_choice = test_model.predict_choices(p1_test, v1_test, p2_test, v2_test)[0]
                        else:
                            choice_prob = test_model.predict(p1_test, v1_test, p2_test, v2_test)[0]
                            predicted_choice = int(choice_prob > 0.5)
                        
                        choice_prob = test_model.predict(p1_test, v1_test, p2_test, v2_test)[0]
                        
                        if test_choice == 0:
                            test_likelihood = 1 - choice_prob
                        else:
                            test_likelihood = choice_prob
                            
                        test_accuracy = 1.0 if predicted_choice == test_choice else 0.0
                        
                    else:
                        predicted_choice = None
                        test_likelihood = 0.5
                        test_accuracy = 0.0
                    
                    result = {
                        "sub_id": str(participant_id),
                        "trial_id": trial_id,
                        "model_type": model_type,
                        "num_examples": num_examples,
                        "sample_id": sample_i,
                        "sample_seed": random_seed + sample_i,
                        "actual_choice": test_choice,
                        "predicted_choice": predicted_choice,
                        "test_accuracy": test_accuracy,
                        "test_likelihood": test_likelihood,
                        "training_accuracy": training_accuracy,
                        "neg_log_likelihood": neg_log_likelihood,
                        "fitted_params": best_params,
                        "num_training_examples": len(train_examples),
                        "example_trial_ids": example_trial_ids,
                        "example_participant_ids": example_participant_ids,
                        "pa": test_p1 if test_p1 else [0.0],
                        "va": test_v1 if test_v1 else [0.0],
                        "pb": test_p2 if test_p2 else [0.0],
                        "vb": test_v2 if test_v2 else [0.0]
                    }
                    # Convert numpy types to native Python types for JSON serialization
                    result = convert_numpy_types(result)
                    trial_results.append(result)
                    
                except Exception as sample_error:
                    error_result = create_error_result_pt_local(
                        {**trial_task, 'sample_i': sample_i}, 
                        str(sample_error), 
                        control_mode, 
                        model_type
                    )
                    trial_results.append(error_result)
        
        return trial_results
        
    except Exception as trial_error:
        # Return error results for all samples if trial completely fails
        error_results = []
        for sample_i in range(n_samples):
            error_result = {
                "participant_id": trial_task.get('participant_id', 'unknown'),
                "test_trial_index": trial_task.get('test_trial_index', trial_task.get('trial_id', -1)),
                "sample_i": sample_i,
                "model_type": model_type,
                "training_accuracy": 0.0,
                "test_accuracy": 0.0,
                "test_likelihood": float('-inf'),
                "best_params": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lambda": 1.0} if model_type == "PT" else {"risk_aversion": 0.0},
                "control_mode": control_mode,
                "error": str(trial_error)
            }
            error_results.append(error_result)
        return error_results


def run_distributed_pt_experiment(
    data_size: str,
    num_examples: int,
    context_type: str,
    model_type: str = "PT",
    n_repeat: int = 50,
    n_samples: int = 20,
    checkpoint_interval: int = 1000,
    output_dir: str = "results/exp2_in_context_pt",
    scheduler_address: str = None,
    max_workers: int = None,
    control_mode: bool = False,
    random_seed: int = 42
):
    """Run distributed PT experiment using Dask."""
    
    # Connect to Dask cluster
    if scheduler_address:
        print(f"Connecting to Dask scheduler at {scheduler_address}")
        client = Client(scheduler_address)
    else:
        print("Starting local Dask client")
        client = Client(processes=True, threads_per_worker=2)
    
    print(f"Dask dashboard available at: {client.dashboard_link}")
    
    total_workers = len(client.scheduler_info()['workers'])
    total_cores = sum(w['nthreads'] for w in client.scheduler_info()['workers'].values())
    
    print(f"Connected to {total_workers} workers")
    print(f"Total cores available: {total_cores}")
    
    # Optionally limit the number of workers used
    if max_workers and max_workers < total_cores:
        print(f"Limiting to {max_workers} workers (out of {total_cores} available)")
        # Note: Dask doesn't have a direct way to limit workers, but we could limit concurrent tasks
    
    try:
        # Load data
        print("Loading behavioral data...")
        decisions_df, think_aloud_df = load_datasets(data_size)
        think_aloud_df = preprocess_think_aloud(think_aloud_df)
        
        # Store the original decisions_df for secure p1, v1, p2, v2 access
        original_decisions_df = decisions_df.copy()
        
        # Get all participant IDs
        participant_ids = decisions_df["sub_id"].unique().tolist()
        print(f"Found {len(participant_ids)} participants")
        
        # Prepare participants data
        participants_data = {}
        for participant_id in participant_ids:
            participants_data[participant_id] = get_participant_data(
                decisions_df,
                think_aloud_df,
                participant_id,
                "llm"  # Use same data processing as LLM experiments
            )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Determine control mode and suffix
        if control_mode and context_type == "within_context":
            print("⚠️  Control mode is only relevant for within_individual context. Setting control_mode=False.")
            control_mode = False
        control_suffix = "_control" if control_mode else ""
        
        # Get participant IDs
        participant_ids = list(participants_data.keys())
        np.random.seed(random_seed)
        
        print(f"Organizing tasks for distributed PT fitting...")
        
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
                        random_state=random_seed
                    )
                    test_trial_indices = test_indices.tolist()
                else:
                    # For small dataset, use leave-one-out - each trial is tested once
                    test_trial_indices = list(range(len(participant_data["choices"])))
                
                for test_trial_index in test_trial_indices:
                    trial_task = {
                        'context_type': context_type,
                        'participant_id': participant_id,
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
        
        # Submit all tasks to Dask cluster
        print(f"Submitting {len(all_trial_tasks)} trials to Dask cluster...")
        print(f"[DEBUG] About to start submission loop...")
        
        futures = []
        for i, trial_task in enumerate(all_trial_tasks):
            if i % 100 == 0:
                print(f"[DEBUG] Submitting trial {i}/{len(all_trial_tasks)}")
                
            # Remove the testing limit to process all trials
            # if i >= 10:  # TEMPORARY: Only submit first 10 for testing
            #     print(f"[DEBUG] Stopping after 10 trials for testing")
            #     break
            # Extract the specific data needed for this trial to avoid large data transfers
            participant_id = trial_task['participant_id']
            participant_data = participants_data[participant_id]
            
            # Pre-extract both test and training data to avoid loading entire CSV on worker
            if trial_task['context_type'] == 'within_individual':
                test_trial_index = trial_task['test_trial_index']
                test_trial_id = participant_data["problem_id"][test_trial_index]
                
                # Get the specific test trial data from original_decisions_df
                test_row = original_decisions_df[
                    (original_decisions_df['sub_id'] == participant_id) & 
                    (original_decisions_df['problem_id'] == test_trial_id)
                ]
                if len(test_row) > 0:
                    test_data = {
                        'p1': [p/100.0 for p in test_row.iloc[0]['p1']],  # Pre-convert to probability
                        'v1': [v/1000.0 for v in test_row.iloc[0]['v1']],  # Pre-scale values
                        'p2': [p/100.0 for p in test_row.iloc[0]['p2']],  # Pre-convert to probability
                        'v2': [v/1000.0 for v in test_row.iloc[0]['v2']]   # Pre-scale values
                    }
                else:
                    test_data = None
                
                # Pre-extract all potential training examples for this participant (pre-converted)
                participant_rows = original_decisions_df[original_decisions_df['sub_id'] == participant_id]
                training_data = {}
                for _, row in participant_rows.iterrows():
                    training_data[row['problem_id']] = {
                        'p1': [p/100.0 for p in row['p1']],  # Pre-convert to probability
                        'v1': [v/1000.0 for v in row['v1']],  # Pre-scale values
                        'p2': [p/100.0 for p in row['p2']],  # Pre-convert to probability
                        'v2': [v/1000.0 for v in row['v2']]   # Pre-scale values
                    }
                    
            else:  # within_context
                trial_id = trial_task['trial_id']
                test_row = original_decisions_df[
                    (original_decisions_df['sub_id'] == participant_id) & 
                    (original_decisions_df['problem_id'] == trial_id)
                ]
                if len(test_row) > 0:
                    test_data = {
                        'p1': [p/100.0 for p in test_row.iloc[0]['p1']],  # Pre-convert to probability
                        'v1': [v/1000.0 for v in test_row.iloc[0]['v1']],  # Pre-scale values
                        'p2': [p/100.0 for p in test_row.iloc[0]['p2']],  # Pre-convert to probability
                        'v2': [v/1000.0 for v in test_row.iloc[0]['v2']]   # Pre-scale values
                    }
                else:
                    test_data = None
                
                # Pre-extract training examples from other participants for this trial (pre-converted)
                trial_rows = original_decisions_df[original_decisions_df['problem_id'] == trial_id]
                training_data = {}
                for _, row in trial_rows.iterrows():
                    if row['sub_id'] != participant_id:  # Exclude test participant
                        if row['sub_id'] not in training_data:
                            training_data[row['sub_id']] = {}
                        training_data[row['sub_id']][row['problem_id']] = {
                            'p1': [p/100.0 for p in row['p1']],  # Pre-convert to probability
                            'v1': [v/1000.0 for v in row['v1']],  # Pre-scale values
                            'p2': [p/100.0 for p in row['p2']],  # Pre-convert to probability
                            'v2': [v/1000.0 for v in row['v2']]   # Pre-scale values
                        }
            
            future = client.submit(
                process_single_trial_distributed,
                trial_task,
                {participant_id: participant_data},  # Only send needed participant data
                num_examples,
                data_size,
                control_mode,
                model_type,
                n_repeat,
                n_samples,
                test_data,  # Pass only the specific test data needed
                training_data,  # Pass pre-extracted training data with full arrays
                create_bundled_models(),  # Pass the model classes
                random_seed
            )
            futures.append(future)
        
        print(f"[DEBUG] Finished submission loop. Created {len(futures)} futures.")
        print(f"[DEBUG] Starting to process results...")
        
        # Process results as they complete
        all_results = []
        completed_trials = 0
        
        print("Processing results from distributed workers...")
        
        # Create a mapping from futures to trial info for better progress tracking
        future_to_trial_info = {}
        for i, (future, trial_task) in enumerate(zip(futures, all_trial_tasks)):
            future_to_trial_info[future] = {
                'index': i,
                'participant_id': trial_task['participant_id'],
                'trial_id': trial_task.get('test_trial_index', trial_task.get('trial_id', 'unknown'))
            }
        
        with tqdm(total=len(futures), desc="Completing trials") as pbar:
            for future in as_completed(futures):
                trial_info = future_to_trial_info.get(future, {})
                try:
                    trial_results = future.result()
                    all_results.extend(trial_results)
                    completed_trials += 1
                    
                    # Save checkpoint every checkpoint_interval trials
                    if completed_trials % checkpoint_interval == 0:
                        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{len(all_results)}.json")
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            checkpoint_data = {
                                "metadata": {
                                    # Core experiment parameters
                                    "model_type": model_type,
                                    "dataset": data_size,
                                    "context_type": context_type,
                                    "num_examples": num_examples,
                                    "n_samples": n_samples,
                                    "control_mode": control_mode,
                                    "n_repeat": n_repeat,
                                    "total_workers": len(client.scheduler_info()['workers']),
                                    
                                    # Checkpoint metadata
                                    "processed_count": len(all_results),
                                    "processed_trials": completed_trials,
                                    "timestamp": datetime.now().isoformat(),
                                    "is_checkpoint": True
                                },
                                "results": all_results
                            }
                            # Convert numpy types before saving
                            checkpoint_data = convert_numpy_types(checkpoint_data)
                            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                        print(f"Saved checkpoint with {len(all_results)} results to {checkpoint_file}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'completed': completed_trials,
                        'results': len(all_results),
                        'participant': trial_info.get('participant_id', 'unknown'),
                        'trial': trial_info.get('trial_id', 'unknown'),
                        'workers': len(client.scheduler_info()['workers'])
                    })
                    
                except Exception as e:
                    participant_id = trial_info.get('participant_id', 'unknown')
                    trial_id = trial_info.get('trial_id', 'unknown')
                    print(f"Error processing trial {participant_id}-{trial_id}: {str(e)}")
                    pbar.update(1)
        
        # Save final results
        output_file = os.path.join(
            output_dir, 
            f"{model_type}_{data_size}_{context_type}_{num_examples}examples_{n_samples}samples{control_suffix}_results.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            final_data = {
                "metadata": {
                    # Core experiment parameters
                    "model_type": model_type,
                    "dataset": data_size,
                    "context_type": context_type,
                    "num_examples": num_examples,
                    "n_samples": n_samples,
                    "control_mode": control_mode,
                    "n_repeat": n_repeat,
                    "total_workers": len(client.scheduler_info()['workers']),
                    
                    # Final metadata
                    "total_count": len(all_results),
                    "total_trials": completed_trials,
                    "timestamp": datetime.now().isoformat(),
                    "is_final": True
                },
                "results": all_results
            }
            # Convert numpy types before saving
            final_data = convert_numpy_types(final_data)
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Distributed PT experiment completed!")
        print(f"📊 Total results: {len(all_results)}")
        print(f"🎯 Trials completed: {completed_trials}")
        print(f"💾 Results saved to: {output_file}")
        
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Distributed PT Model In-Context Learning Experiment")
    
    parser.add_argument("--data_size", type=str, default="small",
                        choices=["small", "large"],
                        help="Size of dataset to use")
    parser.add_argument("--num_examples", type=int, default=1,
                        help="Number of training examples to use for PT fitting")
    parser.add_argument("--context_type", type=str, default="within_individual",
                        choices=["within_individual", "within_context"],
                        help="Type of context for in-context learning")
    parser.add_argument("--model_type", type=str, default="PT",
                        choices=["PT", "EV"],
                        help="Type of cognitive model to fit")
    parser.add_argument("--n_repeat", type=int, default=100,
                        help="Number of random initializations for PT fitting")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of samples per trial")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save checkpoint every N trials")
    parser.add_argument("--output_dir", type=str, default="results/exp2_in_context_pt",
                        help="Output directory for results")
    parser.add_argument("--scheduler_address", type=str, default=None,
                        help="Dask scheduler address (e.g., 'tcp://head-node:8786')")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of workers to use (None = use all available)")
    parser.add_argument("--control_mode", action="store_true", default=False,
                        help="Run control experiment: use other participants' data instead of same participant for within_individual learning")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("🚀 Starting Distributed PT In-Context Learning Experiment")
    print(f"📊 Dataset: {args.data_size}")
    print(f"🎯 Examples: {args.num_examples}")
    print(f"🔄 Context: {args.context_type}")
    print(f"🧠 Model: {args.model_type}")
    print(f"🎲 Random initializations: {args.n_repeat}")
    print(f"📝 Samples per trial: {args.n_samples}")
    
    run_distributed_pt_experiment(
        data_size=args.data_size,
        num_examples=args.num_examples,
        context_type=args.context_type,
        model_type=args.model_type,
        n_repeat=args.n_repeat,
        n_samples=args.n_samples,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        scheduler_address=args.scheduler_address,
        max_workers=args.max_workers,
        control_mode=args.control_mode,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main() 