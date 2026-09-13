"""
Data processing utilities for the think-aloud analysis project.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import SMALL_DATA_PATH, LARGE_DATA_PATH, DATA_SIZE_CONFIGS

def load_datasets(data_size: str = "large") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load decision and think-aloud datasets.
    
    Args:
        data_size: Size of dataset to use ("small", "large", or "all")
        
    Returns:
        Tuple of (decisions_df, think_aloud_df)
    """
    if data_size == "small":
        data_file = "data/behavioral_text_data.csv"
    elif data_size == "large":
        data_file = "data/behavioral_text_data_expanded.csv"
    elif data_size == "all":
        # For "all", load both datasets and combine them
        small_df = pd.read_csv("data/behavioral_text_data.csv")
        large_df = pd.read_csv("data/behavioral_text_data_expanded.csv")
        
        # Clean both datasets
        for df in [small_df, large_df]:
            df.drop(columns=['0'], errors='ignore', inplace=True)
            df.columns = ['sub_id', 'choice', 'p1', 'v1', 'p2', 'v2', 'problem_id', 'rt', 'think_aloud', 'word_count']
            for column in ['p1', 'v1', 'p2', 'v2']:
                df[column] = df[column].apply(eval)
        
        # Combine datasets (adjust subject IDs to avoid conflicts)
        max_sub_id = small_df['sub_id'].max()
        large_df['sub_id'] = large_df['sub_id'] + max_sub_id + 1
        
        # Combine
        df = pd.concat([small_df, large_df], ignore_index=True)
    else:
        data_file = "data/behavioral_text_data_expanded.csv"  # Default to large
    
    # Load single file if not "all"
    if data_size != "all":
        df = pd.read_csv(data_file)
        df.drop(columns=['0'], errors='ignore', inplace=True)
        df.columns = ['sub_id', 'choice', 'p1', 'v1', 'p2', 'v2', 'problem_id', 'rt', 'think_aloud', 'word_count']
        
        # Process lists in the DataFrame
        for column in ['p1', 'v1', 'p2', 'v2']:
            df[column] = df[column].apply(eval)
    

    
    # Split into decisions and think-aloud dataframes
    decisions_df = df[['sub_id', 'choice', 'p1', 'v1', 'p2', 'v2', 'problem_id']].copy()
    think_aloud_df = df[['sub_id', 'problem_id', 'think_aloud', 'word_count']].copy()
    
    # Encode subject IDs
    le = LabelEncoder()
    decisions_df['sub_id'] = le.fit_transform(decisions_df['sub_id'])
    think_aloud_df['sub_id'] = le.transform(think_aloud_df['sub_id'])
    
    return decisions_df, think_aloud_df

def preprocess_think_aloud(think_aloud_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess think-aloud text data.
    
    Args:
        think_aloud_df: DataFrame containing think-aloud data
        
    Returns:
        Preprocessed DataFrame
    """
    # Fill NaN values with empty string
    think_aloud_df['think_aloud'] = think_aloud_df['think_aloud'].fillna('')
    
    # Add processed text column (can be extended with more preprocessing steps)
    think_aloud_df['processed_text'] = think_aloud_df['think_aloud']
    
    return think_aloud_df

def split_data_by_participant(
    decisions_df: pd.DataFrame,
    data_size: str,
    train_ratio: float = 0.9,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split data by participant, using leave-one-out for small dataset and train/test split for large dataset.
    
    Args:
        decisions_df: DataFrame containing decision data
        data_size: Size of dataset ("small" or "large")
        train_ratio: Ratio of training data (for large dataset)
        test_ratio: Ratio of test data (for large dataset)
        random_state: Random seed
        
    Returns:
        Dictionary mapping participant IDs to their train/test indices
    """
    splits = {}
    
    for sub_id in decisions_df['sub_id'].unique():
        participant_data = decisions_df[decisions_df['sub_id'] == sub_id]
        
        if data_size == "small":
            # Leave-one-out cross-validation
            trial_numbers = participant_data['problem_id'].values
            splits[sub_id] = {
                'train_idx': [],
                'test_idx': []
            }
            
            for test_trial in np.unique(trial_numbers):
                train_idx = np.where(trial_numbers != test_trial)[0]
                test_idx = np.where(trial_numbers == test_trial)[0]
                splits[sub_id]['train_idx'].append(train_idx)
                splits[sub_id]['test_idx'].append(test_idx)
        else:
            # Train/test split for large dataset
            indices = np.arange(len(participant_data))
            train_idx, test_idx = train_test_split(
                indices,
                train_size=train_ratio,
                test_size=test_ratio,
                random_state=random_state
            )
            splits[sub_id] = {
                'train_idx': [train_idx],
                'test_idx': [test_idx]
            }
    
    return splits

def question_prompt_generate(row: Dict) -> str:
    """
    Generate question prompt from a row of data.
    
    Args:
        row: Dictionary containing p1, v1, p2, v2 arrays
        
    Returns:
        Formatted question prompt
    """
    fixed_start = 'Which option do you prefer? '
    option_A = 'Option A: '
    option_B = 'Option B: '
    
    p1, v1, p2, v2 = np.array(row['p1']), np.array(row['v1']), np.array(row['p2']), np.array(row['v2'])
    
    for i in range(len(p1)):
        option_A += f"{v1[i]} dollars with {p1[i]}% chance"
        if i < len(p1) - 1:
            option_A += ", "
        else:
            option_A += ". "
    
    for i in range(len(p2)):
        option_B += f"{v2[i]} dollars with {p2[i]}% chance"
        if i < len(p2) - 1:
            option_B += ", "
        else:
            option_B += "."
    
    prompt = fixed_start + option_A + ' ' + option_B
    return prompt

def prepare_model_inputs(
    decisions_df: pd.DataFrame,
    think_aloud_df: Optional[pd.DataFrame],
    model_type: str
) -> Dict:
    """
    Prepare inputs for different model types.
    
    Args:
        decisions_df: DataFrame containing decision data
        think_aloud_df: DataFrame containing think-aloud data (optional)
        model_type: Type of model ("cognitive", "neural", or "llm")
        
    Returns:
        Dictionary of model inputs
    """
    # Basic inputs for all models
    inputs = {
        "probabilities": np.column_stack([
            decisions_df['p1'].apply(lambda x: x[0]).values,
            decisions_df['p2'].apply(lambda x: x[0]).values
        ]),
        "outcomes": np.column_stack([
            decisions_df['v1'].apply(lambda x: x[0]).values,
            decisions_df['v2'].apply(lambda x: x[0]).values
        ]),
        "choices": decisions_df['choice'].values,
        "subject_ids": decisions_df['sub_id'].values,
        "problem_id": decisions_df['problem_id'].values
    }
    
    # Add think-aloud data for neural and LLM models
    if model_type in ["neural", "llm"] and think_aloud_df is not None:
        inputs["think_aloud"] = think_aloud_df["processed_text"].values
        
        # Add question context for LLM models
        if model_type == "llm":
            # Generate question context for each row
            question_contexts = []
            for _, row in decisions_df.iterrows():
                row_dict = {
                    'p1': row['p1'],
                    'v1': row['v1'],
                    'p2': row['p2'],
                    'v2': row['v2']
                }
                question_contexts.append(question_prompt_generate(row_dict))
            inputs["question_context"] = np.array(question_contexts)
    
    return inputs

def get_participant_data(
    decisions_df: pd.DataFrame,
    think_aloud_df: Optional[pd.DataFrame],
    sub_id: int,
    model_type: str
) -> Dict:
    """
    Get data for a specific participant.
    
    Args:
        decisions_df: DataFrame containing decision data
        think_aloud_df: DataFrame containing think-aloud data (optional)
        sub_id: Subject ID
        model_type: Type of model
        
    Returns:
        Dictionary of participant data
    """
    # Get participant's decisions
    participant_decisions = decisions_df[decisions_df['sub_id'] == sub_id]
    
    # Get participant's think-aloud data if available
    participant_think_aloud = None
    if think_aloud_df is not None:
        participant_think_aloud = think_aloud_df[think_aloud_df['sub_id'] == sub_id]
    
    # Prepare model inputs
    return prepare_model_inputs(
        participant_decisions,
        participant_think_aloud,
        model_type
    )

def save_results(
    results: Dict,
    experiment_name: str,
    model_name: str,
    data_size: str
) -> None:
    """
    Save experiment results to disk.
    
    Args:
        results: Dictionary of results
        experiment_name: Name of the experiment
        model_name: Name of the model
        data_size: Size of dataset used
    """
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Save results
    filename = f"results/{experiment_name}_{model_name}_{data_size}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")

def load_results(
    experiment_name: str,
    model_name: str,
    data_size: str
) -> Dict:
    """
    Load experiment results from disk.
    
    Args:
        experiment_name: Name of the experiment
        model_name: Name of the model
        data_size: Size of dataset used
        
    Returns:
        Dictionary of results
    """
    filename = f"results/{experiment_name}_{model_name}_{data_size}.json"
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results 