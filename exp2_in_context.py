"""
Experiment 2: In-Context Learning
This experiment tests how well LLMs can learn from few-shot examples
in both within-individual and within-context settings.

Control Experiment:
The --control_mode flag enables a control experiment to validate whether
within-individual learning truly captures individual characteristics.

In normal within-individual learning:
- Examples come from the same participant's other trials
- Tests if the model learns individual decision patterns

In control mode within-individual learning:
- Examples come from other participants' data for the same trial contexts
- Same trial contexts but different participants' choices/think-aloud
- Tests if performance depends on individual patterns vs. general patterns

For small dataset:
- Uses same trial selection logic (leave-one-out)
- Keeps same trial IDs but samples from other participants
- Controls for context effects while changing individual patterns

For large dataset:
- Uses same trial selection logic (90-10 split)
- Keeps same trial IDs but samples from other participants
- Additional control: ensures test context not in example contexts
- Prevents contamination from seeing the same context during training

Comprehensive Test Mode:
The --mode comprehensive_test enables systematic trial-by-trial analysis to create
a 19×19 matrix showing how think-aloud helps generalize across different trials.

For each participant, each trial is tested against every other trial as a single example:
- Tests both with and without think-aloud in the example
- Creates matrix where columns = example trials, rows = test trials
- Values show performance boost when think-aloud is provided vs baseline

Usage:
    Normal: python exp2_in_context.py --context_type within_individual
    Control: python exp2_in_context.py --context_type within_individual --control_mode
    Comprehensive: python exp2_in_context.py --mode comprehensive_test --llm_mode base
"""

import os

# Set critical environment variables BEFORE any other imports
# These must be set early so vLLM worker processes inherit them
os.environ["VLLM_DO_NOT_TRACK"] = "1"  # Disable usage reporting to avoid disk quota issues
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # Increase HuggingFace download timeout
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable faster downloads
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warnings

import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import random
from sklearn.model_selection import train_test_split

from models.llm_model import LLMModel
from utils.data_processing import (
    load_datasets,
    preprocess_think_aloud,
    get_participant_data
)
from utils.prompt_templates import OPENAI_PROMPTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key: set OPENAI_API_KEY in the environment; never commit secrets.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def format_example(
    question_context: str,
    think_aloud: str,
    choice: int,
    example_type: str = "think_aloud"
) -> str:
    """
    Format a single example for the prompt.
    
    Args:
        question_context: The decision scenario
        think_aloud: The think-aloud text
        choice: The choice made (0 for A, 1 for B)
        example_type: Type of information to include ("think_aloud", "choice", "both")
        
    Returns:
        Formatted example string
    """
    choice_text = "A" if choice == 0 else "B"
    example = f"Decision Scenario:\n{question_context}\n"
    
    # Always present think-aloud first, then choice when both are configured
    if example_type in ["think_aloud", "both"] and think_aloud:
        example += f"Think-aloud:\n{think_aloud}\n"
    
    if example_type in ["choice", "both"]:
        example += f"I chose Option {choice_text}.\n"
    
    return example

def format_examples(examples: List[Dict], example_type: str = "think_aloud") -> str:
    """
    Format examples into a string for the prompt.
    
    Args:
        examples: List of example dictionaries
        example_type: Type of information to include ("think_aloud", "choice", "both")
        
    Returns:
        Formatted examples string
    """
    formatted_examples = []
    for i, example in enumerate(examples, 1):
        formatted_example = f"Example {i}:\n"
        formatted_example += f"Decision Scenario:\n{example['question_context']}\n"
        # Always present think-aloud first, then choice when both are configured
        if example_type in ["think_aloud", "both"] and example.get('think_aloud'):
            formatted_example += f"Think-aloud:\n{example['think_aloud']}\n"
        if example_type in ["choice", "both"] and 'choice' in example:
            choice_text = "A" if example['choice'] == 0 else "B"
            formatted_example += f"I chose Option {choice_text}.\n"
        formatted_examples.append(formatted_example)
    return "\n".join(formatted_examples)

def prepare_within_individual_examples(
    participant_data: Dict,
    participant_id: int,
    test_trial_index: int,
    num_examples: int,
    data_size: str = "small",
    random_state: int = 42,
    control_mode: bool = False,
    all_participants_data: Dict = None
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Prepare examples for within-individual learning.
    
    Args:
        participant_data: Dictionary containing participant's data
        participant_id: ID of the participant (integer from LabelEncoder)
        test_trial_index: Index of the test trial
        num_examples: Number of examples to prepare
        data_size: Size of dataset ("small" or "large")
        random_state: Random seed for reproducibility
        control_mode: If True, sample from other participants' data instead of same participant
        all_participants_data: Dictionary of all participants' data (required for control_mode)
        
    Returns:
        Tuple of (list of example dictionaries, list of example trial IDs, list of example participant IDs)
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
        for idx in selected_indices:
            example = {
                "trial_id": participant_data["problem_id"][idx],
                "probabilities": participant_data["probabilities"][idx],
                "outcomes": participant_data["outcomes"][idx],
                "think_aloud": participant_data["think_aloud"][idx],
                "question_context": participant_data["question_context"][idx],
                "choice": participant_data["choices"][idx]  # Add choice to example
            }
            examples.append(example)
            example_trial_ids.append(str(participant_data["problem_id"][idx]))
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
                    
                    example = {
                        "trial_id": trial_id,
                        "probabilities": selected_participant_data["probabilities"][selected_trial_index],
                        "outcomes": selected_participant_data["outcomes"][selected_trial_index],
                        "think_aloud": selected_participant_data["think_aloud"][selected_trial_index],
                        "question_context": selected_participant_data["question_context"][selected_trial_index],
                        "choice": selected_participant_data["choices"][selected_trial_index]
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
                        "think_aloud": selected_participant_data["think_aloud"][selected_trial_index],
                        "question_context": selected_participant_data["question_context"][selected_trial_index],
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
    random_state: int = 2024
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Prepare examples for within-context learning by finding trials with the same ID
    from other participants.
    
    Args:
        all_participants_data: Dictionary containing all participants' data
        test_participant_id: ID of the test participant
        test_trial_id: ID of the test trial
        num_examples: Number of examples to prepare
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (list of example dictionaries, list of example trial IDs, list of example participant IDs)
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
        
        example_dict = {
            "trial_id": test_trial_id,
            "probabilities": participant_data["probabilities"][trial_index],
            "outcomes": participant_data["outcomes"][trial_index],
            "think_aloud": participant_data["think_aloud"][trial_index],
            "question_context": participant_data["question_context"][trial_index],
            "choice": participant_data["choices"][trial_index],  # Add choice to example
            "participant_id": participant_id
        }
        examples.append(example_dict)
        example_trial_ids.append(str(test_trial_id))
        example_participant_ids.append(str(participant_id))
    
    return examples, example_trial_ids, example_participant_ids

def extract_cot(generation: str) -> Optional[str]:
    """
    Extract chain-of-thought reasoning from model generation.
    
    Args:
        generation: Full model generation text
        
    Returns:
        Extracted chain-of-thought reasoning or None if not found
    """
    # Split by "I will choose" to get the reasoning part
    parts = generation.split("I will choose")
    if len(parts) > 1:
        # Get everything before the choice
        reasoning = parts[0].strip()
        return reasoning if reasoning else None
    return None

def process_participant_trial(
    participant_id: int,
    participant_data: Dict,
    test_trial_index: int,
    current_mode: str,
    num_examples: int,
    model: LLMModel,
    data_size: str = "small",
    example_type: str = "think_aloud",
    sample_i: int = 0,
    sample_seed: int = 2025,
    control_mode: bool = False,
    all_participants_data: Dict = None
) -> Dict:
    """Process a single participant's trial."""
    # Get test data
    test_think_aloud = participant_data["think_aloud"][test_trial_index]
    test_choice = participant_data["choices"][test_trial_index]
    test_context = participant_data["question_context"][test_trial_index]
    
    # Prepare examples with sample-specific seed
    examples, example_trial_ids, example_participant_ids = prepare_within_individual_examples(
        participant_data,
        participant_id,
        test_trial_index,
        num_examples,
        data_size=data_size,
        random_state=sample_seed,
        control_mode=control_mode,
        all_participants_data=all_participants_data
    )
    formatted_examples = format_examples(examples, example_type)
    
    # Get the complete prompt structure using the model's _format_prompt method
    prompt = model._format_prompt(
        current_mode,
        participant_data["probabilities"][test_trial_index],
        participant_data["outcomes"][test_trial_index],
        test_think_aloud if "cot" in current_mode else None,
        test_context,
        formatted_examples,
        example_type
    )
    
    # Get model prediction
    prediction = model.predict(
        probabilities=participant_data["probabilities"][test_trial_index],
        outcomes=participant_data["outcomes"][test_trial_index],
        think_aloud=test_think_aloud if "cot" in current_mode else None,
        question_context=test_context,
        mode=current_mode,
        examples=formatted_examples,
        example_type=example_type
    )
    
    # Store result
    result = {
        "sub_id": str(participant_id),
        "trial_id": participant_data["problem_id"][test_trial_index],
        "mode": current_mode,
        "num_examples": num_examples,
        "example_type": example_type,
        "sample_id": sample_i,
        "sample_seed": sample_seed,
        "control_mode": control_mode,  # Add control mode indicator
        "think_aloud": test_think_aloud,
        "actual_choice": test_choice,
        "extracted_choice": int(prediction["choice"]),
        "probability_option_a": float(prediction["probabilities"]["A"]),
        "probability_option_b": float(prediction["probabilities"]["B"]),
        "logits": prediction.get("logits", None),
        "full_generation": prediction["generation"],
        "question_context": test_context,
        "examples": formatted_examples,
        "example_trial_ids": example_trial_ids,
        "example_participant_ids": example_participant_ids,
        "pa": float(participant_data["probabilities"][test_trial_index][0]),
        "va": float(participant_data["outcomes"][test_trial_index][0]),
        "pb": float(participant_data["probabilities"][test_trial_index][1]),
        "vb": float(participant_data["outcomes"][test_trial_index][1]),
        "complete_prompt": prompt
    }
    return result

def process_trial_participant(
    trial_id: str,
    test_participant_id: int,
    participants_data: Dict,
    current_mode: str,
    num_examples: int,
    model: LLMModel,
    example_type: str = "think_aloud",
    sample_i: int = 0,
    sample_seed: int = 2025
) -> Dict:
    """Process a single trial for a participant in within-context mode."""
    # Get test participant's data for this trial
    test_participant_data = participants_data[test_participant_id]
    test_trial_index = np.where(test_participant_data["problem_id"] == trial_id)[0][0]
    
    # Get test data
    test_think_aloud = test_participant_data["think_aloud"][test_trial_index]
    test_choice = test_participant_data["choices"][test_trial_index]
    test_context = test_participant_data["question_context"][test_trial_index]
    
    # Prepare examples from other participants
    available_examples = []
    for participant_id, participant_data in participants_data.items():
        if participant_id != test_participant_id and trial_id in participant_data["problem_id"]:
            trial_index = np.where(participant_data["problem_id"] == trial_id)[0][0]
            available_examples.append({
                "participant_id": participant_id,
                "trial_index": trial_index,
                "participant_data": participant_data
            })
    
    # Randomly select examples with sample-specific seed
    random.seed(sample_seed)
    selected_examples = random.sample(available_examples, min(num_examples, len(available_examples)))
    
    # Format examples
    formatted_examples = []
    example_trial_ids = []
    example_participant_ids = []
    
    for i, example in enumerate(selected_examples, 1):
        participant_id = example["participant_id"]
        trial_index = example["trial_index"]
        participant_data = example["participant_data"]
        
        formatted_example = f"Example {i}:\n"
        formatted_example += f"Decision Scenario:\n{participant_data['question_context'][trial_index]}\n"
        
        # Include information based on type
        if example_type in ["think_aloud", "both"] and participant_data['think_aloud'][trial_index]:
            formatted_example += f"Think-aloud: {participant_data['think_aloud'][trial_index]}\n"
        
        if example_type in ["choice", "both"]:
            choice_text = "A" if participant_data['choices'][trial_index] == 0 else "B"
            formatted_example += f"I chose Option {choice_text}.\n"
        
        formatted_examples.append(formatted_example)
        example_trial_ids.append(str(trial_id))
        example_participant_ids.append(str(participant_id))
    
    formatted_examples_str = "\n".join(formatted_examples)
    
    # Get the complete prompt structure using the model's _format_prompt method
    prompt = model._format_prompt(
        current_mode,
        test_participant_data["probabilities"][test_trial_index],
        test_participant_data["outcomes"][test_trial_index],
        test_think_aloud if "cot" in current_mode else None,
        test_context,
        formatted_examples_str,
        example_type
    )
    
    # Get model prediction
    prediction = model.predict(
        probabilities=test_participant_data["probabilities"][test_trial_index],
        outcomes=test_participant_data["outcomes"][test_trial_index],
        think_aloud=test_think_aloud if "cot" in current_mode else None,
        question_context=test_context,
        mode=current_mode,
        examples=formatted_examples_str,
        example_type=example_type
    )
    
    # Store result
    result = {
        "sub_id": str(test_participant_id),
        "trial_id": trial_id,
        "mode": current_mode,
        "num_examples": num_examples,
        "example_type": example_type,
        "sample_id": sample_i,
        "sample_seed": sample_seed,
        "think_aloud": test_think_aloud,
        "actual_choice": test_choice,
        "extracted_choice": int(prediction["choice"]),
        "probability_option_a": float(prediction["probabilities"]["A"]),
        "probability_option_b": float(prediction["probabilities"]["B"]),
        "logits": prediction.get("logits", None),
        "full_generation": prediction["generation"],
        "question_context": test_context,
        "examples": formatted_examples_str,
        "example_trial_ids": example_trial_ids,
        "example_participant_ids": example_participant_ids,
        "pa": float(test_participant_data["probabilities"][test_trial_index][0]),
        "va": float(test_participant_data["outcomes"][test_trial_index][0]),
        "pb": float(test_participant_data["probabilities"][test_trial_index][1]),
        "vb": float(test_participant_data["outcomes"][test_trial_index][1]),
        "complete_prompt": prompt
    }
    return result

def process_batch(
    batch_items: List[Tuple],
    current_mode: str,
    num_examples: int,
    model: LLMModel,
    num_workers: int,
    example_type: str = "think_aloud"
) -> List[Dict]:
    """Process a batch of items in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for item in batch_items:
            if "within_context" in current_mode:
                trial_id, test_participant_id, participants_data = item
                future = executor.submit(
                    process_trial_participant,
                    trial_id,
                    test_participant_id,
                    participants_data,
                    current_mode,
                    num_examples,
                    model,
                    example_type
                )
            else:
                participant_id, participant_data, test_trial_index = item
                future = executor.submit(
                    process_participant_trial,
                    participant_id,
                    participant_data,
                    test_trial_index,
                    current_mode,
                    num_examples,
                    model,
                    "small",  # data_size parameter
                    example_type
                )
            futures.append(future)
        
        # Collect results as they complete
        for future in tqdm(futures, desc="Processing batch", position=1, leave=False):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
    
    return results

def process_vllm_batch_group(
    batch_group: List[Dict],
    model: LLMModel,
    participants_data: Dict,
    num_examples: int,
    example_type: str,
    data_size: str,
    control_mode: bool = False
) -> List[Dict]:
    """
    Process a batch group (multiple samples of the same trial) through vLLM optimal batching.
    All prompts in the group are for the same trial but with different example samplings.
    
    Args:
        batch_group: List of task dictionaries for the same trial
        model: LLMModel instance
        participants_data: All participants data
        num_examples: Number of examples to use
        example_type: Type of examples to include
        data_size: Dataset size
        control_mode: If True, sample from other participants' data instead of same participant
        
    Returns:
        List of result dictionaries
    """
    if not batch_group:
        return []
    
    # All tasks in the group should be for the same trial and mode
    first_task = batch_group[0]
    participant_id = first_task['participant_id']
    participant_data = first_task['participant_data']
    test_trial_index = first_task['test_trial_index']
    current_mode = first_task['current_mode']
    
    # Get test data (same for all samples)
    test_think_aloud = participant_data["think_aloud"][test_trial_index]
    test_choice = participant_data["choices"][test_trial_index]
    test_context = participant_data["question_context"][test_trial_index]
    
    # Prepare prompts for all samples in the batch group
    batch_prompts = []
    batch_metadata = []
    
    for task in batch_group:
        sample_i = task['sample_i']
        sample_seed = task['sample_seed']
        
        try:
            # Prepare examples with sample-specific seed
            if "within_context" in current_mode:
                trial_id = participant_data["problem_id"][test_trial_index]
                examples, example_trial_ids, example_participant_ids = prepare_within_context_examples(
                    participants_data,
                    participant_id,
                    trial_id,
                    num_examples,
                    random_state=sample_seed
                )
            else:
                examples, example_trial_ids, example_participant_ids = prepare_within_individual_examples(
                    participant_data,
                    participant_id,
                    test_trial_index,
                    num_examples,
                    data_size=data_size,
                    random_state=sample_seed,
                    control_mode=control_mode,
                    all_participants_data=participants_data
                )
            
            formatted_examples = format_examples(examples, example_type)
            
            # Get the complete prompt structure using the model's _format_prompt method
            prompt = model._format_prompt(
                current_mode,
                participant_data["probabilities"][test_trial_index],
                participant_data["outcomes"][test_trial_index],
                test_think_aloud if "cot" in current_mode else None,
                test_context,
                formatted_examples,
                example_type
            )
            
            batch_prompts.append(prompt)
            batch_metadata.append({
                'task': task,
                'examples': formatted_examples,
                'example_trial_ids': example_trial_ids,
                'example_participant_ids': example_participant_ids
            })
            
        except Exception as e:
            print(f"Error preparing prompt for sample {sample_i}: {e}")
            # Add empty prompt and metadata for this sample
            batch_prompts.append("")
            batch_metadata.append({
                'task': task,
                'examples': "",
                'example_trial_ids': [],
                'example_participant_ids': [],
                'error': str(e)
            })
    
    # Batch inference through vLLM (optimal for KV caching)
    if model.model_type == "openai":
        # For OpenAI, process individually (no KV caching benefit)
        batch_responses = []
        for prompt in batch_prompts:
            if isinstance(prompt, str) and prompt == "":
                batch_responses.append(("", None, None))
            else:
                response, processed_prompt, api_response = model._generate_single_openai(prompt, model._get_max_tokens(current_mode))
                batch_responses.append((response, processed_prompt, api_response))
    else:
        # For HuggingFace/vLLM, use batch generation (optimal KV caching)
        batch_responses = model._generate_batch(batch_prompts, current_mode)
    
    # Process results
    results = []
    for i, (task_metadata, response_data) in enumerate(zip(batch_metadata, batch_responses)):
        task = task_metadata['task']
        
        try:
            if 'error' in task_metadata:
                # Handle preparation error
                result = create_error_result(task, task_metadata['error'], control_mode)
            else:
                # Process successful response
                response = model._process_response(response_data, current_mode)
                choice = model._extract_choice(response)
                
                # Get choice probabilities
                if model.model_type == "openai":
                    prob_A, prob_B = model._get_choice_probabilities(batch_prompts[i], response, response_data[2])
                else:
                    prob_A, prob_B = model._get_choice_probabilities(batch_prompts[i], response, response_data[2])
                
                result = {
                    "sub_id": str(participant_id),
                    "trial_id": participant_data["problem_id"][test_trial_index],
                    "mode": current_mode,
                    "num_examples": num_examples,
                    "example_type": example_type,
                    "sample_id": task['sample_i'],
                    "sample_seed": task['sample_seed'],
                    "control_mode": control_mode,  # Add control mode indicator
                    "think_aloud": test_think_aloud,
                    "actual_choice": test_choice,
                    "extracted_choice": int(choice) if choice is not None else None,
                    "probability_option_a": float(prob_A),
                    "probability_option_b": float(prob_B),
                    "logits": None,  # Not available in this pipeline
                    "full_generation": response,
                    "question_context": test_context,
                    "examples": task_metadata['examples'],
                    "example_trial_ids": task_metadata['example_trial_ids'],
                    "example_participant_ids": task_metadata['example_participant_ids'],
                    "pa": float(participant_data["probabilities"][test_trial_index][0]),
                    "va": float(participant_data["outcomes"][test_trial_index][0]),
                    "pb": float(participant_data["probabilities"][test_trial_index][1]),
                    "vb": float(participant_data["outcomes"][test_trial_index][1]),
                    "complete_prompt": batch_prompts[i] if i < len(batch_prompts) else ""
                }
                
        except Exception as e:
            print(f"Error processing result for sample {task['sample_i']}: {e}")
            result = create_error_result(task, str(e), control_mode)
        
        results.append(result)
    
    return results

def create_error_result(task: Dict, error_message: str, control_mode: bool = False) -> Dict:
    """Create an error result for failed processing."""
    participant_id = task['participant_id']
    participant_data = task['participant_data']
    test_trial_index = task['test_trial_index']
    current_mode = task['current_mode']
    
    return {
        "sub_id": str(participant_id),
        "trial_id": participant_data["problem_id"][test_trial_index],
        "mode": current_mode,
        "num_examples": 0,
        "example_type": "error",
        "sample_id": task['sample_i'],
        "sample_seed": task['sample_seed'],
        "control_mode": control_mode,  # Add control mode indicator
        "think_aloud": participant_data["think_aloud"][test_trial_index],
        "actual_choice": participant_data["choices"][test_trial_index],
        "extracted_choice": None,
        "probability_option_a": 0.5,
        "probability_option_b": 0.5,
        "logits": None,
        "full_generation": f"ERROR: {error_message}",
        "question_context": participant_data["question_context"][test_trial_index],
        "examples": "",
        "example_trial_ids": [],
        "example_participant_ids": [],
        "pa": float(participant_data["probabilities"][test_trial_index][0]),
        "va": float(participant_data["outcomes"][test_trial_index][0]),
        "pb": float(participant_data["probabilities"][test_trial_index][1]),
        "vb": float(participant_data["outcomes"][test_trial_index][1]),
        "complete_prompt": ""
    }

def run_in_context_experiment(
    data_size: str = "small",
    mode: str = "all",
    context_type: str = "all",
    num_examples: int = 0,
    model_name: str = "gpt-3.5-turbo",
    model_type: str = "openai",
    output_dir: str = "results/exp2_in_context",
    num_workers: int = 10,
    checkpoint_interval: int = 1000,
    random_state: int = 42,
    example_type: str = "think_aloud",
    tensor_parallel_size: Optional[int] = None,
    n_samples: int = 1,
    control_mode: bool = False
) -> None:
    """
    Run the in-context learning experiment.
    
    Args:
        data_size: Size of dataset ("small" or "large")
        mode: LLM mode ("base", "cot", or "all")
        context_type: Context learning type:
            - "all": Both within_individual and within_context (small dataset only)
            - "within_individual": Learn from same participant's other trials
            - "within_context": Learn from other participants with same trial
        num_examples: Number of examples to use for in-context learning
        model_name: Name of the model to use
        model_type: Type of model ("openai" or "huggingface")
        output_dir: Directory to save results
        num_workers: Number of worker threads for OpenAI API calls
        checkpoint_interval: Number of trials to process before saving checkpoint
        random_state: Random seed for reproducibility
        example_type: Type of information to include in examples ("think_aloud", "choice", "both")
        tensor_parallel_size: Optional number of GPUs for tensor parallelism (auto-detect if None)
        n_samples: Number of different example samplings per test trial (default: 1)
        control_mode: If True, use other participants' data instead of same participant for within_individual learning (control experiment)
    """
    # Validate example_type
    if example_type not in ["think_aloud", "choice", "both"]:
        raise ValueError(f"example_type must be one of ['think_aloud', 'choice', 'both'], got {example_type}")
    
    # Validate control_mode
    if control_mode and context_type not in ["within_individual", "all"]:
        raise ValueError(f"control_mode can only be used with within_individual context type, got context_type={context_type}")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # For large dataset, only within_individual context is supported
    if data_size == "large" and context_type in ["within_context", "all"]:
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
            "llm"
        )
    
    # Check GPU availability for HuggingFace models
    if model_type == "huggingface":
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                             for i in range(num_gpus)) / 1024**3
            print(f"Available: {num_gpus} GPUs, Total Memory: {total_memory:.1f}GB")
            print(f"Tensor parallel size: {tensor_parallel_size if tensor_parallel_size else 'auto-detect'}")
        else:
            print("⚠️ CUDA not available! This will be very slow.")

    # Initialize model
    print(f"\nInitializing {model_type} model: {model_name}...")
    model = LLMModel(
        model_name=model_name,
        model_type=model_type,
        mode=f"exp2_{mode}",
        num_workers=num_workers,
        tensor_parallel_size=tensor_parallel_size,
        api_key=OPENAI_API_KEY if model_type == "openai" else None
    )
    
    # Log which backend is being used
    if hasattr(model, 'vllm_model') and model.vllm_model is not None:
        print("✅ Using vLLM backend for distributed inference")
    elif hasattr(model, 'model') and model.model is not None:
        print("✅ Using HuggingFace backend")
    else:
        print("✅ Using OpenAI API backend")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint directory - clearly show key parameters in logical order
    model_name_safe = model_name.replace("/", "_").replace(":", "_")
    control_suffix = "_control" if control_mode else ""
    checkpoint_dir = os.path.join(output_dir, "checkpoints", f"{model_name_safe}_{data_size}_{mode}_{context_type}_{num_examples}examples_{example_type}_{n_samples}samples{control_suffix}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Collect all results
    all_results = []
    
    # Determine which LLM modes to run
    if mode == "all":
        llm_modes = ["base", "cot"]
    else:
        llm_modes = [mode]
    
    # Determine which context types to run
    if context_type == "all":
        context_types = ["within_individual", "within_context"]
    else:
        context_types = [context_type]
    
    # For large dataset, only run within-individual context
    if data_size == "large":
        if "within_context" in context_types:
            print(f"⚠️  Large dataset detected. Overriding context_type to within_individual only.")
            context_types = ["within_individual"]
    
    # Build the actual mode combinations to run
    modes_to_run = []
    for llm_mode in llm_modes:
        for ctx_type in context_types:
            modes_to_run.append(f"{llm_mode}_{ctx_type}")
    
    print(f"\nLLM modes: {llm_modes}")
    print(f"Context types: {context_types}")
    print(f"Actual modes to run: {modes_to_run}")
    print(f"Total combinations: {len(modes_to_run)} mode(s) × {n_samples} sample(s) = {len(modes_to_run) * n_samples}x base workload")
    
    # Prepare all tasks for optimal vLLM batching: mode → context_type → trial → samples
    print("\nOrganizing tasks for optimal vLLM batching...")
    print("Order: mode → context_type → trial → samples (maximizes KV cache reuse)")
    
    all_batch_groups = []  # Each group contains similar prompts for optimal batching
    
    # First, collect all test trials for each participant
    participant_test_trials = {}
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
            participant_test_trials[participant_id] = test_indices.tolist()
        else:
            # For small dataset, use leave-one-out - each trial is tested once
            participant_test_trials[participant_id] = list(range(len(participant_data["choices"])))
    
    # Organize by mode → context_type → trial → samples for optimal vLLM batching
    for llm_mode in llm_modes:  # ["base", "cot"]
        for ctx_type in context_types:  # ["within_individual", "within_context"]
            current_mode = f"{llm_mode}_{ctx_type}"
            print(f"\nPreparing batches for {current_mode}...")
            
            # Collect all trials for this mode+context combination
            mode_trials = []
            for participant_id in participant_ids:
                participant_data = participants_data[participant_id]
                test_indices = participant_test_trials[participant_id]
                
                for test_trial_index in test_indices:
                    # Group all samples of the same trial together (optimal for KV caching)
                    trial_samples = []
                    for sample_i in range(n_samples):
                        sample_seed = 2025 + sample_i
                        trial_samples.append({
                            'participant_id': participant_id,
                            'participant_data': participant_data,
                            'test_trial_index': test_trial_index,
                            'current_mode': current_mode,
                            'sample_i': sample_i,
                            'sample_seed': sample_seed
                        })
                    
                    # Add this trial's samples as a batch group
                    if trial_samples:
                        all_batch_groups.append(trial_samples)
            
            print(f"  Created {len([g for g in all_batch_groups if g[0]['current_mode'] == current_mode])} trial groups with {n_samples} samples each")
    
    # Flatten into tasks while maintaining the optimal ordering
    all_tasks = []
    for batch_group in all_batch_groups:
        for task in batch_group:
            all_tasks.append((
                task['participant_id'],
                task['participant_data'], 
                task['test_trial_index'],
                task['current_mode'],
                task['sample_i'],
                task['sample_seed']
            ))
    
    print(f"\nTotal organized tasks: {len(all_tasks)} (with {n_samples} sample(s) per trial)")
    print(f"Batch groups for vLLM: {len(all_batch_groups)} (each group has {n_samples} similar prompts)")
    
    # Process in optimal vLLM batches
    print(f"\nProcessing {len(all_batch_groups)} batch groups with vLLM optimal batching...")
    processed_groups = 0
    processed_count = 0
    
    # Calculate how many batch groups to process before checkpointing
    # Aim for checkpoint_interval total tasks
    groups_per_checkpoint = max(1, checkpoint_interval // n_samples) if n_samples > 0 else checkpoint_interval
    
    for i in tqdm(range(0, len(all_batch_groups), groups_per_checkpoint), desc="Processing checkpoint batches"):
        checkpoint_batch_groups = all_batch_groups[i:i + groups_per_checkpoint]
        checkpoint_results = []
        
        # Process each batch group through vLLM optimal batching
        for batch_group in tqdm(checkpoint_batch_groups, desc="Processing vLLM batches", leave=False):
            try:
                group_results = process_vllm_batch_group(
                    batch_group, 
                    model, 
                    participants_data, 
                    num_examples, 
                    example_type, 
                    data_size,
                    control_mode
                )
                checkpoint_results.extend(group_results)
                processed_groups += 1
                processed_count += len(group_results)
            except Exception as e:
                print(f"Error processing batch group: {e}")
                # Add error results for this batch group
                for task in batch_group:
                    error_result = create_error_result(task, str(e), control_mode)
                    checkpoint_results.append(error_result)
                    processed_count += 1
        
        # Add checkpoint results to all results
        all_results.extend(checkpoint_results)
        
        # Save checkpoint
        if processed_count >= checkpoint_interval or i + groups_per_checkpoint >= len(all_batch_groups):
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{processed_count}.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        # Core experiment parameters (highlighted)
                        "model_name": model_name,
                        "dataset": data_size,
                        "mode": mode,
                        "context_type": context_type,
                        "num_examples": num_examples,
                        
                        # Additional configuration
                        "example_type": example_type,
                        "n_samples": n_samples,
                        "model_type": model_type,
                        "tensor_parallel_size": tensor_parallel_size,
                        "control_mode": control_mode,
                        
                        # Checkpoint metadata
                        "processed_count": processed_count,
                        "processed_groups": processed_groups,
                        "timestamp": datetime.now().isoformat(),
                        "is_checkpoint": True
                    },
                    "results": all_results
                }, f, indent=2, ensure_ascii=False)
            print(f"\nCheckpoint saved: {checkpoint_file} ({processed_groups} groups, {processed_count} total results)")
    
    # Save final results - clearly show key parameters: model_name, dataset, mode, context_type, num_examples
    output_file = os.path.join(output_dir, f"{model_name_safe}_{data_size}_{mode}_{context_type}_{num_examples}examples_{example_type}_{n_samples}samples{control_suffix}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                # Core experiment parameters (highlighted)
                "model_name": model_name,
                "dataset": data_size,
                "mode": mode,
                "context_type": context_type,
                "num_examples": num_examples,
                
                # Additional configuration
                "example_type": example_type,
                "n_samples": n_samples,
                "model_type": model_type,
                "tensor_parallel_size": tensor_parallel_size,
                "control_mode": control_mode,
                
                # Execution metadata
                "timestamp": datetime.now().isoformat(),
                "total_trials": len(all_results),
                "modes_executed": list(set([r.get("mode", "unknown") for r in all_results]))
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to:")
    print(f"  {output_file}")
    print(f"Total results: {len(all_results)}")
    
    # Show clear file naming breakdown
    filename_parts = output_file.split("/")[-1].replace("_results.json", "").split("_")
    print(f"\nFile naming breakdown:")
    if len(filename_parts) >= 7:
        print(f"  Model: {filename_parts[0].replace('_', '/')}")
        print(f"  Dataset: {filename_parts[1]}")
        print(f"  Mode: {filename_parts[2]}")
        print(f"  Context: {filename_parts[3]}")
        print(f"  Examples: {filename_parts[4]}")
        print(f"  Type: {filename_parts[5]}")
        print(f"  Samples: {filename_parts[6]}")
        if len(filename_parts) >= 8 and filename_parts[7] == "control":
            print(f"  Control Mode: {filename_parts[7]}")
    
    # Summary by mode and sample
    if len(all_results) > 0:
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETION SUMMARY")
        print("="*60)
        
        # Show key experiment parameters
        print("KEY PARAMETERS:")
        print(f"  Model: {model_name}")
        print(f"  Dataset: {data_size}")
        print(f"  Mode: {mode}")
        print(f"  Context Type: {context_type}")
        print(f"  Number of Examples: {num_examples}")
        print(f"  Example Type: {example_type}")
        print(f"  Number of Samples: {n_samples}")
        print(f"  Control Mode: {control_mode}{' (applies to within_individual only)' if control_mode and context_type == 'all' else ''}")
        
        # Show vLLM batching optimization details
        if model_type != "openai":
            print(f"\nvLLM BATCHING OPTIMIZATION:")
            print(f"  Batch Groups Processed: {len(all_batch_groups)}")
            print(f"  Samples per Group: {n_samples}")
            print(f"  Optimal Batching Order: mode → context_type → trial → samples")
            print(f"  KV Cache Efficiency: Maximized for similar prompts")
        
        # Count results by mode
        mode_counts = {}
        for result in all_results:
            result_mode = result.get("mode", "unknown")
            mode_counts[result_mode] = mode_counts.get(result_mode, 0) + 1
        
        print(f"\nRESULTS BY MODE:")
        for result_mode, count in sorted(mode_counts.items()):
            print(f"  {result_mode}: {count} results")
        
        if n_samples > 1:
            print(f"\nSAMPLE DISTRIBUTION:")
            sample_counts = {}
            for result in all_results:
                sample_id = result.get("sample_id", 0)
                sample_counts[sample_id] = sample_counts.get(sample_id, 0) + 1
            
            for sample_id, count in sorted(sample_counts.items()):
                print(f"  Sample {sample_id}: {count} results")
            
            # Show batching efficiency
            total_unique_trials = len(all_batch_groups)
            total_inferences = len(all_results)
            avg_batch_size = total_inferences / total_unique_trials if total_unique_trials > 0 else 0
            print(f"\nBATCHING EFFICIENCY:")
            print(f"  Unique Trials: {total_unique_trials}")
            print(f"  Total Inferences: {total_inferences}")
            print(f"  Average Batch Size: {avg_batch_size:.1f} (ideal: {n_samples})")
        
        print("="*60)

def run_comprehensive_test(
    data_size: str = "small",
    model_name: str = "gpt-3.5-turbo",
    model_type: str = "openai",
    output_dir: str = "results/exp2_in_context",
    num_workers: int = 10,
    checkpoint_interval: int = 100,
    random_state: int = 42,
    example_type: str = "think_aloud",
    tensor_parallel_size: Optional[int] = None,
    llm_mode: str = "base"
) -> None:
    """
    Run comprehensive test: systematic leave-one-out where each trial is tested 
    against every other trial as a single example, comparing performance with and without think-aloud.
    
    Creates a matrix where:
    - Columns represent example trials
    - Rows represent test trials  
    - Values show performance boost when think-aloud is provided vs baseline (no think-aloud)
    
    Args:
        data_size: Must be "small" (only supported for comprehensive test)
        model_name: Name of the model to use
        model_type: Type of model ("openai" or "huggingface")
        output_dir: Directory to save results
        num_workers: Number of worker threads for OpenAI API calls
        checkpoint_interval: Number of trials to process before saving checkpoint
        random_state: Random seed for reproducibility
        example_type: Type of information to include in examples ("think_aloud", "choice", "both")
        tensor_parallel_size: Optional number of GPUs for tensor parallelism
        llm_mode: LLM mode ("base" or "cot")
    """
    if data_size != "small":
        raise ValueError("Comprehensive test only supports small dataset")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Load and preprocess data
    print(f"\nLoading {data_size} dataset for comprehensive test...")
    decisions_df, think_aloud_df = load_datasets(data_size)
    think_aloud_df = preprocess_think_aloud(think_aloud_df)
    
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
            "llm"
        )
    
    # Initialize model
    print(f"\nInitializing {model_type} model: {model_name}...")
    model = LLMModel(
        model_name=model_name,
        model_type=model_type,
        mode=f"exp2_{llm_mode}",
        num_workers=num_workers,
        tensor_parallel_size=tensor_parallel_size,
        api_key=OPENAI_API_KEY if model_type == "openai" else None
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint directory
    model_name_safe = model_name.replace("/", "_").replace(":", "_")
    checkpoint_dir = os.path.join(output_dir, "checkpoints", f"{model_name_safe}_{data_size}_comprehensive_{llm_mode}_{example_type}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Collect all results
    all_results = []
    
    # Get number of trials per participant (should be same for all in small dataset)
    first_participant_data = participants_data[participant_ids[0]]
    num_trials = len(first_participant_data["problem_id"])
    print(f"Number of trials per participant: {num_trials}")
    print(f"Expected matrix size: {num_trials} × {num_trials}")
    
    # For each participant, test each trial against every other trial as single example
    total_combinations = len(participant_ids) * num_trials * (num_trials - 1) * 2  # ×2 for with/without think-aloud
    print(f"Total combinations to test: {total_combinations}")
    
    processed_count = 0
    
    for participant_id in tqdm(participant_ids, desc="Processing participants"):
        participant_data = participants_data[participant_id]
        
        # For each trial as test trial
        for test_trial_idx in range(num_trials):
            batch_tasks = []
            
            # For each other trial as example trial
            for example_trial_idx in range(num_trials):
                if example_trial_idx == test_trial_idx:
                    continue  # Skip same trial
                
                # Test both with and without think-aloud in the example
                for include_think_aloud in [True, False]:
                    batch_tasks.append({
                        'participant_id': participant_id,
                        'participant_data': participant_data,
                        'test_trial_idx': test_trial_idx,
                        'example_trial_idx': example_trial_idx,
                        'include_think_aloud': include_think_aloud,
                        'llm_mode': llm_mode
                    })
            
            # Process batch
            if model.model_type == "openai":
                # Process tasks in parallel for OpenAI
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for task in batch_tasks:
                        future = executor.submit(process_comprehensive_task, task, model, example_type)
                        futures.append(future)
                    
                    batch_results = []
                    for future in tqdm(futures, desc=f"Processing test trial {test_trial_idx}", leave=False):
                        try:
                            result = future.result()
                            batch_results.append(result)
                        except Exception as e:
                            print(f"Error processing task: {e}")
                            continue
            else:
                # Process tasks in batch for HuggingFace/vLLM
                batch_results = process_comprehensive_batch(batch_tasks, model, example_type)
            
            all_results.extend(batch_results)
            processed_count += len(batch_results)
            
            # Save checkpoint
            if processed_count % checkpoint_interval == 0:
                checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{processed_count}.json")
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": {
                            "model_name": model_name,
                            "dataset": data_size,
                            "mode": "comprehensive_test",
                            "llm_mode": llm_mode,
                            "example_type": example_type,
                            "processed_count": processed_count,
                            "num_trials": num_trials,
                            "timestamp": datetime.now().isoformat(),
                            "is_checkpoint": True
                        },
                        "results": all_results
                    }, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved: {checkpoint_file} ({processed_count} results)")
    
    # Save final results
    output_file = os.path.join(output_dir, f"{model_name_safe}_{data_size}_comprehensive_{llm_mode}_{example_type}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model_name": model_name,
                "dataset": data_size,
                "mode": "comprehensive_test",
                "llm_mode": llm_mode,
                "example_type": example_type,
                "num_trials": num_trials,
                "total_combinations": len(all_results),
                "timestamp": datetime.now().isoformat()
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nComprehensive test completed!")
    print(f"Results saved to: {output_file}")
    print(f"Total results: {len(all_results)}")
    print(f"Matrix size: {num_trials} × {num_trials} × {len(participant_ids)} participants")
    
    # Show summary statistics
    if len(all_results) > 0:
        print("\nResult breakdown:")
        think_aloud_results = [r for r in all_results if r['include_think_aloud']]
        no_think_aloud_results = [r for r in all_results if not r['include_think_aloud']]
        
        print(f"  With think-aloud: {len(think_aloud_results)} results")
        print(f"  Without think-aloud: {len(no_think_aloud_results)} results")
        
        # Show accuracy comparison if available
        if think_aloud_results and no_think_aloud_results:
            think_aloud_acc = np.mean([r['accuracy'] for r in think_aloud_results if r.get('accuracy') is not None])
            no_think_aloud_acc = np.mean([r['accuracy'] for r in no_think_aloud_results if r.get('accuracy') is not None])
            
            print(f"  Average accuracy with think-aloud: {think_aloud_acc:.3f}")
            print(f"  Average accuracy without think-aloud: {no_think_aloud_acc:.3f}")
            print(f"  Think-aloud boost: {think_aloud_acc - no_think_aloud_acc:.3f}")

def process_comprehensive_task(task: Dict, model: LLMModel, example_type: str) -> Dict:
    """Process a single comprehensive test task."""
    participant_id = task['participant_id']
    participant_data = task['participant_data']
    test_trial_idx = task['test_trial_idx']
    example_trial_idx = task['example_trial_idx']
    include_think_aloud = task['include_think_aloud']
    llm_mode = task['llm_mode']
    
    # Get test trial data
    test_think_aloud = participant_data["think_aloud"][test_trial_idx]
    test_choice = participant_data["choices"][test_trial_idx]
    test_context = participant_data["question_context"][test_trial_idx]
    
    # Get example trial data
    example_trial_data = {
        "trial_id": participant_data["problem_id"][example_trial_idx],
        "probabilities": participant_data["probabilities"][example_trial_idx],
        "outcomes": participant_data["outcomes"][example_trial_idx],
        "think_aloud": participant_data["think_aloud"][example_trial_idx] if include_think_aloud else "",
        "question_context": participant_data["question_context"][example_trial_idx],
        "choice": participant_data["choices"][example_trial_idx]
    }
    
    # Format single example
    formatted_example = format_examples([example_trial_data], example_type if include_think_aloud else "choice")
    
    # Create mode string
    current_mode = f"{llm_mode}_within_individual"
    
    # Get the complete prompt structure
    prompt = model._format_prompt(
        current_mode,
        participant_data["probabilities"][test_trial_idx],
        participant_data["outcomes"][test_trial_idx],
        test_think_aloud if "cot" in llm_mode else None,
        test_context,
        formatted_example,
        example_type if include_think_aloud else "choice"
    )
    
    # Get model prediction
    prediction = model.predict(
        probabilities=participant_data["probabilities"][test_trial_idx],
        outcomes=participant_data["outcomes"][test_trial_idx],
        think_aloud=test_think_aloud if "cot" in llm_mode else None,
        question_context=test_context,
        mode=current_mode,
        examples=formatted_example,
        example_type=example_type if include_think_aloud else "choice"
    )
    
    # Calculate accuracy
    accuracy = 1.0 if int(prediction["choice"]) == test_choice else 0.0
    
    # Store result
    result = {
        "sub_id": str(participant_id),
        "test_trial_id": participant_data["problem_id"][test_trial_idx],
        "test_trial_idx": test_trial_idx,
        "example_trial_id": participant_data["problem_id"][example_trial_idx],
        "example_trial_idx": example_trial_idx,
        "include_think_aloud": include_think_aloud,
        "mode": current_mode,
        "llm_mode": llm_mode,
        "example_type": example_type,
        "think_aloud": test_think_aloud,
        "actual_choice": test_choice,
        "extracted_choice": int(prediction["choice"]),
        "accuracy": accuracy,
        "probability_option_a": float(prediction["probabilities"]["A"]),
        "probability_option_b": float(prediction["probabilities"]["B"]),
        "full_generation": prediction["generation"],
        "question_context": test_context,
        "example_context": example_trial_data["question_context"],
        "example_think_aloud": example_trial_data["think_aloud"],
        "example_choice": example_trial_data["choice"],
        "formatted_example": formatted_example,
        "pa": float(participant_data["probabilities"][test_trial_idx][0]),
        "va": float(participant_data["outcomes"][test_trial_idx][0]),
        "pb": float(participant_data["probabilities"][test_trial_idx][1]),
        "vb": float(participant_data["outcomes"][test_trial_idx][1]),
        "complete_prompt": prompt
    }
    
    return result

def process_comprehensive_batch(batch_tasks: List[Dict], model: LLMModel, example_type: str) -> List[Dict]:
    """Process a batch of comprehensive test tasks using vLLM."""
    if not batch_tasks:
        return []
    
    # Prepare prompts for all tasks
    batch_prompts = []
    batch_metadata = []
    
    for task in batch_tasks:
        participant_id = task['participant_id']
        participant_data = task['participant_data']
        test_trial_idx = task['test_trial_idx']
        example_trial_idx = task['example_trial_idx']
        include_think_aloud = task['include_think_aloud']
        llm_mode = task['llm_mode']
        
        # Get test trial data
        test_think_aloud = participant_data["think_aloud"][test_trial_idx]
        test_choice = participant_data["choices"][test_trial_idx]
        test_context = participant_data["question_context"][test_trial_idx]
        
        # Get example trial data
        example_trial_data = {
            "trial_id": participant_data["problem_id"][example_trial_idx],
            "probabilities": participant_data["probabilities"][example_trial_idx],
            "outcomes": participant_data["outcomes"][example_trial_idx],
            "think_aloud": participant_data["think_aloud"][example_trial_idx] if include_think_aloud else "",
            "question_context": participant_data["question_context"][example_trial_idx],
            "choice": participant_data["choices"][example_trial_idx]
        }
        
        # Format single example
        formatted_example = format_examples([example_trial_data], example_type if include_think_aloud else "choice")
        
        # Create mode string
        current_mode = f"{llm_mode}_within_individual"
        
        # Get the complete prompt structure
        prompt = model._format_prompt(
            current_mode,
            participant_data["probabilities"][test_trial_idx],
            participant_data["outcomes"][test_trial_idx],
            test_think_aloud if "cot" in llm_mode else None,
            test_context,
            formatted_example,
            example_type if include_think_aloud else "choice"
        )
        
        batch_prompts.append(prompt)
        batch_metadata.append({
            'task': task,
            'participant_data': participant_data,
            'test_trial_idx': test_trial_idx,
            'example_trial_idx': example_trial_idx,
            'include_think_aloud': include_think_aloud,
            'llm_mode': llm_mode,
            'test_choice': test_choice,
            'formatted_example': formatted_example,
            'example_trial_data': example_trial_data,
            'current_mode': current_mode
        })
    
    # Batch inference
    if model.model_type == "openai":
        # For OpenAI, process individually
        batch_responses = []
        for prompt in batch_prompts:
            response, processed_prompt, api_response = model._generate_single_openai(prompt, model._get_max_tokens(current_mode))
            batch_responses.append((response, processed_prompt, api_response))
    else:
        # For HuggingFace/vLLM, use batch generation
        batch_responses = model._generate_batch(batch_prompts, current_mode)
    
    # Process results
    results = []
    for i, (metadata, response_data) in enumerate(zip(batch_metadata, batch_responses)):
        try:
            task = metadata['task']
            participant_id = task['participant_id']
            participant_data = metadata['participant_data']
            test_trial_idx = metadata['test_trial_idx']
            example_trial_idx = metadata['example_trial_idx']
            
            # Process response
            response = model._process_response(response_data, metadata['current_mode'])
            choice = model._extract_choice(response)
            
            # Get choice probabilities
            if model.model_type == "openai":
                prob_A, prob_B = model._get_choice_probabilities(batch_prompts[i], response, response_data[2])
            else:
                prob_A, prob_B = model._get_choice_probabilities(batch_prompts[i], response, response_data[2])
            
            # Calculate accuracy
            accuracy = 1.0 if int(choice) == metadata['test_choice'] else 0.0
            
            result = {
                "sub_id": str(participant_id),
                "test_trial_id": participant_data["problem_id"][test_trial_idx],
                "test_trial_idx": test_trial_idx,
                "example_trial_id": participant_data["problem_id"][example_trial_idx],
                "example_trial_idx": example_trial_idx,
                "include_think_aloud": metadata['include_think_aloud'],
                "mode": metadata['current_mode'],
                "llm_mode": metadata['llm_mode'],
                "example_type": example_type,
                "think_aloud": participant_data["think_aloud"][test_trial_idx],
                "actual_choice": metadata['test_choice'],
                "extracted_choice": int(choice) if choice is not None else None,
                "accuracy": accuracy,
                "probability_option_a": float(prob_A),
                "probability_option_b": float(prob_B),
                "full_generation": response,
                "question_context": participant_data["question_context"][test_trial_idx],
                "example_context": metadata['example_trial_data']["question_context"],
                "example_think_aloud": metadata['example_trial_data']["think_aloud"],
                "example_choice": metadata['example_trial_data']["choice"],
                "formatted_example": metadata['formatted_example'],
                "pa": float(participant_data["probabilities"][test_trial_idx][0]),
                "va": float(participant_data["outcomes"][test_trial_idx][0]),
                "pb": float(participant_data["probabilities"][test_trial_idx][1]),
                "vb": float(participant_data["outcomes"][test_trial_idx][1]),
                "complete_prompt": batch_prompts[i]
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing comprehensive task: {e}")
            continue
    
    return results

def main():
    """
    Main function with optimized vLLM batching.
    
    vLLM Batching Optimization:
    - Tasks are organized by: mode → context_type → trial → samples
    - All samples of the same trial are batched together for optimal KV cache reuse
    - This maximizes inference speed for HuggingFace models with vLLM
    
    Usage Examples:
    
    # Single mode with multiple samples (optimal for vLLM batching):
    python exp2_in_context.py --mode base --context_type within_individual --n_samples 5
    python exp2_in_context.py --mode cot --context_type within_individual --n_samples 5
    
    # Multiple modes with single sample (original behavior):
    python exp2_in_context.py --mode all --context_type within_individual --n_samples 1
    
    # All modes and contexts (use with caution for large n_samples):
    python exp2_in_context.py --mode all --context_type all --n_samples 1
    
    # Control experiment (tests if within_individual really learns individual characteristics):
    python exp2_in_context.py --mode base --context_type within_individual --control_mode
    
    # Comprehensive test (systematic trial-by-trial analysis for 19×19 matrix):
    python exp2_in_context.py --mode comprehensive_test --llm_mode base
    python exp2_in_context.py --mode comprehensive_test --llm_mode cot --example_type think_aloud
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run in-context learning experiment")
    parser.add_argument("--model_name", type=str, default="gpt-4",
                      help="Name of the model to use")
    parser.add_argument("--model_type", type=str, default="openai",
                      choices=["openai", "huggingface"],
                      help="Type of model to use")
    parser.add_argument("--data_size", type=str, default="small",
                      choices=["small", "large"],
                      help="Size of dataset to use")
    parser.add_argument("--num_examples", type=int, default=2,
                      help="Number of examples to use")
    parser.add_argument("--mode", type=str, default="all",
                      choices=["base", "cot", "all", "comprehensive_test"],
                      help="LLM mode (use specific modes like 'base' for large n_samples) or 'comprehensive_test' for systematic trial-by-trial analysis")
    parser.add_argument("--context_type", type=str, default="all",
                      choices=["within_individual", "within_context", "all"],
                      help="Context learning type")
    parser.add_argument("--example_type", type=str, default="think_aloud",
                      choices=["think_aloud", "choice", "both"],
                      help="Type of information to include in examples")
    parser.add_argument("--num_workers", type=int, default=10,
                      help="Number of worker threads for OpenAI API calls")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                      help="Number of trials to process before saving checkpoint")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                      help="Number of GPUs to use for tensor parallelism (auto-detect if not specified)")
    parser.add_argument("--n_samples", type=int, default=1,
                      help="Number of different example samplings per test trial (default: 1)")
    parser.add_argument("--output_dir", type=str, default="./results/exp2_in_context",
                      help="Directory to save results")
    parser.add_argument("--control_mode", action="store_true", default=False,
                      help="Run control experiment: use other participants' data instead of same participant for within_individual learning")
    parser.add_argument("--llm_mode", type=str, default="base",
                      choices=["base", "cot"],
                      help="LLM mode for comprehensive test (base or cot)")
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("IN-CONTEXT LEARNING EXPERIMENT")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Model Type: {args.model_type}")
    print(f"Data Size: {args.data_size}")
    print(f"LLM Mode: {args.mode}")
    if args.mode == "comprehensive_test":
        print(f"Comprehensive Test LLM Mode: {args.llm_mode}")
    print(f"Context Type: {args.context_type}")
    print(f"Number of Examples: {args.num_examples}")
    print(f"Example Type: {args.example_type}")
    print(f"Number of Samples: {args.n_samples}")
    print(f"Control Mode: {args.control_mode}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size if args.tensor_parallel_size else 'auto-detect'}")
    print("="*60)
    
    # Warn about control_mode with within_context
    if args.control_mode and args.context_type == "within_context":
        print("⚠️  WARNING: Control mode is designed for within_individual context only.")
        print("   Control mode will be ignored for within_context experiments.")
        print("="*60)
    
    # Warn about computational load for large n_samples with multiple modes
    if args.n_samples > 1 and (args.mode == "all" or args.context_type == "all"):
        # Calculate expected multiplier
        mode_multiplier = 2 if args.mode == "all" else 1
        context_multiplier = 2 if args.context_type == "all" and args.data_size == "small" else 1
        total_multiplier = mode_multiplier * context_multiplier
        
        if total_multiplier > 1:
            estimated_total = args.n_samples * total_multiplier
            print(f"⚠️  WARNING: Running {args.mode} mode + {args.context_type} context with {args.n_samples} samples will generate")
            print(f"   {estimated_total}x more results than baseline (n_samples × {total_multiplier} combinations)")
            print(f"   Consider using specific settings like '--mode base --context_type within_individual' for separate runs")
            print("="*60)
    
    # Run experiment
    if args.mode == "comprehensive_test":
        # Run comprehensive test (systematic trial-by-trial analysis)
        if args.data_size != "small":
            print("⚠️  Comprehensive test only supports small dataset. Switching to small dataset.")
            args.data_size = "small"
        
        if args.context_type != "within_individual":
            print("⚠️  Comprehensive test only supports within_individual context. Switching to within_individual.")
            args.context_type = "within_individual"
        
        # For comprehensive test, we need to specify the LLM mode (base or cot)
        if args.n_samples > 1:
            print("⚠️  Comprehensive test uses n_samples=1. Overriding n_samples.")
            args.n_samples = 1
        
        print(f"Running comprehensive test with LLM mode: {args.llm_mode}")
        
        run_comprehensive_test(
            data_size=args.data_size,
            model_name=args.model_name,
            model_type=args.model_type,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            checkpoint_interval=args.checkpoint_interval,
            random_state=42,
            example_type=args.example_type,
            tensor_parallel_size=args.tensor_parallel_size,
            llm_mode=args.llm_mode
        )
    else:
        # Run regular in-context experiment
        run_in_context_experiment(
            model_name=args.model_name,
            model_type=args.model_type,
            data_size=args.data_size,
            num_examples=args.num_examples,
            mode=args.mode,
            context_type=args.context_type,
            example_type=args.example_type,
            num_workers=args.num_workers,
            checkpoint_interval=args.checkpoint_interval,
            tensor_parallel_size=args.tensor_parallel_size,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            control_mode=args.control_mode
        )
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main() 