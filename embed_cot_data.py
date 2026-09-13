#!/usr/bin/env python3
"""
Chain-of-Thought Embedding Script

This script embeds four different types of chain-of-thought data using Qwen-3-embedding-8B:
1. Model generated zero-shot (from exp1)
2. Few-shot prompting (with examples, from exp2)  
3. Permuted few-shot (examples from other participants, from exp2 control mode)
4. Human think-aloud (directly from data)

The script ensures proper text processing (removing final action claims), maintains 
embedding order consistency with original data, and provides indexing for safety.

Usage:
    python embed_cot_data.py --data_source human --data_size small
    python embed_cot_data.py --data_source model_zeroshot --data_size large
    python embed_cot_data.py --data_source model_fewshot --data_size small
    python embed_cot_data.py --data_source model_permuted --data_size small
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import re
import warnings

# Set critical environment variables for vLLM
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import vLLM after setting environment variables
from vllm import LLM

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.data_processing import load_datasets, preprocess_think_aloud, get_participant_data

# Suppress warnings
warnings.filterwarnings('ignore')

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction for embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'

def clean_cot_text(text: str) -> str:
    """
    Clean chain-of-thought text by removing final action claims.
    Based on prompt design, models should consistently end with "I will choose A/B".
    
    Args:
        text: Raw CoT text that may contain final decisions
        
    Returns:
        Cleaned CoT text without final action claims
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove the standard final choice statement that prompts are designed to produce
    # This should handle the main pattern: "I will choose A" or "I will choose B"
    cleaned_text = re.sub(r'\bI will choose [AB]\.?$', '', text.strip(), flags=re.IGNORECASE)
    
    # Also handle slight variations that might occur
    cleaned_text = re.sub(r'\bI will choose option [AB]\.?$', '', cleaned_text.strip(), flags=re.IGNORECASE)
    
    # Remove any trailing whitespace or punctuation
    cleaned_text = cleaned_text.strip()
    if cleaned_text.endswith('.') or cleaned_text.endswith(','):
        cleaned_text = cleaned_text[:-1].strip()
    
    return cleaned_text

def load_human_think_aloud_data(data_size: str) -> List[Dict]:
    """
    Load human think-aloud data directly from the dataset.
    
    Args:
        data_size: Dataset size ("small" or "large")
        
    Returns:
        List of dictionaries with human think-aloud data
    """
    print(f"Loading human think-aloud data for {data_size} dataset...")
    
    # Load datasets
    decisions_df, think_aloud_df = load_datasets(data_size)
    think_aloud_df = preprocess_think_aloud(think_aloud_df)
    
    # Get all participant IDs
    participant_ids = decisions_df["sub_id"].unique().tolist()
    print(f"Found {len(participant_ids)} participants")
    
    # Collect all human think-aloud texts
    human_data = []
    total_count = 0
    
    for participant_id in tqdm(participant_ids, desc="Loading participant data"):
        participant_data = get_participant_data(
            decisions_df,
            think_aloud_df,  
            participant_id,
            "llm"
        )
        
        for i, think_aloud in enumerate(participant_data["think_aloud"]):
            # Keep ALL trials, even those with empty think-aloud
            raw_text = think_aloud if think_aloud and isinstance(think_aloud, str) else ""
            cleaned_text = clean_cot_text(raw_text) if raw_text.strip() else ""
            
            human_data.append({
                "index": total_count,
                "participant_id": participant_id,
                "trial_index": i,
                "trial_id": participant_data["problem_id"][i],
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "data_source": "human",
                "data_size": data_size
            })
            total_count += 1
    
    print(f"Loaded {len(human_data)} human think-aloud texts")
    return human_data

def load_model_generated_data(data_source: str, data_size: str) -> Dict[int, List[Dict]]:
    """
    Load model-generated CoT data from experiment results.
    
    Args:
        data_source: Type of model data ("model_zeroshot", "model_fewshot", "model_permuted")
        data_size: Dataset size ("small" or "large")
        
    Returns:
        Dictionary mapping number of examples to list of data dictionaries
    """
    print(f"Loading {data_source} data for {data_size} dataset...")
    
    # Determine the directory and file patterns based on data source
    if data_source == "model_zeroshot":
        # From exp1 results - zero-shot generation (only Meta-Llama-3.1-70B with cot)
        results_dir = "results/exp1_llm_prediction"
        return load_zeroshot_data(results_dir, data_size)
    elif data_source == "model_fewshot":
        # From exp2 results - few-shot with same participant examples (cot WITHOUT control)
        results_dir = "results/exp2_in_context"
        return load_fewshot_data(results_dir, data_size, control_mode=False)
    elif data_source == "model_permuted":
        # From exp2 control results - few-shot with other participants examples (cot WITH control)
        results_dir = "results/exp2_in_context"
        return load_fewshot_data(results_dir, data_size, control_mode=True)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

def load_zeroshot_data(results_dir: str, data_size: str) -> Dict[int, List[Dict]]:
    """Load zero-shot data from exp1 results.
    Strategy:
    1) Prefer files that already contain CoT in their filename (e.g., *_cot_*.json)
    2) If none are found, fall back to the consolidated results file and filter by mode=='cot'
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Model filter (current focus: Llama 3.1-70B)
    model_key = "meta-llama_Meta-Llama-3.1-70B-Instruct"
    
    cot_files: List[Path] = []
    fallback_files: List[Path] = []
    
    # Scan directory once and categorize
    for filename in results_path.iterdir():
        if filename.suffix != '.json' or 'checkpoint' in filename.name:
            continue
        name = filename.name
        # Focus on target model and dataset size in filename for quick filtering
        if model_key in name and f"_{data_size}_" in name:
            if 'cot' in name.lower():
                cot_files.append(filename)
            # Standard consolidated file naming (no explicit cot in name)
            if name == f"{model_key}_{data_size}_results.json":
                fallback_files.append(filename)
    
    selected_files: List[Path] = []
    if cot_files:
        print(f"Found {len(cot_files)} CoT-specific file(s) for zero-shot: {[p.name for p in cot_files]}")
        selected_files = cot_files
        filter_by_mode = True  # still filter by mode=cot for safety
    else:
        if fallback_files:
            print(f"No CoT-specific files found. Falling back to consolidated file(s): {[p.name for p in fallback_files]}")
            selected_files = fallback_files
            filter_by_mode = True  # must filter by mode=cot in consolidated file
        else:
            # As a last resort, try the standard consolidated filename directly
            target_file = results_path / f"{model_key}_{data_size}_results.json"
            if target_file.exists():
                print(f"Using fallback consolidated file: {target_file.name}")
                selected_files = [target_file]
                filter_by_mode = True
            else:
                raise FileNotFoundError(
                    f"No suitable zero-shot files found in {results_dir} for {model_key} {data_size}"
                )
    
    print(f"Found {len(selected_files)} result file(s) to load")
    
    # For zero-shot, we use 0 as the key since there are no examples
    model_data: Dict[int, List[Dict]] = {0: []}
    total_count = 0
    
    for result_file in tqdm(selected_files, desc="Loading result files"):
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate dataset size
        file_data_size = data.get('metadata', {}).get('data_size') or data.get('metadata', {}).get('dataset')
        expected_data_size = data_size
        if file_data_size != expected_data_size:
            print(f"  Skipping {result_file.name}: data_size mismatch ({file_data_size} != {expected_data_size})")
            continue
        
        results = data.get('results', [])
        print(f"  {result_file.name}: {len(results)} total results")
        
        # Filter for CoT mode if requested
        if filter_by_mode:
            filtered_results = [r for r in results if r.get('mode') == 'cot']
        else:
            filtered_results = results
        print(f"    CoT mode results (filtered by mode='cot'): {len(filtered_results)}")
        
        for result in filtered_results:
            raw_text = result.get('full_generation', '')
            # Keep ALL results, including empty ones for ordering consistency
            if not raw_text or not isinstance(raw_text, str):
                raw_text = ""
            cleaned_text = clean_cot_text(raw_text) if raw_text.strip() else ""
            
            model_data[0].append({
                "index": total_count,
                "participant_id": result.get('sub_id', ''),
                "trial_id": result.get('trial_id', ''),
                "mode": "cot",
                "num_examples": 0,
                "sample_id": result.get('sample_id', 0),
                "sample_seed": result.get('sample_seed', 0),
                "control_mode": False,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "data_source": "model_zeroshot",
                "original_file": result_file.name
            })
            total_count += 1
    
    print(f"Loaded {len(model_data[0])} zero-shot CoT texts")
    print(f"🔍 DEBUG load_zeroshot_data: Returning model_data keys: {list(model_data.keys())}")
    print(f"🔍 DEBUG load_zeroshot_data: model_data[0] length: {len(model_data[0])}")
    print(f"🔍 DEBUG load_zeroshot_data: model_data type: {type(model_data)}")
    return model_data

def load_fewshot_data(results_dir: str, data_size: str, control_mode: bool) -> Dict[int, List[Dict]]:
    """Load few-shot CoT data from exp2 results.

    This implementation scans all relevant result files for the given
    dataset size and control mode, infers the number of examples from
    the filename (e.g., "_10examples" -> 10), and groups results by that.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Focus on Llama 70B as in zero-shot loader to keep datasets consistent
    model_key = "meta-llama_Meta-Llama-3.1-70B-Instruct"

    # Collect all candidate files recursively
    candidate_files: List[Path] = []
    for path in results_path.rglob("*.json"):
        # Skip checkpoints or non-result artifacts
        if "checkpoint" in path.name:
            continue
        # Ensure target model and dataset size appear in filename
        name = path.name
        if model_key not in name or f"_{data_size}_" not in name:
            continue
        # Must be CoT runs
        if "cot" not in name:
            continue
        candidate_files.append(path)

    if not candidate_files:
        print(
            f"⚠️  No few-shot CoT files found for data_size='{data_size}', "
            f"control_mode={control_mode} in {results_dir}"
        )
        return {}

    print(f"Discovered {len(candidate_files)} candidate few-shot CoT file(s)")

    # Group results by inferred number of examples
    model_data: Dict[int, List[Dict]] = {}

    for result_file in sorted(candidate_files):
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verify dataset size via metadata when available
        file_data_size = data.get('metadata', {}).get('data_size') or data.get('metadata', {}).get('dataset')
        if file_data_size and file_data_size != data_size:
            print(
                f"  Skipping {result_file.name}: data_size mismatch (" 
                f"{file_data_size} != {data_size})"
            )
            continue

        # Validate control mode via metadata when present
        md = data.get('metadata', {})
        md_control = md.get('control_mode')
        if isinstance(md_control, bool):
            if control_mode and not md_control:
                continue
            if not control_mode and md_control:
                continue
        else:
            # Fallback to filename if metadata missing/invalid
            name = result_file.name
            contains_control = "_control_" in name or name.endswith("_control_results.json")
            if control_mode and not contains_control:
                continue
            if not control_mode and contains_control:
                continue

        # Prefer to infer number of examples from metadata, then results, then filename
        n_examples: Optional[int] = None
        for key in [
            'num_examples', 'n_examples', 'k_examples', 'k', 'num_shots', 'shots', 'examples'
        ]:
            value = md.get(key)
            if isinstance(value, int):
                n_examples = value
                break
            if isinstance(value, str) and value.isdigit():
                n_examples = int(value)
                break

        if n_examples is None:
            results_preview = data.get('results', [])
            if results_preview:
                candidate = results_preview[0].get('num_examples')
                if isinstance(candidate, int):
                    n_examples = candidate
                elif isinstance(candidate, str) and candidate.isdigit():
                    n_examples = int(candidate)

        if n_examples is None:
            match = re.search(r"_(\d+)examples", result_file.name)
            if match:
                n_examples = int(match.group(1))

        if n_examples is None:
            print(f"  ⚠️  Skipping (cannot infer #examples): {result_file.name}")
            continue

        results = data.get('results', [])
        print(f"  {result_file.name}: {len(results)} results (inferred {n_examples} examples)")

        # Initialize bucket and running index
        if n_examples not in model_data:
            model_data[n_examples] = []
        total_count = len(model_data[n_examples])

        for result in results:
            raw_text = result.get('full_generation', '')
            if not raw_text or not isinstance(raw_text, str):
                raw_text = ""
            cleaned_text = clean_cot_text(raw_text) if raw_text.strip() else ""

            model_data[n_examples].append({
                "index": total_count,
                "participant_id": result.get('sub_id', ''),
                "trial_id": result.get('trial_id', ''),
                "mode": result.get('mode', ''),
                "num_examples": n_examples,
                "sample_id": result.get('sample_id', 0),
                "sample_seed": result.get('sample_seed', 0),
                "control_mode": control_mode,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "data_source": "model_fewshot" if not control_mode else "model_permuted",
                "original_file": result_file.name
            })
            total_count += 1

    # Final reporting
    for n_examples, items in sorted(model_data.items()):
        print(f"Loaded {len(items)} CoT texts for {n_examples} examples")

    return model_data

def setup_embedding_model() -> LLM:
    """
    Set up the Qwen-3-embedding-8B model with vLLM.
    
    Returns:
        Initialized vLLM embedding model
    """
    print("Setting up Qwen-3-embedding-8B model with vLLM...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                         for i in range(num_gpus)) / 1024**3
        print(f"Available: {num_gpus} GPUs, Total Memory: {total_memory:.1f}GB")
    else:
        print("⚠️ CUDA not available! This will be very slow.")
    
    # Initialize embedding model
    model = LLM(
        model="Qwen/Qwen3-Embedding-8B",
        task="embed",
        tensor_parallel_size=1,  # Single GPU should be enough for 8B model
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )
    
    print("✅ Embedding model initialized successfully")
    return model

def embed_texts_batch(
    texts: List[str], 
    model: LLM, 
    task_description: str,
    batch_size: int = 64
) -> torch.Tensor:
    """
    Embed texts in batches using the embedding model.
    
    Args:
        texts: List of texts to embed
        model: vLLM embedding model
        task_description: Task description for instruct formatting
        batch_size: Batch size for processing
        
    Returns:
        Tensor of embeddings
    """
    print(f"Embedding {len(texts)} texts in batches of {batch_size}...")
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Format texts with instruction
        formatted_texts = [
            get_detailed_instruct(task_description, text) 
            for text in batch_texts
        ]
        
        # Get embeddings
        outputs = model.embed(formatted_texts)
        batch_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    return embeddings

def save_embeddings_with_metadata(
    embeddings: torch.Tensor,
    metadata: List[Dict],
    data_source: str,
    data_size: str,
    output_dir: str = "results/embeddings",
    num_examples: Optional[int] = None
) -> None:
    """
    Save embeddings with metadata for later use.
    
    Args:
        embeddings: Tensor of embeddings
        metadata: List of metadata dictionaries
        data_source: Source of the data
        data_size: Size of the dataset
        output_dir: Output directory
        num_examples: Number of examples (for few-shot data)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    if num_examples is not None:
        output_file = os.path.join(output_dir, f"{data_source}_{data_size}_{num_examples}examples_embeddings.pt")
        metadata_file = os.path.join(output_dir, f"{data_source}_{data_size}_{num_examples}examples_metadata.json")
    else:
        output_file = os.path.join(output_dir, f"{data_source}_{data_size}_embeddings.pt")
        metadata_file = os.path.join(output_dir, f"{data_source}_{data_size}_metadata.json")
    
    # Save embeddings
    torch.save(embeddings, output_file)
    
    # Save metadata
    metadata_dict = {
        "data_source": data_source,
        "data_size": data_size,
        "num_embeddings": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "metadata": metadata
    }
    
    if num_examples is not None:
        metadata_dict["num_examples"] = num_examples
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Saved embeddings to: {output_file}")
    print(f"Saved metadata to: {metadata_file}")
    print(f"Embeddings shape: {embeddings.shape}")

def main():
    """Main function to embed CoT data."""
    parser = argparse.ArgumentParser(description="Embed chain-of-thought data using Qwen-3-embedding-8B")
    
    parser.add_argument(
        "--data_source", 
        type=str, 
        required=True,
        choices=["human", "model_zeroshot", "model_fewshot", "model_permuted"],
        help="Source of chain-of-thought data to embed"
    )
    
    parser.add_argument(
        "--data_size",
        type=str,
        required=True, 
        choices=["small", "large"],
        help="Size of dataset to process"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/embeddings",
        help="Directory to save embeddings"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding processing"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("CHAIN-OF-THOUGHT EMBEDDING SCRIPT")
    print("="*60)
    print(f"Data Source: {args.data_source}")
    print(f"Data Size: {args.data_size}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print("="*60)
    
    # Load data based on source
    if args.data_source == "human":
        # Human data returns a simple list
        data = load_human_think_aloud_data(args.data_size)
        if not data:
            print("❌ No human data loaded. Exiting.")
            return
        
        # Process human data as a single batch
        data_batches = {None: data}  # Use None as key for human data
    else:
        # Model data returns a dictionary mapping num_examples to data lists
        data_dict = load_model_generated_data(args.data_source, args.data_size)
        if not data_dict:
            print("❌ No model data loaded. Exiting.")
            return
        
        # Debug: Print what we actually loaded
        print(f"🔍 DEBUG: data_dict keys: {list(data_dict.keys())}")
        for key, value in data_dict.items():
            print(f"🔍 DEBUG: data_dict[{key}] type: {type(value)}, length: {len(value) if value else 0}")
            if value:
                print(f"🔍 DEBUG: data_dict[{key}][0] keys: {list(value[0].keys()) if len(value) > 0 else 'empty list'}")
        
        data_batches = data_dict
    
    # Set up embedding model
    model = setup_embedding_model()
    
    # Define task description based on data source
    task_descriptions = {
        "human": "Given a decision-making scenario, analyze the human reasoning process and thought patterns in this think-aloud text",
        "model_zeroshot": "Given a decision-making scenario, analyze the AI model's zero-shot reasoning process and chain-of-thought",
        "model_fewshot": "Given a decision-making scenario with examples, analyze the AI model's few-shot reasoning process and chain-of-thought",
        "model_permuted": "Given a decision-making scenario with permuted examples, analyze the AI model's reasoning process and chain-of-thought"
    }
    
    task_description = task_descriptions[args.data_source]
    
    # Process each data batch separately
    total_embeddings_created = 0
    
    for num_examples, data in data_batches.items():
        print(f"🔍 DEBUG: Processing num_examples={num_examples}, data type={type(data)}, data length={len(data) if data else 0}")
        print(f"🔍 DEBUG: bool(data) = {bool(data)}, 'not data' = {not data}")
        
        if not data:
            print(f"⚠️  No data for {num_examples} examples, skipping...")
            continue
        
        print(f"\n" + "="*40)
        if num_examples is None:
            print(f"Processing {args.data_source} data...")
        else:
            print(f"Processing {args.data_source} data with {num_examples} examples...")
        print("="*40)
        
        # Extract texts for embedding (keep ALL texts, including empty ones)
        texts = [item["cleaned_text"] for item in data]
        print(f"Prepared {len(texts)} texts for embedding")
        
        # Count empty vs non-empty texts for info
        empty_count = sum(1 for text in texts if not text.strip())
        non_empty_count = len(texts) - empty_count
        print(f"  Non-empty texts: {non_empty_count}")
        print(f"  Empty texts: {empty_count}")
        print(f"  Total texts: {len(texts)} (keeping ALL for order consistency)")
        
        if len(texts) == 0:
            print(f"❌ No texts for {num_examples} examples. Skipping.")
            continue
        
        # Embed texts (including empty ones for order consistency)
        embeddings = embed_texts_batch(
            texts, 
            model, 
            task_description,
            args.batch_size
        )
        
        # Save embeddings with metadata
        save_embeddings_with_metadata(
            embeddings,
            data,
            args.data_source,
            args.data_size,
            args.output_dir,
            num_examples
        )
        
        total_embeddings_created += len(embeddings)
    
    print("\n" + "="*60)
    print("EMBEDDING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Data Source: {args.data_source}")
    print(f"Data Size: {args.data_size}")
    print(f"Total Embeddings Created: {total_embeddings_created}")
    
    if args.data_source == "human":
        print(f"Output Files:")
        print(f"  Embeddings: {args.output_dir}/{args.data_source}_{args.data_size}_embeddings.pt")
        print(f"  Metadata: {args.output_dir}/{args.data_source}_{args.data_size}_metadata.json")
    else:
        print(f"Output Files (by number of examples):")
        for num_examples in sorted(data_batches.keys()):
            if data_batches[num_examples]:  # Only show if data exists
                print(f"  {num_examples} examples:")
                print(f"    Embeddings: {args.output_dir}/{args.data_source}_{args.data_size}_{num_examples}examples_embeddings.pt")
                print(f"    Metadata: {args.output_dir}/{args.data_source}_{args.data_size}_{num_examples}examples_metadata.json")
    
    print("="*60)

if __name__ == "__main__":
    main() 