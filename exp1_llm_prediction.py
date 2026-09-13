"""
Experiment 1: LLM prediction for risky decision-making tasks.
Final optimized version with simplified vLLM integration.
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
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from models.llm_model import LLMModel
from utils.data_processing import (
    load_datasets,
    preprocess_think_aloud,
    get_participant_data
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI API key: set OPENAI_API_KEY in the environment; never commit secrets.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def permute_think_aloud_by_choice(
    decisions_df: pd.DataFrame,
    think_aloud_df: pd.DataFrame,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Permute think-aloud texts among trials with the same final choice.
    This creates a control condition to test if LLMs are simply using action claims
    in think-aloud rather than meaningful cognitive processes.
    
    Args:
        decisions_df: DataFrame with decision data including choices
        think_aloud_df: DataFrame with think-aloud data
        random_seed: Fixed random seed for reproducible permutation
        
    Returns:
        DataFrame with permuted think-aloud data
    """
    logger.info(f"Permuting think-aloud data with random seed {random_seed}...")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Merge dataframes to get choice information with think-aloud
    merged_df = pd.merge(
        think_aloud_df, 
        decisions_df[['sub_id', 'problem_id', 'choice']], 
        on=['sub_id', 'problem_id'], 
        how='inner'
    )
    
    # Create a copy for permutation
    permuted_df = think_aloud_df.copy()
    
    # Group by choice (0 for Option A, 1 for Option B)
    choice_groups = merged_df.groupby('choice')
    
    logger.info(f"Found {len(choice_groups)} choice groups:")
    for choice, group in choice_groups:
        choice_name = "Option A" if choice == 0 else "Option B"
        logger.info(f"  {choice_name}: {len(group)} trials")
    
    # Permute think-aloud within each choice group
    for choice, group in choice_groups:
        choice_name = "Option A" if choice == 0 else "Option B"
        logger.info(f"Permuting think-aloud for {choice_name} ({len(group)} trials)...")
        
        # Get think-aloud texts for this choice group
        think_aloud_texts = group['think_aloud'].values.copy()
        
        # Permute the think-aloud texts
        np.random.shuffle(think_aloud_texts)
        
        # Create mapping from (sub_id, problem_id) to permuted think-aloud
        for i, (_, row) in enumerate(group.iterrows()):
            sub_id = row['sub_id']
            problem_id = row['problem_id']
            
            # Find the corresponding row in permuted_df and update think-aloud
            mask = (permuted_df['sub_id'] == sub_id) & (permuted_df['problem_id'] == problem_id)
            permuted_df.loc[mask, 'think_aloud'] = think_aloud_texts[i]
    
    # Update processed_text as well if it exists
    if 'processed_text' in permuted_df.columns:
        permuted_df['processed_text'] = permuted_df['think_aloud']
    
    logger.info("Think-aloud permutation completed.")
    return permuted_df


def remove_superficial_action_claims(
    think_aloud_df: pd.DataFrame,
    superficial_claims_csv: str,
    mask_token: str = "[ACTION_REMOVED]",
) -> pd.DataFrame:
    """
    Remove or mask superficial action claims from think-aloud transcripts.

    This function aligns the original think-aloud dataframe with the
    superficial-claims extraction output (from utils/superficial_action_extraction.py),
    and replaces each extracted claim span with a mask token.

    Args:
        think_aloud_df: Original think-aloud dataframe (must contain sub_id, problem_id, think_aloud).
        superficial_claims_csv: Path to CSV with columns
            sub_id, problem_id, has_superficial_claim, superficial_claims_json.
        mask_token: Token to insert in place of each superficial claim.

    Returns:
        A copy of think_aloud_df with superficial action claims masked out.
    """
    logger.info(f"Loading superficial action claims from: {superficial_claims_csv}")
    claims_df = pd.read_csv(superficial_claims_csv)

    required_cols = {"sub_id", "problem_id", "superficial_claims_json"}
    missing = required_cols - set(claims_df.columns)
    if missing:
        raise ValueError(
            f"superficial_claims_csv is missing required columns: {sorted(missing)}"
        )

    # Restrict to the columns we actually need for alignment
    claims_df = claims_df[["sub_id", "problem_id", "superficial_claims_json"]].copy()

    # Merge so that every row in think_aloud_df can see its claims (if any)
    merged = pd.merge(
        think_aloud_df,
        claims_df,
        on=["sub_id", "problem_id"],
        how="left",
        suffixes=("", "_claims"),
    )

    def _mask_row(row: pd.Series) -> str:
        text = row.get("think_aloud", "")
        if not isinstance(text, str) or not text:
            return text

        claims_json = row.get("superficial_claims_json")
        if not isinstance(claims_json, str) or not claims_json.strip():
            return text

        try:
            claims = json.loads(claims_json)
        except Exception:
            return text

        # claims is expected to be a list of {"text": ..., "normalized_choice": ...}
        # If multiple matches exist within the same transcript, we only mask the
        # *last* occurrence overall (most likely the action/choice commitment).
        last_idx = -1
        last_claim_text = ""
        last_claim_order = -1

        for claim_order, item in enumerate(claims or []):
            try:
                claim_text = str(item.get("text", "")).strip()
            except Exception:
                claim_text = ""
            if not claim_text:
                continue

            idx = text.rfind(claim_text)
            if idx == -1:
                continue
            if idx > last_idx or (idx == last_idx and claim_order > last_claim_order):
                last_idx = idx
                last_claim_text = claim_text
                last_claim_order = claim_order

        if last_idx == -1:
            return text

        # Replace only that last match with a mask token.
        return (
            text[:last_idx]
            + f" {mask_token} "
            + text[last_idx + len(last_claim_text) :]
        )

    logger.info("Applying action-removal mask to think-aloud transcripts...")
    masked = merged.copy()
    masked["think_aloud"] = masked.apply(_mask_row, axis=1)

    # Keep schema consistent with original think_aloud_df
    if "processed_text" in masked.columns:
        masked["processed_text"] = masked["think_aloud"]

    # Drop helper column and return in original column order where possible
    masked = masked.drop(columns=["superficial_claims_json"], errors="ignore")
    masked = masked[think_aloud_df.columns]  # align column order

    logger.info("Superficial action claims have been masked out.")
    return masked

def run_llm_prediction_optimized(
    model_name: str,
    model_type: str,
    data_size: str = "small",
    mode: str = "all",
    num_workers: int = 10,
    batch_size_per_device: int = 4,
    checkpoint_interval: int = 1000,
    output_dir: str = "results",
    force_hf_fallback: bool = False,
    tensor_parallel_size: Optional[int] = None,
    superficial_claims_csv: Optional[str] = None,
) -> List[Dict]:
    """
    Run optimized LLM prediction experiment using batch processing.
    
    Args:
        model_name: Name of the model to use
        model_type: Type of model (huggingface or openai)
        data_size: Size of dataset to use (small, large, all)
        mode: Experiment mode (base, cot, human, permutation, or all)
        num_workers: Number of worker threads for OpenAI API calls
        batch_size_per_device: Total batch size for processing
        checkpoint_interval: Number of results before saving checkpoint
        output_dir: Directory to save results
        force_hf_fallback: Force HuggingFace fallback instead of vLLM
        tensor_parallel_size: Optional tensor parallel size
    
    Returns:
        List of all results
    """
    # Handle "all" data_size by running experiments on each dataset separately
    if data_size == "all":
        all_results = []
        for dataset_size in ["small", "large"]:
            logger.info(f"Running experiment on {dataset_size} dataset...")
            dataset_results = run_llm_prediction_optimized(
                model_name=model_name,
                model_type=model_type,
                data_size=dataset_size,
                mode=mode,
                num_workers=num_workers,
                batch_size_per_device=batch_size_per_device,
                checkpoint_interval=checkpoint_interval,
                output_dir=output_dir,
                force_hf_fallback=force_hf_fallback,
                tensor_parallel_size=tensor_parallel_size
            )
            all_results.extend(dataset_results)
        return all_results
    
    # Load and preprocess data
    logger.info(f"Loading {data_size} dataset...")
    try:
        decisions_df, think_aloud_df = load_datasets(data_size)

        # Optionally create an action-removed version of the think-aloud data
        if mode == "action_removal":
            if superficial_claims_csv is None:
                raise ValueError(
                    "action_removal mode requires --superficial_claims_csv "
                    "pointing to the superficial claims extraction CSV."
                )
            ta_for_action_removal = remove_superficial_action_claims(
                think_aloud_df, superficial_claims_csv=superficial_claims_csv
            )
            think_aloud_df_action_removed = preprocess_think_aloud(ta_for_action_removal)
        else:
            think_aloud_df_action_removed = None

        # Standard preprocessing for the original think-aloud data
        think_aloud_df = preprocess_think_aloud(think_aloud_df)
        
        # Apply permutation if mode is "permutation" or "all" (which includes permutation)
        if mode == "permutation" or mode == "all":
            logger.info("Creating permuted think-aloud data for permutation control...")
            permuted_think_aloud_df = permute_think_aloud_by_choice(
                decisions_df, think_aloud_df, random_seed=42
            )
        else:
            permuted_think_aloud_df = None
            
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise
    
    # Get all participant IDs
    participant_ids = decisions_df["sub_id"].unique().tolist()
    logger.info(f"Found {len(participant_ids)} participants")
    
    # Prepare participants data
    participants_data = {}
    participants_data_permuted = {}
    participants_data_action_removed = {}
    logger.info("Loading participant data...")
    for participant_id in tqdm(participant_ids, desc="Loading participants"):
        try:
            # Load normal data
            participants_data[participant_id] = get_participant_data(
                decisions_df,
                think_aloud_df,
                participant_id,
                "llm"
            )
            
            # Load permuted data if needed
            if permuted_think_aloud_df is not None:
                participants_data_permuted[participant_id] = get_participant_data(
                    decisions_df,
                    permuted_think_aloud_df,
                    participant_id,
                    "llm"
                )

            # Load action-removed data if needed
            if think_aloud_df_action_removed is not None:
                participants_data_action_removed[participant_id] = get_participant_data(
                    decisions_df,
                    think_aloud_df_action_removed,
                    participant_id,
                    "llm"
                )
        except Exception as e:
            logger.error(f"Failed to load data for participant {participant_id}: {e}")
            continue
    
    logger.info(f"Successfully loaded data for {len(participants_data)} participants")
    
    # Initialize model with optimized settings
    logger.info(f"Initializing {model_type} model: {model_name}...")
    
    # Check GPU availability for HuggingFace models
    if model_type == "huggingface":
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                             for i in range(num_gpus)) / 1024**3
            logger.info(f"Available: {num_gpus} GPUs, Total Memory: {total_memory:.1f}GB")
        else:
            logger.warning("CUDA not available! This will be very slow.")
    
    # Determine which modes to run
    if mode == "all":
        modes_to_run = ["base", "cot", "human", "permutation"]
    else:
        modes_to_run = [mode]
    
    all_results = []
    
    # Run each mode
    for current_mode in modes_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running mode: {current_mode.upper()}")
        logger.info(f"{'='*50}")
        
        # Use appropriate data based on mode
        if current_mode == "permutation":
            current_participants_data = participants_data_permuted
            actual_mode = "human"  # Use human mode logic but with permuted data
        elif current_mode == "action_removal":
            current_participants_data = participants_data_action_removed
            actual_mode = "human"  # Same prompting logic as human mode, but with claims removed
        else:
            current_participants_data = participants_data
            actual_mode = current_mode
        
        # Force HuggingFace fallback if requested
        if force_hf_fallback and model_type == "huggingface":
            logger.info("🔒 Forcing HuggingFace fallback mode...")
            import models.llm_model as llm_module
            original_vllm = llm_module.VLLM_AVAILABLE
            llm_module.VLLM_AVAILABLE = False
            
            try:
                model = LLMModel(
                    model_name=model_name,
                    model_type=model_type,
                    mode=actual_mode,
                    num_workers=num_workers,
                    batch_size_per_device=batch_size_per_device,
                    gpu_memory_utilization=0.80,  # Conservative for stability
                    api_key=OPENAI_API_KEY if model_type == "openai" else None,
                    tensor_parallel_size=tensor_parallel_size
                )
            finally:
                llm_module.VLLM_AVAILABLE = original_vllm
        else:
            # Normal initialization (tries vLLM first, fallback to HF)
            model = LLMModel(
                model_name=model_name,
                model_type=model_type,
                mode=actual_mode,
                num_workers=num_workers,
                batch_size_per_device=batch_size_per_device,
                gpu_memory_utilization=0.85,  # Optimized for H200
                api_key=OPENAI_API_KEY if model_type == "openai" else None,
                tensor_parallel_size=tensor_parallel_size
            )
        
        # Log which backend is being used
        if hasattr(model, 'vllm_model') and model.vllm_model is not None:
            logger.info("✅ Using vLLM backend")
        elif hasattr(model, 'model') and model.model is not None:
            logger.info("✅ Using HuggingFace backend")
        else:
            logger.info("✅ Using OpenAI API backend")
        
        # Test probability extraction if using vLLM
        if hasattr(model, 'vllm_model') and model.vllm_model is not None:
            logger.info("🧪 Testing probability extraction...")
            try:
                # Create a simple test prompt and response
                test_prompt = "You need to choose between two options. I will choose Option "
                test_response = "After considering the options carefully, I will choose Option "
                
                # Test the probability extraction
                prob_A, prob_B = model._get_choice_probabilities_vllm(test_prompt, test_response, None, actual_mode)
                logger.info(f"✅ Probability extraction test: A={prob_A:.4f}, B={prob_B:.4f}")
                
                # Check if probabilities are not default values
                if prob_A == 0.5 and prob_B == 0.5:
                    logger.warning("⚠️  Probability extraction returned default values - check debug logs")
                else:
                    logger.info("✅ Probability extraction appears to be working correctly")
                    
            except Exception as e:
                logger.error(f"❌ Probability extraction test failed: {e}")
        
        # Check for existing checkpoints
        latest_checkpoint = model.find_latest_checkpoint(output_dir, data_size)
        if latest_checkpoint:
            existing_results, checkpoint_metadata = model.load_checkpoint(latest_checkpoint)
            logger.info(f"Found checkpoint with {len(existing_results)} existing results")
            logger.info("Starting fresh for this experiment.")
        
        # Run batch processing
        logger.info("Starting batch processing...")
        print(f"🔍 STARTING BATCH PROCESSING with {len(current_participants_data)} participants for mode: {current_mode}")
        
        try:
            results = model.predict_batch(
                participants_data=current_participants_data,
                participant_ids=list(current_participants_data.keys()),  # Only use successfully loaded participants
                data_size=data_size,
                output_dir=output_dir,
                save_results=False,  # We'll save manually with proper naming
                checkpoint_interval=checkpoint_interval,
                num_examples=0  # exp1 doesn't use examples
            )
            
            # Tag results with the actual mode (including permutation)
            for result in results:
                result['mode'] = current_mode
                if current_mode == "permutation":
                    result['is_permuted'] = True
                    result['permutation_seed'] = 42
                
            # Save results with mode-specific filename
            if results:
                model_name_safe = model.model_name.replace("/", "_").replace(":", "_")
                if current_mode == "permutation":
                    mode_suffix = "_permuted"
                else:
                    mode_suffix = f"_{current_mode}"
                output_file = os.path.join(
                    output_dir,
                    f"{model_name_safe}_{data_size}{mode_suffix}_results.json"
                )
                
                # Prepare metadata
                metadata = {
                    "mode": current_mode,
                    "is_permuted": current_mode == "permutation",
                    "permutation_seed": 42 if current_mode == "permutation" else None,
                    "action_removal": current_mode == "action_removal",
                    "superficial_claims_csv": superficial_claims_csv if current_mode == "action_removal" else None,
                    "participants_processed": list(current_participants_data.keys()),
                    "total_participants": len(current_participants_data),
                    "processing_method": "mode_specific_batch_processing"
                }
                
                model.save_results(
                    results,
                    output_file,
                    data_size,
                    metadata=metadata,
                    num_examples=0
                )
                
                logger.info(f"Results for {current_mode} mode saved to: {output_file}")
            
            all_results.extend(results)
            
        except Exception as e:
            logger.error(f"Batch processing failed for mode {current_mode}: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        
        logger.info(f"Batch processing completed for mode {current_mode}. Generated {len(results)} results.")
        
        # Analyze probability extraction results for this mode
        print(f"\n🔍 PROBABILITY EXTRACTION ANALYSIS for {current_mode.upper()}:")
        total_results = len(results)
        default_prob_count = 0
        meaningful_prob_count = 0
        
        for result in results[:10]:  # Check first 10 results as sample
            prob_A = result.get('probability_option_a', 0.5)
            prob_B = result.get('probability_option_b', 0.5)
            
            if abs(prob_A - 0.5) < 0.001 and abs(prob_B - 0.5) < 0.001:
                default_prob_count += 1
            else:
                meaningful_prob_count += 1
                print(f"🔍 Sample meaningful probability: A={prob_A:.6f}, B={prob_B:.6f} (mode: {result.get('mode', 'unknown')})")
        
        print(f"🔍 In first 10 results: {meaningful_prob_count} meaningful, {default_prob_count} default (0.5, 0.5)")
        
        if default_prob_count > meaningful_prob_count:
            print(f"🔍 ⚠️  WARNING: Most probabilities are default values - check debugging output above")
        else:
            print(f"🔍 ✅ Good: Most probabilities appear meaningful")
    
    logger.info(f"All modes completed successfully. Generated {len(all_results)} total results.")
    return all_results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run LLM prediction experiment (Optimized Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                      help="Name of the model to use")
    parser.add_argument("--model_type", type=str, default="huggingface",
                      choices=["openai", "huggingface"],
                      help="Type of model to use")
    parser.add_argument("--data_size", type=str, default="small",
                      choices=["small", "large", "all"],
                      help="Size of dataset to use")
    parser.add_argument("--mode", type=str, default="all",
                      choices=["base", "cot", "human", "permutation", "action_removal", "all"],
                      help="Experiment mode. 'permutation' uses human mode with think-aloud permuted by choice to test action claim hypothesis. "
                           "'action_removal' uses human mode with superficial action claims masked out from the think-aloud.")
    parser.add_argument("--num_workers", type=int, default=10,
                      help="Number of worker threads for OpenAI API calls")
    parser.add_argument("--batch_size_per_device", type=int, default=4,
                      help="Total batch size for processing")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                      help="Number of results before saving checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results/exp1_llm_prediction",
                      help="Directory to save results")
    parser.add_argument("--force_hf_fallback", action="store_true",
                      help="Force HuggingFace fallback instead of trying vLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                      help="Number of GPUs to use for tensor parallelism (auto-detect if not specified)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    parser.add_argument(
        "--superficial_claims_csv",
        type=str,
        default=None,
        help="Path to superficial-claims extraction CSV (from utils/superficial_action_extraction.py). "
             "Required when mode='action_removal'.",
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also set debug for specific loggers
        logging.getLogger('models.llm_model').setLevel(logging.DEBUG)
        print("🔍 DEBUG MODE ENABLED - Enhanced probability extraction debugging active")
    else:
        # Always show probability debugging for now until issue is resolved
        print("🔍 PROBABILITY DEBUGGING ENABLED - Will show detailed extraction info")
    
    # Print configuration
    logger.info("="*60)
    logger.info("LLM PREDICTION EXPERIMENT - FINAL VERSION")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Data Size: {args.data_size}")
    logger.info(f"Mode: {args.mode}")
    if args.mode == "permutation":
        logger.info("  PERMUTATION MODE: Think-aloud will be permuted among trials with same choice")
        logger.info("  This tests if LLMs use action claims vs. meaningful cognitive processes")
    if args.mode == "action_removal":
        logger.info("  ACTION_REMOVAL MODE: Superficial action claims will be masked out from think-aloud before prediction")
        logger.info(f"  Superficial claims CSV: {args.superficial_claims_csv}")
    logger.info(f"Batch Size: {args.batch_size_per_device}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size if args.tensor_parallel_size else 'auto-detect'}")
    logger.info(f"Force HF Fallback: {args.force_hf_fallback}")
    logger.info("="*60)
    
    # Run experiment with error handling
    try:
        results = run_llm_prediction_optimized(
            model_name=args.model_name,
            model_type=args.model_type,
            data_size=args.data_size,
            mode=args.mode,
            num_workers=args.num_workers,
            batch_size_per_device=args.batch_size_per_device,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=args.output_dir,
            force_hf_fallback=args.force_hf_fallback,
            tensor_parallel_size=args.tensor_parallel_size,
            superficial_claims_csv=args.superficial_claims_csv,
        )
        
        # Print completion summary
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total results: {len(results)}")
        
        # Count results by mode and dataset
        if args.data_size == "all":
            # Group by both mode and dataset if available
            dataset_mode_counts = {}
            for result in results:
                # Try to determine dataset from metadata or make assumption
                mode = result.get("mode", "unknown")
                dataset_mode_counts[mode] = dataset_mode_counts.get(mode, 0) + 1
            
            logger.info(f"Results processed for datasets: {args.data_size}")
            for mode, count in dataset_mode_counts.items():
                logger.info(f"  {mode.upper()}: {count} trials")
        else:
            # Count results by mode only
            mode_counts = {}
            for result in results:
                mode = result.get("mode", "unknown")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            logger.info(f"Results for {args.data_size} dataset:")
            for mode, count in mode_counts.items():
                logger.info(f"  {mode.upper()}: {count} trials")
                if mode == "permutation":
                    logger.info(f"    (Permuted think-aloud with seed 42)")
        
        # Calculate accuracy if possible
        total_with_choice = sum(1 for r in results if r.get("extracted_choice") is not None)
        if total_with_choice > 0:
            accuracy_rate = total_with_choice / len(results) * 100
            logger.info(f"Choice extraction success rate: {accuracy_rate:.1f}%")
        
        # Final probability extraction analysis
        print(f"\n🔍 FINAL PROBABILITY EXTRACTION SUMMARY:")
        total_default = sum(1 for r in results 
                           if abs(r.get('probability_option_a', 0.5) - 0.5) < 0.001 
                           and abs(r.get('probability_option_b', 0.5) - 0.5) < 0.001)
        total_meaningful = len(results) - total_default
        
        print(f"🔍 Total results: {len(results)}")
        print(f"🔍 Meaningful probabilities: {total_meaningful} ({total_meaningful/len(results)*100:.1f}%)")
        print(f"🔍 Default probabilities (0.5): {total_default} ({total_default/len(results)*100:.1f}%)")
        
        if total_default > total_meaningful:
            print(f"🔍 ❌ ISSUE: {total_default/len(results)*100:.1f}% of probabilities are default values")
            print(f"🔍 This suggests the probability extraction is not working correctly")
        else:
            print(f"🔍 ✅ SUCCESS: {total_meaningful/len(results)*100:.1f}% of probabilities are meaningful")
        
        # Special analysis for permutation mode
        permutation_results = [r for r in results if r.get('mode') == 'permutation']
        human_results = [r for r in results if r.get('mode') == 'human']
        
        if permutation_results and human_results:
            print(f"\n🔍 PERMUTATION ANALYSIS:")
            print(f"🔍 Human mode trials: {len(human_results)}")
            print(f"🔍 Permutation mode trials: {len(permutation_results)}")
            
            # Calculate accuracy for both
            human_accuracy = sum(1 for r in human_results if r.get('extracted_choice') == r.get('actual_choice')) / len(human_results)
            perm_accuracy = sum(1 for r in permutation_results if r.get('extracted_choice') == r.get('actual_choice')) / len(permutation_results)
            
            print(f"🔍 Human mode accuracy: {human_accuracy:.3f}")
            print(f"🔍 Permutation mode accuracy: {perm_accuracy:.3f}")
            print(f"🔍 Accuracy difference: {human_accuracy - perm_accuracy:.3f}")
            
            if abs(human_accuracy - perm_accuracy) < 0.05:
                print(f"🔍 ⚠️  WARNING: Small accuracy difference suggests LLM may be using action claims")
            else:
                print(f"🔍 ✅ GOOD: Significant accuracy difference suggests LLM uses meaningful cognitive processes")
        
        logger.info("Experiment completed successfully! 🎉")
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 