"""
Experiment: LLM-based extraction of superficial action claims from think-aloud
transcripts, followed by a verification experiment.

Stage 1 (extraction):
    - Use an open-source LLM (e.g., LLaMA-3.1-70B) to scan each think-aloud
      transcript and extract any *superficial action claims* – explicit phrases
      like "I will choose A" that directly reveal the final choice without
      requiring contextual reasoning.

Stage 2 (verification):
    - Feed the *same* LLM the decision context with all numeric information
      masked (probabilities, outcomes) and, for the think-aloud part, provide
      only the extracted superficial claims.
    - Measure how accurately the model can predict choices using only these
      superficial claims plus non-numeric context.

This script is designed to work with the existing LLMModel infrastructure and
data loading utilities in this repository.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import pandas as pd

from models.llm_model import LLMModel
from utils.data_processing import (
    load_datasets,
    preprocess_think_aloud,
    question_prompt_generate,
)
from utils.superficial_action_extraction import (
    build_extraction_prompt,
    build_extraction_messages,
    extract_superficial_claims_for_dataframe,
    mask_numeric_text,
    should_run_verification,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _sanitize_model_name(model_name: str) -> str:
    """
    Create a filesystem-friendly tag from a HuggingFace model name.

    Examples:
        "meta-llama/Meta-Llama-3.1-70B-Instruct"
        -> "meta-llama_Meta-Llama-3.1-70B-Instruct"
    """
    return model_name.replace("/", "_").replace(" ", "_")


def build_structural_masked_question_context(row_dict: dict) -> str:
    """
    Build a question context where each numeric quantity is replaced by a
    structural placeholder like pa1, va1, pb1, vb1 instead of the actual
    probability/value numbers.

    This mirrors the structure of `question_prompt_generate`, but with
    symbolic tokens so that no numeric information is available to the model
    in the verification condition.
    """
    fixed_start = "Which option do you prefer? "
    option_A = "Option A: "
    option_B = "Option B: "

    p1 = list(row_dict["p1"])
    v1 = list(row_dict["v1"])
    p2 = list(row_dict["p2"])
    v2 = list(row_dict["v2"])

    for i in range(len(p1)):
        idx = i + 1
        option_A += f"va{idx} dollars with pa{idx}% chance"
        if i < len(p1) - 1:
            option_A += ", "
        else:
            option_A += ". "

    for i in range(len(p2)):
        idx = i + 1
        option_B += f"vb{idx} dollars with pb{idx}% chance"
        if i < len(p2) - 1:
            option_B += ", "
        else:
            option_B += "."

    return fixed_start + option_A + " " + option_B


def run_extraction(
    model_name: str,
    data_size: str = "small",
    batch_size_per_device: int = 4,
    model_type: str = "huggingface",
    openai_api_key: str | None = None,
    openai_num_workers: int = 10,
    openai_max_retries: int = 6,
) -> pd.DataFrame:
    """
    Run Stage 1: extract superficial action claims for all think-aloud transcripts.

    Returns:
        think_aloud_df with additional superficial-claim columns.
    """
    logger.info(f"Loading datasets for data_size='{data_size}'...")
    decisions_df, think_aloud_df = load_datasets(data_size)
    think_aloud_df = preprocess_think_aloud(think_aloud_df)

    logger.info(
        f"Loaded {len(think_aloud_df)} think-aloud transcripts for extraction."
    )

    logger.info(
        f"Initializing model '{model_name}' (type={model_type}) for extraction "
        "(deterministic decoding: temperature=0.0 where supported)..."
    )
    llm = LLMModel(
        model_name=model_name,
        model_type=model_type,
        mode="base",
        batch_size_per_device=batch_size_per_device,
        api_key=openai_api_key,
    )

    # Build prompts that include BOTH question context and think-aloud,
    # mirroring the structure used for prediction prompts.
    # First, index decisions by (sub_id, problem_id) so we can recover the
    # corresponding gamble description for each think-aloud row.
    decisions_indexed = decisions_df.set_index(["sub_id", "problem_id"])

    prompts: List = []
    for _, ta_row in think_aloud_df.iterrows():
        key = (ta_row["sub_id"], ta_row["problem_id"])
        if key in decisions_indexed.index:
            d_row = decisions_indexed.loc[key]
            row_dict = {
                "p1": d_row["p1"],
                "v1": d_row["v1"],
                "p2": d_row["p2"],
                "v2": d_row["v2"],
            }
            question_context = question_prompt_generate(row_dict)
        else:
            # Fallback: no matching decision row; provide empty context.
            question_context = ""

        transcript = str(ta_row["processed_text"])
        if model_type == "openai":
            prompts.append(build_extraction_messages(question_context, transcript))
        else:
            prompts.append(build_extraction_prompt(question_context, transcript))

    logger.info("Running LLM extraction over all transcripts (deterministic decoding)...")
    if model_type == "openai":
        # IMPORTANT: keep project-wide OpenAI defaults unchanged; only extraction is deterministic.
        max_tokens = llm._get_max_tokens("base")

        def _call_one(idx_and_msg: Tuple[int, list]) -> Tuple[int, str]:
            idx, msg = idx_and_msg
            for attempt in range(openai_max_retries):
                try:
                    # Some newer OpenAI chat models use `max_completion_tokens`
                    # instead of `max_tokens`.
                    try:
                        resp = llm.client.chat.completions.create(
                            model=llm.model_name,
                            messages=msg,
                            max_completion_tokens=max_tokens,
                            temperature=0.0,
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0,
                        )
                    except Exception as inner_e:
                        if "max_completion_tokens" in str(inner_e) or "Unsupported parameter" in str(inner_e):
                            resp = llm.client.chat.completions.create(
                                model=llm.model_name,
                                messages=msg,
                                max_tokens=max_tokens,
                                temperature=0.0,
                                top_p=1.0,
                                frequency_penalty=0.0,
                                presence_penalty=0.0,
                            )
                        else:
                            raise
                    text = ""
                    if resp and getattr(resp, "choices", None):
                        choice0 = resp.choices[0]
                        if choice0 and getattr(choice0, "message", None) and choice0.message:
                            text = (choice0.message.content or "").strip()
                    return idx, text
                except Exception as e:
                    # Simple deterministic backoff (no jitter) for rate limits / transient failures
                    wait_s = min(60.0, 1.5 ** attempt)
                    logger.warning(f"OpenAI error at {idx} (attempt {attempt+1}/{openai_max_retries}): {e} | sleeping {wait_s:.1f}s")
                    time.sleep(wait_s)
            logger.error(f"OpenAI extraction failed at {idx} after {openai_max_retries} retries")
            return idx, ""

        responses = [""] * len(prompts)
        n_workers = max(1, int(openai_num_workers))
        logger.info(f"OpenAI extraction: running with openai_num_workers={n_workers}")

        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_call_one, (i, prompts[i])) for i in range(len(prompts))]
            for fut in as_completed(futures):
                idx, text = fut.result()
                responses[idx] = text
                done += 1
                if done % 250 == 0 or done == len(prompts):
                    logger.info(f"Processed {done} / {len(prompts)} transcripts")
    else:
        # Deterministic generation: temperature=0.0, greedy, no sampling.
        # We bypass the generic helper and call the underlying vLLM / HF model
        # with fixed decoding parameters.
        import torch  # local import to avoid unused in other paths
        from vllm import SamplingParams  # type: ignore[import-untyped]

        max_tokens = llm._get_max_tokens("base")  # small JSON responses
        responses: List[str] = []

        for start in range(0, len(prompts), batch_size_per_device):
            end = min(start + batch_size_per_device, len(prompts))
            batch_prompts = prompts[start:end]
            batch_responses: List[str] = []

            # vLLM path (if available)
            if hasattr(llm, "vllm_model") and llm.vllm_model is not None:
                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    stop=None,
                    skip_special_tokens=True,
                    logprobs=None,
                )
                outputs = llm.vllm_model.generate(batch_prompts, sampling_params)
                for out in outputs:
                    if out.outputs:
                        text = (out.outputs[0].text or "").strip()
                    else:
                        text = ""
                    batch_responses.append(text)
            else:
                # Standard HuggingFace path
                inputs = llm.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = llm.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        do_sample=False,
                        pad_token_id=llm.tokenizer.pad_token_id,
                        eos_token_id=llm.tokenizer.eos_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                for seq in outputs:
                    new_tokens = seq[input_len:]
                    text = llm.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    batch_responses.append(text.strip())

            responses.extend(batch_responses)
            logger.info(f"Processed {end} / {len(prompts)} transcripts")

    logger.info("Parsing extraction outputs and attaching to dataframe...")
    ta_with_claims = extract_superficial_claims_for_dataframe(
        think_aloud_df, responses
    )
    return ta_with_claims


def run_verification(
    model_name: str,
    decisions_df: pd.DataFrame,
    ta_with_claims: pd.DataFrame,
    data_size: str = "small",
    mode: str = "human",
    batch_size_per_device: int = 4,
    model_type: str = "huggingface",
    openai_api_key: str | None = None,
) -> pd.DataFrame:
    """
    Run Stage 2: verification using masked numeric context + superficial claims.

    For each trial where a superficial claim was found, we:
      - Build the usual question prompt from probabilities and outcomes.
      - Mask all numeric information (probabilities, amounts) in that prompt.
      - Use only the extracted superficial-only text as the think-aloud input.
      - Ask the LLM (in 'human' mode) to predict the choice.
    """
    logger.info(f"Initializing model '{model_name}' (type={model_type}) for verification...")
    llm = LLMModel(
        model_name=model_name,
        model_type=model_type,
        mode=mode,
        batch_size_per_device=batch_size_per_device,
        api_key=openai_api_key,
    )

    # Merge decisions with think-aloud (including superficial-claim columns)
    merged = pd.merge(
        decisions_df,
        ta_with_claims[
            ["sub_id", "problem_id", "has_superficial_claim", "superficial_only_text"]
        ],
        on=["sub_id", "problem_id"],
        how="left",
    )

    # Filter to trials that actually have at least one superficial claim
    valid_mask = merged["has_superficial_claim"].fillna(False)
    eval_df = merged[valid_mask].reset_index(drop=True)
    logger.info(
        f"Verification will run on {len(eval_df)} trials "
        f"with at least one superficial action claim."
    )

    # Prepare prompts and labels
    gold_choices = eval_df["choice"].to_numpy()
    predictions = []

    for idx, row in eval_df.iterrows():
        # Build question context using structural placeholders (pa1, va1, ...)
        row_dict = {
            "p1": row["p1"],
            "v1": row["v1"],
            "p2": row["p2"],
            "v2": row["v2"],
        }
        masked_context = build_structural_masked_question_context(row_dict)

        superficial_text = str(row["superficial_only_text"] or "").strip()

        # Provide dummy probability/outcome vectors – they are not used when
        # question_context is explicitly supplied to _format_prompt.
        dummy_probs = [0.0, 0.0]
        dummy_outcomes = [0.0, 0.0]

        result = llm.predict(
            probabilities=dummy_probs,
            outcomes=dummy_outcomes,
            think_aloud=superficial_text,
            question_context=masked_context,
            mode=mode,
        )

        pred_choice = result.get("choice", None)
        predictions.append(pred_choice)

        logger.info(
            f"Trial {idx+1}/{len(eval_df)}: gold={gold_choices[idx]}, "
            f"pred={pred_choice}, has_superficial={bool(superficial_text)}"
        )

    predictions_arr = np.array(predictions, dtype=float)
    accuracy = np.mean(predictions_arr == gold_choices)
    logger.info(
        f"Verification accuracy using ONLY superficial claims "
        f"(with numeric-masked context): {accuracy:.4f}"
    )

    eval_df = eval_df.copy()
    eval_df["pred_choice_superficial_only"] = predictions_arr
    eval_df["correct_superficial_only"] = (
        eval_df["pred_choice_superficial_only"] == eval_df["choice"]
    )

    return eval_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM-based extraction of superficial action claims from "
            "think-aloud transcripts and a verification experiment."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="HuggingFace model name for extraction and verification.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="huggingface",
        choices=["huggingface", "openai"],
        help="Backend to use for extraction/verification.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var). Only used when --model_type openai.",
    )
    parser.add_argument(
        "--openai_num_workers",
        type=int,
        default=10,
        help="Number of concurrent OpenAI requests during extraction (only for --model_type openai).",
    )
    parser.add_argument(
        "--openai_max_retries",
        type=int,
        default=6,
        help="Max retries per request for OpenAI extraction (only for --model_type openai).",
    )
    parser.add_argument(
        "--data_size",
        type=str,
        default="small",
        choices=["small", "large", "all"],
        help="Dataset size to use.",
    )
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=4,
        help="Total batch size per device for HuggingFace/vLLM generation.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="results/superficial_claims",
        help="Prefix for saving intermediate and final CSVs (model name and data size are appended automatically).",
    )
    args = parser.parse_args()

    # Stage 1: extraction
    ta_with_claims = run_extraction(
        model_name=args.model_name,
        data_size=args.data_size,
        batch_size_per_device=args.batch_size_per_device,
        model_type=args.model_type,
        openai_api_key=(args.openai_api_key or os.getenv("OPENAI_API_KEY")),
        openai_num_workers=args.openai_num_workers,
        openai_max_retries=args.openai_max_retries,
    )

    model_tag = _sanitize_model_name(args.model_name)

    # Save extraction results
    extraction_out_path = (
        f"{args.output_prefix}_extraction_{args.data_size}_{model_tag}.csv"
    )
    ta_with_claims.to_csv(extraction_out_path, index=False)
    logger.info(f"Saved superficial-claim annotations to {extraction_out_path}")

    # Stage 2: verification
    verification_out_path = (
        f"{args.output_prefix}_verification_{args.data_size}_{model_tag}.csv"
    )

    if not should_run_verification(args.model_type):
        # Cancel Stage 2 verification for OpenAI runs to avoid extra OpenAI
        # calls / potential API-side restrictions. We still write a CSV with
        # expected columns so downstream scripts can load it safely.
        empty_cols = [
            "sub_id",
            "problem_id",
            "has_superficial_claim",
            "superficial_only_text",
            "choice",
            "pred_choice_superficial_only",
            "correct_superficial_only",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(verification_out_path, index=False)
        logger.info(
            "Skipped Stage 2 verification for model_type='openai'. "
            f"Saved empty verification CSV to {verification_out_path}"
        )
    else:
        decisions_df, _ = load_datasets(args.data_size)
        eval_df = run_verification(
            model_name=args.model_name,
            decisions_df=decisions_df,
            ta_with_claims=ta_with_claims,
            data_size=args.data_size,
            mode="human",
            batch_size_per_device=args.batch_size_per_device,
            model_type=args.model_type,
            openai_api_key=(args.openai_api_key or os.getenv("OPENAI_API_KEY")),
        )

        eval_df.to_csv(verification_out_path, index=False)
        logger.info(
            f"Saved verification results (superficial-only) to {verification_out_path}"
        )


if __name__ == "__main__":
    main()


