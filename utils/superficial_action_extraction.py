"""
Utilities for extracting superficial action claims from think-aloud transcripts
and running LLM-based verification experiments.

The key idea is to identify explicit, surface-level statements like
\"I will choose A\" or \"I will go for the left option\" that reveal the
participant's choice without requiring any additional context or reasoning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SuperficialClaim:
    """Container for a single superficial action claim."""

    text: str
    normalized_choice: str  # e.g. "A", "B", "left", "right", "first", "second", "other"


EXTRACTION_SYSTEM_INSTRUCTIONS = """
You are analyzing a think-aloud transcript from a risky decision-making task.

Your goal is to extract ONLY **choice-committing statements**.

## Definition: Choice-Committing Statement

A choice-committing statement is a span where the speaker indicates which option
they are choosing, planning to choose, about to choose, or leaning toward choosing.

This includes:
- direct commitments: "I choose...", "I pick...", "I will go with..."
- soft or hedged commitments: "I think I'll go with...", "I'd probably take...",
  "I'm leaning toward...", "I guess I'll take..."

A valid span must indicate a choice tendency or commitment, not just describe an option.

The span may rely on local context such as:
- "this one"
- "that one"
- "the first one"
- "the second one"
- "the left one"
- "the right one"

These ARE valid if the speaker is clearly committing to that option.

## Minimum Completeness Requirement

A valid extracted span must be a minimally complete phrase that can stand alone
and clearly indicate WHAT is being chosen.

Do NOT extract incomplete or unclear fragments such as:
- "I guess"
- "This one"
- "I will go with"
- "I am doing"
- "take the"
- "going to go with dollars"

If the phrase does not clearly indicate the chosen option, do NOT extract it.

## NOT Choice-Committing Statements

Do NOT extract:

1. Option descriptions
- "I would lose 6000 with a 0.1 percent chance"
- "There is a 0.2 percent chance of losing 3000"
- "This gives 1000 with 50 percent probability"

2. Problem rephrasings
- "One option has a 0.1 probability"
- "The second option gives 3000"

3. Reasoning without commitment
- "0.1 is lower than 0.2"
- "This seems safer"
- "This one is better because..."

4. Pure evaluation without choosing
- "this makes more sense"
- "that seems better"
- "this is the best option"

These are NOT sufficient unless the speaker also indicates choosing it.

Mentioning an option is NOT enough.
The speaker must indicate choosing it, intending to choose it, or leaning toward it.

## Weak Commitment Rule

Weak or informal commitments are VALID only if they clearly refer to a specific option.

Valid:
- "probably the second one"
- "I guess I'll take the right one"
- "I'd go with the first one"

Invalid:
- "probably"
- "I guess"

## Mixed Sentences

If a sentence contains both commitment and reasoning, extract only the commitment part.

Example:
"I think I will go with the first option because it is safer"
Extract:
"I think I will go with the first option"

## Additional Guidance

- Extract the shortest continuous span that expresses the commitment
- Keep hedges if they are part of the commitment, such as:
  "I think", "probably", "I guess", "I'd rather"
- Do NOT include extra explanation if it is separable
- Choice statements often appear near the end, but not always

## Output Format

Return a JSON object:

{
  "claims": [
    {"text": "...", "normalized_choice": "A"},
    {"text": "...", "normalized_choice": "B"}
  ]
}

Use labels like:
"A", "B", "left", "right", "first", "second", or "other"

If no valid claims exist, return:
{"claims": []}

## Examples

Think-aloud:
"I think I will go with the second option because it has lower risk"
Output:
{"claims": [{"text": "I think I will go with the second option", "normalized_choice": "second"}]}

Think-aloud:
"I would lose 6000 with a 0.1 percent chance"
Output:
{"claims": []}

Think-aloud:
"Okay, I'm going with that"
Output:
{"claims": [{"text": "I'm going with that", "normalized_choice": "other"}]}

Think-aloud:
"Probably the one on the right"
Output:
{"claims": [{"text": "Probably the one on the right", "normalized_choice": "right"}]}

Think-aloud:
"The second option gives 3000 with 0.2 percent chance"
Output:
{"claims": []}

Think-aloud:
"I'd take the left one"
Output:
{"claims": [{"text": "I'd take the left one", "normalized_choice": "left"}]}
""".strip()


def build_extraction_prompt(question_context: str, transcript: str) -> str:
    """
    Build a single-string prompt for an open-source chat-style LLM.

    Both the decision problem context and the think-aloud transcript are
    provided explicitly, mirroring the structure used when asking the model
    to predict choices ("question context: ...; Think-Aloud: ...").
    """
    question_context = (question_context or "").strip()
    transcript = (transcript or "").strip()

    q_block = question_context if question_context else "<<<EMPTY_CONTEXT>>>"
    ta_block = transcript if transcript else "<<<EMPTY_TRANSCRIPT>>>"

    return (
        EXTRACTION_SYSTEM_INSTRUCTIONS
        + "\n\nNow process the following:\n\n### Decision problem context:\n<<<\n"
        + q_block
        + "\n>>>\n\n### Think-aloud transcript:\n<<<\n"
        + ta_block
        + "\n>>>\n\n### Output:\n"
    )


def build_extraction_messages(question_context: str, transcript: str) -> List[Dict[str, str]]:
    """
    Build OpenAI-chat style messages for extraction.

    We pass the extraction instructions as a system message, and the
    (context, transcript) payload as the user message.
    """
    question_context = (question_context or "").strip()
    transcript = (transcript or "").strip()

    q_block = question_context if question_context else "<<<EMPTY_CONTEXT>>>"
    ta_block = transcript if transcript else "<<<EMPTY_TRANSCRIPT>>>"

    user_content = (
        "Now process the following:\n\n### Decision problem context:\n<<<\n"
        + q_block
        + "\n>>>\n\n### Think-aloud transcript:\n<<<\n"
        + ta_block
        + "\n>>>\n\n### Output:\n"
    )

    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_content},
    ]


def _extract_json_blob(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object substring from model output."""
    if not text:
        return None

    # Find the first '{' and last '}' and attempt to parse the enclosed text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_extraction_response(raw_response: str) -> List[SuperficialClaim]:
    """Parse model output into a list of SuperficialClaim objects."""
    raw_response = (raw_response or "").strip()
    if not raw_response:
        return []

    json_blob = _extract_json_blob(raw_response)
    if json_blob is None:
        return []

    try:
        obj = json.loads(json_blob)
    except json.JSONDecodeError:
        return []

    # Allow the model to return null / empty / non-dict top-level values.
    if not isinstance(obj, dict):
        return []

    claims_raw = obj.get("claims", [])
    # If "claims" is missing, null, or not a list, treat as no claims.
    if not isinstance(claims_raw, list):
        return []
    claims: List[SuperficialClaim] = []
    for item in claims_raw:
        try:
            text = str(item.get("text", "")).strip()
            norm = str(item.get("normalized_choice", "")).strip()
        except Exception:
            continue
        if not text:
            continue
        if not norm:
            norm = "other"
        claims.append(SuperficialClaim(text=text, normalized_choice=norm))
    return claims


NUMERIC_PATTERN = re.compile(
    r"""
    (?:
        [+-]?\d+(?:\.\d+)?      # integer or decimal
        (?:\s*%|\s*percent)?    # optional percent sign/word
    )
    """,
    re.VERBOSE,
)


def mask_numeric_text(text: str, mask_token: str = "<NUM>") -> str:
    """Mask numeric information (probabilities, amounts, etc.) in free text."""
    if not isinstance(text, str):
        return ""
    return NUMERIC_PATTERN.sub(mask_token, text)


def build_superficial_only_text(claims: List[SuperficialClaim]) -> str:
    """Concatenate superficial claim texts into a single snippet."""
    if not claims:
        return ""
    return " ".join(c.text.strip() for c in claims if c.text.strip())


def extract_superficial_claims_for_dataframe(
    think_aloud_df: pd.DataFrame,
    response_texts: List[str],
    transcript_col: str = "think_aloud",
) -> pd.DataFrame:
    """
    Attach superficial-claim annotations to a think-aloud dataframe.

    Args:
        think_aloud_df: DataFrame with a column containing transcripts.
        response_texts: Raw LLM outputs, aligned 1:1 with dataframe rows.
        transcript_col: Name of the transcript column.
    """
    if len(think_aloud_df) != len(response_texts):
        raise ValueError(
            f"Mismatched lengths: dataframe has {len(think_aloud_df)} rows, "
            f"but got {len(response_texts)} LLM responses."
        )

    all_claims: List[List[SuperficialClaim]] = []
    for raw in response_texts:
        claims = parse_extraction_response(raw)
        all_claims.append(claims)

    # Construct new columns
    has_claim = [len(c_list) > 0 for c_list in all_claims]
    superficial_text = [build_superficial_only_text(c_list) for c_list in all_claims]
    claims_json = [
        json.dumps(
            [
                {"text": c.text, "normalized_choice": c.normalized_choice}
                for c in c_list
            ],
            ensure_ascii=False,
        )
        for c_list in all_claims
    ]

    out_df = think_aloud_df.copy()
    out_df["has_superficial_claim"] = np.array(has_claim, dtype=bool)
    out_df["superficial_only_text"] = superficial_text
    out_df["superficial_claims_json"] = claims_json
    return out_df


def should_run_verification(model_type: str) -> bool:
    """
    Return True if Stage 2 verification should be executed.

    For OpenAI-backed extraction runs, we typically skip verification to
    avoid additional OpenAI calls and API-side restrictions.
    """
    return (model_type or "").strip().lower() != "openai"


