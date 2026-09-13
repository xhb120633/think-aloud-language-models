"""
Prompt templates for LLM experiments.
"""

# Prompt templates for LLM simulation of risky decision-making tasks

# === Prompt dictionary for HuggingFace models (completion style) ===
HF_PROMPTS = {
    "base": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "Read the decision scenario and make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "{question_context}"
        },
        {
            "role": "assistant",
            "content": "I will choose Option"
        }
    ],
    "cot": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "Read the decision scenario and think through your decision process step by step. "
                "When you have finished thinking, make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "{question_context}"
        },
        {
            "role": "assistant",
            "content": "Let me think step by step:\n"
        }
    ],
    "human": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "Read the decision scenario and think through your decision process step by step. "
                "When you have finished thinking, make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "{question_context}"
        },
        {
            "role": "assistant",
            "content": "Let me think step by step:\n{think_aloud}\n\nI will choose Option"
        }
    ],
    # New in-context learning templates
    "base_within_individual": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "You will be shown some examples of your previous {example_content_type} and then asked to make a new decision. "
                "Read the decision scenario and make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "Here are some examples of your previous {example_content_type}:\n\n{examples}\n\nNow, here is your new decision scenario:\n\n{question_context}"
        },
        {
            "role": "assistant",
            "content": "I will choose Option"
        }
    ],
    "cot_within_individual": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "You will be shown some examples of your previous {example_content_type} and then asked to make a new decision. "
                "Read the decision scenario and think through your decision process step by step. "
                "When you have finished thinking, make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "Here are some examples of your previous {example_content_type}:\n\n{examples}\n\nNow, here is your new decision scenario:\n\n{question_context}"
        },
        {
            "role": "assistant",
            "content": "Let me think step by step:\n"
        }
    ],
    "base_within_context": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "You will be shown a decision scenario, followed by examples of other participants' {example_content_type} in the same scenario. "
                "Read the decision scenario and make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "Here is your decision scenario:\n\n{question_context}\n\nHere are some examples of other participants' {example_content_type} in similar scenarios:\n\n{examples}"
        },
        {
            "role": "assistant",
            "content": "I will choose Option"
        }
    ],
    "cot_within_context": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "You will be shown a decision scenario, followed by examples of other participants' {example_content_type} in the same scenario. "
                "Read the decision scenario and think through your decision process step by step. "
                "When you have finished thinking, make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "Here is your decision scenario:\n\n{question_context}\n\nHere are some examples of other participants' {example_content_type} in the same scenario:\n\n{examples}"
        },
        {
            "role": "assistant",
            "content": "Let me think step by step:\n"
        }
    ]
}

# === Prompt dictionary for OpenAI models (chat style) ===
OPENAI_PROMPTS = {
    "base": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "Read the decision scenario and make your choice. "
                "You must respond with ONLY one of these two exact phrases: "
                "'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "{question_context}"
        }
    ],
    "cot": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "Read the decision scenario and think through your decision process step by step. "
                "When you have finished thinking, make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "{question_context}\n\nLet me think step by step:"
        }
    ],
    "human": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "Read the decision scenario. You have indicated your reasoning process before. "
                "Based on your own reasoning (even if it is not informative), directlymake your final choice. "
                "You must respond with ONLY one of these two exact phrases: "
                "'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "{question_context}\n\n Let me think step by step:\n{think_aloud}\n\nMake your choice:"
        }
    ],
    # New in-context learning templates
    "base_within_individual": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "You will be shown some examples of your previous {example_content_type} and then asked to make a new decision. "
                "You must respond with ONLY one of these two exact phrases: "
                "'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "Here are some examples of your previous {example_content_type}:\n\n{examples}\n\nNow, here is your new decision scenario:\n\n{question_context}"
        }
    ],
    "cot_within_individual": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "You will be shown some examples of your previous {example_content_type} and then asked to make a new decision. "
                "Read the decision scenario and think through your decision process step by step. "
                "When you have finished thinking, make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "Here are some examples of your previous {example_content_type}:\n\n{examples}\n\nNow, here is your new decision scenario:\n\n{question_context}\n\nLet me think step by step:"
        }
    ],
    "base_within_context": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "You will be shown a decision scenario, followed by examples of other participants' {example_content_type} in the same scenario. "
                "You must respond with ONLY one of these two exact phrases: "
                "'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "Here is your decision scenario:\n\n{question_context}\n\nHere are some examples of other participants' {example_content_type} in similar scenarios:\n\n{examples}"
        }
    ],
    "cot_within_context": [
        {
            "role": "system",
            "content": (
                "You are a human participant in a risky decision-making task. "
                "You will be shown a decision scenario, followed by examples of other participants' {example_content_type} in the same scenario. "
                "Read the decision scenario and think through your decision process step by step. "
                "When you have finished thinking, make your choice. "
                "Your response must end with: 'I will choose Option A.' or 'I will choose Option B.'"
            )
        },
        {
            "role": "user",
            "content": "Here is your decision scenario:\n\n{question_context}\n\nHere are some examples of other participants' {example_content_type} in the same scenario:\n\n{examples}\n\nLet me think step by step:"
        }
    ]
}

# For backward compatibility
LLM_PROMPTS = HF_PROMPTS


