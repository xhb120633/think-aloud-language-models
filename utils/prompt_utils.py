"""
Prompt utilities for risky decision-making experiments.
"""

import numpy as np

def question_prompt_generate(row):
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

def build_llm_prompt(question_prompt, think_aloud=None, mode="base"):
    """
    Build the full LLM prompt for a given trial.
    Args:
        question_prompt: The question description string
        think_aloud: The think-aloud string (if any)
        mode: Prompting mode ("base", "cot", "human")
    Returns:
        The full prompt string
    """
    if mode == "human" and think_aloud:
        return f'The question context is "{question_prompt}" The think aloud response is "{think_aloud}". Please predict which option the participant will choose: Option A or Option B.'
    elif mode == "cot":
        return f'{question_prompt}\nPlease think through your decision process step by step, then make your choice: Option A or Option B.'
    else:
        return f'{question_prompt}\nPlease make your choice: Option A or Option B.' 