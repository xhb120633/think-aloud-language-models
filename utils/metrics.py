"""
Evaluation metrics for the think-aloud analysis project.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute prediction accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)

def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return precision, recall, f1

def compute_sentence_contribution(
    model,
    text: str,
    sentences: List[str]
) -> List[float]:
    """
    Compute contribution of each sentence to the final prediction.
    
    Args:
        model: Trained model
        text: Full text
        sentences: List of sentences
        
    Returns:
        List of contribution scores
    """
    # Get full text prediction
    full_pred = model.predict(text)
    
    # Get predictions with each sentence removed
    contributions = []
    for sentence in sentences:
        # Remove sentence from text
        modified_text = text.replace(sentence, '')
        
        # Get prediction
        modified_pred = model.predict(modified_text)
        
        # Compute contribution as difference in prediction
        contribution = abs(full_pred - modified_pred)
        contributions.append(contribution)
    
    return contributions

def compute_in_context_performance(
    model,
    few_shot_examples: List[Tuple[str, int]],
    test_cases: List[Tuple[str, int]]
) -> float:
    """
    Compute model performance with in-context learning.
    
    Args:
        model: Trained model
        few_shot_examples: List of (text, label) pairs for few-shot learning
        test_cases: List of (text, label) pairs for testing
        
    Returns:
        Accuracy score
    """
    # Format few-shot examples
    examples = "\n".join([f"Text: {text}\nLabel: {label}" for text, label in few_shot_examples])
    
    # Make predictions
    predictions = []
    for text, _ in test_cases:
        # Combine examples and test case
        prompt = f"{examples}\n\nText: {text}\nLabel:"
        
        # Get prediction
        pred = model.predict(prompt)
        predictions.append(pred)
    
    # Compute accuracy
    true_labels = [label for _, label in test_cases]
    return compute_accuracy(np.array(true_labels), np.array(predictions))

def compute_ablation_performance(
    model,
    text: str,
    true_label: int,
    ablation_type: str = "shuffle"
) -> float:
    """
    Compute model performance with different ablation strategies.
    
    Args:
        model: Trained model
        text: Input text
        true_label: True label
        ablation_type: Type of ablation ("shuffle", "reverse", "remove")
        
    Returns:
        Accuracy score
    """
    if ablation_type == "shuffle":
        # Shuffle sentences
        sentences = text.split('.')
        np.random.shuffle(sentences)
        modified_text = '.'.join(sentences)
    elif ablation_type == "reverse":
        # Reverse sentence order
        sentences = text.split('.')
        modified_text = '.'.join(reversed(sentences))
    elif ablation_type == "remove":
        # Remove every other sentence
        sentences = text.split('.')
        modified_text = '.'.join(sentences[::2])
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    # Get prediction
    pred = model.predict(modified_text)
    
    return int(pred == true_label) 