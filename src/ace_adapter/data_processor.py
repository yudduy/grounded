"""Custom DataProcessor for equation discovery domain.

Adapts the ACE framework's DataProcessor interface for evaluating
mathematical equation predictions against ground truth.
"""
import numpy as np
from typing import Dict, List


class EquationDataProcessor:
    """DataProcessor for equation discovery tasks.

    Implements the interface required by ACE:
    - process_task_data(raw_data) -> List[Dict]
    - answer_is_correct(predicted, ground_truth) -> bool
    - evaluate_accuracy(predictions, ground_truths) -> float
    """

    def __init__(self, mse_threshold: float = 0.1):
        self.mse_threshold = mse_threshold

    def process_task_data(self, raw_data) -> List[Dict]:
        """Convert raw experiment data to ACE format.

        Each sample has:
        - context: accumulated data table
        - question: "what equation fits this data?"
        - target: ground truth expression (for evaluation)
        """
        return raw_data  # Already in correct format

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted MSE is below threshold.

        In equation discovery, 'predicted' is the MSE as a string,
        and 'ground_truth' is the threshold.
        """
        try:
            mse = float(predicted)
            return mse < self.mse_threshold
        except (ValueError, TypeError):
            return False

    def evaluate_accuracy(self, predictions: List[str],
                          ground_truths: List[str]) -> float:
        """Compute fraction of predictions below MSE threshold."""
        correct = sum(
            1 for p in predictions
            if self.answer_is_correct(p, "")
        )
        return correct / max(len(predictions), 1)
