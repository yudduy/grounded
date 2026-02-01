"""Condition C: Static + Gradient fitting.

Same as Condition A but after LLM proposes an equation form,
parameters are optimized via template kernel gradient fitting.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))
sys.path.insert(0, str(_repo_root / "src"))

from shinka.llm.llm import LLMClient
from environments.base import BaseEnvironment
from loop.orchestrator import LoopState, RoundResult
from conditions.static import StaticCondition
from gradient.fitter import fit_expression, make_predict_fn

logger = logging.getLogger(__name__)


class GradientCondition(StaticCondition):
    """Condition C: Static baseline + gradient parameter fitting.

    Inherits choose_inputs, hypothesize, reflect, curate from StaticCondition.
    Overrides fit() to use template kernel gradient fitting.
    """

    def __init__(self, points_per_round: int = 5, total_rounds: int = 100,
                 use_slurm: bool = False, max_templates: int = 3):
        super().__init__(points_per_round=points_per_round,
                         total_rounds=total_rounds)
        self.use_slurm = use_slurm
        self.max_templates = max_templates

    def fit(self, expr_str: str, inputs: np.ndarray, outputs: np.ndarray,
            env: BaseEnvironment) -> Optional[Callable]:
        """Fit expression parameters using template library."""
        result = fit_expression(
            expr_str, inputs, outputs,
            max_templates=self.max_templates)
        if result is not None:
            logger.info(f"Gradient fit: {result.template_name} MSE={result.train_mse:.6f}")
            return make_predict_fn(result)
        return None
