"""Condition D: ACE + Gradient fitting.

Combines ACE playbook evolution (Condition B) with gradient parameter
fitting (Condition C).
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))
sys.path.insert(0, str(_repo_root / "src"))

from environments.base import BaseEnvironment
from conditions.ace_condition import ACECondition
from gradient.fitter import fit_expression, make_predict_fn

logger = logging.getLogger(__name__)


class ACEGradientCondition(ACECondition):
    """Condition D: ACE playbook + gradient fitting.

    Inherits playbook evolution from ACECondition.
    Adds gradient fitting from GradientCondition.
    """

    def __init__(self, max_templates: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_templates = max_templates

    def fit(self, expr_str: str, inputs: np.ndarray, outputs: np.ndarray,
            env: BaseEnvironment) -> Optional[Callable]:
        """Fit expression parameters using template library."""
        result = fit_expression(expr_str, inputs, outputs,
                                max_templates=self.max_templates)
        if result is not None:
            logger.info(f"Gradient fit: {result.template_name} MSE={result.train_mse:.6f}")
            return make_predict_fn(result)
        return None
