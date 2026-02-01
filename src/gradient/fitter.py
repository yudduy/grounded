"""Gradient-based parameter fitting using template kernels.

Uses scipy L-BFGS-B for CPU fitting. When Warp is available,
can use wp.Tape() for GPU-accelerated gradients via SLURM jobs.
"""
import logging
import numpy as np
from typing import Optional, Callable, Tuple
from environments.kernels import TemplateLibrary, FitResult

logger = logging.getLogger(__name__)


def fit_expression(expr_str: str, inputs: np.ndarray, targets: np.ndarray,
                   max_templates: int = 3, max_iter: int = 200,
                   n_restarts: int = 3) -> Optional[FitResult]:
    """Fit an LLM expression using template library with multiple restarts.

    Args:
        expr_str: LLM expression string
        inputs: (n, d) input data
        targets: (n,) target values
        max_templates: number of templates to try
        max_iter: optimization iterations per attempt
        n_restarts: number of random restarts per template
    Returns:
        Best FitResult, or None if all fail
    """
    n_inputs = inputs.shape[1]
    matches = TemplateLibrary.match_expression(expr_str, n_inputs)[:max_templates]

    best_result = None
    for match in matches:
        for restart in range(n_restarts):
            try:
                init_params = np.random.randn(len(match.param_names)) * 0.5
                result = TemplateLibrary.fit(
                    match.template_name, inputs, targets,
                    initial_params=init_params, max_iter=max_iter)
                if best_result is None or result.train_mse < best_result.train_mse:
                    best_result = result
            except Exception as e:
                logger.debug(f"Fit failed for {match.template_name} restart {restart}: {e}")

    return best_result


def make_predict_fn(fit_result: FitResult) -> Callable:
    """Create a predict function from a FitResult."""
    return fit_result.predict_fn
