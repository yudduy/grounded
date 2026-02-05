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
                   n_restarts: int = 5) -> Optional[FitResult]:
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

    # Nelder-Mead fallback if L-BFGS-B produced poor results (normalized MSE threshold)
    target_var = np.var(targets)
    nmse = (best_result.train_mse / target_var) if (best_result is not None and target_var > 0) else float("inf")
    if best_result is not None and nmse > 0.01:
        from scipy.optimize import minimize as scipy_minimize
        n_nm_restarts = 3
        for match in matches:
            for nm_restart in range(n_nm_restarts):
                try:
                    def loss(p, _name=match.template_name):
                        try:
                            pred = TemplateLibrary.evaluate_template(_name, p, inputs)
                            mse = np.mean((pred - targets) ** 2)
                            return 1e10 if (np.isnan(mse) or np.isinf(mse)) else mse
                        except Exception:
                            return 1e10

                    init = np.random.randn(len(match.param_names)) * 0.5
                    res = scipy_minimize(loss, init, method="Nelder-Mead",
                                         options={"maxiter": 500, "xatol": 1e-8})
                    if res.fun < best_result.train_mse:
                        final_params = res.x
                        best_result = FitResult(
                            template_name=match.template_name,
                            params=final_params,
                            param_names=match.param_names,
                            train_mse=float(res.fun),
                            predict_fn=lambda x, n=match.template_name, p=final_params: TemplateLibrary.evaluate_template(n, p, x),
                        )
                except Exception as e:
                    logger.debug(f"Nelder-Mead fallback failed for {match.template_name} restart {nm_restart}: {e}")

    return best_result


def make_predict_fn(fit_result: FitResult) -> Callable:
    """Create a predict function from a FitResult."""
    return fit_result.predict_fn
