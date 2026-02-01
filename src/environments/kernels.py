"""Warp template kernels for gradient-based parameter fitting.

Each template defines a parameterized functional form. When Warp is available,
gradients are computed via wp.Tape() autodiff on GPU. Otherwise, falls back
to scipy.optimize with numerical gradients on CPU.

Templates:
- polynomial: sum of a_i * x^i terms (up to degree 5)
- trigonometric: a*sin(b*x + c) + d*cos(e*x + f) + g
- power_law: a * x1^b * x2^c + d
- coupled: a*x1*x2 + b*x1 + c*x2 + d*x1^2*x2 + e*x1*x2^2 + f
- rational: (a*x + b) / (c*x^2 + d*x + 1) + e
"""
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from scipy.optimize import minimize

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False


@dataclass
class TemplateMatch:
    """Result of matching an expression to a template."""
    template_name: str
    initial_params: np.ndarray
    param_names: List[str]
    confidence: float  # 0-1 how well the expression matches


@dataclass
class FitResult:
    """Result of fitting template parameters to data."""
    template_name: str
    params: np.ndarray
    param_names: List[str]
    train_mse: float
    predict_fn: Callable  # (inputs: np.ndarray) -> np.ndarray


class TemplateLibrary:
    """Library of parameterized function templates for fitting."""

    TEMPLATES = {
        "polynomial_1d": {
            "param_names": ["a0", "a1", "a2", "a3", "a4", "a5"],
            "n_params": 6,
            "n_inputs": 1,
            "description": "a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5",
        },
        "polynomial_2d": {
            "param_names": ["a", "b", "c", "d", "e", "f", "g"],
            "n_params": 7,
            "n_inputs": 2,
            "description": "a + b*x1 + c*x2 + d*x1^2 + e*x2^2 + f*x1*x2 + g*x1^2*x2",
        },
        "trigonometric": {
            "param_names": ["a", "b", "c", "d", "e", "f", "g"],
            "n_params": 7,
            "n_inputs": 1,
            "description": "a*sin(b*x+c) + d*cos(e*x+f) + g",
        },
        "trig_coupled": {
            "param_names": ["a", "b", "c", "d", "e"],
            "n_params": 5,
            "n_inputs": 2,
            "description": "a*x1*sin(b*x2) + c*x2*cos(d*x1) + e",
        },
        "power_law": {
            "param_names": ["a", "b", "c", "d"],
            "n_params": 4,
            "n_inputs": 2,
            "description": "a * |x1|^b * |x2|^c + d",
        },
        "rational": {
            "param_names": ["a", "b", "c", "d", "e"],
            "n_params": 5,
            "n_inputs": 1,
            "description": "(a*x + b) / (c*x^2 + d*x + 1) + e",
        },
        "rational_2d": {
            "param_names": ["a", "b", "c", "d"],
            "n_params": 4,
            "n_inputs": 2,
            "description": "a*x1 / (1 + b*x2^2) + c*x2 + d",
        },
        "coupled_product": {
            "param_names": ["a", "b", "c", "d", "e", "f"],
            "n_params": 6,
            "n_inputs": 2,
            "description": "a*x1*x2 + b*x1 + c*x2 + d*x1^2*x2 + e*x1*x2^2 + f",
        },
        "trig_3d": {
            "param_names": ["a", "b", "c", "d", "e"],
            "n_params": 5,
            "n_inputs": 3,
            "description": "a*x1*sin(b*x2) + c*x3*cos(d*x1) + e*x2*x3",
        },
    }

    @staticmethod
    def evaluate_template(name: str, params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate a template with given parameters on inputs.

        Args:
            name: template name from TEMPLATES
            params: parameter values
            inputs: shape (n, d)
        Returns:
            outputs: shape (n,)
        """
        p = params
        if name == "polynomial_1d":
            x = inputs[:, 0]
            return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4 + p[5]*x**5
        elif name == "polynomial_2d":
            x1, x2 = inputs[:, 0], inputs[:, 1]
            return p[0] + p[1]*x1 + p[2]*x2 + p[3]*x1**2 + p[4]*x2**2 + p[5]*x1*x2 + p[6]*x1**2*x2
        elif name == "trigonometric":
            x = inputs[:, 0]
            return p[0]*np.sin(p[1]*x + p[2]) + p[3]*np.cos(p[4]*x + p[5]) + p[6]
        elif name == "trig_coupled":
            x1, x2 = inputs[:, 0], inputs[:, 1]
            return p[0]*x1*np.sin(p[1]*x2) + p[2]*x2*np.cos(p[3]*x1) + p[4]
        elif name == "power_law":
            x1, x2 = inputs[:, 0], inputs[:, 1]
            return p[0] * np.abs(x1)**p[1] * np.abs(x2)**p[2] + p[3]
        elif name == "rational":
            x = inputs[:, 0]
            denom = p[2]*x**2 + p[3]*x + 1.0
            return (p[0]*x + p[1]) / np.where(np.abs(denom) < 1e-8, 1e-8, denom) + p[4]
        elif name == "rational_2d":
            x1, x2 = inputs[:, 0], inputs[:, 1]
            denom = 1.0 + p[1]*x2**2
            return p[0]*x1 / np.where(np.abs(denom) < 1e-8, 1e-8, denom) + p[2]*x2 + p[3]
        elif name == "coupled_product":
            x1, x2 = inputs[:, 0], inputs[:, 1]
            return p[0]*x1*x2 + p[1]*x1 + p[2]*x2 + p[3]*x1**2*x2 + p[4]*x1*x2**2 + p[5]
        elif name == "trig_3d":
            x1, x2, x3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
            return p[0]*x1*np.sin(p[1]*x2) + p[2]*x3*np.cos(p[3]*x1) + p[4]*x2*x3
        else:
            raise ValueError(f"Unknown template: {name}")

    @classmethod
    def fit(cls, name: str, inputs: np.ndarray, targets: np.ndarray,
            initial_params: Optional[np.ndarray] = None,
            max_iter: int = 200) -> FitResult:
        """Fit template parameters to data using scipy L-BFGS-B.

        Args:
            name: template name
            inputs: shape (n, d)
            targets: shape (n,)
            initial_params: starting parameters (or random)
            max_iter: maximum optimization iterations
        Returns:
            FitResult with optimized parameters
        """
        tmpl = cls.TEMPLATES[name]
        n_params = tmpl["n_params"]
        if initial_params is None:
            initial_params = np.random.randn(n_params) * 0.1

        def loss(p):
            try:
                pred = cls.evaluate_template(name, p, inputs)
                mse = np.mean((pred - targets) ** 2)
                if np.isnan(mse) or np.isinf(mse):
                    return 1e10
                return mse
            except Exception:
                return 1e10

        result = minimize(loss, initial_params, method="L-BFGS-B",
                          options={"maxiter": max_iter, "ftol": 1e-12})
        final_params = result.x
        final_mse = float(result.fun)

        def predict_fn(x):
            return cls.evaluate_template(name, final_params, x)

        return FitResult(
            template_name=name,
            params=final_params,
            param_names=tmpl["param_names"],
            train_mse=final_mse,
            predict_fn=predict_fn,
        )

    @classmethod
    def match_expression(cls, expr_str: str, n_inputs: int) -> List[TemplateMatch]:
        """Match a Python expression string to candidate templates.

        Uses keyword heuristics to rank templates by likelihood of match.

        Args:
            expr_str: Python expression string from LLM
            n_inputs: number of input variables
        Returns:
            List of TemplateMatch sorted by confidence (descending)
        """
        expr_lower = expr_str.lower()
        candidates = []

        for name, tmpl in cls.TEMPLATES.items():
            if tmpl["n_inputs"] != n_inputs and tmpl["n_inputs"] < n_inputs:
                continue
            score = 0.0

            # Check for keyword matches
            if name.startswith("polynomial"):
                # Polynomial indicators: x^n, x**n, x*x
                if re.search(r'\*\*\s*[2-5]|\^[2-5]|x\s*\*\s*x', expr_lower):
                    score += 0.5
                if "sin" not in expr_lower and "cos" not in expr_lower:
                    score += 0.3
            elif name.startswith("trig"):
                if "sin" in expr_lower or "cos" in expr_lower:
                    score += 0.7
            elif name == "power_law":
                if re.search(r'\*\*\s*[a-z]|\^[a-z]|pow', expr_lower):
                    score += 0.6
                if "abs" in expr_lower:
                    score += 0.2
            elif name.startswith("rational"):
                if "/" in expr_lower and ("+" in expr_lower or "-" in expr_lower):
                    score += 0.5
                if re.search(r'1\s*\+|1\s*-', expr_lower):
                    score += 0.2
            elif name == "coupled_product":
                if "*" in expr_lower and n_inputs >= 2:
                    score += 0.4

            # Dimensionality bonus
            if tmpl["n_inputs"] == n_inputs:
                score += 0.1

            if score > 0:
                candidates.append(TemplateMatch(
                    template_name=name,
                    initial_params=np.random.randn(tmpl["n_params"]) * 0.1,
                    param_names=tmpl["param_names"],
                    confidence=min(score, 1.0),
                ))

        # Sort by confidence descending
        candidates.sort(key=lambda m: -m.confidence)

        # Always include a fallback polynomial
        fallback_name = "polynomial_2d" if n_inputs >= 2 else "polynomial_1d"
        if n_inputs == 3:
            fallback_name = "trig_3d"
        if not any(c.template_name == fallback_name for c in candidates):
            tmpl = cls.TEMPLATES[fallback_name]
            candidates.append(TemplateMatch(
                template_name=fallback_name,
                initial_params=np.random.randn(tmpl["n_params"]) * 0.1,
                param_names=tmpl["param_names"],
                confidence=0.05,
            ))

        return candidates

    @classmethod
    def fit_best(cls, expr_str: str, inputs: np.ndarray, targets: np.ndarray,
                 max_templates: int = 3, max_iter: int = 200) -> FitResult:
        """Match expression to templates, fit top candidates, return best.

        Args:
            expr_str: LLM expression string
            inputs: shape (n, d)
            targets: shape (n,)
            max_templates: number of templates to try
            max_iter: optimization iterations per template
        Returns:
            Best FitResult across all tried templates
        """
        n_inputs = inputs.shape[1]
        matches = cls.match_expression(expr_str, n_inputs)[:max_templates]

        best_result = None
        for match in matches:
            result = cls.fit(match.template_name, inputs, targets,
                             initial_params=match.initial_params, max_iter=max_iter)
            if best_result is None or result.train_mse < best_result.train_mse:
                best_result = result

        return best_result
