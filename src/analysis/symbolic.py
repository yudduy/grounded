"""Symbolic recovery checking using SymPy.

Determines whether discovered equations match ground truth
up to algebraic simplification and parameter renaming.
"""
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import sympy
    from sympy import simplify, symbols, sympify, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


def check_symbolic_match(discovered: str, ground_truth: str,
                         variable_names: list,
                         tol: float = 1e-4) -> Dict:
    """Check if discovered expression matches ground truth symbolically.

    Args:
        discovered: Python expression string
        ground_truth: ground truth expression string
        variable_names: list of variable names
        tol: tolerance for numerical comparison
    Returns:
        Dict with matched (bool), simplified forms, difference
    """
    if not SYMPY_AVAILABLE:
        return {"matched": False, "error": "sympy not available"}

    try:
        syms = symbols(variable_names)
        sym_dict = dict(zip(variable_names, syms))
        sym_dict["np"] = sympy
        sym_dict["sin"] = sympy.sin
        sym_dict["cos"] = sympy.cos
        sym_dict["exp"] = sympy.exp
        sym_dict["log"] = sympy.log
        sym_dict["sqrt"] = sympy.sqrt
        sym_dict["abs"] = sympy.Abs
        sym_dict["sign"] = sympy.sign
        sym_dict["pi"] = sympy.pi

        disc_expr = sympify(discovered, locals=sym_dict)
        truth_expr = sympify(ground_truth, locals=sym_dict)

        diff = simplify(disc_expr - truth_expr)
        matched = diff == 0

        # If exact match fails, try numerical evaluation
        if not matched:
            test_vals = {s: np.random.uniform(-2, 2) for s in syms}
            num_diff = float(abs(N(diff.subs(test_vals))))
            matched = num_diff < tol

        return {
            "matched": matched,
            "discovered_simplified": str(simplify(disc_expr)),
            "truth_simplified": str(simplify(truth_expr)),
            "difference": str(diff),
        }
    except Exception as e:
        return {"matched": False, "error": str(e)}


def symbolic_recovery_rate(results: list, env_ground_truths: Dict[str, str],
                           env_variables: Dict[str, list]) -> Dict:
    """Compute symbolic recovery rate across all runs.

    Args:
        results: list of run result dicts with env_name, best_expression
        env_ground_truths: {env_name: ground_truth_expr}
        env_variables: {env_name: [var_names]}
    Returns:
        Dict with overall rate and per-condition rates
    """
    by_cond = {}
    for r in results:
        env = r["env_name"]
        cond = r["condition"]
        expr = r.get("best_expression")
        if not expr or env not in env_ground_truths:
            continue
        match = check_symbolic_match(
            expr, env_ground_truths[env], env_variables[env])
        by_cond.setdefault(cond, []).append(match.get("matched", False))

    rates = {}
    for cond, matches in sorted(by_cond.items()):
        rates[cond] = sum(matches) / max(len(matches), 1)
    overall = sum(m for ms in by_cond.values() for m in ms) / max(
        sum(len(ms) for ms in by_cond.values()), 1)
    return {"overall": overall, "by_condition": rates}
