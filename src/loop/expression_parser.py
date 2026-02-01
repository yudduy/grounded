"""Parse LLM-generated Python expressions into callable functions.

Handles common LLM output quirks: markdown fences, variable name mismatches,
numpy vs math functions, etc.
"""
import re
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


# Allowed names in eval context
SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,
    "pi": np.pi,
    "e": np.e,
}


def clean_expression(raw: str) -> str:
    """Clean LLM output to extract a Python expression.

    Handles: markdown code fences, 'y = ...' prefix, math notation,
    multiple lines (takes last expression-like line).
    """
    text = raw.strip()
    # Remove markdown fences
    text = re.sub(r'```(?:python)?\s*', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()

    # If multiple lines, find the one that looks most like an expression
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    expr_line = lines[-1]  # Default to last line
    for line in lines:
        # Prefer lines starting with y= or output= or just an expression
        if re.match(r'^(y|output|f|force|result)\s*=\s*', line, re.IGNORECASE):
            expr_line = re.sub(r'^[a-zA-Z_]+\s*=\s*', '', line)
            break

    # Replace math notation
    expr = expr_line
    expr = expr.replace('^', '**')
    expr = re.sub(r'(?<!\w)math\.', 'np.', expr)
    expr = re.sub(r'(?<!\w)numpy\.', 'np.', expr)

    # Replace common function names without np prefix
    for fn in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'sign']:
        expr = re.sub(rf'(?<!\w)(?<!np\.){fn}\s*\(', f'{fn}(', expr)

    return expr.strip().rstrip(';')


def parse_expression(expr_str: str, input_names: List[str]
                     ) -> Tuple[Optional[Callable], Optional[str]]:
    """Parse a cleaned expression string into a callable.

    Args:
        expr_str: cleaned Python expression
        input_names: list of variable names (e.g., ["x", "y"])
    Returns:
        (predict_fn, error_msg) — predict_fn takes (n, d) array -> (n,)
        On failure, predict_fn is None and error_msg describes the issue.
    """
    cleaned = clean_expression(expr_str)

    def predict_fn(inputs: np.ndarray) -> np.ndarray:
        local_vars = dict(SAFE_GLOBALS)
        for i, name in enumerate(input_names):
            local_vars[name] = inputs[:, i]
        result = eval(cleaned, {"__builtins__": {}}, local_vars)
        result = np.asarray(result, dtype=np.float64)
        # Broadcast scalar to array
        if result.ndim == 0:
            result = np.full(inputs.shape[0], float(result))
        return result

    # Test with dummy data to catch syntax/name errors
    try:
        dummy = np.ones((2, len(input_names)))
        out = predict_fn(dummy)
        if not np.all(np.isfinite(out)):
            return None, f"Expression produces non-finite values: {cleaned}"
        return predict_fn, None
    except Exception as e:
        return None, f"Failed to evaluate '{cleaned}': {e}"
