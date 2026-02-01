"""Parse LLM-generated Python expressions into callable functions.

Handles common LLM output quirks: markdown fences, variable name mismatches,
numpy vs math functions, reasoning model chain-of-thought output.
"""
import re
import warnings
import numpy as np
from typing import Callable, List, Optional, Tuple


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


def _normalize_math(expr: str) -> str:
    """Normalize a candidate expression string to Python/numpy syntax."""
    expr = expr.replace('≈', '').replace('×', '*').replace('·', '*')
    expr = expr.replace('−', '-').replace('π', 'pi')
    expr = expr.replace('^', '**')
    expr = re.sub(r'(?<!\w)math\.', 'np.', expr)
    expr = re.sub(r'(?<!\w)numpy\.', 'np.', expr)
    expr = expr.strip().rstrip(';').rstrip('?').rstrip('.').strip()
    expr = re.sub(r'\s+or\s+\w+$', '', expr)
    expr = re.sub(r'\s+etc\.?$', '', expr)
    _math_fns = {'exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'abs', 'sign', 'pi', 'np'}
    m = re.search(r'\.\s*([A-Za-z]\w*)$', expr)
    if m and m.group(1).lower() not in _math_fns and not m.group(1)[0].isdigit():
        expr = expr[:m.start()]
    expr = re.sub(r'\s+[A-Z][a-z]+.*$', '', expr)
    for fn in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'sign']:
        expr = re.sub(rf'(?<!\w)(?<!np\.){fn}\s*\(', f'{fn}(', expr)
    return expr


def _common_aliases(input_names: List[str]) -> dict[str, str]:
    """Generate common variable aliases (t->time, x->position, etc.)."""
    aliases = {}
    first_letter_counts = {}
    for name in input_names:
        first_letter_counts[name[0]] = first_letter_counts.get(name[0], 0) + 1
    for name in input_names:
        if first_letter_counts[name[0]] == 1:
            aliases[name[0]] = name
        if name == "time":
            aliases["t"] = "time"
        elif name == "position":
            aliases["x"] = "position"
        elif name == "velocity":
            aliases["v"] = "velocity"
        elif name == "y_offset":
            aliases["offset"] = "y_offset"
    return aliases


def _substitute_aliases(expr: str, input_names: List[str]) -> str:
    """Replace common variable aliases with actual input names."""
    aliases = _common_aliases(input_names)
    for alias, real in sorted(aliases.items(), key=lambda x: -len(x[0])):
        if alias != real and alias not in input_names:
            expr = re.sub(rf'\b{re.escape(alias)}\b', real, expr)
    return expr


def _try_eval(expr: str, input_names: List[str]) -> bool:
    """Test whether an expression evaluates successfully with dummy inputs."""
    if not expr or len(expr) < 3:
        return False
    try:
        local_vars = dict(SAFE_GLOBALS)
        for name in input_names:
            local_vars[name] = np.array([1.0, 2.0])
        aliases = _common_aliases(input_names)
        for alias, real in aliases.items():
            if real in input_names:
                local_vars[alias] = local_vars[real]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            result = eval(expr, {"__builtins__": {}}, local_vars)
        result = np.asarray(result, dtype=np.float64)
        return np.all(np.isfinite(result)) and result.size > 0
    except Exception:
        return False


def _extract_candidates(text: str) -> List[str]:
    """Extract candidate expression strings from LLM output."""
    candidates = []

    text = re.sub(r'```(?:python)?\s*', '', text)
    text = re.sub(r'```', '', text)

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    for line in lines:
        m = re.search(r'(?:^|\s)(?:y|output|f|force|result)\s*=\s*(.+)', line, re.IGNORECASE)
        if m:
            candidates.append(m.group(1))

        m = re.search(r'(?:y|f)\s*\([^)]*\)\s*=\s*(.+)', line, re.IGNORECASE)
        if m:
            candidates.append(m.group(1))

        m = re.search(r'(?:^|\s)(\w+)\s*=\s*([^=].+)', line)
        if m:
            rhs = m.group(2).strip()
            rhs = re.split(r'\.\.\.|(?<=[)\d])\s+[A-Z]', rhs)[0].strip()
            if re.search(r'[+\-*/^]', rhs) and len(rhs) < 120:
                candidates.append(rhs)

        english_words = {"the", "is", "are", "was", "let", "use", "try",
                         "now", "this", "that", "from", "have", "with",
                         "compute", "solve", "substitute", "calculate",
                         "note", "since", "because", "therefore", "thus",
                         "where", "check", "verify", "seems", "maybe",
                         "approximate", "about", "should", "could", "would"}
        words_in_line = set(re.findall(r'\b[a-z]+\b', line.lower()))
        has_english = len(words_in_line & english_words) > 0
        has_operator = bool(re.search(r'[+\-*/^]', line))

        if has_operator and not has_english and len(line) < 120:
            candidates.append(line)

        has_math_fn = bool(re.search(r'\b(sin|cos|exp|log|sqrt|abs)\b', line))
        if (has_math_fn or has_operator) and len(line) < 200:
            for sep in [':', '=']:
                if sep in line:
                    after = line.split(sep, 1)[1].strip()
                    after = re.split(r'\.\.\.|(?<=[)\d])\s+[A-Z]', after)[0].strip()
                    if len(after) > 3 and len(after) < 120:
                        candidates.append(after)

        m = re.search(r'[≈~]\s*(.+)', line)
        if m:
            candidates.append(m.group(1))

    if lines:
        candidates.append(lines[-1])

    return candidates


def clean_expression(raw: str, input_names: Optional[List[str]] = None) -> str:
    """Clean LLM output to extract a Python expression.

    Tries each candidate from reasoning chain-of-thought and returns the last
    one that evaluates as valid Python.
    """
    if not raw or not raw.strip():
        return ""

    candidates = _extract_candidates(raw)
    if not candidates:
        return ""

    normalized = [_normalize_math(c) for c in candidates]

    # Pick the last valid candidate (reasoning models build up to the answer)
    if input_names:
        valid = []
        for expr in normalized:
            if _try_eval(expr, input_names):
                valid.append(expr)
            else:
                # Try with alias substitution
                subst = _substitute_aliases(expr, input_names)
                if subst != expr and _try_eval(subst, input_names):
                    valid.append(subst)
        if valid:
            return valid[-1]

    # Fallback: score heuristically
    best_expr = ""
    best_score = -999

    for expr in normalized:
        if len(expr) > 100:
            continue
        score = 0.0
        for op in ['+', '-', '*', '/', '**']:
            if op in expr:
                score += 1.0
        for fn in ['sin', 'cos', 'exp', 'log', 'sqrt', 'np.']:
            if fn in expr:
                score += 2.0
        if len(expr) < 3:
            score -= 5.0
        if score > best_score:
            best_score = score
            best_expr = expr

    return best_expr


def parse_expression(expr_str: str, input_names: List[str]
                     ) -> Tuple[Optional[Callable], Optional[str]]:
    """Parse expression string into a callable (n,d) array -> (n,) array.

    Returns (predict_fn, error_msg). On failure, predict_fn is None.
    """
    cleaned = clean_expression(expr_str, input_names=input_names)

    if not cleaned:
        return None, "No valid expression found in LLM output"

    def predict_fn(inputs: np.ndarray) -> np.ndarray:
        local_vars = dict(SAFE_GLOBALS)
        for i, name in enumerate(input_names):
            local_vars[name] = inputs[:, i]
        aliases = _common_aliases(input_names)
        for alias, real in aliases.items():
            idx = input_names.index(real) if real in input_names else -1
            if idx >= 0:
                local_vars[alias] = inputs[:, idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            result = eval(cleaned, {"__builtins__": {}}, local_vars)
        result = np.asarray(result, dtype=np.float64)
        if result.ndim == 0:
            result = np.full(inputs.shape[0], float(result))
        return result

    try:
        dummy = np.ones((2, len(input_names)))
        out = predict_fn(dummy)
        if not np.all(np.isfinite(out)):
            return None, f"Expression produces non-finite values: {cleaned}"
        return predict_fn, None
    except Exception as e:
        return None, f"Failed to evaluate '{cleaned}': {e}"
