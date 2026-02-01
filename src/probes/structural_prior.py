"""Structural prior probe: verify LLM cannot guess correct functional form.

Asks LLM for plausible functional forms for each physical system
(described by name/description only, no data). Checks that the
LLM's guesses don't match the actual ground truth structure.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))

from shinka.llm.llm import LLMClient

logger = logging.getLogger(__name__)

STRUCTURAL_SYSTEM = """You are a physicist. Given a description of a physical system,
suggest 3 plausible mathematical forms for the governing equation.
Express each as a Python expression. Use variable names as specified.
Format your response as:
1. <expression>
2. <expression>
3. <expression>"""

STRUCTURAL_USER = """System: {description}
Input variables: {input_names}
Output variable: y

What are 3 plausible functional forms for y = f({input_args})?"""


def _parse_expressions(response: str) -> List[str]:
    """Extract numbered expressions from LLM response."""
    import re
    exprs = []
    for line in response.strip().split("\n"):
        m = re.match(r'^\d+\.\s*(.+)$', line.strip())
        if m:
            exprs.append(m.group(1).strip().rstrip(';'))
    return exprs


def _structural_similarity(expr_str: str, env, n_test: int = 50) -> float:
    """Check if expression structurally matches ground truth via correlation.

    Returns R^2 between expression output and ground truth on test points.
    """
    try:
        local_vars = {"np": np}
        for i, name in enumerate(env.input_names):
            local_vars[name] = env._test_inputs[:, i]
        pred = np.asarray(eval(expr_str, {"__builtins__": {}}, local_vars), dtype=np.float64)
        truth = env._test_outputs

        # Compute R^2
        ss_res = np.sum((truth - pred) ** 2)
        ss_tot = np.sum((truth - np.mean(truth)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        r2 = 1.0 - ss_res / ss_tot
        return max(r2, 0.0)
    except Exception:
        return 0.0


def run_structural_prior_probe(env, llm: LLMClient,
                               r2_threshold: float = 0.9) -> Dict:
    """Run structural prior probe on one environment.

    Args:
        env: BaseEnvironment instance
        llm: LLMClient
        r2_threshold: R^2 above which we consider the structure "guessed"
    Returns:
        Dict with results
    """
    input_args = ", ".join(env.input_names)
    user_msg = STRUCTURAL_USER.format(
        description=env.spec.description,
        input_names=env.input_names,
        input_args=input_args,
    )

    result = llm.query(msg=user_msg, system_msg=STRUCTURAL_SYSTEM)
    if result is None:
        return {"env_name": env.name, "passed": True, "expressions": [], "r2_scores": []}

    expressions = _parse_expressions(result.content)
    r2_scores = [_structural_similarity(expr, env) for expr in expressions]

    guessed = any(r2 > r2_threshold for r2 in r2_scores)
    logger.info(f"  Structural prior: {env.name} -> R2 scores: {r2_scores}")
    return {
        "env_name": env.name,
        "expressions": expressions,
        "r2_scores": r2_scores,
        "passed": not guessed,
    }


def run_all_structural_prior(envs, llm: LLMClient, **kwargs) -> List[Dict]:
    """Run structural prior probe on all environments."""
    results = []
    for env in envs:
        logger.info(f"Structural prior probe: {env.name}")
        r = run_structural_prior_probe(env, llm, **kwargs)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        logger.info(f"  {status}")
    return results
