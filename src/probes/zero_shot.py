"""Zero-shot probe: verify LLM cannot recover laws from data alone.

Gives LLM 20 data points and asks it to propose the governing equation.
Success rate must be <20% across all 9 environments for the experiment
to be valid (ensures the task is non-trivial).
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Path setup for imports
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))

from shinka.llm.llm import LLMClient

logger = logging.getLogger(__name__)


def format_data_table(input_names: List[str], inputs: np.ndarray,
                      outputs: np.ndarray) -> str:
    """Format data as a readable table for the LLM prompt."""
    header = " | ".join(input_names + ["output"])
    rows = []
    for i in range(len(outputs)):
        vals = [f"{inputs[i, j]:.4f}" for j in range(inputs.shape[1])]
        vals.append(f"{outputs[i]:.4f}")
        rows.append(" | ".join(vals))
    return header + "\n" + "-" * len(header) + "\n" + "\n".join(rows)


ZERO_SHOT_SYSTEM = """You are a physicist analyzing experimental data.
Given a table of input-output measurements, propose the mathematical equation
that best describes the relationship. Express it as a Python expression using
the given variable names. Use numpy functions (np.sin, np.cos, np.sqrt, etc.)
where needed. Reply with ONLY the Python expression, nothing else."""

ZERO_SHOT_USER = """Here are {n} measurements from a physical system.

Input variables: {input_names}
Output variable: y

{data_table}

What is the equation y = f({input_args})? Reply with only the Python expression."""


def run_zero_shot_probe(env, llm: LLMClient, n_points: int = 20,
                        n_trials: int = 5, mse_threshold: float = 0.1) -> Dict:
    """Run zero-shot probe on a single environment.

    Args:
        env: BaseEnvironment instance
        llm: LLMClient for querying
        n_points: number of data points to show
        n_trials: number of independent LLM attempts
        mse_threshold: MSE below which we count as "success"
    Returns:
        Dict with success_rate, expressions, mses
    """
    inputs = env.sample_inputs(n_points)
    outputs = env.evaluate(inputs)
    data_table = format_data_table(env.input_names, inputs, outputs)
    input_args = ", ".join(env.input_names)

    user_msg = ZERO_SHOT_USER.format(
        n=n_points, input_names=env.input_names,
        data_table=data_table, input_args=input_args,
    )

    successes = 0
    expressions = []
    mses = []

    for trial in range(n_trials):
        result = llm.query(msg=user_msg, system_msg=ZERO_SHOT_SYSTEM)
        if result is None:
            expressions.append(None)
            mses.append(float("inf"))
            continue

        expr_str = result.content.strip()
        expressions.append(expr_str)

        try:
            # Build a predict function from the expression
            local_vars = {"np": np}
            for i, name in enumerate(env.input_names):
                local_vars[name] = env._test_inputs[:, i]
            pred = eval(expr_str, {"__builtins__": {}}, local_vars)
            pred = np.asarray(pred, dtype=np.float64)
            mse = float(np.mean((pred - env._test_outputs) ** 2))
            mses.append(mse)
            if mse < mse_threshold:
                successes += 1
                logger.info(f"  Trial {trial}: SUCCESS (MSE={mse:.6f}) expr={expr_str}")
            else:
                logger.info(f"  Trial {trial}: fail (MSE={mse:.6f}) expr={expr_str}")
        except Exception as e:
            mses.append(float("inf"))
            logger.info(f"  Trial {trial}: error ({e}) expr={expr_str}")

    success_rate = successes / n_trials
    return {
        "env_name": env.name,
        "success_rate": success_rate,
        "expressions": expressions,
        "mses": mses,
        "passed": success_rate < 0.2,
    }


def run_all_zero_shot(envs, llm: LLMClient, **kwargs) -> List[Dict]:
    """Run zero-shot probe on all environments."""
    results = []
    for env in envs:
        logger.info(f"Zero-shot probe: {env.name}")
        r = run_zero_shot_probe(env, llm, **kwargs)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        logger.info(f"  {status}: success_rate={r['success_rate']:.2f}")
    return results
