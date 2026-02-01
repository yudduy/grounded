"""Random search baseline.

Generates random expressions from a grammar instead of using an LLM.
Same loop structure for fair comparison.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from shinka.llm.llm import LLMClient
from environments.base import BaseEnvironment
from loop.orchestrator import LoopState, RoundResult

logger = logging.getLogger(__name__)


class RandomSearchCondition:
    """Random expression search baseline.

    Generates random expressions from grammar:
    {+, -, *, /, **, sin, cos, exp, log, sqrt}
    """

    def __init__(self, points_per_round: int = 5, total_rounds: int = 100,
                 seed: int = 0, max_depth: int = 4):
        self.points_per_round = points_per_round
        self.total_rounds = total_rounds
        self.rng = np.random.RandomState(seed)
        self.max_depth = max_depth

    def _random_expr(self, var_names: List[str], depth: int = 0) -> str:
        """Generate a random expression from grammar."""
        if depth >= self.max_depth or self.rng.random() < 0.3:
            # Terminal: variable or constant
            if self.rng.random() < 0.6:
                return self.rng.choice(var_names)
            else:
                c = self.rng.uniform(-5, 5)
                return f"{c:.2f}"

        op_type = self.rng.choice(["binary", "unary"], p=[0.6, 0.4])
        if op_type == "binary":
            op = self.rng.choice(["+", "-", "*", "/", "**"])
            left = self._random_expr(var_names, depth + 1)
            right = self._random_expr(var_names, depth + 1)
            if op == "**":
                # Limit exponent to small integers
                right = str(self.rng.randint(1, 4))
            if op == "/":
                return f"({left} / np.where(np.abs({right}) < 1e-8, 1.0, {right}))"
            return f"({left} {op} {right})"
        else:
            fn = self.rng.choice(["np.sin", "np.cos", "np.exp", "np.log", "np.sqrt", "np.abs"])
            arg = self._random_expr(var_names, depth + 1)
            if fn in ("np.log", "np.sqrt"):
                return f"{fn}(np.abs({arg}) + 1e-8)"
            if fn == "np.exp":
                return f"{fn}(np.clip({arg}, -10, 10))"
            return f"{fn}({arg})"

    def choose_inputs(self, env: BaseEnvironment, state: LoopState,
                      round_num: int, llm: LLMClient) -> np.ndarray:
        return env.sample_inputs(self.points_per_round)

    def hypothesize(self, env: BaseEnvironment, state: LoopState,
                    round_num: int, llm: LLMClient) -> str:
        return self._random_expr(env.input_names)

    def fit(self, expr_str: str, inputs: np.ndarray, outputs: np.ndarray,
            env: BaseEnvironment) -> Optional[Callable]:
        return None

    def reflect(self, env: BaseEnvironment, state: LoopState,
                round_result: RoundResult, llm: LLMClient) -> str:
        return ""

    def curate(self, state: LoopState, round_num: int,
               llm: LLMClient) -> str:
        return ""
