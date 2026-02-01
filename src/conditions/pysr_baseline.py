"""Condition F: PySR symbolic regression baseline.

Runs PySR on accumulated data at rounds 25, 50, 75, 100.
Reports best symbolic expression found.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from environments.base import BaseEnvironment
from loop.orchestrator import LoopState, RoundResult
from conditions.static import StaticCondition

logger = logging.getLogger(__name__)

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False


class PySRCondition(StaticCondition):
    """Condition F: PySR symbolic regression at milestone rounds.

    Runs PySR on all accumulated data at rounds 25, 50, 75, 100.
    Between milestones, uses the best PySR expression found so far.
    """

    def __init__(self, milestone_rounds=None, pysr_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.milestone_rounds = milestone_rounds or [25, 50, 75, 100]
        self.pysr_kwargs = pysr_kwargs or {}
        self._best_pysr_fn = None
        self._best_pysr_expr = None

    def hypothesize(self, env: BaseEnvironment, state: LoopState,
                    round_num: int, llm: LLMClient) -> str:
        """At milestones, run PySR. Otherwise use LLM."""
        if round_num in self.milestone_rounds and PYSR_AVAILABLE:
            return self._run_pysr(env, state)
        if self._best_pysr_expr:
            return self._best_pysr_expr
        return super().hypothesize(env, state, round_num, llm)

    def _run_pysr(self, env: BaseEnvironment, state: LoopState) -> str:
        """Run PySR on accumulated data."""
        try:
            model = PySRRegressor(
                niterations=40,
                binary_operators=["+", "-", "*", "/", "^"],
                unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],
                populations=15,
                population_size=33,
                maxsize=25,
                timeout_in_seconds=120,
                temp_equation_file=True,
                **self.pysr_kwargs,
            )
            model.fit(state.all_inputs, state.all_outputs,
                      variable_names=env.input_names)

            best_expr = str(model.sympy())
            # Convert to Python/numpy syntax
            best_expr = best_expr.replace("^", "**")

            self._best_pysr_expr = best_expr
            logger.info(f"PySR found: {best_expr}")
            return best_expr
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            if self._best_pysr_expr:
                return self._best_pysr_expr
            return "0"

    def fit(self, expr_str: str, inputs: np.ndarray, outputs: np.ndarray,
            env: BaseEnvironment) -> Optional[Callable]:
        """No additional fitting for PySR (it already fits)."""
        return None
