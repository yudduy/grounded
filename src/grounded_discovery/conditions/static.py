"""Condition A: Static baseline.

Fixed system prompt, no playbook, no gradient fitting.
LLM proposes equations given accumulated data points.
"""
import re
import sys
import ast
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))
sys.path.insert(0, str(_repo_root / "src"))

from shinka.llm.llm import LLMClient
from environments.base import BaseEnvironment
from loop.orchestrator import LoopState, RoundResult
from loop import prompt_templates as pt

logger = logging.getLogger(__name__)


class StaticCondition:
    """Condition A: Static baseline with no evolving context."""

    def __init__(self, points_per_round: int = 5, total_rounds: int = 100):
        self.points_per_round = points_per_round
        self.total_rounds = total_rounds

    def choose_inputs(self, env: BaseEnvironment, state: LoopState,
                      round_num: int, llm: LLMClient) -> np.ndarray:
        """Ask LLM to choose input points, fall back to random."""
        input_args = ", ".join(env.input_names)
        ranges_str = ", ".join(
            f"{name}: [{lo:.1f}, {hi:.1f}]"
            for name, (lo, hi) in env.input_ranges.items()
        )

        sys_msg = pt.CHOOSE_SYSTEM.format(
            n_inputs=len(env.input_names),
            input_names=env.input_names,
            input_ranges=ranges_str,
            input_args=input_args,
            n_points=self.points_per_round,
            playbook_section="",
        )
        obs_summary = pt.format_observation_summary(
            state.all_inputs, state.all_outputs, env.input_names
        ) if state.all_inputs is not None else "(no observations yet)"

        user_msg = pt.CHOOSE_USER.format(
            round_num=round_num,
            total_rounds=self.total_rounds,
            n_obs=len(state.all_outputs) if state.all_outputs is not None else 0,
            observation_summary=obs_summary,
            reflection_section="",
            n_points=self.points_per_round,
        )

        result = llm.query(msg=user_msg, system_msg=sys_msg)
        if result is None:
            return env.sample_inputs(self.points_per_round)

        try:
            # Parse the list of lists from LLM output
            text = result.content.strip()
            # Remove markdown fences
            text = text.replace("```python", "").replace("```", "").strip()
            # Extract the list using regex before literal_eval
            m = re.search(r'(\[\s*\[.*?\]\s*\])', text, re.DOTALL)
            if m:
                text = m.group(1)
            parsed = ast.literal_eval(text)
            return np.array(parsed, dtype=np.float64)
        except Exception:
            return env.sample_inputs(self.points_per_round)

    def hypothesize(self, env: BaseEnvironment, state: LoopState,
                    round_num: int, llm: LLMClient) -> str:
        """Ask LLM to propose an equation."""
        input_args = ", ".join(env.input_names)
        ranges_str = ", ".join(
            f"{name}: [{lo:.1f}, {hi:.1f}]"
            for name, (lo, hi) in env.input_ranges.items()
        )

        sys_msg = pt.HYPOTHESIZE_SYSTEM.format(
            input_names=env.input_names,
            input_ranges=ranges_str,
            input_args=input_args,
            playbook_section="",
        )

        data_table = pt.format_data_table(
            state.all_inputs, state.all_outputs, env.input_names)
        prev_hyp = pt.format_previous_hypotheses(state.history)

        user_msg = pt.HYPOTHESIZE_USER.format(
            round_num=round_num,
            total_rounds=self.total_rounds,
            n_obs=len(state.all_outputs),
            data_table=data_table,
            previous_hypotheses=prev_hyp,
            reflection_section="",
            input_args=input_args,
        )

        result = llm.query(msg=user_msg, system_msg=sys_msg)
        if result is None:
            raise RuntimeError("LLM returned None for hypothesis")
        return result.content

    def fit(self, expr_str: str, inputs: np.ndarray, outputs: np.ndarray,
            env: BaseEnvironment) -> Optional[Callable]:
        """No fitting in static condition."""
        return None

    def reflect(self, env: BaseEnvironment, state: LoopState,
                round_result: RoundResult, llm: LLMClient) -> str:
        """Basic reflection on round results."""
        sys_msg = pt.REFLECT_SYSTEM

        # Compute residual summary if we have predictions
        residual_summary = ""
        if round_result.expression_clean and state.all_inputs is not None:
            try:
                from loop.expression_parser import parse_expression
                fn, _ = parse_expression(round_result.expression_clean, env.input_names)
                if fn is not None:
                    preds = fn(state.all_inputs)
                    residual_summary = pt.format_residual_summary(
                        state.all_inputs, state.all_outputs, preds, env.input_names)
            except Exception:
                pass

        user_msg = pt.REFLECT_USER.format(
            round_num=round_result.round_num,
            total_rounds=self.total_rounds,
            expression=round_result.expression_clean or "(failed to parse)",
            train_mse=round_result.train_mse,
            test_mse=round_result.test_mse,
            best_mse=state.best_test_mse,
            residual_summary=residual_summary,
        )

        result = llm.query(msg=user_msg, system_msg=sys_msg)
        return result.content if result else ""

    def curate(self, state: LoopState, round_num: int,
               llm: LLMClient) -> str:
        """No curation in static condition."""
        return state.playbook
