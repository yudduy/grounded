"""Condition E: DrSR-style with P/N/I sections.

Uses ACE framework but with Positive/Negative/Invalid sections
instead of helpful/harmful bullet tagging. Accumulates equations
into categorized sections based on their effect on MSE.
"""
import sys
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
from conditions.static import StaticCondition
from ace_adapter.drsr_adapter import categorize_equation, build_pni_playbook

logger = logging.getLogger(__name__)

DRSR_PLAYBOOK_TEMPLATE = """## POSITIVE (equations that improved MSE)

## NEGATIVE (equations that worsened MSE)

## INVALID (equations that failed to parse/evaluate)
"""


class DrSRCondition(StaticCondition):
    """Condition E: DrSR-style P/N/I idea library.

    Maintains a library of equations categorized as:
    - Positive: improved MSE over previous best
    - Negative: worsened MSE
    - Invalid: failed to parse or evaluate
    """

    def __init__(self, max_entries_per_section: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._positive = []
        self._negative = []
        self._invalid = []
        self._insights = []
        self._max_entries = max_entries_per_section
        self._prev_best_mse = float("inf")

    def _build_playbook(self) -> str:
        """Build P/N/I playbook from accumulated entries."""
        playbook = build_pni_playbook(
            self._positive, self._negative, self._invalid,
            max_per_section=self._max_entries,
        )

        if self._insights:
            insight_lines = [
                "",
                "## INSIGHTS (extracted patterns from successful equations)",
            ]
            for insight in self._insights[-5:]:
                insight_lines.append(f"  {insight}")
            playbook += "\n" + "\n".join(insight_lines)

        return playbook

    def hypothesize(self, env: BaseEnvironment, state: LoopState,
                    round_num: int, llm: LLMClient) -> str:
        """Propose equation with P/N/I context."""
        input_args = ", ".join(env.input_names)
        ranges_str = ", ".join(
            f"{name}: [{lo:.1f}, {hi:.1f}]"
            for name, (lo, hi) in env.input_ranges.items()
        )

        playbook = self._build_playbook()
        playbook_section = f"\n--- IDEA LIBRARY ---\n{playbook}\n---\n"

        sys_msg = pt.HYPOTHESIZE_SYSTEM.format(
            input_names=env.input_names,
            input_ranges=ranges_str,
            input_args=input_args,
            playbook_section=playbook_section,
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
            raise RuntimeError("LLM returned None")
        return result.content

    def reflect(self, env: BaseEnvironment, state: LoopState,
                round_result: RoundResult, llm: LLMClient) -> str:
        """Categorize this round's equation into P/N/I."""
        expr = round_result.expression_clean or "(none)"
        mse = round_result.test_mse

        category, entry = categorize_equation(
            expr, mse, self._prev_best_mse,
            parse_error=round_result.parse_error,
        )
        if category == "INVALID":
            self._invalid.append(entry)
        elif category == "POSITIVE":
            self._positive.append(entry)
            self._prev_best_mse = mse
        else:
            self._negative.append(entry)

        # Inductive idea extraction every 5 rounds from all 3 categories
        if round_result.round_num % 5 == 0 and (len(self._positive) + len(self._negative) + len(self._invalid)) >= 3:
            sections = []
            if self._positive:
                sections.append("POSITIVE (what works):\n" + "\n".join(self._positive[-5:]))
            if self._negative:
                sections.append("NEGATIVE (what to avoid):\n" + "\n".join(self._negative[-5:]))
            if self._invalid:
                sections.append("INVALID (common errors):\n" + "\n".join(self._invalid[-5:]))
            insight_sys = ("Analyze these categorized equations. Extract 2-3 actionable insights: "
                           "what functional forms work (from positives), what to avoid (from negatives), "
                           "and common errors to prevent (from invalids).")
            insight_msg = "\n\n".join(sections)
            try:
                insight_result = llm.query(msg=insight_msg, system_msg=insight_sys)
                if insight_result and insight_result.content:
                    self._insights.append(f"[Round {round_result.round_num}] {insight_result.content.strip()}")
            except Exception:
                pass

        return f"Categorized as {category}: {expr} MSE={mse:.6f}"

    def curate(self, state: LoopState, round_num: int,
               llm: LLMClient) -> str:
        """P/N/I library self-curates by accumulation."""
        return self._build_playbook()
