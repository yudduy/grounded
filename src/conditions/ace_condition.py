"""Condition B: ACE (evolving playbook).

Uses ACE's Generator/Reflector/Curator to maintain an evolving playbook
of equation discovery strategies. No gradient fitting.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))
sys.path.insert(0, str(_repo_root / "ace"))
sys.path.insert(0, str(_repo_root / "src"))

from shinka.llm.llm import LLMClient
from environments.base import BaseEnvironment
from loop.orchestrator import LoopState, RoundResult
from loop import prompt_templates as pt
from conditions.static import StaticCondition
from ace_adapter.prompts import (
    EQUATION_PLAYBOOK_TEMPLATE,
    EQUATION_REFLECTOR_CONTEXT,
    EQUATION_CURATOR_CONTEXT,
    format_bullets_used,
)

logger = logging.getLogger(__name__)


class ACECondition(StaticCondition):
    """Condition B: ACE with evolving playbook.

    Inherits base loop from StaticCondition.
    Overrides:
    - hypothesize: includes playbook in prompt
    - reflect: tags playbook bullets as helpful/harmful
    - curate: evolves playbook every N rounds
    """

    def __init__(self, points_per_round: int = 5, total_rounds: int = 100,
                 curate_interval: int = 10,
                 api_provider: str = "openai",
                 reflector_model: str = "gpt-4o",
                 curator_model: str = "gpt-4o"):
        super().__init__(points_per_round=points_per_round,
                         total_rounds=total_rounds)
        self.curate_interval = curate_interval
        self.api_provider = api_provider
        self.reflector_model = reflector_model
        self.curator_model = curator_model
        self._playbook = EQUATION_PLAYBOOK_TEMPLATE
        self._next_global_id = 1
        self._playbook_stats = {}
        self._ace_initialized = False
        self._reflector = None
        self._curator = None
        self._api_client = None

    def _init_ace(self):
        """Lazy initialization of ACE components."""
        if self._ace_initialized:
            return
        try:
            from llm import get_api_client
            self._api_client = get_api_client(self.api_provider)
            from ace.core import Reflector, Curator
            self._reflector = Reflector(
                self._api_client, self.api_provider, self.reflector_model)
            self._curator = Curator(
                self._api_client, self.api_provider, self.curator_model)
            try:
                from ace.core.bulletpoint_analyzer import BulletpointAnalyzer
                self._analyzer = BulletpointAnalyzer(
                    self._api_client, self.curator_model,
                    embedding_model_name='all-mpnet-base-v2')
            except Exception:
                self._analyzer = None
            self._ace_initialized = True
        except ImportError as e:
            logger.warning(f"ACE import failed, using LLM-based fallback: {e}")
            self._ace_initialized = True  # Don't retry

    def hypothesize(self, env: BaseEnvironment, state: LoopState,
                    round_num: int, llm: LLMClient) -> str:
        """Propose equation with playbook context."""
        input_args = ", ".join(env.input_names)
        ranges_str = ", ".join(
            f"{name}: [{lo:.1f}, {hi:.1f}]"
            for name, (lo, hi) in env.input_ranges.items()
        )

        playbook_section = f"\n--- PLAYBOOK ---\n{self._playbook}\n---\n" if self._playbook else ""

        sys_msg = pt.HYPOTHESIZE_SYSTEM.format(
            input_names=env.input_names,
            input_ranges=ranges_str,
            input_args=input_args,
            playbook_section=playbook_section,
        )

        data_table = pt.format_data_table(
            state.all_inputs, state.all_outputs, env.input_names)
        prev_hyp = pt.format_previous_hypotheses(state.history)

        # Include last reflection if available
        last_reflection = ""
        if state.history and len(state.history) > 0:
            last_h = state.history[-1]
            if "reflection" in last_h:
                last_reflection = f"Last reflection: {last_h['reflection']}"

        user_msg = pt.HYPOTHESIZE_USER.format(
            round_num=round_num,
            total_rounds=self.total_rounds,
            n_obs=len(state.all_outputs),
            data_table=data_table,
            previous_hypotheses=prev_hyp,
            reflection_section=last_reflection,
            input_args=input_args,
        )

        result = llm.query(msg=user_msg, system_msg=sys_msg)
        if result is None:
            raise RuntimeError("LLM returned None for hypothesis")
        return result.content

    def reflect(self, env: BaseEnvironment, state: LoopState,
                round_result: RoundResult, llm: LLMClient) -> str:
        """Reflect using ACE Reflector if available, else LLM fallback."""
        self._init_ace()

        improved = "Yes" if round_result.test_mse < state.best_test_mse else "No"

        if self._reflector is not None:
            try:
                bullets_used = format_bullets_used(state.history, playbook=self._playbook)
                reflection, bullet_tags, call_info = self._reflector.reflect(
                    question=f"Discover equation for {env.name}",
                    reasoning_trace=round_result.expression_clean or "",
                    predicted_answer=f"{round_result.test_mse:.6f}",
                    ground_truth=f"{state.best_test_mse:.6f}",
                    environment_feedback=f"MSE improved: {improved}",
                    bullets_used=bullets_used,
                    use_ground_truth=True,
                )
                # Update stats
                for tag in (bullet_tags or []):
                    bid = tag.get("id", "")
                    label = tag.get("tag", "")
                    if bid not in self._playbook_stats:
                        self._playbook_stats[bid] = {"helpful": 0, "harmful": 0}
                    if label in ("helpful", "harmful"):
                        self._playbook_stats[bid][label] += 1
                return reflection
            except Exception as e:
                logger.warning(f"ACE Reflector failed: {e}")

        # Fallback to basic reflection
        return super().reflect(env, state, round_result, llm)

    def curate(self, state: LoopState, round_num: int,
               llm: LLMClient) -> str:
        """Evolve playbook every curate_interval rounds."""
        self._rounds_since_curate = getattr(self, '_rounds_since_curate', 0) + 1
        self._rounds_since_improvement = getattr(self, '_rounds_since_improvement', 0)

        # Detect performance plateau
        if len(state.history) >= 5:
            recent_mses = [h.get("test_mse", float("inf")) for h in state.history[-5:]]
            if all(m < float("inf") for m in recent_mses):
                mse_var = np.var(recent_mses)
                if mse_var < 0.001:
                    self._rounds_since_improvement += 1
                else:
                    self._rounds_since_improvement = 0

        # Adaptive interval: curate more often during plateaus
        interval = min(5, self.curate_interval) if self._rounds_since_improvement >= 3 else self.curate_interval
        if self._rounds_since_curate < interval:
            return self._playbook
        self._rounds_since_curate = 0

        self._init_ace()

        if self._curator is not None:
            try:
                perf_summary = format_bullets_used(state.history, n_recent=self.curate_interval, playbook=self._playbook)
                last_reflection = ""
                if state.history:
                    last_reflection = state.history[-1].get("reflection", "")

                updated, next_id, operations, call_info = self._curator.curate(
                    current_playbook=self._playbook,
                    recent_reflection=last_reflection,
                    question_context=f"Equation discovery, round {round_num}/{self.total_rounds}",
                    current_step=round_num,
                    total_samples=self.total_rounds,
                    token_budget=80000,
                    playbook_stats=self._playbook_stats,
                    next_global_id=self._next_global_id,
                )
                self._playbook = updated
                self._next_global_id = next_id
                logger.info(f"Curated playbook at round {round_num}: "
                            f"{len(operations)} operations")

                # Semantic dedup every 20 rounds
                if getattr(self, '_analyzer', None) and round_num % 20 == 0:
                    try:
                        self._playbook = self._analyzer.analyze(
                            self._playbook, threshold=0.90, merge=True)
                        logger.info(f"Deduplicated playbook at round {round_num}")
                    except Exception as e:
                        logger.debug(f"Dedup failed: {e}")

                return self._playbook
            except Exception as e:
                logger.warning(f"ACE Curator failed: {e}")

        # Fallback: ask LLM to update playbook
        ctx = EQUATION_CURATOR_CONTEXT.format(
            performance_summary=format_bullets_used(state.history, self.curate_interval, playbook=self._playbook))
        sys_msg = "You are curating an equation discovery playbook. Update it based on recent results."
        result = llm.query(msg=ctx, system_msg=sys_msg)
        if result and result.content:
            self._playbook = result.content
        return self._playbook
