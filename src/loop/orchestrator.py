"""Discovery loop orchestrator.

Runs the interactive equation discovery loop:
CHOOSE → OBSERVE → HYPOTHESIZE → FIT → EVALUATE → REFLECT → CURATE

Different experimental conditions plug in via the ConditionStrategy interface.
"""
import sys
import json
import logging
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Protocol

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))
sys.path.insert(0, str(_repo_root / "src"))

from shinka.llm.llm import LLMClient
from environments.base import BaseEnvironment
from loop.expression_parser import parse_expression, clean_expression
from loop import prompt_templates as pt

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Results from a single round of the discovery loop."""
    round_num: int
    chosen_inputs: Optional[np.ndarray] = None
    observed_outputs: Optional[np.ndarray] = None
    expression_raw: Optional[str] = None
    expression_clean: Optional[str] = None
    train_mse: float = float("inf")
    test_mse: float = float("inf")
    reflection: Optional[str] = None
    llm_cost: float = 0.0
    wall_time: float = 0.0
    parse_error: Optional[str] = None


@dataclass
class LoopState:
    """Accumulated state across rounds."""
    all_inputs: Optional[np.ndarray] = None  # (N, d)
    all_outputs: Optional[np.ndarray] = None  # (N,)
    history: List[Dict] = field(default_factory=list)
    best_test_mse: float = float("inf")
    best_expression: Optional[str] = None
    best_round: int = -1
    playbook: str = ""
    total_cost: float = 0.0

    def add_observations(self, inputs: np.ndarray, outputs: np.ndarray):
        """Accumulate new observations into the loop state."""
        if self.all_inputs is None:
            self.all_inputs = inputs.copy()
            self.all_outputs = outputs.copy()
        else:
            self.all_inputs = np.vstack([self.all_inputs, inputs])
            self.all_outputs = np.concatenate([self.all_outputs, outputs])


class ConditionStrategy(Protocol):
    """Interface for experimental conditions (A-F).

    Each condition implements different subsets of the loop steps.
    """

    def choose_inputs(self, env: BaseEnvironment, state: LoopState,
                      round_num: int, llm: LLMClient) -> np.ndarray:
        """Choose input points for this round."""
        ...

    def hypothesize(self, env: BaseEnvironment, state: LoopState,
                    round_num: int, llm: LLMClient) -> str:
        """Propose an equation given accumulated data."""
        ...

    def fit(self, expr_str: str, inputs: np.ndarray, outputs: np.ndarray,
            env: BaseEnvironment) -> Optional[Callable]:
        """Optionally fit parameters. Returns predict_fn or None."""
        ...

    def reflect(self, env: BaseEnvironment, state: LoopState,
                round_result: RoundResult, llm: LLMClient) -> str:
        """Reflect on this round's results."""
        ...

    def curate(self, state: LoopState, round_num: int,
               llm: LLMClient) -> str:
        """Update playbook/context. Returns updated playbook."""
        ...


class DiscoveryLoop:
    """Main orchestrator for the equation discovery experiment."""

    def __init__(self, env: BaseEnvironment, llm: LLMClient,
                 strategy: ConditionStrategy,
                 total_rounds: int = 100,
                 points_per_round: int = 5,
                 seed: int = 0):
        """Initialize the discovery loop.

        Args:
            env: The counterfactual physics environment.
            llm: LLM client for prompting.
            strategy: Condition strategy implementing the loop steps.
            total_rounds: Number of discovery rounds to run.
            points_per_round: Number of new data points per round.
            seed: Random seed for reproducibility.
        """
        self.env = env
        self.llm = llm
        self.strategy = strategy
        self.total_rounds = total_rounds
        self.points_per_round = points_per_round
        self.seed = seed
        self.state = LoopState()
        self.results: List[RoundResult] = []

    def _track_cost(self, result) -> float:
        """Extract cost from an LLM QueryResult if available."""
        if result is not None and hasattr(result, 'cost') and result.cost is not None:
            return float(result.cost)
        return 0.0

    def run(self, start_round: int = 1) -> List[RoundResult]:
        """Run the discovery loop from start_round to total_rounds.

        Args:
            start_round: Starting round number (for resuming from checkpoint).

        Returns:
            List of RoundResult objects, one per round.
        """
        for round_num in range(start_round, self.total_rounds + 1):
            t0 = time.time()
            rr = self._run_round(round_num)
            rr.wall_time = time.time() - t0
            self.state.total_cost += rr.llm_cost
            self.results.append(rr)

            logger.info(
                f"Round {round_num}/{self.total_rounds}: "
                f"test_mse={rr.test_mse:.6f} best={self.state.best_test_mse:.6f} "
                f"expr={rr.expression_clean}"
            )

        return self.results

    def _run_round(self, round_num: int) -> RoundResult:
        """Execute one complete round of the discovery loop.

        Steps:
        1. CHOOSE: select input points
        2. OBSERVE: query environment
        3. HYPOTHESIZE: propose equation
        4. FIT: optionally fit parameters
        5. EVALUATE: compute train/test MSE
        6. REFLECT: analyze results
        7. CURATE: update playbook

        Args:
            round_num: Current round number.

        Returns:
            RoundResult with all outputs and metrics.
        """
        rr = RoundResult(round_num=round_num)

        # 1. CHOOSE inputs
        try:
            chosen = self.strategy.choose_inputs(
                self.env, self.state, round_num, self.llm)
            chosen = np.asarray(chosen, dtype=np.float64)
            if chosen.ndim == 1:
                chosen = chosen.reshape(1, -1)
        except Exception as e:
            logger.warning(f"Round {round_num} CHOOSE failed: {e}, using random")
            chosen = self.env.sample_inputs(self.points_per_round)

        # 2. OBSERVE
        try:
            inputs, outputs = self.env.choose_inputs(chosen)
            rr.chosen_inputs = inputs
            rr.observed_outputs = outputs
            self.state.add_observations(inputs, outputs)
        except Exception as e:
            logger.warning(f"Round {round_num} OBSERVE failed: {e}")
            rr.parse_error = str(e)
            self._record_history(rr, round_num)
            return rr

        # 3. HYPOTHESIZE
        try:
            expr_raw = self.strategy.hypothesize(
                self.env, self.state, round_num, self.llm)
            rr.expression_raw = expr_raw
            rr.expression_clean = clean_expression(expr_raw)
        except Exception as e:
            logger.warning(f"Round {round_num} HYPOTHESIZE failed: {e}")
            rr.parse_error = str(e)
            self._record_history(rr, round_num)
            return rr

        # 4. FIT (optional, depends on condition)
        predict_fn, parse_err = parse_expression(
            rr.expression_clean, self.env.input_names)
        if predict_fn is None:
            rr.parse_error = parse_err
            logger.warning(f"Round {round_num} parse failed: {parse_err}")
            self._record_history(rr, round_num)
            return rr

        # Try condition-specific fitting
        try:
            fitted_fn = self.strategy.fit(
                rr.expression_clean, self.state.all_inputs,
                self.state.all_outputs, self.env)
            if fitted_fn is not None:
                predict_fn = fitted_fn
        except Exception as e:
            logger.warning(f"Round {round_num} FIT failed: {e}, using base parse")

        # 5. EVALUATE
        try:
            train_pred = predict_fn(self.state.all_inputs)
            if not np.all(np.isfinite(train_pred)):
                rr.parse_error = "Prediction contains NaN/Inf"
                logger.warning(f"Round {round_num} EVALUATE: non-finite predictions")
                self._record_history(rr, round_num)
                return rr
            rr.train_mse = float(np.mean(
                (train_pred - self.state.all_outputs) ** 2))
            rr.test_mse = self.env.test_mse(predict_fn)
        except Exception as e:
            logger.warning(f"Round {round_num} EVALUATE failed: {e}")
            rr.parse_error = str(e)
            self._record_history(rr, round_num)
            return rr

        # Update best
        if rr.test_mse < self.state.best_test_mse:
            self.state.best_test_mse = rr.test_mse
            self.state.best_expression = rr.expression_clean
            self.state.best_round = round_num

        # 6. REFLECT
        try:
            rr.reflection = self.strategy.reflect(
                self.env, self.state, rr, self.llm)
        except Exception as e:
            logger.warning(f"Round {round_num} REFLECT failed: {e}")

        # 7. CURATE
        try:
            self.state.playbook = self.strategy.curate(
                self.state, round_num, self.llm)
        except Exception as e:
            logger.warning(f"Round {round_num} CURATE failed: {e}")

        self._record_history(rr, round_num)
        return rr

    def _record_history(self, rr: RoundResult, round_num: int):
        """Record this round in the loop's history."""
        self.state.history.append({
            "round": round_num,
            "expression": rr.expression_clean,
            "train_mse": rr.train_mse,
            "test_mse": rr.test_mse,
        })

    def get_summary(self) -> Dict:
        """Return summary statistics of the run.

        Returns:
            Dictionary with best MSE, expression, round, final MSE, and MSE curve.
        """
        mses = [r.test_mse for r in self.results if r.test_mse < float("inf")]
        return {
            "env_name": self.env.name,
            "total_rounds": len(self.results),
            "best_test_mse": self.state.best_test_mse,
            "best_expression": self.state.best_expression,
            "best_round": self.state.best_round,
            "final_test_mse": mses[-1] if mses else float("inf"),
            "total_cost": self.state.total_cost,
            "mse_curve": [r.test_mse for r in self.results],
        }

    def save_results(self, output_path: str):
        """Save results to a JSON file.

        Args:
            output_path: Path to save results.
        """
        summary = self.get_summary()
        summary["results"] = [
            {
                "round": r.round_num,
                "expression_raw": r.expression_raw,
                "expression_clean": r.expression_clean,
                "train_mse": r.train_mse,
                "test_mse": r.test_mse,
                "reflection": r.reflection,
                "parse_error": r.parse_error,
                "wall_time": r.wall_time,
            }
            for r in self.results
        ]
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load state from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint JSON.

        Returns:
            Round number to resume from (i.e., next round after checkpoint).
        """
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)

        # Reconstruct state from checkpoint
        if "all_inputs" in ckpt and ckpt["all_inputs"] is not None:
            self.state.all_inputs = np.array(ckpt["all_inputs"], dtype=np.float64)
            self.state.all_outputs = np.array(ckpt["all_outputs"], dtype=np.float64)
        self.state.best_test_mse = ckpt.get("best_test_mse", float("inf"))
        self.state.best_expression = ckpt.get("best_expression")
        self.state.best_round = ckpt.get("best_round", -1)
        self.state.playbook = ckpt.get("playbook", "")
        self.state.total_cost = ckpt.get("total_cost", 0.0)
        self.state.history = ckpt.get("history", [])

        # Reconstruct results
        for r_dict in ckpt.get("results", []):
            rr = RoundResult(
                round_num=r_dict["round"],
                expression_raw=r_dict.get("expression_raw"),
                expression_clean=r_dict.get("expression_clean"),
                train_mse=r_dict.get("train_mse", float("inf")),
                test_mse=r_dict.get("test_mse", float("inf")),
                reflection=r_dict.get("reflection"),
                parse_error=r_dict.get("parse_error"),
                wall_time=r_dict.get("wall_time", 0.0),
            )
            self.results.append(rr)

        last_round = ckpt.get("total_rounds", 0)
        logger.info(f"Loaded checkpoint: {last_round} rounds completed, "
                    f"best test MSE = {self.state.best_test_mse:.6f}")
        return last_round + 1

    def save_checkpoint(self, checkpoint_path: str):
        """Save current state to a checkpoint file.

        Args:
            checkpoint_path: Path to save checkpoint.
        """
        ckpt = {
            "all_inputs": self.state.all_inputs.tolist() if self.state.all_inputs is not None else None,
            "all_outputs": self.state.all_outputs.tolist() if self.state.all_outputs is not None else None,
            "best_test_mse": self.state.best_test_mse,
            "best_expression": self.state.best_expression,
            "best_round": self.state.best_round,
            "playbook": self.state.playbook,
            "total_cost": self.state.total_cost,
            "history": self.state.history,
            "total_rounds": len(self.results),
            "results": [
                {
                    "round": r.round_num,
                    "expression_raw": r.expression_raw,
                    "expression_clean": r.expression_clean,
                    "train_mse": r.train_mse,
                    "test_mse": r.test_mse,
                    "reflection": r.reflection,
                    "parse_error": r.parse_error,
                    "wall_time": r.wall_time,
                }
                for r in self.results
            ],
        }
        with open(checkpoint_path, "w") as f:
            json.dump(ckpt, f, indent=2)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
