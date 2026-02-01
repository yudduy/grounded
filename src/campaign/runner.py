"""Campaign runner for 162 experimental runs.

Orchestrates all environment x condition x seed combinations.
Supports checkpoint/resume from SQLite.
"""
import sys
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))
sys.path.insert(0, str(_repo_root / "src"))

from shinka.llm.llm import LLMClient
from environments import ALL_ENVIRONMENTS
from loop.orchestrator import DiscoveryLoop
from conditions.static import StaticCondition
from conditions.ace_condition import ACECondition
from conditions.gradient import GradientCondition
from conditions.ace_gradient import ACEGradientCondition
from conditions.drsr import DrSRCondition
from conditions.pysr_baseline import PySRCondition
from conditions.random_search import RandomSearchCondition
from campaign.config import CampaignConfig, RunConfig

logger = logging.getLogger(__name__)

ENV_MAP = {cls.__name__: cls for cls in ALL_ENVIRONMENTS}

CONDITION_MAP = {
    "A": StaticCondition,
    "B": ACECondition,
    "C": GradientCondition,
    "D": ACEGradientCondition,
    "E": DrSRCondition,
    "F": PySRCondition,
    "R": RandomSearchCondition,
}


class CampaignRunner:
    """Orchestrates all experimental runs with checkpoint/resume."""

    def __init__(self, config: CampaignConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.total_cost = 0.0

    def _init_db(self):
        """Initialize SQLite database for results tracking."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                env_name TEXT,
                condition TEXT,
                seed INTEGER,
                status TEXT DEFAULT 'pending',
                best_test_mse REAL,
                best_expression TEXT,
                best_round INTEGER,
                final_test_mse REAL,
                total_cost REAL DEFAULT 0,
                mse_curve TEXT,
                started_at TEXT,
                completed_at TEXT,
                PRIMARY KEY (env_name, condition, seed)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS round_results (
                env_name TEXT,
                condition TEXT,
                seed INTEGER,
                round_num INTEGER,
                expression TEXT,
                train_mse REAL,
                test_mse REAL,
                reflection TEXT,
                wall_time REAL,
                PRIMARY KEY (env_name, condition, seed, round_num)
            )
        """)
        conn.commit()
        conn.close()

    def _is_completed(self, run: RunConfig) -> bool:
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT status FROM runs WHERE env_name=? AND condition=? AND seed=?",
            (run.env_name, run.condition, run.seed)
        ).fetchone()
        conn.close()
        return row is not None and row[0] == "completed"

    def _save_run_result(self, run: RunConfig, summary: Dict):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT OR REPLACE INTO runs
            (env_name, condition, seed, status, best_test_mse, best_expression,
             best_round, final_test_mse, total_cost, mse_curve, completed_at)
            VALUES (?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            run.env_name, run.condition, run.seed,
            summary.get("best_test_mse"),
            summary.get("best_expression"),
            summary.get("best_round"),
            summary.get("final_test_mse"),
            summary.get("total_cost", 0),
            json.dumps(summary.get("mse_curve", [])),
        ))
        conn.commit()
        conn.close()

    def _make_strategy(self, run: RunConfig):
        """Create the condition strategy for a run."""
        cond_class = CONDITION_MAP[run.condition]
        kwargs = {
            "points_per_round": run.points_per_round,
            "total_rounds": run.total_rounds,
        }
        if run.condition == "R":
            kwargs["seed"] = run.seed
        return cond_class(**kwargs)

    def run_single(self, run: RunConfig, llm: LLMClient) -> Dict:
        """Execute a single experimental run."""
        logger.info(f"Starting: {run.env_name} / {run.condition} / seed={run.seed}")

        env_cls = ENV_MAP[run.env_name]
        env = env_cls(seed=run.seed)
        strategy = self._make_strategy(run)

        loop = DiscoveryLoop(
            env=env, llm=llm, strategy=strategy,
            total_rounds=run.total_rounds,
            points_per_round=run.points_per_round,
            seed=run.seed,
        )

        results = loop.run()
        summary = loop.get_summary()
        self._save_run_result(run, summary)

        logger.info(f"Completed: {run.env_name}/{run.condition}/seed={run.seed} "
                     f"best_mse={summary['best_test_mse']:.6f}")
        return summary

    def run_all(self, llm: LLMClient):
        """Run all configured experiments, skipping completed ones."""
        runs = self.config.generate_runs()
        completed = 0
        skipped = 0

        for i, run in enumerate(runs):
            if self._is_completed(run):
                skipped += 1
                continue

            if self.total_cost >= self.config.budget_total:
                logger.warning(f"Budget exceeded (${self.total_cost:.2f}), stopping")
                break

            try:
                summary = self.run_single(run, llm)
                self.total_cost += summary.get("total_cost", 0)
                completed += 1
            except Exception as e:
                logger.error(f"Run failed: {run.env_name}/{run.condition}/seed={run.seed}: {e}")

            logger.info(f"Progress: {completed + skipped}/{len(runs)} "
                         f"(completed={completed}, skipped={skipped})")

        logger.info(f"Campaign finished: {completed} new, {skipped} skipped, "
                     f"total_cost=${self.total_cost:.2f}")

    def get_results_table(self) -> List[Dict]:
        """Get all results as a list of dicts."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM runs ORDER BY env_name, condition, seed").fetchall()
        conn.close()
        return [dict(r) for r in rows]
