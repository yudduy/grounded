"""Configuration for the experiment campaign.

Defines all experimental conditions, environments, and run parameters.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class RunConfig:
    """Configuration for a single experimental run."""
    env_name: str
    condition: str  # "A", "B", "C", "D", "E", "F", "R"
    seed: int
    total_rounds: int = 100
    points_per_round: int = 5
    llm_model: str = "gpt-4o-mini"
    budget_per_run: float = 10.0  # USD


@dataclass
class CampaignConfig:
    """Configuration for the full experiment campaign."""
    environments: List[str] = field(default_factory=lambda: [
        "ExponentialDampedGravity",
        "AsymmetricDrag",
        "NonReciprocalSpring",
        "VelocityDependentMass",
        "CoupledNonlinearDamping",
        "FractionalDrag",
        "NonPolynomialConserved",
        "CrossCoupledDynamics",
        "HistoryDependentForce",
    ])
    conditions: List[str] = field(default_factory=lambda: [
        "A", "B", "C", "D", "E", "F",
    ])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    total_rounds: int = 100
    points_per_round: int = 5
    llm_model: str = "gpt-4o-mini"
    budget_total: float = 500.0
    budget_per_run: float = 10.0
    results_dir: str = "results"
    db_path: str = "results/campaign.sqlite"

    def generate_runs(self) -> List[RunConfig]:
        """Generate all run configurations."""
        runs = []
        for env_name in self.environments:
            for cond in self.conditions:
                for seed in self.seeds:
                    runs.append(RunConfig(
                        env_name=env_name,
                        condition=cond,
                        seed=seed,
                        total_rounds=self.total_rounds,
                        points_per_round=self.points_per_round,
                        llm_model=self.llm_model,
                        budget_per_run=self.budget_per_run,
                    ))
        return runs

    @property
    def total_runs(self) -> int:
        return len(self.environments) * len(self.conditions) * len(self.seeds)
