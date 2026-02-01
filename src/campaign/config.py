"""Configuration for the experiment campaign."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class RunConfig:
    env_name: str
    condition: str  # "A", "B", "C", "D", "E", "F", "R"
    seed: int
    total_rounds: int = 100
    points_per_round: int = 5
    # TEMP: Using Hyperbolic GPT-OSS 20B for PoC (~$6 for full campaign)
    # It's a reasoning model — expression_parser extracts equations from chain-of-thought.
    # Production: switch to "gpt-4o-mini" or "gpt-4.1-nano"
    llm_model: str = "hyperbolic/openai/gpt-oss-20b"
    budget_per_run: float = 1.0  # USD (TEMP: lowered for cheap PoC model)


@dataclass
class CampaignConfig:
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
    # TEMP: Using Hyperbolic GPT-OSS 20B for PoC (~$6 for full campaign)
    # Production: switch back to "gpt-4o-mini" or "gpt-4.1-nano"
    llm_model: str = "hyperbolic/openai/gpt-oss-20b"
    budget_total: float = 20.0  # TEMP: lowered for cheap PoC model
    budget_per_run: float = 1.0  # TEMP: lowered for cheap PoC model
    results_dir: str = "results"
    db_path: str = "results/campaign.sqlite"

    def generate_runs(self) -> List[RunConfig]:
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
