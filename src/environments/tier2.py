"""Tier 2 environments: Multi-system interactions requiring integrated knowledge."""
import numpy as np
from .base import BaseEnvironment, EnvironmentSpec


class NonPolynomialConserved(BaseEnvironment):
    """
    Coupled system with transcendental nonlinearity and implicit conservation.

    Ground truth involves sin/cos cross-coupling that requires symbolic reasoning.
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="NonPolynomialConserved",
            description="Transcendental coupling with implicit conservation law",
            input_names=["x", "y"],
            input_ranges={"x": (-np.pi, np.pi), "y": (-np.pi, np.pi)},
            ground_truth_expr="output = x*sin(y) - y*cos(x)",
            tier=2,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs[:, 0]
        y = inputs[:, 1]
        return x * np.sin(y) - y * np.cos(x)


class CrossCoupledDynamics(BaseEnvironment):
    """
    Predator-prey-like system with multiplicative state interaction.

    Ground truth: output = x*(1 - x - 0.5*y) + y*(-0.3 + 0.4*x)*(1 + 0.1*x*y)

    Requires understanding how x and y interact multiplicatively.
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="CrossCoupledDynamics",
            description="Nonlinear population dynamics with cross-coupling",
            input_names=["x", "y"],
            input_ranges={"x": (0.1, 2.0), "y": (0.1, 2.0)},
            ground_truth_expr="f = x*(1-x-0.5*y) + y*(-0.3+0.4*x)*(1+0.1*x*y)",
            tier=2,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs[:, 0]
        y = inputs[:, 1]
        term1 = x * (1.0 - x - 0.5 * y)
        term2 = y * (-0.3 + 0.4 * x) * (1.0 + 0.1 * x * y)
        return term1 + term2


class HistoryDependentForce(BaseEnvironment):
    """
    Three-variable system with implicit temporal coupling.

    Ground truth involves x, y, z interactions that require integrated understanding.
    output = x*sin(y) + 0.5*z*cos(x) - 0.3*y*z

    Demonstrates how intermediate variables couple final output.
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="HistoryDependentForce",
            description="Three-variable system with implicit temporal coupling",
            input_names=["x", "y", "z"],
            input_ranges={"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (-3.0, 3.0)},
            ground_truth_expr="f = x*sin(y) + 0.5*z*cos(x) - 0.3*y*z",
            tier=2,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs[:, 0]
        y = inputs[:, 1]
        z = inputs[:, 2]
        return x * np.sin(y) + 0.5 * z * np.cos(x) - 0.3 * y * z
