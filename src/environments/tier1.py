"""Tier 1 environments: Single system changes affecting one equation."""
import numpy as np
from .base import BaseEnvironment, EnvironmentSpec


class ExponentialDampedGravity(BaseEnvironment):
    """
    Gravity with exponential state-dependent damping.

    Ground truth: y = -4.9*t^2 * exp(-0.05*|y_offset|)
    Exponential correction harder to guess than linear.
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="ExponentialDampedGravity",
            description="Gravity with exponential state-dependent damping",
            input_names=["time", "y_offset"],
            input_ranges={"time": (0.0, 5.0), "y_offset": (-10.0, 10.0)},
            ground_truth_expr="y = -4.9*t^2 * exp(-0.05*|y_offset|)",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        time = inputs[:, 0]
        y_offset = inputs[:, 1]
        return -4.9 * time ** 2 * np.exp(-0.05 * np.abs(y_offset))


class AsymmetricDrag(BaseEnvironment):
    """
    Quadratic drag with asymmetry.

    Ground truth: force = -velocity - sign(velocity)*velocity^2
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="AsymmetricDrag",
            description="Linear spring with asymmetric quadratic damping",
            input_names=["velocity", "position"],
            input_ranges={"velocity": (-5.0, 5.0), "position": (-5.0, 5.0)},
            ground_truth_expr="force = -position - sign(velocity)*velocity^2",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        velocity = inputs[:, 0]
        position = inputs[:, 1]
        asymmetry = np.sign(velocity) * velocity ** 2
        return -position - asymmetry


class NonReciprocalSpring(BaseEnvironment):
    """
    Non-reciprocal spring coupling with multiplicative interaction.

    Ground truth: force = -position + 0.2*position*velocity
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="NonReciprocalSpring",
            description="Spring force with state-dependent coupling",
            input_names=["position", "velocity"],
            input_ranges={"position": (-5.0, 5.0), "velocity": (-5.0, 5.0)},
            ground_truth_expr="force = -position + 0.2*position*velocity",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        position = inputs[:, 0]
        velocity = inputs[:, 1]
        return -position + 0.2 * position * velocity


class VelocityDependentMass(BaseEnvironment):
    """
    Oscillator where effective mass changes with velocity.

    Ground truth: force = -position / (1 + 0.1*velocity^2)
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="VelocityDependentMass",
            description="Spring force with velocity-dependent effective mass",
            input_names=["position", "velocity"],
            input_ranges={"position": (-5.0, 5.0), "velocity": (-5.0, 5.0)},
            ground_truth_expr="force = -position / (1 + 0.1*velocity^2)",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        position = inputs[:, 0]
        velocity = inputs[:, 1]
        return -position / (1.0 + 0.1 * velocity ** 2)


class CoupledNonlinearDamping(BaseEnvironment):
    """
    Nonlinear damping with position-velocity coupling.

    Ground truth: force = -position * (1 + 0.15*velocity*position)
    No direct textbook analog — coupling between damping and displacement.
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="CoupledNonlinearDamping",
            description="Oscillator with nonlinear position-velocity damping coupling",
            input_names=["position", "velocity"],
            input_ranges={"position": (-4.0, 4.0), "velocity": (-4.0, 4.0)},
            ground_truth_expr="force = -position * (1 + 0.15*velocity*position)",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        position = inputs[:, 0]
        velocity = inputs[:, 1]
        return -position * (1.0 + 0.15 * velocity * position)


class FractionalDrag(BaseEnvironment):
    """
    Projectile with non-standard fractional drag exponent.

    Ground truth: force_y = -9.8 - 0.15 * |v|^1.5 * sign(vy)
    Exponent 1.5 is between laminar (1) and turbulent (2) — very rare.
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="FractionalDrag",
            description="2D projectile with fractional-exponent drag",
            input_names=["velocity_x", "velocity_y"],
            input_ranges={"velocity_x": (-10.0, 10.0), "velocity_y": (-10.0, 10.0)},
            ground_truth_expr="force_y = -9.8 - 0.15 * (vx^2+vy^2)^0.75 * sign(vy)",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        vx = inputs[:, 0]
        vy = inputs[:, 1]
        v_mag = np.sqrt(vx ** 2 + vy ** 2)
        return -9.8 - 0.15 * (v_mag ** 1.5) * np.sign(vy)
