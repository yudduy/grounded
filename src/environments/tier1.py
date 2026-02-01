"""Tier 1 environments: Single system changes affecting one equation."""
import numpy as np
from .base import BaseEnvironment, EnvironmentSpec


class ModifiedGravityCoupling(BaseEnvironment):
    """
    Modified gravity coupling: y = -0.5*g*t^2 + 0.1*y_offset*t

    Introduces polynomial nonlinearity to single gravitational system.
    Ground truth: output = -4.9*t^2 + 0.1*y_offset*t
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="ModifiedGravityCoupling",
            description="Gravity with state-dependent correction term",
            input_names=["time", "y_offset"],
            input_ranges={"time": (0.0, 5.0), "y_offset": (-10.0, 10.0)},
            ground_truth_expr="y = -4.9*t^2 + 0.1*y_offset*t",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        time = inputs[:, 0]
        y_offset = inputs[:, 1]
        return -4.9 * time ** 2 + 0.1 * y_offset * time


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


class AnharmonicOscillator(BaseEnvironment):
    """
    Anharmonic oscillator with cubic restoring force nonlinearity.

    Ground truth: force = -position - 0.3*position^3
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="AnharmonicOscillator",
            description="Oscillator with cubic restoring force",
            input_names=["position", "velocity"],
            input_ranges={"position": (-4.0, 4.0), "velocity": (-4.0, 4.0)},
            ground_truth_expr="force = -position - 0.3*position^3",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        position = inputs[:, 0]
        velocity = inputs[:, 1]
        return -position - 0.3 * position ** 3


class ModifiedProjectile(BaseEnvironment):
    """
    Projectile motion with quadratic air resistance.

    Ground truth: force_magnitude = 0.1 * velocity_magnitude
    Applied separately to x and y directions.
    """

    def _make_spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="ModifiedProjectile",
            description="2D projectile with velocity-magnitude-dependent drag",
            input_names=["velocity_x", "velocity_y"],
            input_ranges={"velocity_x": (-10.0, 10.0), "velocity_y": (-10.0, 10.0)},
            ground_truth_expr="drag = 0.1*sqrt(vx^2 + vy^2); force_x = -drag*vx; force_y = -9.8 - drag*vy",
            tier=1,
            noise_sigma_rel=0.01,
        )

    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        vx = inputs[:, 0]
        vy = inputs[:, 1]
        v_mag = np.sqrt(vx ** 2 + vy ** 2)
        drag_coeff = 0.1
        # Return force_y (more interesting than force_x)
        return -9.8 - drag_coeff * vy * v_mag
