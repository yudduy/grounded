"""Dynamical systems environments for equation discovery."""
from .base import BaseEnvironment
from .tier1 import (
    ModifiedGravityCoupling,
    AsymmetricDrag,
    NonReciprocalSpring,
    VelocityDependentMass,
    AnharmonicOscillator,
    ModifiedProjectile,
)
from .tier2 import (
    NonPolynomialConserved,
    CrossCoupledDynamics,
    HistoryDependentForce,
)

__all__ = [
    "BaseEnvironment",
    "ModifiedGravityCoupling",
    "AsymmetricDrag",
    "NonReciprocalSpring",
    "VelocityDependentMass",
    "AnharmonicOscillator",
    "ModifiedProjectile",
    "NonPolynomialConserved",
    "CrossCoupledDynamics",
    "HistoryDependentForce",
]

ALL_ENVIRONMENTS = [
    ModifiedGravityCoupling,
    AsymmetricDrag,
    NonReciprocalSpring,
    VelocityDependentMass,
    AnharmonicOscillator,
    ModifiedProjectile,
    NonPolynomialConserved,
    CrossCoupledDynamics,
    HistoryDependentForce,
]
