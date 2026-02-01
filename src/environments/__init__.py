"""Dynamical systems environments for equation discovery."""
from .base import BaseEnvironment
from .tier1 import (
    ExponentialDampedGravity,
    AsymmetricDrag,
    NonReciprocalSpring,
    VelocityDependentMass,
    CoupledNonlinearDamping,
    FractionalDrag,
)
from .tier2 import (
    NonPolynomialConserved,
    CrossCoupledDynamics,
    HistoryDependentForce,
)

__all__ = [
    "BaseEnvironment",
    "ExponentialDampedGravity",
    "AsymmetricDrag",
    "NonReciprocalSpring",
    "VelocityDependentMass",
    "CoupledNonlinearDamping",
    "FractionalDrag",
    "NonPolynomialConserved",
    "CrossCoupledDynamics",
    "HistoryDependentForce",
]

ALL_ENVIRONMENTS = [
    ExponentialDampedGravity,
    AsymmetricDrag,
    NonReciprocalSpring,
    VelocityDependentMass,
    CoupledNonlinearDamping,
    FractionalDrag,
    NonPolynomialConserved,
    CrossCoupledDynamics,
    HistoryDependentForce,
]
