"""Levin extension analysis suite.

Three probes for understanding how ACE playbooks evolve and transfer:
- L1: Strategy profile coding (strategy_coder.py)
- L2: Intervention experiment (intervention.py)
- L3: Transfer probe (transfer.py)
"""

from .strategy_coder import classify_bullet, code_playbook, strategy_profile
from .intervention import identify_zero_helpful_bullets, remove_bullets, measure_degradation
from .transfer import prepare_transfer_conditions, compare_transfer_results

__all__ = [
    "classify_bullet",
    "code_playbook",
    "strategy_profile",
    "identify_zero_helpful_bullets",
    "remove_bullets",
    "measure_degradation",
    "prepare_transfer_conditions",
    "compare_transfer_results",
]
