"""L3: Transfer probe for Levin extension.

Test whether playbook learned on E1 transfers to E2.
Compare: static playbook, trained playbook, random playbook, ablated playbook.
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def prepare_transfer_conditions(trained_playbook: str,
                                ablated_playbook: str) -> Dict[str, str]:
    """Prepare playbook variants for transfer experiment.

    Args:
        trained_playbook: playbook trained on source environment
        ablated_playbook: playbook with domain-specific bullets removed
    Returns:
        Dict mapping condition name to playbook string
    """
    return {
        "static": "",  # No playbook (Condition A equivalent)
        "trained": trained_playbook,  # Full trained playbook from E1
        "random": _shuffle_bullets(trained_playbook),  # Random reassignment
        "ablated": ablated_playbook,  # Domain-specific bullets removed
    }


def _shuffle_bullets(playbook: str) -> str:
    """Randomly reassign bullets across sections."""
    import random
    lines = playbook.split("\n")
    sections = []
    bullets = []
    for line in lines:
        if line.strip().startswith("##"):
            sections.append(line)
        elif line.strip().startswith("[") or line.strip().startswith("-"):
            bullets.append(line)

    random.shuffle(bullets)

    result = []
    bullet_idx = 0
    for line in lines:
        if line.strip().startswith("##"):
            result.append(line)
        elif line.strip().startswith("[") or line.strip().startswith("-"):
            if bullet_idx < len(bullets):
                result.append(bullets[bullet_idx])
                bullet_idx += 1
        else:
            result.append(line)
    return "\n".join(result)


def compare_transfer_results(results: Dict[str, Dict]) -> Dict:
    """Compare transfer experiment results.

    Args:
        results: {condition_name: {best_test_mse, mse_curve, ...}}
    Returns:
        Comparison summary
    """
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "best_mse": r.get("best_test_mse", float("inf")),
            "final_mse": r.get("final_test_mse", float("inf")),
        }

    # Check if trained playbook transfers (better than static)
    if "static" in summary and "trained" in summary:
        summary["transfer_effective"] = (
            summary["trained"]["best_mse"] < summary["static"]["best_mse"]
        )
    return summary
