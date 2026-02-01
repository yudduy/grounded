"""L2: Intervention experiment for Levin extension.

Remove zero-helpful bullets at round 50 and measure impact on
subsequent discovery performance.
"""
import sys
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def identify_zero_helpful_bullets(playbook: str, stats: Dict) -> List[str]:
    """Find playbook bullet IDs with helpful=0.

    Args:
        playbook: current playbook string
        stats: {bullet_id: {"helpful": int, "harmful": int}}
    Returns:
        List of bullet IDs to remove
    """
    zero_helpful = []
    for bid, counts in stats.items():
        if counts.get("helpful", 0) == 0 and counts.get("harmful", 0) > 0:
            zero_helpful.append(bid)
    return zero_helpful


def remove_bullets(playbook: str, bullet_ids: List[str]) -> str:
    """Remove specified bullets from playbook.

    Args:
        playbook: current playbook string
        bullet_ids: list of bullet IDs to remove
    Returns:
        Modified playbook string
    """
    lines = playbook.split("\n")
    filtered = []
    for line in lines:
        skip = False
        for bid in bullet_ids:
            if f"[{bid}]" in line:
                skip = True
                break
        if not skip:
            filtered.append(line)
    return "\n".join(filtered)


def measure_degradation(mse_curve_before: List[float],
                        mse_curve_after: List[float],
                        window: int = 10) -> Dict:
    """Measure performance degradation after intervention.

    Compares average MSE in a window before and after intervention.

    Args:
        mse_curve_before: MSE values before intervention (last `window` rounds)
        mse_curve_after: MSE values after intervention (first `window` rounds)
        window: number of rounds to compare
    Returns:
        Dict with degradation metrics
    """
    before = np.array(mse_curve_before[-window:])
    after = np.array(mse_curve_after[:window])

    return {
        "mean_before": float(np.mean(before)),
        "mean_after": float(np.mean(after)),
        "relative_change": float((np.mean(after) - np.mean(before)) / max(np.mean(before), 1e-10)),
        "degraded": float(np.mean(after)) > float(np.mean(before)),
    }
