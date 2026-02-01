"""Analysis and evaluation utilities for grounded discovery experiments.

Modules:
- statistics: Paired t-tests, summary tables, effect size calculations
- plots: Learning curves, comparison visualizations
- symbolic: Symbolic recovery checking with SymPy
- levin: Extended analysis probes (strategy coding, intervention, transfer)
"""

from . import statistics, plots, symbolic, levin

__all__ = [
    "statistics",
    "plots",
    "symbolic",
    "levin",
]
