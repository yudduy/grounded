"""DrSR adapter for ACE framework.

Adapts ACE's Reflector and Curator to use P/N/I (Positive/Negative/Invalid)
sections instead of the standard helpful/harmful bullet tagging.
"""
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def categorize_equation(expression: str, test_mse: float, prev_best_mse: float,
                        parse_error: Optional[str] = None) -> Tuple[str, str]:
    """Categorize an equation into P/N/I.

    Args:
        expression: the equation string
        test_mse: test MSE achieved
        prev_best_mse: previous best MSE
        parse_error: if not None, equation is Invalid
    Returns:
        (category, formatted_entry) tuple
    """
    if parse_error:
        return "INVALID", f"{expression} (error: {parse_error})"
    if test_mse < prev_best_mse:
        return "POSITIVE", f"{expression} (MSE={test_mse:.6f})"
    return "NEGATIVE", f"{expression} (MSE={test_mse:.6f})"


def build_pni_playbook(positive: List[str], negative: List[str],
                       invalid: List[str], max_per_section: int = 20) -> str:
    """Build a P/N/I formatted playbook string."""
    sections = []
    sections.append("## POSITIVE (equations that improved MSE)")
    for entry in positive[-max_per_section:]:
        sections.append(f"  {entry}")
    sections.append("")
    sections.append("## NEGATIVE (equations that worsened MSE)")
    for entry in negative[-max_per_section:]:
        sections.append(f"  {entry}")
    sections.append("")
    sections.append("## INVALID (equations that failed to parse/evaluate)")
    for entry in invalid[-max_per_section:]:
        sections.append(f"  {entry}")
    return "\n".join(sections)
