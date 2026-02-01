"""L1: Strategy profile coding for Levin extension.

Taxonomizes playbook bullets by type (exploration, structural,
parametric, meta-strategic) for qualitative analysis.
"""
import re
from typing import Dict, List, Tuple


STRATEGY_CATEGORIES = {
    "exploration": ["explore", "try", "test", "probe", "vary", "boundary", "extreme"],
    "structural": ["form", "polynomial", "sin", "cos", "trig", "power", "rational",
                    "linear", "quadratic", "cubic", "exponential"],
    "parametric": ["parameter", "coefficient", "constant", "scale", "shift",
                    "magnitude", "sign", "range"],
    "meta": ["combine", "simplify", "compare", "pattern", "residual",
             "systematic", "improve", "avoid"],
}


def classify_bullet(text: str) -> str:
    """Classify a playbook bullet into a strategy category."""
    text_lower = text.lower()
    scores = {}
    for cat, keywords in STRATEGY_CATEGORIES.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def code_playbook(playbook: str) -> List[Dict]:
    """Parse and classify all bullets in a playbook.

    Args:
        playbook: full playbook string
    Returns:
        List of {id, text, category, section}
    """
    results = []
    current_section = "unknown"
    for line in playbook.split("\n"):
        line = line.strip()
        if line.startswith("##"):
            current_section = line.lstrip("#").strip()
            continue
        # Match bulleted items with optional IDs like [str-00001]
        m = re.match(r'\[([^\]]+)\].*?::\s*(.*)', line)
        if m:
            bid, text = m.group(1), m.group(2)
        elif line.startswith("-") or line.startswith("*"):
            bid = None
            text = line.lstrip("-* ").strip()
        else:
            continue
        if text:
            results.append({
                "id": bid,
                "text": text,
                "category": classify_bullet(text),
                "section": current_section,
            })
    return results


def strategy_profile(playbook: str) -> Dict[str, int]:
    """Compute strategy category distribution."""
    coded = code_playbook(playbook)
    profile = {}
    for item in coded:
        cat = item["category"]
        profile[cat] = profile.get(cat, 0) + 1
    return profile
