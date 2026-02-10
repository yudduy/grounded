"""Kimi-based reflector for ACE-style bullet tagging."""

import json
import re
from typing import Dict, List, Tuple, Optional

from .kimi_client import kimi_chat, REFLECT_MODEL

REFLECT_SYSTEM = (
    "You are a math reasoning analyst. Analyze the solution and whether playbook strategies helped.\n"
    'For each bullet ID used, output a JSON line: {"id": "str-00001", "tag": "helpful"}\n'
    "Tags: helpful, harmful, neutral.\n"
    "End with a brief reflection about what mathematical insight was key."
)


def reflect(
    problem: str,
    solution: str,
    is_correct: bool,
    bullets_used: List[str],
    playbook_text: str,
    client=None,
) -> Tuple[str, Dict[str, str]]:
    """Reflect on a solution and tag bullets.

    Args:
        problem: The math problem text
        solution: The model's solution (may be truncated)
        is_correct: Whether the answer was correct
        bullets_used: List of bullet IDs referenced in solution
        playbook_text: Current playbook as string
        client: Optional pre-created Kimi client

    Returns:
        (reflection_text, {bullet_id: tag}) where tag is helpful/harmful/neutral
    """
    feedback = "CORRECT" if is_correct else "INCORRECT"

    # Build bullets text for context
    bullets_context = ""
    if bullets_used:
        # Extract relevant bullet lines from playbook
        for line in playbook_text.split("\n"):
            for bid in bullets_used:
                if bid in line:
                    bullets_context += f"  {line.strip()}\n"
    if not bullets_context:
        bullets_context = "  (none referenced)"

    user_msg = (
        f"Problem: {problem}\n\n"
        f"Solution:\n{solution[:2000]}\n\n"
        f"Result: {feedback}\n\n"
        f"Bullets referenced:\n{bullets_context}"
    )

    messages = [
        {"role": "system", "content": REFLECT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    try:
        reflection = kimi_chat(messages, model=REFLECT_MODEL, temperature=0.3, max_tokens=512, client=client)
    except Exception as e:
        print(f"  Reflector failed: {e}")
        reflection = ""

    # Extract tags
    tags = extract_tags(reflection, bullets_used, is_correct)
    return reflection, tags


def extract_tags(
    reflection: str, bullets_used: List[str], is_correct: bool
) -> Dict[str, str]:
    """Extract bullet tags from reflection text."""
    tags = {}
    for m in re.finditer(
        r'"id"\s*:\s*"([^"]+)".*?"tag"\s*:\s*"(helpful|harmful|neutral)"', reflection
    ):
        bid, tag = m.group(1), m.group(2)
        if bid in bullets_used:
            tags[bid] = tag

    # Fallback: if no tags extracted, use default based on correctness
    if not tags and bullets_used:
        default_tag = "helpful" if is_correct else "harmful"
        for bid in bullets_used:
            tags[bid] = default_tag

    return tags


def batch_reflect(
    items: List[Dict],
    playbook_text: str,
    client=None,
) -> List[Tuple[str, Dict[str, str]]]:
    """Batch reflect on multiple items.

    Args:
        items: List of dicts with keys: problem, solution, is_correct, bullets_used
        playbook_text: Current playbook text
        client: Optional Kimi client

    Returns:
        List of (reflection_text, tags_dict) tuples
    """
    results = []
    for item in items:
        reflection, tags = reflect(
            problem=item["problem"],
            solution=item["solution"],
            is_correct=item["is_correct"],
            bullets_used=item.get("bullets_used", []),
            playbook_text=playbook_text,
            client=client,
        )
        results.append((reflection, tags))
    return results
