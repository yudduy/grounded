"""Kimi-based curator for playbook evolution."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .kimi_client import kimi_chat, REFLECT_MODEL


@dataclass
class CurateOp:
    """A single curation operation."""
    op: str  # ADD, UPDATE, DELETE, MERGE
    section: str = "STRATEGIES"
    content: str = ""
    target_id: str = ""  # for UPDATE/DELETE
    merge_ids: List[str] = None  # for MERGE

    def __post_init__(self):
        if self.merge_ids is None:
            self.merge_ids = []


def build_curate_system(max_bullets: int, current_size: int) -> str:
    """Build curator system prompt."""
    return (
        "You are a playbook curator for math competition solving. Based on the reflection, "
        "propose operations to improve the playbook.\n"
        "Output a JSON array of operations:\n"
        '[{"op": "ADD", "section": "STRATEGIES", "content": "new insight"},\n'
        ' {"op": "UPDATE", "id": "str-00001", "content": "refined text"},\n'
        ' {"op": "DELETE", "id": "err-00002"}]\n'
        f"Sections: STRATEGIES, COMMON_MISTAKES, SOLUTION_PATTERNS\n"
        f"Max bullets: {max_bullets}. Current: {current_size}.\n"
        "Only propose operations clearly supported by the reflection. Keep it minimal."
    )


def curate(
    playbook,  # Playbook instance
    problem: str,
    reflection: str,
    max_bullets: int = 20,
    client=None,
) -> List[CurateOp]:
    """Curate playbook based on reflection.

    Returns list of CurateOp to apply.
    """
    system = build_curate_system(max_bullets, playbook.size)
    user_msg = (
        f"Question: {problem}\n"
        f"Current playbook:\n{playbook.to_str()}\n\n"
        f"Reflection:\n{reflection}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = kimi_chat(messages, model=REFLECT_MODEL, temperature=0.4, max_tokens=512, client=client)
        return parse_curate_ops(raw)
    except Exception as e:
        print(f"  Curator API failed: {e}")
        return []


def parse_curate_ops(raw: str) -> List[CurateOp]:
    """Parse curator response into operations."""
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        return []
    try:
        ops_data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    ops = []
    for item in ops_data:
        try:
            op_type = item.get("op", "").upper()
            if op_type == "ADD":
                ops.append(CurateOp(
                    op="ADD",
                    section=item.get("section", "STRATEGIES"),
                    content=item.get("content", ""),
                ))
            elif op_type == "UPDATE":
                ops.append(CurateOp(
                    op="UPDATE",
                    target_id=item.get("id", ""),
                    content=item.get("content", ""),
                ))
            elif op_type == "DELETE":
                ops.append(CurateOp(
                    op="DELETE",
                    target_id=item.get("id", ""),
                ))
            elif op_type == "MERGE":
                ops.append(CurateOp(
                    op="MERGE",
                    merge_ids=item.get("ids", []),
                    content=item.get("content", ""),
                    section=item.get("section", "STRATEGIES"),
                ))
        except Exception:
            continue
    return ops


def rule_based_curate(playbook, pending_items: List[Dict], max_bullets: int = 20):
    """Rule-based fallback curator (no API needed).

    Ported from run_modal.py:1193-1213.
    Modifies playbook in-place.
    """
    # Remove bullets where harmful > helpful + 2
    to_remove = [b.id for b in playbook.bullets if b.harmful > b.helpful + 2]
    for bid in to_remove:
        playbook.remove(bid)

    # Tag helpful bullets from correct solutions
    for item in pending_items:
        if item.get("is_correct", False):
            candidates = item.get("candidates", [])
            if candidates:
                best = max(candidates, key=lambda c: len(c.get("bullets_used", [])))
                for bid in best.get("bullets_used", []):
                    playbook.tag(bid, "helpful")

    # Prune to max size
    while playbook.size > max_bullets:
        worst = min(
            playbook.bullets,
            key=lambda b: b.helpful / (b.helpful + b.harmful + 1),
        )
        playbook.remove(worst.id)

    # If empty, reset to initial
    if playbook.size == 0:
        old_next = playbook._next_id
        from .playbook import make_initial_playbook
        new_pb = make_initial_playbook()
        new_pb._next_id = old_next
        playbook.bullets = new_pb.bullets
