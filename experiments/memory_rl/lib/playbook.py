"""Playbook infrastructure for Memory x RL experiment."""

import copy
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Bullet:
    """A single playbook bullet."""
    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0

    def to_str(self) -> str:
        return f"[{self.id}] helpful={self.helpful} harmful={self.harmful} :: {self.content}"


class Playbook:
    """Mutable playbook of strategy bullets."""

    def __init__(self):
        self.bullets: List[Bullet] = []
        self._next_id: int = 1

    def add(self, section: str, content: str) -> str:
        prefix = {
            "STRATEGIES": "str",
            "COMMON_MISTAKES": "err",
            "SOLUTION_PATTERNS": "sol",
        }.get(section, "gen")
        bid = f"{prefix}-{self._next_id:05d}"
        self._next_id += 1
        self.bullets.append(Bullet(id=bid, section=section, content=content))
        return bid

    def remove(self, bid: str):
        self.bullets = [b for b in self.bullets if b.id != bid]

    def update(self, bid: str, content: str):
        for b in self.bullets:
            if b.id == bid:
                b.content = content
                return

    def tag(self, bid: str, label: str):
        for b in self.bullets:
            if b.id == bid:
                if label == "helpful":
                    b.helpful += 1
                elif label == "harmful":
                    b.harmful += 1

    def to_str(self) -> str:
        sections = defaultdict(list)
        for b in self.bullets:
            sections[b.section].append(b.to_str())
        parts = []
        for sec in ["STRATEGIES", "COMMON_MISTAKES", "SOLUTION_PATTERNS"]:
            if sections[sec]:
                parts.append(f"## {sec}")
                parts.extend(sections[sec])
        return "\n".join(parts) if parts else "(empty playbook)"

    def copy(self) -> "Playbook":
        return copy.deepcopy(self)

    @property
    def size(self) -> int:
        return len(self.bullets)

    def snapshot(self) -> Dict:
        return {
            "bullets": [
                {
                    "id": b.id, "section": b.section, "content": b.content,
                    "helpful": b.helpful, "harmful": b.harmful,
                }
                for b in self.bullets
            ],
            "next_id": self._next_id,
        }

    @classmethod
    def from_snapshot(cls, data: Dict) -> "Playbook":
        pb = cls()
        pb._next_id = data.get("next_id", 1)
        for bd in data.get("bullets", []):
            pb.bullets.append(Bullet(**bd))
        return pb

    def prune_harmful(self, threshold: int = 2):
        """Remove bullets where harmful > helpful + threshold."""
        self.bullets = [
            b for b in self.bullets
            if not (b.harmful > b.helpful + threshold)
        ]

    def entropy(self) -> float:
        """Shannon entropy of helpful/harmful ratios across bullets."""
        import math
        if not self.bullets:
            return 0.0
        total = sum(b.helpful + b.harmful for b in self.bullets)
        if total == 0:
            return 0.0
        probs = [(b.helpful + b.harmful) / total for b in self.bullets if (b.helpful + b.harmful) > 0]
        return -sum(p * math.log2(p) for p in probs if p > 0)


def make_initial_playbook() -> Playbook:
    """Create initial playbook with seed strategies for math competitions."""
    pb = Playbook()
    pb.add("STRATEGIES", "Math competition problems often have integer answers. Always verify your answer is an integer when expected.")
    pb.add("STRATEGIES", "Break complex problems into smaller sub-problems and solve each step carefully.")
    pb.add("COMMON_MISTAKES", "Watch for off-by-one errors in counting and combinatorics problems.")
    return pb


class PlaybookManager(ABC):
    """Abstract base for playbook management strategies."""

    @abstractmethod
    def get_context(self) -> str:
        """Return playbook context string for prompt injection."""
        ...

    @abstractmethod
    def snapshot(self) -> Dict:
        """Return serializable snapshot of current state."""
        ...


class NullPlaybook(PlaybookManager):
    """No playbook - returns empty context."""

    def get_context(self) -> str:
        return ""

    def snapshot(self) -> Dict:
        return {"type": "null"}


class StaticPlaybook(PlaybookManager):
    """Frozen playbook - returns constant context, no curation."""

    def __init__(self, playbook: Playbook):
        self._playbook = playbook.copy()  # Freeze via deep copy
        self._context = self._build_context()

    def _build_context(self) -> str:
        if self._playbook.size == 0:
            return ""
        return (
            f"\nPLAYBOOK (use these strategies, reference IDs like [str-00001]):\n"
            f"{self._playbook.to_str()}"
        )

    def get_context(self) -> str:
        return self._context

    def snapshot(self) -> Dict:
        return {"type": "static", "playbook": self._playbook.snapshot()}

    @classmethod
    def from_snapshot(cls, data: Dict) -> "StaticPlaybook":
        pb = Playbook.from_snapshot(data["playbook"])
        return cls(pb)

    @classmethod
    def from_json(cls, path: str) -> "StaticPlaybook":
        """Load frozen playbook from JSON file."""
        with open(path) as f:
            data = json.load(f)
        pb = Playbook.from_snapshot(data)
        return cls(pb)


class ActivePlaybook(PlaybookManager):
    """Evolving playbook with reflect/curate hooks."""

    def __init__(self, playbook: Optional[Playbook] = None):
        self.playbook = playbook or make_initial_playbook()

    def get_context(self) -> str:
        if self.playbook.size == 0:
            return ""
        return (
            f"\nPLAYBOOK (use these strategies, reference IDs like [str-00001]):\n"
            f"{self.playbook.to_str()}"
        )

    def snapshot(self) -> Dict:
        return {"type": "active", "playbook": self.playbook.snapshot()}

    @classmethod
    def from_snapshot(cls, data: Dict) -> "ActivePlaybook":
        pb = Playbook.from_snapshot(data["playbook"])
        return cls(pb)

    def apply_tags(self, tags: Dict[str, str]):
        """Apply bullet tags from reflector."""
        for bid, tag in tags.items():
            self.playbook.tag(bid, tag)

    def apply_ops(self, ops, max_bullets: int = 20):
        """Apply curator operations."""
        from .curator import CurateOp
        for op in ops:
            if isinstance(op, CurateOp):
                if op.op == "ADD" and self.playbook.size < max_bullets:
                    self.playbook.add(op.section, op.content)
                elif op.op == "UPDATE" and op.target_id:
                    self.playbook.update(op.target_id, op.content)
                elif op.op == "DELETE" and op.target_id:
                    self.playbook.remove(op.target_id)
            elif isinstance(op, dict):
                # Dict-style ops for backward compatibility
                op_type = op.get("op", "").upper()
                if op_type == "ADD" and self.playbook.size < max_bullets:
                    self.playbook.add(op.get("section", "STRATEGIES"), op.get("content", ""))
                elif op_type == "UPDATE" and op.get("id"):
                    self.playbook.update(op["id"], op.get("content", ""))
                elif op_type == "DELETE" and op.get("id"):
                    self.playbook.remove(op["id"])

        # Aggressive pruning
        self.playbook.prune_harmful()
