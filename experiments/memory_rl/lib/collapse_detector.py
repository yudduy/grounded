"""Per-epoch entropy monitoring and collapse detection.

Tracks Shannon entropy of answer distributions across training epochs.
Flags collapse when entropy drops below threshold for consecutive epochs.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EpochRecord:
    """Record for a single epoch's answer distribution."""
    epoch: int
    entropy: float
    unique_ratio: float
    total_completions: int
    unique_answers: int
    is_collapsed: bool
    answer_distribution: Dict[str, int] = field(default_factory=dict)


class CollapseDetector:
    """Monitor answer diversity and detect training collapse.

    Collapse is flagged when Shannon entropy of the answer distribution
    drops below `entropy_threshold` for `window_size` consecutive epochs.
    """

    def __init__(self, window_size: int = 5, entropy_threshold: float = 0.1):
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self._history: List[EpochRecord] = []

    def record_epoch(self, completions: List[str], epoch: int = -1) -> EpochRecord:
        """Record answer distribution for one epoch.

        Args:
            completions: All completion strings from this epoch
                         (answers already parsed, or raw text to be parsed).
            epoch: Epoch number (auto-incremented if -1).

        Returns:
            EpochRecord with entropy, collapse flag, etc.
        """
        if epoch < 0:
            epoch = len(self._history)

        # Parse answers if needed (import here to avoid circular dependency)
        from .answer_parsing import parse_answer

        answers = [parse_answer(c) for c in completions]

        # Normalize answers
        normalized = []
        for a in answers:
            try:
                normalized.append(str(int(float(a.replace(",", "")))))
            except (ValueError, TypeError):
                normalized.append(a.strip())

        # Compute distribution
        counter = Counter(normalized)
        total = len(normalized) or 1
        unique = len(counter)

        # Shannon entropy
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        unique_ratio = unique / total

        # Check collapse: entropy below threshold for window_size consecutive epochs
        is_collapsed = False
        if len(self._history) >= self.window_size - 1:
            recent = self._history[-(self.window_size - 1) :]
            all_below = all(r.entropy < self.entropy_threshold for r in recent)
            if all_below and entropy < self.entropy_threshold:
                is_collapsed = True

        record = EpochRecord(
            epoch=epoch,
            entropy=entropy,
            unique_ratio=unique_ratio,
            total_completions=total,
            unique_answers=unique,
            is_collapsed=is_collapsed,
            answer_distribution=dict(counter.most_common(20)),
        )
        self._history.append(record)

        if is_collapsed:
            print(
                f"  WARNING: Collapse detected at epoch {epoch}! "
                f"Entropy={entropy:.4f} < {self.entropy_threshold} "
                f"for {self.window_size} consecutive epochs."
            )

        return record

    def get_history(self) -> List[Dict]:
        """Return history as list of dicts (JSON-serializable)."""
        return [
            {
                "epoch": r.epoch,
                "entropy": r.entropy,
                "unique_ratio": r.unique_ratio,
                "total_completions": r.total_completions,
                "unique_answers": r.unique_answers,
                "is_collapsed": r.is_collapsed,
                "top_answers": r.answer_distribution,
            }
            for r in self._history
        ]

    def is_collapsed(self) -> bool:
        """Check if the most recent epoch was flagged as collapsed."""
        if not self._history:
            return False
        return self._history[-1].is_collapsed

    def latest_entropy(self) -> float:
        """Return entropy of the most recent epoch."""
        if not self._history:
            return float("inf")
        return self._history[-1].entropy
