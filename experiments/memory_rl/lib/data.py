"""Data loaders for Memory x RL experiment datasets."""

import json
import os
from pathlib import Path
from typing import Dict, List


def load_math500_l4l5() -> List[Dict]:
    """Load MATH-500 filtered to levels 4 and 5."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for row in ds:
        if row["level"] in ["Level 4", "Level 5"]:
            problems.append({
                "id": f"math500-{len(problems)}",
                "problem": row["problem"],
                "answer": row["answer"],
            })
    return problems


def load_aime_2025() -> List[Dict]:
    """Load AIME 2025 from local HuggingFace arrow cache."""
    from datasets import load_from_disk
    data_path = str(Path(__file__).resolve().parents[3] / "dynamic-cheatsheet" / "data" / "AIME_2025")
    ds = load_from_disk(data_path)
    problems = []
    for i, row in enumerate(ds):
        problems.append({
            "id": f"aime2025-{i}",
            "problem": row["input"],
            "answer": str(row["target"]).strip(),
        })
    return problems


def load_olympmath_easy() -> List[Dict]:
    """Load OlymMATH EASY tier from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("RUC-AIBOX/OlymMATH", split="test")
    problems = []
    for row in ds:
        difficulty = row.get("difficulty", row.get("level", ""))
        if "easy" in str(difficulty).lower():
            problems.append({
                "id": f"olympmath-{len(problems)}",
                "problem": row.get("problem", row.get("question", "")),
                "answer": str(row.get("answer", "")).strip(),
            })
    return problems


def load_amc12() -> List[Dict]:
    """Load AMC 12 2024-2025 from local JSON."""
    data_path = Path(__file__).parent / "data" / "amc12_2024_2025.json"
    if not data_path.exists():
        print(f"WARNING: AMC 12 data not found at {data_path}")
        print("Create this file with manual entries from AoPS wiki.")
        return []
    with open(data_path) as f:
        return json.load(f)


def load_all_datasets() -> Dict[str, List[Dict]]:
    """Load all datasets."""
    return {
        "math500_l4l5": load_math500_l4l5(),
        "aime_2025": load_aime_2025(),
        "olympmath_easy": load_olympmath_easy(),
        "amc12": load_amc12(),
    }


def load_training_data() -> List[Dict]:
    """Load training data (MATH-500 L4-5)."""
    return load_math500_l4l5()


def load_eval_datasets() -> Dict[str, List[Dict]]:
    """Load evaluation datasets only."""
    return {
        "aime_2025": load_aime_2025(),
        "olympmath_easy": load_olympmath_easy(),
        "amc12": load_amc12(),
    }
