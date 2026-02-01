"""Plotting utilities for experiment results.

Learning curves, comparison plots, and summary visualizations.
"""
import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def plot_learning_curves(db_path: str, env_name: str,
                         output_path: str = "learning_curves.png",
                         conditions: Optional[List[str]] = None):
    """Plot MSE vs round for all conditions on one environment.

    Args:
        db_path: path to campaign SQLite database
        env_name: environment name to plot
        output_path: save path for the plot
        conditions: list of conditions to include (default: all)
    """
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs WHERE env_name=? AND status='completed'",
        (env_name,)
    ).fetchall()
    conn.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"A": "gray", "B": "blue", "C": "green", "D": "red",
              "E": "orange", "F": "purple", "R": "brown"}

    for row in rows:
        cond = row["condition"]
        if conditions and cond not in conditions:
            continue
        curve = json.loads(row["mse_curve"]) if row["mse_curve"] else []
        if not curve:
            continue
        rounds = list(range(1, len(curve) + 1))
        ax.plot(rounds, curve, color=colors.get(cond, "black"),
                alpha=0.3, linewidth=0.8)

    # Plot condition means
    by_cond = {}
    for row in rows:
        cond = row["condition"]
        if conditions and cond not in conditions:
            continue
        curve = json.loads(row["mse_curve"]) if row["mse_curve"] else []
        if curve:
            by_cond.setdefault(cond, []).append(curve)

    for cond, curves in sorted(by_cond.items()):
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        mean = np.mean(arr, axis=0)
        rounds = list(range(1, min_len + 1))
        ax.plot(rounds, mean, color=colors.get(cond, "black"),
                linewidth=2, label=f"Condition {cond}")

    ax.set_xlabel("Round")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Learning Curves: {env_name}")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_final_mse_comparison(db_path: str, output_path: str = "mse_comparison.png"):
    """Bar chart of final MSE by condition, averaged across environments."""
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs WHERE status='completed'"
    ).fetchall()
    conn.close()

    by_cond = {}
    for row in rows:
        c = row["condition"]
        mse = row["best_test_mse"]
        if mse is not None:
            by_cond.setdefault(c, []).append(mse)

    fig, ax = plt.subplots(figsize=(8, 5))
    conds = sorted(by_cond.keys())
    means = [np.mean(by_cond[c]) for c in conds]
    stds = [np.std(by_cond[c]) for c in conds]
    colors = {"A": "gray", "B": "blue", "C": "green", "D": "red",
              "E": "orange", "F": "purple", "R": "brown"}

    bars = ax.bar(conds, means, yerr=stds, capsize=5,
                  color=[colors.get(c, "black") for c in conds])
    ax.set_xlabel("Condition")
    ax.set_ylabel("Best Test MSE (mean ± std)")
    ax.set_title("Final MSE by Condition")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
