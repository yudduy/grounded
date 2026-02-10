"""Statistical analysis and visualization for Memory x RL experiment."""

import json
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

def bootstrap_ci(
    outcomes: List[bool],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    stat_fn: Callable = np.mean,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for binary outcomes.

    Returns: (mean, lower, upper)
    """
    rng = np.random.default_rng(seed)
    data = np.array(outcomes, dtype=float)
    boot_stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = float(np.percentile(boot_stats, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_stats, (1 + ci) / 2 * 100))
    return float(stat_fn(data)), lower, upper


def mcnemar_test(
    outcomes_a: List[bool], outcomes_b: List[bool]
) -> Tuple[float, int, int]:
    """McNemar's test for paired binary outcomes.

    Returns: (p_value, n01, n10)
      n01: A wrong, B right
      n10: A right, B wrong
    """
    from scipy import stats

    a = np.array(outcomes_a, dtype=bool)
    b = np.array(outcomes_b, dtype=bool)
    n01 = int(np.sum(~a & b))  # A wrong, B right
    n10 = int(np.sum(a & ~b))  # A right, B wrong
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    p_value = stats.binomtest(n01, n, 0.5).pvalue
    return float(p_value), n01, n10


def holm_bonferroni(
    comparisons: List[Tuple[str, float]],
    alpha: float = 0.05,
) -> List[Tuple[str, float, float, bool]]:
    """Holm-Bonferroni correction for multiple comparisons.

    Args:
        comparisons: List of (label, p_value) tuples
        alpha: Family-wise error rate

    Returns:
        List of (label, p_raw, p_adjusted, significant) tuples
    """
    n = len(comparisons)
    # Sort by p-value
    sorted_comps = sorted(comparisons, key=lambda x: x[1])

    results = []
    rejected = True  # Track if we should keep rejecting
    for i, (label, p_raw) in enumerate(sorted_comps):
        adjusted_alpha = alpha / (n - i)
        if rejected and p_raw <= adjusted_alpha:
            p_adj = p_raw * (n - i)  # Adjusted p-value
            results.append((label, p_raw, min(p_adj, 1.0), True))
        else:
            rejected = False
            p_adj = p_raw * (n - i)
            results.append((label, p_raw, min(p_adj, 1.0), False))

    # Re-sort to original order
    label_order = {label: i for i, (label, _) in enumerate(comparisons)}
    results.sort(key=lambda x: label_order.get(x[0], 0))
    return results


def cohens_g(outcomes_a: List[bool], outcomes_b: List[bool]) -> float:
    """Cohen's g effect size for paired binary data.

    g = proportion of discordant pairs favoring one direction
    Interpretation: 0.05 = small, 0.15 = medium, 0.25 = large
    """
    a = np.array(outcomes_a, dtype=bool)
    b = np.array(outcomes_b, dtype=bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    total_discordant = n01 + n10
    if total_discordant == 0:
        return 0.0
    return abs(n01 / total_discordant - 0.5)


def run_all_comparisons(
    binary_outcomes: Dict[str, List[bool]],
    comparison_pairs: List[Tuple[str, str, str]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Run full statistical analysis on all condition pairs.

    Args:
        binary_outcomes: {condition_name: [True/False per problem]}
        comparison_pairs: [(cond_a, cond_b, label), ...]
        alpha: Significance level

    Returns:
        Dict with "comparisons", "accuracy_table", "holm_corrected"
    """
    raw_results = []
    for cond_a, cond_b, label in comparison_pairs:
        p_val, n01, n10 = mcnemar_test(binary_outcomes[cond_a], binary_outcomes[cond_b])
        g = cohens_g(binary_outcomes[cond_a], binary_outcomes[cond_b])
        raw_results.append({
            "label": label,
            "cond_a": cond_a,
            "cond_b": cond_b,
            "p_value": p_val,
            "n01": n01,
            "n10": n10,
            "cohens_g": g,
        })

    # Holm-Bonferroni correction
    p_values = [(r["label"], r["p_value"]) for r in raw_results]
    corrected = holm_bonferroni(p_values, alpha)
    correction_map = {label: (p_adj, sig) for label, _, p_adj, sig in corrected}

    for r in raw_results:
        p_adj, sig = correction_map[r["label"]]
        r["p_adjusted"] = p_adj
        r["significant"] = sig

    # Accuracy table
    accuracy_table = {}
    for cond, outcomes in binary_outcomes.items():
        mean, lo, hi = bootstrap_ci(outcomes)
        accuracy_table[cond] = {"mean": mean, "ci_lower": lo, "ci_upper": hi, "n": len(outcomes)}

    return {
        "comparisons": raw_results,
        "accuracy_table": accuracy_table,
    }


# ---------------------------------------------------------------------------
# Tactic Classification
# ---------------------------------------------------------------------------

TACTIC_TAXONOMY = [
    "modular_arithmetic",
    "casework",
    "pigeonhole",
    "algebraic_manipulation",
    "generating_functions",
    "induction",
    "bounding",
    "symmetry",
    "parity",
    "constructive",
    "coordinate_geometry",
    "trigonometric",
    "polynomial_roots",
    "number_theoretic",
    "other",
]

CLASSIFY_SYSTEM = (
    "You are a math competition tactic classifier. Given a solution to a math problem, "
    "identify the PRIMARY mathematical tactic used. Respond with ONLY one of these categories:\n"
    + "\n".join(f"- {t}" for t in TACTIC_TAXONOMY)
    + "\n\nRespond with just the category name, nothing else."
)


def classify_tactic(solution_text: str, kimi_client=None) -> str:
    """Classify the primary tactic in a solution via Kimi API.

    Returns one of the 15 tactic categories.
    """
    from .kimi_client import kimi_chat, CLASSIFY_MODEL

    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": f"Solution:\n{solution_text[:3000]}"},
    ]

    try:
        response = kimi_chat(
            messages, model=CLASSIFY_MODEL, temperature=0.1, max_tokens=50, client=kimi_client
        )
        tactic = response.strip().lower().replace(" ", "_")
        if tactic in TACTIC_TAXONOMY:
            return tactic
        # Fuzzy match
        for t in TACTIC_TAXONOMY:
            if t in tactic or tactic in t:
                return t
        return "other"
    except Exception as e:
        print(f"  Tactic classification failed: {e}")
        return "other"


def batch_classify_tactics(
    solutions: List[str], kimi_client=None
) -> List[str]:
    """Classify tactics for a batch of solutions."""
    return [classify_tactic(s, kimi_client) for s in solutions]


# ---------------------------------------------------------------------------
# Diversity Metrics
# ---------------------------------------------------------------------------

def shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy of a distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def tactic_distribution(
    solutions: List[str], kimi_client=None
) -> Dict[str, int]:
    """Get tactic distribution for a set of solutions."""
    tactics = batch_classify_tactics(solutions, kimi_client)
    return dict(Counter(tactics))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    """Configure matplotlib for non-interactive use."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })
    return plt


CONDITION_COLORS = {
    "B": "#FF9800",
    "C": "#2196F3",
    "D": "#4CAF50",
    "E": "#9C27B0",
    "C-abs": "#90CAF9",
    "D-abs": "#A5D6A7",
}

CONDITION_LABELS = {
    "B": "B: GRPO-only",
    "C": "C: Static Playbook",
    "D": "D: Active Playbook",
    "E": "E: Novelty Reward",
    "C-abs": "C-abs: C w/o playbook",
    "D-abs": "D-abs: D w/o playbook",
}


def plot_training_curves(
    results: Dict[str, Any],
    save_path: str,
    conditions: List[str] = None,
):
    """Figure 1: Training curves (reward accuracy per epoch)."""
    plt = _setup_matplotlib()

    if conditions is None:
        conditions = ["B", "C", "D", "E"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions:
        if cond not in results:
            continue
        data = results[cond]
        # Expect data to have "epoch_metrics" with per-epoch accuracy
        metrics = data.get("epoch_metrics", [])
        if not metrics:
            continue
        epochs = [m.get("epoch", i) + 1 for i, m in enumerate(metrics)]
        accs = [m.get("reward_accuracy", m.get("accuracy", 0)) for m in metrics]
        ax.plot(
            epochs, accs,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLORS.get(cond, "#888"),
            linewidth=2,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward Accuracy")
    ax.set_title("Training Curves: Reward Accuracy per Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_eval_accuracy(
    accuracy_table: Dict[str, Dict],
    save_path: str,
    conditions: List[str] = None,
):
    """Figure 2: Eval accuracy bar chart with 95% CI."""
    plt = _setup_matplotlib()

    if conditions is None:
        conditions = ["B", "C", "C-abs", "D", "D-abs", "E"]

    # Filter to available conditions
    conditions = [c for c in conditions if c in accuracy_table]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(conditions))
    means = [accuracy_table[c]["mean"] for c in conditions]
    ci_lo = [accuracy_table[c]["mean"] - accuracy_table[c]["ci_lower"] for c in conditions]
    ci_hi = [accuracy_table[c]["ci_upper"] - accuracy_table[c]["mean"] for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#888") for c in conditions]

    bars = ax.bar(x_pos, means, color=colors, alpha=0.8, edgecolor="black")
    ax.errorbar(x_pos, means, yerr=[ci_lo, ci_hi], fmt="none", ecolor="black", capsize=5, linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Evaluation Accuracy (95% CI)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{mean:.1%}",
            ha="center", va="bottom", fontweight="bold", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_absorption_delta(
    accuracy_table: Dict[str, Dict],
    save_path: str,
):
    """Figure 3: Absorption delta (D vs D-abs, C vs C-abs)."""
    plt = _setup_matplotlib()

    pairs = [("C", "C-abs"), ("D", "D-abs")]
    available_pairs = [(a, b) for a, b in pairs if a in accuracy_table and b in accuracy_table]

    if not available_pairs:
        print("No absorption pairs available to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = []
    deltas = []
    colors_list = []

    for with_pb, without_pb in available_pairs:
        delta = accuracy_table[with_pb]["mean"] - accuracy_table[without_pb]["mean"]
        label = f"{with_pb} - {without_pb}"
        labels.append(label)
        deltas.append(delta)
        colors_list.append(CONDITION_COLORS.get(with_pb, "#888"))

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, deltas, color=colors_list, alpha=0.8, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Accuracy Delta")
    ax.set_title("Absorption Test: With Playbook - Without Playbook")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, delta in zip(bars, deltas):
        va = "bottom" if delta >= 0 else "top"
        offset = 0.005 if delta >= 0 else -0.005
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            f"{delta:+.1%}",
            ha="center", va=va, fontweight="bold", fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_playbook_evolution(
    results: Dict[str, Any],
    save_path: str,
):
    """Figure 4: Playbook evolution for condition D (size + entropy)."""
    plt = _setup_matplotlib()

    d_data = results.get("D", {})
    epoch_metrics = d_data.get("epoch_metrics", [])

    if not epoch_metrics:
        print("No D epoch metrics for playbook evolution plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = [m.get("epoch", i) + 1 for i, m in enumerate(epoch_metrics)]
    sizes = [m.get("pb_size", 0) for m in epoch_metrics]
    entropies = [m.get("pb_entropy", 0) for m in epoch_metrics]

    ax1.plot(epochs, sizes, color=CONDITION_COLORS["D"], linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Playbook Size (bullets)")
    ax1.set_title("Playbook Size Over Training")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, entropies, color=CONDITION_COLORS["D"], linewidth=2, marker="s", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Playbook Entropy (bits)")
    ax2.set_title("Playbook Entropy Over Training")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Condition D: Playbook Evolution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_answer_entropy(
    results: Dict[str, Any],
    save_path: str,
    conditions: List[str] = None,
):
    """Figure 5: Answer entropy over training (collapse detection)."""
    plt = _setup_matplotlib()

    if conditions is None:
        conditions = ["B", "C", "D", "E"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions:
        if cond not in results:
            continue
        data = results[cond]
        metrics = data.get("epoch_metrics", [])
        if not metrics:
            continue
        epochs = [m.get("epoch", i) + 1 for i, m in enumerate(metrics)]
        entropies = [m.get("answer_entropy", 0) for m in metrics]
        ax.plot(
            epochs, entropies,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLORS.get(cond, "#888"),
            linewidth=2,
        )

    # Collapse threshold
    ax.axhline(y=0.1, color="red", linewidth=1, linestyle="--", alpha=0.5, label="Collapse threshold")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Answer Entropy (bits)")
    ax.set_title("Answer Distribution Entropy (Collapse Detection)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_tactic_heatmap(
    tactic_data: Dict[str, Dict[str, int]],
    save_path: str,
):
    """Tactic distribution heatmap (conditions x tactics)."""
    plt = _setup_matplotlib()

    conditions = sorted(tactic_data.keys())
    tactics = TACTIC_TAXONOMY

    # Build matrix (normalized per condition)
    matrix = np.zeros((len(conditions), len(tactics)))
    for i, cond in enumerate(conditions):
        total = sum(tactic_data[cond].values()) or 1
        for j, tactic in enumerate(tactics):
            matrix[i, j] = tactic_data[cond].get(tactic, 0) / total

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(tactics)))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tactics], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([CONDITION_LABELS.get(c, c) for c in conditions])

    # Add text annotations
    for i in range(len(conditions)):
        for j in range(len(tactics)):
            val = matrix[i, j]
            if val > 0.01:
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=7)

    ax.set_title("Tactic Distribution by Condition")
    fig.colorbar(im, ax=ax, label="Proportion")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def generate_summary_report(
    results: Dict[str, Any],
    binary_outcomes: Dict[str, List[bool]],
    comparison_pairs: List[Tuple[str, str, str]],
) -> str:
    """Generate a text summary report of the experiment."""
    analysis = run_all_comparisons(binary_outcomes, comparison_pairs)

    lines = []
    lines.append("=" * 60)
    lines.append("Memory x RL Interaction Experiment - Summary Report")
    lines.append("=" * 60)

    # Accuracy table
    lines.append("\nAccuracy Table:")
    lines.append(f"{'Condition':20s} {'Accuracy':>10s} {'95% CI':>20s} {'N':>6s}")
    lines.append("-" * 60)
    for cond, stats in analysis["accuracy_table"].items():
        lines.append(
            f"{CONDITION_LABELS.get(cond, cond):20s} "
            f"{stats['mean']:10.1%} "
            f"[{stats['ci_lower']:.1%}, {stats['ci_upper']:.1%}] "
            f"{stats['n']:6d}"
        )

    # Pairwise comparisons
    lines.append("\nPairwise Comparisons (Holm-Bonferroni corrected):")
    lines.append(f"{'Comparison':35s} {'p-raw':>10s} {'p-adj':>10s} {'g':>8s} {'Sig':>5s}")
    lines.append("-" * 70)
    for r in analysis["comparisons"]:
        sig = "***" if r["p_adjusted"] < 0.001 else "**" if r["p_adjusted"] < 0.01 else "*" if r["p_adjusted"] < 0.05 else "ns"
        lines.append(
            f"{r['label']:35s} "
            f"{r['p_value']:10.4f} "
            f"{r['p_adjusted']:10.4f} "
            f"{r['cohens_g']:8.3f} "
            f"{sig:>5s}"
        )

    return "\n".join(lines)
