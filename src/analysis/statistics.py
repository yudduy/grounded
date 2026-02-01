"""Statistical analysis for experiment results.

Paired t-tests comparing ACE conditions vs static baselines,
summary tables, and effect size calculations.
"""
import json
import sqlite3
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import false_discovery_control


def load_results(db_path: str) -> List[Dict]:
    """Load all results from campaign database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs WHERE status='completed' ORDER BY env_name, condition, seed"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _extract_pairs(results: List[Dict], cond_a: str, cond_b: str,
                    metric: str = "best_test_mse"):
    """Extract matched pairs for two conditions.

    Returns:
        (pairs_a, pairs_b) as lists, or (None, None) if insufficient.
    """
    lookup = {}
    for r in results:
        key = (r["env_name"], r["seed"])
        lookup.setdefault(key, {})[r["condition"]] = r.get(metric)

    pairs_a, pairs_b = [], []
    for key, conds in lookup.items():
        if cond_a in conds and cond_b in conds:
            va, vb = conds[cond_a], conds[cond_b]
            if va is not None and vb is not None:
                pairs_a.append(va)
                pairs_b.append(vb)
    return pairs_a, pairs_b


def paired_t_test(results: List[Dict], cond_a: str, cond_b: str,
                  metric: str = "best_test_mse",
                  log_transform: bool = False) -> Dict:
    """Run paired t-test comparing two conditions.

    Pairs are matched by (env_name, seed).

    Args:
        results: list of run result dicts
        cond_a: first condition label
        cond_b: second condition label
        metric: column to compare
        log_transform: if True, apply np.log1p to values before testing
            (useful for right-skewed MSE distributions)
    Returns:
        Dict with t_stat, p_value, mean_diff, effect_size (Cohen's d)
    """
    pairs_a, pairs_b = _extract_pairs(results, cond_a, cond_b, metric)

    if len(pairs_a) < 2:
        return {"error": "insufficient pairs", "n_pairs": len(pairs_a)}

    a, b = np.array(pairs_a), np.array(pairs_b)
    if log_transform:
        a, b = np.log1p(a), np.log1p(b)
    diff = a - b
    t_stat, p_value = stats.ttest_rel(a, b)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n_pairs": len(pairs_a),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": mean_diff,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05,
        "log_transformed": log_transform,
    }


def wilcoxon_test(results: List[Dict], cond_a: str, cond_b: str,
                  metric: str = "best_test_mse") -> Dict:
    """Run Wilcoxon signed-rank test comparing two conditions.

    Non-parametric alternative to paired t-test. Robust to non-normal
    distributions and outliers common in MSE data.

    Pairs are matched by (env_name, seed).

    Args:
        results: list of run result dicts
        cond_a: first condition label
        cond_b: second condition label
        metric: column to compare
    Returns:
        Dict with statistic, p_value, n_pairs
    """
    pairs_a, pairs_b = _extract_pairs(results, cond_a, cond_b, metric)

    if len(pairs_a) < 2:
        return {"error": "insufficient pairs", "n_pairs": len(pairs_a)}

    a, b = np.array(pairs_a), np.array(pairs_b)
    diff = a - b
    # Filter out zero differences (wilcoxon requires non-zero diffs)
    nonzero = diff != 0
    if nonzero.sum() < 2:
        return {"error": "insufficient non-zero differences", "n_pairs": len(pairs_a)}

    try:
        w_stat, p_value = stats.wilcoxon(a, b)
    except ValueError as e:
        return {"error": str(e), "n_pairs": len(pairs_a)}

    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n_pairs": len(pairs_a),
        "test": "wilcoxon",
        "statistic": float(w_stat),
        "p_value": float(p_value),
        "median_diff": float(np.median(diff)),
        "significant": p_value < 0.05,
    }


def summary_table(results: List[Dict]) -> Dict:
    """Generate summary statistics grouped by condition."""
    by_cond = {}
    for r in results:
        c = r["condition"]
        by_cond.setdefault(c, []).append(r)

    table = {}
    for cond, runs in sorted(by_cond.items()):
        mses = [r["best_test_mse"] for r in runs if r.get("best_test_mse") is not None]
        table[cond] = {
            "n_runs": len(runs),
            "mean_mse": float(np.mean(mses)) if mses else None,
            "std_mse": float(np.std(mses)) if mses else None,
            "median_mse": float(np.median(mses)) if mses else None,
            "min_mse": float(np.min(mses)) if mses else None,
            "max_mse": float(np.max(mses)) if mses else None,
        }
    return table


def run_all_comparisons(results: List[Dict]) -> List[Dict]:
    """Run paired t-tests for key comparisons with FDR correction."""
    comparisons = [
        ("A", "B"),  # Static vs ACE
        ("A", "C"),  # Static vs Gradient
        ("A", "D"),  # Static vs ACE+Gradient
        ("A", "E"),  # Static vs DrSR
        ("B", "D"),  # ACE vs ACE+Gradient
        ("C", "D"),  # Gradient vs ACE+Gradient
        ("B", "E"),  # ACE vs DrSR
    ]
    results_list = [paired_t_test(results, a, b) for a, b in comparisons]

    # Wilcoxon signed-rank tests as robustness check
    wilcoxon_results = [wilcoxon_test(results, a, b) for a, b in comparisons]
    for t_res, w_res in zip(results_list, wilcoxon_results):
        if "error" not in w_res:
            t_res["wilcoxon_statistic"] = w_res["statistic"]
            t_res["wilcoxon_p_value"] = w_res["p_value"]
            t_res["wilcoxon_significant"] = w_res["significant"]

    # Benjamini-Hochberg FDR correction
    p_values = [r.get("p_value", 1.0) for r in results_list if "error" not in r]
    if len(p_values) >= 2:
        try:
            adjusted = false_discovery_control(p_values, method='bh')
            idx = 0
            for r in results_list:
                if "error" not in r:
                    r["p_value_adjusted"] = float(adjusted[idx])
                    r["significant_adjusted"] = float(adjusted[idx]) < 0.05
                    idx += 1
        except Exception:
            pass  # scipy version may not have this

    return results_list
