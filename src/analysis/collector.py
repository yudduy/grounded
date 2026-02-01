"""Data collector and mid-campaign analyzer.

Safe to run while campaign is in progress (reads from WAL-mode SQLite).
"""
import json
import csv
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def collect_all(db_path: str, results_dir: str, output_dir: str = None):
    """Run full data collection and analysis."""
    if output_dir is None:
        output_dir = str(Path(results_dir) / "analysis")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append(f"# Experiment Analysis Report")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")

    if not Path(db_path).exists():
        report_lines.append("## No campaign data yet")
        report_lines.append(f"Database not found: {db_path}")
        report_path = str(Path(output_dir) / "report.md")
        Path(report_path).write_text("\n".join(report_lines))
        return report_path

    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    if "runs" not in tables:
        report_lines.append("## No campaign data yet")
        report_lines.append("Database exists but no runs table.")
        report_path = str(Path(output_dir) / "report.md")
        Path(report_path).write_text("\n".join(report_lines))
        return report_path

    progress = _campaign_progress(db_path)
    report_lines.append("## Campaign Progress")
    report_lines.append(f"- Completed: {progress['completed']}/{progress['total']}")
    report_lines.append(f"- Running: {progress['running']}")
    report_lines.append(f"- Failed/Pending: {progress['pending']}")
    report_lines.append(f"- Total cost: ${progress['total_cost']:.4f}")
    report_lines.append("")

    summary = _summary_by_condition(db_path)
    report_lines.append("## Results by Condition")
    report_lines.append("| Condition | N | Mean MSE | Median MSE | Std MSE | Min MSE | Mean Cost |")
    report_lines.append("|-----------|---|----------|------------|---------|---------|-----------|")
    for cond, s in sorted(summary.items()):
        report_lines.append(
            f"| {cond:9s} | {s['n']:1d} | {s['mean_mse']:.4e} | {s['median_mse']:.4e} | "
            f"{s['std_mse']:.4e} | {s['min_mse']:.4e} | ${s['mean_cost']:.4f} |"
        )
    report_lines.append("")

    env_summary = _summary_by_env(db_path)
    report_lines.append("## Results by Environment")
    for env_name, conds in sorted(env_summary.items()):
        report_lines.append(f"\n### {env_name}")
        report_lines.append("| Condition | Seeds | Best MSE | Best Expr | Best Round |")
        report_lines.append("|-----------|-------|----------|-----------|------------|")
        for cond, runs in sorted(conds.items()):
            for r in runs:
                expr = (r['best_expression'] or "")[:40]
                report_lines.append(
                    f"| {cond} | s{r['seed']} | {r['best_test_mse']:.4e} | "
                    f"`{expr}` | {r['best_round']} |"
                )
    report_lines.append("")

    stats_report = _statistical_comparisons(db_path)
    if stats_report:
        report_lines.append("## Statistical Comparisons (Paired t-tests)")
        report_lines.append("| Comparison | N pairs | t-stat | p-value | Cohen's d | Significant |")
        report_lines.append("|------------|---------|--------|---------|-----------|-------------|")
        for s in stats_report:
            if "error" in s:
                report_lines.append(f"| {s['cond_a']} vs {s['cond_b']} | {s['n_pairs']} | - | - | - | insufficient data |")
            else:
                sig = "YES" if s.get('significant', False) else "no"
                report_lines.append(
                    f"| {s['cond_a']} vs {s['cond_b']} | {s['n_pairs']} | "
                    f"{s['t_stat']:.3f} | {s['p_value']:.4f} | {s['cohens_d']:.3f} | {sig} |"
                )
        report_lines.append("")

    diversity = _expression_diversity(db_path)
    report_lines.append("## Expression Diversity")
    report_lines.append(f"- Total unique expressions: {diversity['total_unique']}")
    report_lines.append(f"- Unique best expressions: {diversity['unique_best']}")
    report_lines.append("- Most common best expressions:")
    for expr, count in diversity['top_expressions'][:10]:
        report_lines.append(f"  - `{expr}` (×{count})")
    report_lines.append("")

    cost = _cost_breakdown(db_path)
    report_lines.append("## Cost Breakdown")
    report_lines.append(f"- Total: ${cost['total']:.4f}")
    for cond, c in sorted(cost['by_condition'].items()):
        report_lines.append(f"  - Condition {cond}: ${c:.4f}")
    report_lines.append("")

    report_path = str(Path(output_dir) / "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    _export_round_csv(db_path, str(Path(output_dir) / "round_results.csv"))
    _export_run_csv(db_path, str(Path(output_dir) / "run_summary.csv"))

    if MPL_AVAILABLE:
        _plot_all(db_path, output_dir)

    return report_path


def _campaign_progress(db_path: str) -> Dict:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT status, total_cost FROM runs").fetchall()
    conn.close()
    completed = sum(1 for r in rows if r[0] == "completed")
    running = sum(1 for r in rows if r[0] == "running")
    pending = len(rows) - completed - running
    total_cost = sum(r[1] or 0 for r in rows)
    total = 162  # 9 envs x 6 conditions x 3 seeds
    return {"completed": completed, "running": running, "pending": pending,
            "total": total, "total_cost": total_cost}


def _summary_by_condition(db_path: str) -> Dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM runs WHERE status='completed'").fetchall()
    conn.close()

    by_cond = {}
    for r in rows:
        c = r["condition"]
        by_cond.setdefault(c, []).append(dict(r))

    summary = {}
    for cond, runs in by_cond.items():
        mses = [r["best_test_mse"] for r in runs if r["best_test_mse"] is not None]
        costs = [r["total_cost"] for r in runs if r["total_cost"] is not None]
        summary[cond] = {
            "n": len(runs),
            "mean_mse": float(np.mean(mses)) if mses else 0,
            "median_mse": float(np.median(mses)) if mses else 0,
            "std_mse": float(np.std(mses)) if mses else 0,
            "min_mse": float(np.min(mses)) if mses else 0,
            "mean_cost": float(np.mean(costs)) if costs else 0,
        }
    return summary


def _summary_by_env(db_path: str) -> Dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs WHERE status='completed' ORDER BY env_name, condition, seed"
    ).fetchall()
    conn.close()

    by_env = {}
    for r in rows:
        env = r["env_name"]
        cond = r["condition"]
        by_env.setdefault(env, {}).setdefault(cond, []).append(dict(r))
    return by_env


def _statistical_comparisons(db_path: str) -> List[Dict]:
    from analysis.statistics import load_results, paired_t_test

    results = load_results(db_path)
    if len(results) < 4:
        return []

    comparisons = [
        ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"),
        ("B", "D"), ("C", "D"), ("B", "E"),
    ]
    return [paired_t_test(results, a, b) for a, b in comparisons]


def _expression_diversity(db_path: str) -> Dict:
    conn = sqlite3.connect(db_path)
    all_exprs = conn.execute(
        "SELECT DISTINCT expression FROM round_results WHERE expression IS NOT NULL"
    ).fetchall()
    best_exprs = conn.execute(
        "SELECT best_expression FROM runs WHERE status='completed' AND best_expression IS NOT NULL"
    ).fetchall()
    conn.close()

    from collections import Counter
    best_counter = Counter(r[0] for r in best_exprs)

    return {
        "total_unique": len(all_exprs),
        "unique_best": len(set(r[0] for r in best_exprs)),
        "top_expressions": best_counter.most_common(10),
    }


def _cost_breakdown(db_path: str) -> Dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT condition, total_cost FROM runs WHERE status='completed'").fetchall()
    conn.close()

    by_cond = {}
    total = 0
    for r in rows:
        c = r["condition"]
        cost = r["total_cost"] or 0
        by_cond[c] = by_cond.get(c, 0) + cost
        total += cost
    return {"total": total, "by_condition": by_cond}


def _export_round_csv(db_path: str, output_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM round_results ORDER BY env_name, condition, seed, round_num"
    ).fetchall()
    conn.close()

    if not rows:
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))


def _export_run_csv(db_path: str, output_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT env_name, condition, seed, status, best_test_mse, best_expression, "
        "best_round, final_test_mse, total_cost, completed_at "
        "FROM runs ORDER BY env_name, condition, seed"
    ).fetchall()
    conn.close()

    if not rows:
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))


def _plot_all(db_path: str, output_dir: str):
    from analysis.plots import plot_learning_curves, plot_final_mse_comparison

    conn = sqlite3.connect(db_path)
    envs = [r[0] for r in conn.execute(
        "SELECT DISTINCT env_name FROM runs WHERE status='completed'"
    ).fetchall()]
    conn.close()

    plots_dir = str(Path(output_dir) / "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    for env_name in envs:
        try:
            plot_learning_curves(
                db_path, env_name,
                output_path=str(Path(plots_dir) / f"learning_{env_name}.png"),
            )
        except Exception:
            pass

    try:
        plot_final_mse_comparison(
            db_path,
            output_path=str(Path(plots_dir) / "mse_comparison.png"),
        )
    except Exception:
        pass

    try:
        _plot_round_progression(db_path, plots_dir)
    except Exception:
        pass


def _plot_round_progression(db_path: str, plots_dir: str):
    if not MPL_AVAILABLE:
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT condition, round_num, test_mse FROM round_results "
        "WHERE test_mse < 1e10 ORDER BY condition, round_num"
    ).fetchall()
    conn.close()

    if not rows:
        return

    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["condition"]][r["round_num"]].append(r["test_mse"])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"A": "gray", "B": "blue", "C": "green", "D": "red",
              "E": "orange", "F": "purple", "R": "brown"}

    for cond in sorted(data.keys()):
        rounds_data = data[cond]
        rounds = sorted(rounds_data.keys())
        means = [np.mean(rounds_data[r]) for r in rounds]
        ax.plot(rounds, means, color=colors.get(cond, "black"),
                linewidth=2, label=f"Condition {cond}", alpha=0.8)
        stds = [np.std(rounds_data[r]) for r in rounds]
        ax.fill_between(rounds,
                       [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)],
                       color=colors.get(cond, "black"), alpha=0.1)

    ax.set_xlabel("Round")
    ax.set_ylabel("Test MSE (mean ± std)")
    ax.set_title("MSE Progression by Condition (all environments)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(Path(plots_dir) / "round_progression.png"), dpi=150)
    plt.close(fig)
