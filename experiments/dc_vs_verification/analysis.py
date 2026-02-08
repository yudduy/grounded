#!/usr/bin/env python3
"""
Analysis script for DC vs Verification experiment.
Reads results, computes statistics, and outputs the decision.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results():
    combined = RESULTS_DIR / "all_results.json"
    if combined.exists():
        with open(combined) as f:
            return json.load(f)
    # Fallback: load individual seed files
    results = []
    for p in sorted(RESULTS_DIR.glob("seed_*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def analyze(results):
    if not results:
        print("ERROR: No results found. Run the experiment first.", file=sys.stderr)
        sys.exit(1)

    conditions = ["baseline", "self_consistency", "dc", "dc_verified"]
    labels = {
        "baseline": "Baseline (CoT Pass@1)",
        "self_consistency": "Self-Consistency (N=16)",
        "dc": "Dynamic Cheatsheet (DC)",
        "dc_verified": "DC + Outcome Verification",
    }

    print("=" * 70)
    print("DC vs Verification on AIME 2024 — Results")
    print("=" * 70)
    print(f"Seeds: {[r['seed'] for r in results]}")
    print(f"Problems per seed: {results[0]['n_problems']}")
    print()

    # Per-condition summary
    stats = {}
    print(f"{'Condition':35s} {'Mean':>7s} {'Std':>7s} {'Seeds':>20s} {'Calls':>8s} {'Time':>7s}")
    print("-" * 70)

    for cond in conditions:
        accs = [r[cond]["accuracy"] for r in results]
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
        avg_calls = sum(sum(r[cond]["calls"].values()) for r in results) / len(results)
        avg_time = sum(r[cond]["time_s"] for r in results) / len(results)
        seeds_str = " ".join(f"{a:.0%}" for a in accs)

        stats[cond] = {"mean": mean_acc, "std": std_acc, "calls": avg_calls, "time": avg_time}
        print(f"{labels[cond]:35s} {mean_acc:6.1%} {std_acc:6.1%}   [{seeds_str:>16s}]  {avg_calls:7.0f}  {avg_time:5.0f}s")

    # DC learning curve analysis
    print("\n" + "-" * 70)
    print("DC Learning Curve (problem-by-problem accuracy):")
    for cond in ["dc", "dc_verified"]:
        n_problems = results[0]["n_problems"]
        per_problem = []
        for i in range(n_problems):
            correct_count = sum(1 for r in results if r[cond]["results"][i]["correct"])
            per_problem.append(correct_count / len(results))

        # Running average over last 10
        if n_problems >= 10:
            last10 = per_problem[-10:]
            first10 = per_problem[:10]
            print(f"  {labels[cond]:35s}: first10={sum(first10)/10:.0%}  last10={sum(last10)/10:.0%}")

    # DC+V specific: how often did any candidate get the right answer?
    print("\n" + "-" * 70)
    print("DC+V Oracle Analysis (any of N=8 candidates correct):")
    for r in results:
        oracle_correct = sum(1 for res in r["dc_verified"]["results"] if res.get("any_candidate_correct", False))
        oracle_acc = oracle_correct / r["n_problems"]
        actual_acc = r["dc_verified"]["accuracy"]
        print(f"  Seed {r['seed']}: oracle={oracle_acc:.0%} actual={actual_acc:.0%} (gap={oracle_acc-actual_acc:+.0%})")

    # Decision framework
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    dc_mean = stats["dc"]["mean"]
    dcv_mean = stats["dc_verified"]["mean"]
    gap = dcv_mean - dc_mean
    sc_mean = stats["self_consistency"]["mean"]

    print(f"  DC accuracy:           {dc_mean:.1%}")
    print(f"  DC+V accuracy:         {dcv_mean:.1%}")
    print(f"  Self-Consistency:      {sc_mean:.1%}")
    print(f"  Gap (DC+V - DC):       {gap:+.1%}")
    print()

    if gap < 0.05:
        verdict = "A"
        verdict_text = "Verifier over-engineered"
        next_step = "Commit to DC-only. Focus on cheatsheet quality, strategy diversity."
    elif gap < 0.15:
        verdict = "A+"
        verdict_text = "Marginal value"
        next_step = "DC-first with lightweight outcome verification (self-consistency as proxy)."
    else:
        verdict = "B"
        verdict_text = "Verifier justified"
        next_step = "Build distilled verifier pipeline. Invest in verifier diversity."

    print(f"  VERDICT: Path {verdict} — {verdict_text}")
    print(f"  NEXT:    {next_step}")
    print()

    # Additional context
    dc_vs_sc = dc_mean - sc_mean
    print(f"  DC vs Self-Consistency gap: {dc_vs_sc:+.1%}")
    if dc_vs_sc > 0.05:
        print("  → Playbook learning adds value beyond just sampling more.")
    elif dc_vs_sc > -0.05:
        print("  → Playbook learning roughly matches extra sampling (compute-neutral).")
    else:
        print("  → Playbook learning hurts — investigate strategy quality.")

    # Save analysis
    analysis = {
        "stats": {k: {"mean": v["mean"], "std": v["std"], "calls": v["calls"], "time_s": v["time"]} for k, v in stats.items()},
        "gap_dcv_minus_dc": gap,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "next_step": next_step,
    }
    outfile = RESULTS_DIR / "analysis.json"
    with open(outfile, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {outfile}")


if __name__ == "__main__":
    results = load_results()
    analyze(results)
