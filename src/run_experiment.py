#!/usr/bin/env python3
"""Main entry point for the grounded discovery experiment.

Usage:
    python3 src/run_experiment.py                  # prelim + full campaign
    python3 src/run_experiment.py --prelim-only    # just the sanity check
    python3 src/run_experiment.py --skip-prelim    # skip straight to campaign
"""
import sys
import argparse
import logging
import traceback
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "ShinkaEvolve"))
sys.path.insert(0, str(_repo_root / "src"))

from shinka.llm.llm import LLMClient
from environments import ALL_ENVIRONMENTS
from loop.orchestrator import DiscoveryLoop
from conditions.static import StaticCondition
from campaign.config import CampaignConfig, RunConfig
from campaign.runner import CampaignRunner
from analysis.collector import collect_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_repo_root / "src" / "results" / "experiment.log")),
    ],
)
logger = logging.getLogger("experiment")


def run_prelim(llm: LLMClient) -> bool:
    """Run a minimal sanity check: 1 env x Condition A x 5 rounds."""
    logger.info("=" * 60)
    logger.info("PRELIMINARY SANITY CHECK")
    logger.info("=" * 60)

    env_cls = ALL_ENVIRONMENTS[0]  # First tier-1 environment
    env = env_cls(seed=42)
    logger.info(f"Environment: {env.name}")
    logger.info(f"Input names: {env.input_names}")
    logger.info(f"Input ranges: {env.input_ranges}")

    logger.info("[1/5] Testing LLM API connectivity...")
    try:
        result = llm.query(msg="Reply with just the number 42.", system_msg="You are a helpful assistant.")
        if result is None or not result.content:
            logger.error("LLM returned empty response")
            return False
        logger.info(f"  LLM response: {result.content[:100]}")
        logger.info(f"  Cost: ${result.cost:.6f}" if result.cost else "  Cost: N/A")
    except Exception as e:
        logger.error(f"  LLM API failed: {e}")
        traceback.print_exc()
        return False

    logger.info("[2/5] Testing environment evaluation...")
    try:
        inputs = env.sample_inputs(5)
        _, outputs = env.choose_inputs(inputs)
        logger.info(f"  Sample inputs shape: {inputs.shape}")
        logger.info(f"  Outputs: {outputs[:3]}...")
    except Exception as e:
        logger.error(f"  Environment failed: {e}")
        return False

    logger.info("[3/5] Running 5-round discovery loop...")
    strategy = StaticCondition(points_per_round=5, total_rounds=5)
    loop = DiscoveryLoop(
        env=env, llm=llm, strategy=strategy,
        total_rounds=5, points_per_round=5, seed=42,
    )

    ckpt_path = str(_repo_root / "src" / "results" / "prelim_checkpoint.json")
    results_path = str(_repo_root / "src" / "results" / "prelim_results.json")
    try:
        results = loop.run(checkpoint_path=ckpt_path)
        logger.info(f"  Completed {len(results)} rounds")
        for r in results:
            logger.info(f"  Round {r.round_num}: test_mse={r.test_mse:.6f} "
                       f"expr={r.expression_clean}")
    except Exception as e:
        logger.error(f"  Discovery loop failed: {e}")
        traceback.print_exc()
        return False

    logger.info("[4/5] Testing checkpoint save/load...")
    try:
        loop.save_checkpoint(ckpt_path)
        loop2 = DiscoveryLoop(
            env=env_cls(seed=42), llm=llm, strategy=strategy,
            total_rounds=5, points_per_round=5, seed=42,
        )
        resume_round = loop2.load_checkpoint(ckpt_path)
        assert resume_round == 6, f"Expected resume_round=6, got {resume_round}"
        assert len(loop2.results) == 5, f"Expected 5 results, got {len(loop2.results)}"
        logger.info(f"  Checkpoint OK: would resume from round {resume_round}")
        Path(ckpt_path).unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"  Checkpoint test failed: {e}")
        return False

    logger.info("[5/5] Saving prelim results...")
    try:
        loop.save_results(results_path)
        summary = loop.get_summary()
        logger.info(f"  Best MSE: {summary['best_test_mse']:.6f}")
        logger.info(f"  Best expr: {summary['best_expression']}")
        logger.info(f"  Total cost: ${summary['total_cost']:.6f}")
    except Exception as e:
        logger.error(f"  Results save failed: {e}")
        return False

    logger.info("=" * 60)
    logger.info("PRELIMINARY CHECK PASSED")
    logger.info("=" * 60)
    return True


def run_analysis(config: CampaignConfig):
    logger.info("Running analysis and data collection...")
    try:
        report_path = collect_all(
            db_path=config.db_path,
            results_dir=config.results_dir,
        )
        logger.info(f"Analysis report written to: {report_path}")
    except Exception as e:
        logger.warning(f"Analysis failed (non-fatal): {e}")


def run_campaign(llm: LLMClient):
    logger.info("=" * 60)
    logger.info("STARTING FULL CAMPAIGN")
    logger.info("=" * 60)

    config = CampaignConfig()
    logger.info(f"Model: {config.llm_model}")
    logger.info(f"Total runs: {config.total_runs}")
    logger.info(f"Budget: ${config.budget_total}")

    runner = CampaignRunner(config)

    runs = config.generate_runs()
    completed = 0
    skipped = 0
    failed = 0
    analysis_interval = 10

    for i, run in enumerate(runs):
        if runner._is_completed(run):
            skipped += 1
            continue

        if runner.total_cost >= config.budget_total:
            logger.warning(f"Budget exceeded (${runner.total_cost:.2f}), stopping")
            break

        try:
            summary = runner.run_single(run, llm)
            runner.total_cost += summary.get("total_cost", 0)
            completed += 1
        except Exception as e:
            logger.error(f"Run failed: {run.env_name}/{run.condition}/seed={run.seed}: {e}")
            failed += 1

        logger.info(f"Progress: {completed + skipped}/{len(runs)} "
                     f"(completed={completed}, skipped={skipped}, failed={failed})")

        if completed > 0 and completed % analysis_interval == 0:
            run_analysis(config)

    logger.info(f"Campaign finished: {completed} new, {skipped} skipped, "
                 f"{failed} failed, total_cost=${runner.total_cost:.2f}")

    run_analysis(config)

    results = runner.get_results_table()
    completed_runs = [r for r in results if r.get("status") == "completed"]
    logger.info(f"\nFinal: {len(completed_runs)}/{config.total_runs} runs completed")
    for r in completed_runs:
        logger.info(f"  {r['env_name']:30s} {r['condition']:3s} s{r['seed']} "
                    f"best_mse={r['best_test_mse']:.6f} "
                    f"cost=${r['total_cost']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Grounded Discovery Experiment")
    parser.add_argument("--prelim-only", action="store_true",
                       help="Only run the preliminary sanity check")
    parser.add_argument("--skip-prelim", action="store_true",
                       help="Skip preliminary check, go straight to campaign")
    args = parser.parse_args()

    Path(_repo_root / "src" / "results").mkdir(parents=True, exist_ok=True)

    config = CampaignConfig()
    # TEMP: GPT-OSS uses reasoning tokens from the output budget.
    # Keep max_tokens low so reasoning doesn't consume entire output.
    llm = LLMClient(
        model_names=[config.llm_model],
        temperatures=0.7,
        max_tokens=4096,
    )
    logger.info(f"LLM model: {config.llm_model}")

    if not args.skip_prelim:
        ok = run_prelim(llm)
        if not ok:
            logger.error("Preliminary check FAILED. Fix issues before running campaign.")
            sys.exit(1)
        if args.prelim_only:
            logger.info("Prelim-only mode. Exiting.")
            return

    run_campaign(llm)


if __name__ == "__main__":
    main()
