"""
Memory x RL Interaction Experiment — Modal Serverless Evaluation

Evaluates trained checkpoints on held-out benchmarks:
  - AIME 2025 (30 problems)
  - OlymMATH-EASY (~100 problems)
  - AMC 12 2024-2025 (50 problems)

Handles 6 condition variants:
  B, C, D, E: evaluate with their training-time playbook context
  C-abs: C's trained model evaluated WITHOUT playbook
  D-abs: D's trained model evaluated WITHOUT playbook

Usage:
    # Single condition + seed
    modal run experiments/memory_rl/eval_modal.py --condition B --seed 42

    # All conditions, all seeds
    modal run experiments/memory_rl/eval_modal.py --condition all

    # Download results
    modal volume get memory-rl-results results/ --force
"""

import modal

app = modal.App("memory-rl-eval")

vol = modal.Volume.from_name("memory-rl-results", create_if_missing=True)

lib_mount = modal.Mount.from_local_dir(
    "experiments/memory_rl/lib",
    remote_path="/root/lib",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.10.2",
        "transformers==4.57.3",
        "peft==0.18.1",
        "accelerate==1.12.0",
        "datasets",
        "scipy==1.15.2",
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/results": vol},
    mounts=[lib_mount],
    timeout=4 * 3600,
)
def eval_condition(condition: str, seed: int):
    """Evaluate one (condition, seed) on all eval datasets."""
    import gc
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import torch

    sys.path.insert(0, "/root")

    from lib.config import ExperimentConfig
    from lib.data import load_eval_datasets
    from lib.answer_parsing import parse_answer, check_answer
    from lib.rewards import majority_vote

    CFG = ExperimentConfig()

    # -------------------------------------------------------------------
    # Resolve condition to actual model path
    # -------------------------------------------------------------------
    # C-abs uses C's model, D-abs uses D's model
    model_condition = condition.replace("-abs", "")
    use_playbook = condition in ("C", "D")  # abs variants explicitly skip playbook

    run_id = f"{condition}_seed{seed}"
    result_dir = os.path.join(CFG.RESULTS_DIR, "eval", condition, f"seed{seed}")
    os.makedirs(result_dir, exist_ok=True)

    # Check if already done
    done_marker = os.path.join(result_dir, "eval_done.json")
    if os.path.exists(done_marker):
        print(f"Evaluation already complete for {run_id}. Skipping.")
        with open(done_marker) as f:
            return json.load(f)

    # Find merged model
    merged_path = os.path.join(
        CFG.CHECKPOINTS_DIR, model_condition, f"seed{seed}", "merged"
    )
    if not os.path.exists(merged_path):
        # Try merging from adapter
        adapter_path = os.path.join(
            CFG.CHECKPOINTS_DIR, model_condition, f"seed{seed}", "final_adapter"
        )
        if os.path.exists(adapter_path):
            from lib.evaluation import merge_lora_checkpoint
            merge_lora_checkpoint(CFG.MODEL_NAME, adapter_path, merged_path)
        else:
            raise FileNotFoundError(
                f"No checkpoint found for {model_condition} seed={seed} "
                f"at {merged_path} or {adapter_path}"
            )

    # -------------------------------------------------------------------
    # Load playbook context (if applicable)
    # -------------------------------------------------------------------
    playbook_context = ""
    if use_playbook:
        if condition == "C":
            # Static playbook
            pb_path = os.path.join(CFG.RESULTS_DIR, "static_playbook.json")
            if os.path.exists(pb_path):
                from lib.playbook import StaticPlaybook
                pb_mgr = StaticPlaybook.from_json(pb_path)
                playbook_context = pb_mgr.get_context()
                print(f"Using static playbook from {pb_path}")
        elif condition == "D":
            # Active playbook from training
            pb_path = os.path.join(
                CFG.RESULTS_DIR, "D", f"seed{seed}", "final_playbook.json"
            )
            if os.path.exists(pb_path):
                from lib.playbook import Playbook
                with open(pb_path) as f:
                    pb_data = json.load(f)
                pb = Playbook.from_snapshot(pb_data)
                playbook_context = (
                    f"\nPLAYBOOK (use these strategies, reference IDs like [str-00001]):\n"
                    f"{pb.to_str()}"
                )
                print(f"Using D's playbook from {pb_path} ({pb.size} bullets)")

    print(f"\n{'='*60}")
    print(f"Evaluating: {run_id}")
    print(f"Model: {merged_path}")
    print(f"Playbook: {'yes' if playbook_context else 'no'}")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------
    # Load eval datasets
    # -------------------------------------------------------------------
    datasets = load_eval_datasets()
    for name, probs in datasets.items():
        print(f"  {name}: {len(probs)} problems")

    # -------------------------------------------------------------------
    # Load tokenizer + vLLM engine
    # -------------------------------------------------------------------
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(merged_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompt(problem: str) -> str:
        system = (
            "You are an expert math competition solver. Solve the problem step-by-step.\n"
            "Show all your work clearly. At the end, put your final answer inside \\boxed{}.\n"
        )
        if playbook_context:
            system += playbook_context
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Solve this problem:\n\n{problem}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    print("Initializing vLLM engine...")
    llm = LLM(
        model=merged_path,
        dtype="bfloat16",
        gpu_memory_utilization=CFG.VLLM_GPU_UTIL_FROZEN,
        max_model_len=CFG.VLLM_MAX_MODEL_LEN,
        max_num_seqs=512,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        seed=seed,
    )

    # DeepSeek-R1 stop tokens
    stop_tokens = ["<|endoftext|>", "<\uff5cend\u2581of\u2581sentence\uff5c>"]

    sampling_params = SamplingParams(
        n=CFG.NUM_GENERATIONS,
        temperature=max(CFG.TEMPERATURE, 0.01),
        max_tokens=CFG.MAX_COMPLETION_LENGTH,
        stop=stop_tokens,
    )

    # -------------------------------------------------------------------
    # Evaluate each dataset
    # -------------------------------------------------------------------
    all_results = {}
    total_correct = 0
    total_problems = 0

    for ds_name, problems in datasets.items():
        if not problems:
            print(f"  Skipping {ds_name} (no problems)")
            continue

        print(f"\n  Evaluating {ds_name} ({len(problems)} problems)...")
        t0 = time.time()

        prompts = [format_prompt(p["problem"]) for p in problems]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        ds_results = []
        correct_count = 0

        for problem, output in zip(problems, outputs):
            candidates = []
            for completion in output.outputs:
                text = completion.text
                answer = parse_answer(text)
                candidates.append({"answer": answer, "raw": text[:500]})

            answers = [c["answer"] for c in candidates]
            winner, confidence = majority_vote(answers)
            is_correct = check_answer(winner, problem["answer"])

            correct_count += int(is_correct)
            ds_results.append({
                "id": problem["id"],
                "problem": problem["problem"][:200],
                "predicted": winner,
                "ground_truth": problem["answer"],
                "correct": is_correct,
                "confidence": confidence,
                "n_candidates": len(candidates),
            })

        acc = correct_count / len(problems)
        elapsed = time.time() - t0
        print(f"    {ds_name}: {acc:.1%} ({correct_count}/{len(problems)}) in {elapsed:.0f}s")

        # Save per-dataset results
        ds_path = os.path.join(result_dir, f"{ds_name}.json")
        with open(ds_path, "w") as f:
            json.dump(ds_results, f, indent=2)

        all_results[ds_name] = {
            "accuracy": acc,
            "correct": correct_count,
            "total": len(problems),
            "results": ds_results,
        }

        total_correct += correct_count
        total_problems += len(problems)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    overall_acc = total_correct / total_problems if total_problems > 0 else 0.0
    summary = {
        "condition": condition,
        "seed": seed,
        "model_condition": model_condition,
        "use_playbook": use_playbook,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_problems": total_problems,
        "per_dataset": {
            name: {"accuracy": r["accuracy"], "correct": r["correct"], "total": r["total"]}
            for name, r in all_results.items()
        },
        "timestamp": time.time(),
    }

    with open(done_marker, "w") as f:
        json.dump(summary, f, indent=2)

    # Save summary
    with open(os.path.join(result_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()

    print(f"\n{'='*60}")
    print(f"Evaluation complete: {run_id}")
    print(f"Overall: {overall_acc:.1%} ({total_correct}/{total_problems})")
    for name, r in all_results.items():
        print(f"  {name}: {r['accuracy']:.1%}")
    print(f"{'='*60}")

    # Cleanup
    from vllm.distributed.parallel_state import destroy_model_parallel

    del llm
    destroy_model_parallel()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    condition: str = "B",
    seed: int = 42,
    batch: int = 5,
):
    """Run evaluation for Memory x RL experiment.

    Args:
        condition: B, C, C-abs, D, D-abs, E, or "all"
        seed: Random seed, or -1 for all seeds
        batch: Max concurrent jobs
    """
    from lib.config import ExperimentConfig

    CFG = ExperimentConfig()

    ALL_EVAL_CONDITIONS = CFG.TRAINING_CONDITIONS + CFG.EVAL_ONLY_CONDITIONS

    if condition == "all":
        conditions = ALL_EVAL_CONDITIONS
    else:
        conditions = [condition]

    if seed == -1:
        seeds = CFG.SEEDS
    else:
        seeds = [seed]

    # Build job list
    jobs = []
    for c in conditions:
        for s in seeds:
            jobs.append((c, s))

    print(f"Launching {len(jobs)} evaluation jobs (batch={batch}):")
    for c, s in jobs:
        print(f"  {c} seed={s}")

    if len(jobs) == 1:
        result = eval_condition.remote(jobs[0][0], jobs[0][1])
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        import json

        futures = []
        for i, (c, s) in enumerate(jobs):
            f = eval_condition.spawn(c, s)
            futures.append(f)
            print(f"  Spawned {c} seed={s} ({i+1}/{len(jobs)})")

        print(f"\nWaiting for {len(futures)} jobs to complete...")
        results = []
        for i, f in enumerate(futures):
            try:
                result = f.get()
                results.append(result)
                c, s = jobs[i]
                print(f"  Completed: {c} seed={s} -> {result.get('overall_accuracy', 0):.1%}")
            except Exception as e:
                c, s = jobs[i]
                print(f"  FAILED: {c} seed={s}: {e}")

        print(f"\nAll evaluations done. {len(results)}/{len(jobs)} succeeded.")
        print("Download results: modal volume get memory-rl-results results/ --force")
