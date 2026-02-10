"""
Memory x RL Interaction Experiment — Modal Serverless Training

Studies how context-space learning (evolving playbooks via DC/ACE curation) and
weight-space learning (GRPO) interact on DeepSeek-R1-Distill-Qwen-7B.

Conditions:
  B: GRPO-only (NullPlaybook, majority_vote reward)
  C: Static Playbook + GRPO (frozen playbook from pilot, majority_vote)
  D: Active Playbook + GRPO (evolving playbook, ACE reward + deferred curation)
  E: GRPO + Novelty Reward (NullPlaybook, novelty_augmented_reward)

Each (condition, seed) runs as an independent Modal function.

Usage:
    # Single run
    modal run experiments/memory_rl/run_modal.py --condition B --seed 42

    # All conditions, all seeds
    modal run experiments/memory_rl/run_modal.py --condition all

    # Download results
    modal volume get memory-rl-results results/ --force
"""

import modal

app = modal.App("memory-rl-train")

vol = modal.Volume.from_name("memory-rl-results", create_if_missing=True)

# Mount local lib/ into the container
lib_mount = modal.Mount.from_local_dir(
    "experiments/memory_rl/lib",
    remote_path="/root/lib",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "trl==0.27.2",
        "vllm==0.10.2",
        "transformers==4.57.3",
        "peft==0.18.1",
        "accelerate==1.12.0",
        "datasets",
        "scipy==1.15.2",
        "openai>=1.0",
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/results": vol},
    mounts=[lib_mount],
    timeout=8 * 3600,
    secrets=[modal.Secret.from_name("kimi-api-key", required=False)],
)
def train_condition(condition: str, seed: int, resume: bool = True):
    """Train one (condition, seed) pair."""
    import gc
    import json
    import os
    import random
    import re
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch

    # Add lib to path
    sys.path.insert(0, "/root")

    from lib.config import ExperimentConfig
    from lib.answer_parsing import parse_answer, check_answer, strip_think_blocks
    from lib.rewards import majority_vote, majority_vote_reward, ace_reward_fn, novelty_augmented_reward
    from lib.playbook import (
        Playbook, NullPlaybook, StaticPlaybook, ActivePlaybook,
        make_initial_playbook,
    )
    from lib.curator import rule_based_curate
    from lib.collapse_detector import CollapseDetector

    # -------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------
    CFG = ExperimentConfig()

    # Per-run paths
    run_id = f"{condition}_seed{seed}"
    run_dir = os.path.join(CFG.RESULTS_DIR, condition, f"seed{seed}")
    ckpt_dir = os.path.join(CFG.CHECKPOINTS_DIR, condition, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Seed everything
    # -------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------
    # GPU setup
    # -------------------------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    else:
        print("WARNING: No GPU detected.")

    print(f"\n{'='*60}")
    print(f"Training: condition={condition}, seed={seed}")
    print(f"Model: {CFG.MODEL_NAME}")
    print(f"Epochs: {CFG.GRPO_EPOCHS}")
    print(f"Run dir: {run_dir}")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------
    # Load training data
    # -------------------------------------------------------------------
    from lib.data import load_math500_l4l5

    problems = load_math500_l4l5()
    print(f"Loaded {len(problems)} MATH-500 L4-5 training problems")

    # -------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_system_prompt(playbook_context: str = "") -> str:
        base = (
            "You are an expert math competition solver. Solve the problem step-by-step.\n"
            "Show all your work clearly. At the end, put your final answer inside \\boxed{}.\n"
        )
        if playbook_context:
            base += playbook_context
        return base

    def format_prompt(problem: str, playbook_context: str = "") -> str:
        system = build_system_prompt(playbook_context)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Solve this problem:\n\n{problem}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # -------------------------------------------------------------------
    # Initialize playbook based on condition
    # -------------------------------------------------------------------
    if condition == "B":
        playbook_mgr = NullPlaybook()
    elif condition == "C":
        # Static playbook from pilot — load from JSON if available
        static_pb_path = os.path.join(CFG.RESULTS_DIR, "static_playbook.json")
        if os.path.exists(static_pb_path):
            playbook_mgr = StaticPlaybook.from_json(static_pb_path)
            print(f"Loaded static playbook from {static_pb_path}")
        else:
            # Fallback: use initial playbook as static
            print("WARNING: static_playbook.json not found, using initial playbook")
            playbook_mgr = StaticPlaybook(make_initial_playbook())
    elif condition == "D":
        playbook_mgr = ActivePlaybook()
    elif condition == "E":
        playbook_mgr = NullPlaybook()
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # -------------------------------------------------------------------
    # ACE+TTRL state (for condition D)
    # -------------------------------------------------------------------
    ace_state = {
        "playbook_mgr": playbook_mgr if condition == "D" else None,
        "problem_lookup": {},
        "pending_curate": [],
        "episode_stats": [],
    }

    # -------------------------------------------------------------------
    # Select reward function
    # -------------------------------------------------------------------
    if condition == "D":
        # ACE reward with playbook side-effects
        # We need to wrap ace_reward_fn to pass state
        def reward_fn(prompts, completions, **kwargs):
            return ace_reward_fn(prompts, completions, state=ace_state, **kwargs)
    elif condition == "E":
        # Novelty-augmented reward
        def reward_fn(prompts, completions, **kwargs):
            return novelty_augmented_reward(
                prompts, completions, novelty_weight=CFG.NOVELTY_WEIGHT, **kwargs
            )
    else:
        # Conditions B and C: standard majority vote
        reward_fn = majority_vote_reward

    # -------------------------------------------------------------------
    # GRPO setup
    # -------------------------------------------------------------------
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerState

    peft_config = LoraConfig(
        r=CFG.LORA_RANK,
        lora_alpha=CFG.LORA_ALPHA,
        target_modules=CFG.LORA_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
        use_rslora=True,
    )

    grpo_config = GRPOConfig(
        output_dir=ckpt_dir,
        num_train_epochs=1,  # We do manual epoch loop
        per_device_train_batch_size=CFG.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=CFG.LR,
        max_grad_norm=CFG.MAX_GRAD_NORM,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        num_generations=CFG.NUM_GENERATIONS,
        generation_batch_size=CFG.NUM_GENERATIONS * 2,
        max_completion_length=CFG.MAX_COMPLETION_LENGTH,
        temperature=1.0,
        top_p=0.95,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=CFG.VLLM_GPU_UTIL_COLOCATE,
        vllm_enable_sleep_mode=True,
        beta=CFG.KL_COEFF,
        report_to="none",
    )

    # -------------------------------------------------------------------
    # Check for resume from per-epoch checkpoint
    # -------------------------------------------------------------------
    epoch_ckpt_path = os.path.join(run_dir, "epoch_checkpoint.json")
    start_epoch = 0
    epoch_metrics = []
    collapse_detector = CollapseDetector()

    if resume and os.path.exists(epoch_ckpt_path):
        with open(epoch_ckpt_path) as f:
            epoch_ckpt = json.load(f)
        start_epoch = epoch_ckpt.get("completed_epoch", -1) + 1
        epoch_metrics = epoch_ckpt.get("epoch_metrics", [])

        # Restore playbook for condition D
        if condition == "D" and "playbook_snapshot" in epoch_ckpt:
            playbook_mgr.playbook = Playbook.from_snapshot(epoch_ckpt["playbook_snapshot"])
            ace_state["playbook_mgr"] = playbook_mgr
            print(f"Restored playbook: {playbook_mgr.playbook.size} bullets")

        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Check if already done
    done_marker = os.path.join(run_dir, "training_done.json")
    if resume and os.path.exists(done_marker):
        print(f"Training already complete for {run_id}. Skipping.")
        return

    # -------------------------------------------------------------------
    # Build initial prompts and create trainer
    # -------------------------------------------------------------------
    pb_ctx = playbook_mgr.get_context()
    initial_prompts = [format_prompt(p["problem"], pb_ctx) for p in problems]

    if condition == "D":
        ace_state["problem_lookup"] = {
            prompt: p for prompt, p in zip(initial_prompts, problems)
        }

    grpo_trainer = GRPOTrainer(
        model=CFG.MODEL_NAME,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=Dataset.from_dict({"prompt": initial_prompts}),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Restore LoRA weights if resuming
    if start_epoch > 0:
        lora_ckpt = os.path.join(ckpt_dir, "epoch_lora")
        if os.path.exists(lora_ckpt):
            from peft import set_peft_model_state_dict
            import safetensors.torch

            state_dict = safetensors.torch.load_file(
                os.path.join(lora_ckpt, "adapter_model.safetensors")
            )
            set_peft_model_state_dict(grpo_trainer.model, state_dict)
            print(f"Restored LoRA weights from {lora_ckpt}")

    print(f"GRPOTrainer initialized. Starting epoch loop...")

    # -------------------------------------------------------------------
    # Training epoch loop
    # -------------------------------------------------------------------
    def reset_trainer_state(trainer):
        trainer.state = TrainerState()
        trainer.lr_scheduler = None

    try:
        for epoch in range(start_epoch, CFG.GRPO_EPOCHS):
            epoch_start = time.time()

            # Rebuild prompts with current playbook
            pb_ctx = playbook_mgr.get_context()
            epoch_prompts = [format_prompt(p["problem"], pb_ctx) for p in problems]

            grpo_trainer.train_dataset = Dataset.from_dict({"prompt": epoch_prompts})

            if condition == "D":
                ace_state["problem_lookup"] = {
                    prompt: p for prompt, p in zip(epoch_prompts, problems)
                }
                ace_state["pending_curate"] = []

            # Reset trainer state (critical for repeated .train() calls)
            reset_trainer_state(grpo_trainer)

            # Train 1 epoch
            grpo_trainer.train()

            # -----------------------------------------------------------
            # Post-epoch: condition D deferred curation
            # -----------------------------------------------------------
            if condition == "D" and ace_state["pending_curate"]:
                # Try Kimi API first, fallback to rule-based
                try:
                    kimi_available = os.environ.get("KIMI_API_KEY", "")
                    if kimi_available:
                        from lib.reflector import batch_reflect
                        from lib.curator import curate as kimi_curate

                        reflect_items = ace_state["pending_curate"][:10]  # Limit API calls
                        reflections = batch_reflect(
                            reflect_items, playbook_mgr.playbook.to_str()
                        )
                        for (reflection, tags), item in zip(reflections, reflect_items):
                            playbook_mgr.apply_tags(tags)
                            ops = kimi_curate(
                                playbook_mgr.playbook,
                                item["problem"],
                                reflection,
                                max_bullets=CFG.MAX_BULLETS,
                            )
                            playbook_mgr.apply_ops(ops, max_bullets=CFG.MAX_BULLETS)
                    else:
                        raise ValueError("No Kimi API key")
                except Exception as e:
                    print(f"  Kimi curation failed ({e}), using rule-based fallback")
                    rule_based_curate(
                        playbook_mgr.playbook,
                        ace_state["pending_curate"],
                        max_bullets=CFG.MAX_BULLETS,
                    )

            # -----------------------------------------------------------
            # Collapse detection
            # -----------------------------------------------------------
            # Collect all completions from this epoch for entropy
            # We approximate by re-using pending_curate answers for D,
            # or the prompts count for others
            if condition == "D" and ace_state.get("episode_stats"):
                # Use confidence as proxy for diversity
                stats = ace_state["episode_stats"]
                epoch_answers = [
                    str(s.get("is_correct", False)) for s in stats[-len(problems):]
                ]
            else:
                # For B/C/E we don't have direct access to completions
                # Log placeholder; real detection from training logs
                epoch_answers = ["1"] * len(problems)

            collapse_record = collapse_detector.record_epoch(epoch_answers, epoch=epoch)

            # -----------------------------------------------------------
            # Metrics
            # -----------------------------------------------------------
            pb_size = 0
            pb_entropy = 0.0
            if condition == "D" and hasattr(playbook_mgr, "playbook"):
                pb_size = playbook_mgr.playbook.size
                pb_entropy = playbook_mgr.playbook.entropy()

            epoch_time = time.time() - epoch_start
            metrics = {
                "epoch": epoch,
                "time_s": epoch_time,
                "pb_size": pb_size,
                "pb_entropy": pb_entropy,
                "answer_entropy": collapse_record.entropy,
                "is_collapsed": collapse_record.is_collapsed,
            }
            epoch_metrics.append(metrics)

            print(
                f"  Epoch {epoch+1}/{CFG.GRPO_EPOCHS}: "
                f"pb_size={pb_size} entropy={collapse_record.entropy:.3f} "
                f"time={epoch_time:.0f}s"
            )

            # -----------------------------------------------------------
            # Checkpoint
            # -----------------------------------------------------------
            lora_ckpt_dir = os.path.join(ckpt_dir, "epoch_lora")
            grpo_trainer.save_model(lora_ckpt_dir)

            ckpt_data = {
                "completed_epoch": epoch,
                "condition": condition,
                "seed": seed,
                "epoch_metrics": epoch_metrics,
            }
            if condition == "D":
                ckpt_data["playbook_snapshot"] = playbook_mgr.playbook.snapshot()
            with open(epoch_ckpt_path, "w") as f:
                json.dump(ckpt_data, f, indent=2, default=str)

            vol.commit()

    except Exception as e:
        print(f"Training error at epoch {epoch}: {e}")
        import traceback
        traceback.print_exc()
        # Save what we have
        with open(os.path.join(run_dir, "error.json"), "w") as f:
            json.dump({"error": str(e), "epoch": epoch}, f)
        vol.commit()
        raise

    # -------------------------------------------------------------------
    # Save final adapter
    # -------------------------------------------------------------------
    final_adapter_path = os.path.join(ckpt_dir, "final_adapter")
    grpo_trainer.save_model(final_adapter_path)

    # Save final playbook for condition D
    if condition == "D":
        with open(os.path.join(run_dir, "final_playbook.json"), "w") as f:
            json.dump(playbook_mgr.playbook.snapshot(), f, indent=2)

    # Save training done marker
    with open(done_marker, "w") as f:
        json.dump({
            "completed": True,
            "condition": condition,
            "seed": seed,
            "epochs": CFG.GRPO_EPOCHS,
            "epoch_metrics": epoch_metrics,
            "timestamp": time.time(),
        }, f, indent=2, default=str)

    # Merge LoRA with base for evaluation
    print("Merging LoRA adapter with base model...")
    from lib.evaluation import merge_lora_checkpoint

    merged_path = os.path.join(ckpt_dir, "merged")
    merge_lora_checkpoint(CFG.MODEL_NAME, final_adapter_path, merged_path)

    vol.commit()

    print(f"\n{'='*60}")
    print(f"Training complete: {run_id}")
    print(f"Final adapter: {final_adapter_path}")
    print(f"Merged model: {merged_path}")
    print(f"{'='*60}")

    # Cleanup
    del grpo_trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "condition": condition,
        "seed": seed,
        "epochs": CFG.GRPO_EPOCHS,
        "epoch_metrics": epoch_metrics,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    condition: str = "B",
    seed: int = 42,
    batch: int = 5,
    resume: bool = True,
):
    """Run training for Memory x RL experiment.

    Args:
        condition: B, C, D, E, or "all" for all conditions
        seed: Random seed, or -1 for all seeds
        batch: Max concurrent jobs (for "all" mode)
        resume: Whether to resume from checkpoints
    """
    from lib.config import ExperimentConfig

    CFG = ExperimentConfig()

    if condition == "all":
        conditions = CFG.TRAINING_CONDITIONS
    else:
        conditions = [condition]

    if seed == -1:
        seeds = CFG.SEEDS
    else:
        seeds = [seed]

    # Build job list
    jobs = [(c, s) for c in conditions for s in seeds]
    print(f"Launching {len(jobs)} training jobs (batch={batch}):")
    for c, s in jobs:
        print(f"  {c} seed={s}")

    if len(jobs) == 1:
        # Single job: run synchronously
        result = train_condition.remote(jobs[0][0], jobs[0][1], resume)
        print(f"Result: {result}")
    else:
        # Multiple jobs: spawn with batch control
        futures = []
        for i, (c, s) in enumerate(jobs):
            # Respect batch limit
            while len([f for f in futures if not f.object_id]) >= batch:
                import time
                time.sleep(10)

            f = train_condition.spawn(c, s, resume)
            futures.append(f)
            print(f"  Spawned {c} seed={s} ({i+1}/{len(jobs)})")

        # Wait for all
        print(f"\nWaiting for {len(futures)} jobs to complete...")
        results = []
        for i, f in enumerate(futures):
            try:
                result = f.get()
                results.append(result)
                c, s = jobs[i]
                print(f"  Completed: {c} seed={s}")
            except Exception as e:
                c, s = jobs[i]
                print(f"  FAILED: {c} seed={s}: {e}")

        print(f"\nAll jobs done. {len(results)}/{len(jobs)} succeeded.")
        print("Download results: modal volume get memory-rl-results results/ --force")
