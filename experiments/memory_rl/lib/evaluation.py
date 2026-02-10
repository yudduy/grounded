"""Evaluation pipeline for Memory x RL experiment.

Handles:
- LoRA adapter merging
- vLLM batch inference on eval datasets
- Per-problem binary outcome collection
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .answer_parsing import check_answer, parse_answer
from .rewards import majority_vote


# ---------------------------------------------------------------------------
# LoRA merging
# ---------------------------------------------------------------------------

def merge_lora_checkpoint(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
) -> str:
    """Merge LoRA adapter with base model and save.

    Args:
        base_model_name: HuggingFace model name for base weights
        adapter_path: Path to LoRA adapter directory
        output_path: Where to save merged model

    Returns:
        Path to merged model directory
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Merging LoRA: {adapter_path} -> {output_path}")
    t0 = time.time()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    del base_model, peft_model, merged
    gc.collect()
    print(f"  Merged in {time.time() - t0:.1f}s -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# vLLM evaluation
# ---------------------------------------------------------------------------

def batch_generate_eval(
    llm,
    prompts: List[str],
    n_generations: int = 16,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stop_tokens: Optional[List[str]] = None,
) -> List[List[Dict]]:
    """Run vLLM batch generation for evaluation.

    Returns:
        List of lists of candidate dicts: [{"answer", "raw"}]
    """
    from vllm import SamplingParams

    if stop_tokens is None:
        # DeepSeek-R1 uses special Unicode stop token
        stop_tokens = ["<|endoftext|>", "<\uff5cend\u2581of\u2581sentence\uff5c>"]

    sampling_params = SamplingParams(
        n=n_generations,
        temperature=max(temperature, 0.01),
        max_tokens=max_tokens,
        stop=stop_tokens,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    all_candidates = []
    for request_output in outputs:
        candidates = []
        for completion in request_output.outputs:
            text = completion.text
            answer = parse_answer(text)
            candidates.append({"answer": answer, "raw": text})
        all_candidates.append(candidates)
    return all_candidates


def evaluate_checkpoint(
    model_path: str,
    datasets: Dict[str, List[Dict]],
    n_generations: int = 16,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 4096,
    playbook_context: str = "",
    tokenizer_name: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """Evaluate a merged checkpoint on multiple datasets.

    Args:
        model_path: Path to merged model
        datasets: {dataset_name: [{"id", "problem", "answer"}, ...]}
        n_generations: Number of generations per problem
        temperature: Sampling temperature
        max_tokens: Max completion length
        gpu_memory_utilization: vLLM GPU memory fraction
        max_model_len: Maximum model sequence length
        playbook_context: Optional playbook context to prepend
        tokenizer_name: Tokenizer to use (defaults to model_path)

    Returns:
        {dataset_name: [{"id", "problem", "predicted", "ground_truth", "correct"}, ...]}
    """
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    if tokenizer_name is None:
        tokenizer_name = model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading vLLM engine: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=512,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        seed=42,
    )

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

    # DeepSeek-R1 stop tokens
    stop_tokens = ["<|endoftext|>", "<\uff5cend\u2581of\u2581sentence\uff5c>"]

    all_results = {}

    for ds_name, problems in datasets.items():
        print(f"\n  Evaluating {ds_name} ({len(problems)} problems)...")
        t0 = time.time()

        prompts = [format_prompt(p["problem"]) for p in problems]
        all_candidates = batch_generate_eval(
            llm, prompts, n_generations, temperature, max_tokens, stop_tokens
        )

        ds_results = []
        correct_count = 0
        for problem, candidates in zip(problems, all_candidates):
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

        acc = correct_count / len(problems) if problems else 0.0
        elapsed = time.time() - t0
        print(f"    {ds_name}: {acc:.1%} ({correct_count}/{len(problems)}) in {elapsed:.0f}s")
        all_results[ds_name] = ds_results

    # Cleanup
    from vllm.distributed.parallel_state import destroy_model_parallel

    del llm
    destroy_model_parallel()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


def evaluate_all_conditions(
    conditions_config: Dict[str, Dict],
    checkpoints_dir: str,
    eval_datasets: Dict[str, List[Dict]],
    base_model_name: str,
    n_generations: int = 16,
    output_dir: str = "/results/eval",
) -> Dict[str, Dict]:
    """Evaluate all conditions x seeds on all eval datasets.

    Args:
        conditions_config: {condition: {seeds: [...], playbook_path: ..., ...}}
        checkpoints_dir: Base dir for checkpoints
        eval_datasets: {dataset_name: problems_list}
        base_model_name: HuggingFace model name
        n_generations: Generations per problem
        output_dir: Where to save results

    Returns:
        {condition_seed_key: {dataset: results_list}}
    """
    os.makedirs(output_dir, exist_ok=True)
    all_eval_results = {}

    for condition, config in conditions_config.items():
        seeds = config.get("seeds", [42])
        for seed in seeds:
            key = f"{condition}_seed{seed}"
            result_dir = os.path.join(output_dir, condition, f"seed{seed}")
            os.makedirs(result_dir, exist_ok=True)

            # Check if already done
            done_marker = os.path.join(result_dir, "eval_done.json")
            if Path(done_marker).exists():
                print(f"  {key}: already evaluated, loading results")
                results = {}
                for ds_name in eval_datasets:
                    ds_path = os.path.join(result_dir, f"{ds_name}.json")
                    if Path(ds_path).exists():
                        with open(ds_path) as f:
                            results[ds_name] = json.load(f)
                all_eval_results[key] = results
                continue

            # Determine model path
            adapter_path = os.path.join(
                checkpoints_dir, condition, f"seed{seed}", "final_adapter"
            )
            merged_path = os.path.join(
                checkpoints_dir, condition, f"seed{seed}", "merged"
            )

            if not Path(merged_path).exists():
                if Path(adapter_path).exists():
                    merge_lora_checkpoint(base_model_name, adapter_path, merged_path)
                else:
                    print(f"  WARNING: No checkpoint for {key}, skipping")
                    continue

            # Determine playbook context
            playbook_context = ""
            if condition in ("C", "D"):
                pb_path = config.get("playbook_path", "")
                if pb_path and Path(pb_path).exists():
                    with open(pb_path) as f:
                        pb_data = json.load(f)
                    from .playbook import Playbook

                    pb = Playbook.from_snapshot(pb_data)
                    playbook_context = (
                        f"\nPLAYBOOK (use these strategies):\n{pb.to_str()}"
                    )
            # C-abs and D-abs: explicitly NO playbook
            if condition in ("C-abs", "D-abs"):
                playbook_context = ""

            # Run evaluation
            results = evaluate_checkpoint(
                model_path=merged_path,
                datasets=eval_datasets,
                n_generations=n_generations,
                playbook_context=playbook_context,
            )

            # Save per-dataset results
            for ds_name, ds_results in results.items():
                ds_path = os.path.join(result_dir, f"{ds_name}.json")
                with open(ds_path, "w") as f:
                    json.dump(ds_results, f, indent=2)

            with open(done_marker, "w") as f:
                json.dump({"completed": True, "timestamp": time.time()}, f)

            all_eval_results[key] = results

    return all_eval_results
