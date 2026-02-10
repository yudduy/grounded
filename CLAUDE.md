# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Stanford FarmShare shared cluster (rice-XX nodes). Home directories are NFS-mounted.

### Critical Rules

- **NEVER run tests interactively on the login node.** Always submit compute-heavy work (tests, training, builds) as SLURM batch jobs.
- After spawning any long-running process, track the PID so it can be cleaned up if the session dies.
- Use `timeout` when running tests to prevent zombie processes (e.g., `timeout 300 pytest ...`).
- **Always commit changes when task is complete.** Do not include `Co-Authored-By` lines.

---

## Commands

### Running Experiments

```bash
# TTRL + ACE on AIME 2024 (Modal serverless, A100-80GB)
modal run experiments/ttrl_dc_aime/run_modal.py
modal volume get ace-ttrl-results results/ --force

# DC vs Verification on AIME 2024 (requires local vLLM server)
python experiments/dc_vs_verification/run.py --condition dc --seeds 3

# Dynamic Cheatsheet benchmark (Game of 24, etc.)
cd dynamic-cheatsheet && python run_benchmark.py --task GameOf24

# ACE on finance tasks
cd ace && pip install -r requirements.txt
python -m eval.finance.run --task_name finer --mode offline --save_path results
```

### Running Tests (via SLURM)

```bash
# NEVER run directly on login nodes. Use sbatch:
sbatch --wrap="cd /home/users/duynguy/proj/grounded/src && timeout 300 pytest tests/ -v" --partition=normal

# Single test
sbatch --wrap="cd /home/users/duynguy/proj/grounded/src && timeout 120 pytest tests/test_environments.py::TestBaseEnvironment::test_evaluate_shape -v"
```

### Installing Dependencies

```bash
# src/ package (equation discovery)
cd src && pip install -e ".[dev]"

# ACE framework
cd ace && pip install -r requirements.txt   # openai, tiktoken, faiss-cpu, sentence-transformers

# ShinkaEvolve (code evolution)
cd ShinkaEvolve && pip install -e .
```

---

## Architecture

### Four Subprojects

This repo contains four loosely coupled subprojects. They share concepts (playbooks, bandits, verification) but have separate codebases:

**`src/`** — Equation discovery loop (the "grounded discovery" experiment). Python package `grounded-discovery`.
- `environments/base.py`: `BaseEnvironment` ABC — defines hidden physics laws as `_ground_truth()` + `EnvironmentSpec`. Tier 1 = single-equation, Tier 2 = coupled systems.
- `loop/orchestrator.py`: `DiscoveryLoop` — runs CHOOSE → OBSERVE → HYPOTHESIZE → FIT → EVALUATE → REFLECT → CURATE. Conditions plug in via `ConditionStrategy` protocol.
- `conditions/`: Experimental conditions A-R. `StaticCondition` (baseline) → `ACECondition` (adds playbook) → `ACEGradientCondition` (adds gradient fitting). Each overrides `hypothesize()`, `reflect()`, `curate()`.
- `campaign/runner.py`: Orchestrates all env × condition × seed combinations. Uses SQLite for checkpointing.
- `gradient/fitter.py`: Scipy curve fitting with SLURM wrapper for GPU.

**`ace/`** — ACE framework (Agentic Context Engineering). Three-agent architecture:
- `Generator` → produces answers using playbook context
- `Reflector` → tags playbook bullets as helpful/harmful
- `Curator` → evolves playbook via ADD/UPDATE/MERGE/DELETE ops
- Token budget enforced (~80K default). Playbook bullets have IDs like `[str-00001]` with helpful/harmful counters.
- New domains: implement `DataProcessor` with `process_task_data()`, `answer_is_correct()`, `evaluate_accuracy()`.

**`ShinkaEvolve/`** — Code evolution engine (fork of EvoCodeBench-style system). Evolves Python programs via LLM-guided mutation.
- `shinka/core/runner.py`: `EvolutionConfig` + main evolution loop. Uses island model with LLM-based crossover.
- `shinka/database/`: `ProgramDatabase` tracks evolved programs, fitness, lineage.
- `shinka/launch/`: Job scheduling — `local.py` and `slurm.py` backends.
- `shinka/llm/`: Multi-provider LLM client (OpenAI, Anthropic, Gemini, DeepSeek) with bandit-based dynamic sampling.
- `examples/adas_aime/`: AIME math benchmark with evolved solvers.

**`dynamic-cheatsheet/`** — Reference implementation of Dynamic Cheatsheet (DC) prompting.
- `run_benchmark.py`: Entry point. `--task GameOf24|AIME|...`, `--model_name openai/gpt-4o-mini`.
- `dynamic_cheatsheet/language_model.py`: LLM wrapper with cheatsheet-augmented generation.

### Notebooks

`notebooks/` contains the primary PoC experiments as Jupyter notebooks. These are self-contained — they inline all experiment logic rather than importing from `src/`.
- `search_augmented_ace_poc.ipynb`: GSM8K, 14 conditions (UCB, Thompson, PUCT, Beam Search, etc.)
- `verified_archetype_discovery_poc.ipynb`: Game of 24, archetype clustering pipeline

### Experiments

`experiments/` contains standalone experiment scripts that may pull data from `ShinkaEvolve/` (e.g., AIME CSV datasets) but are otherwise self-contained.

---

## Research Context

Investigates **verification-driven reasoning** and **self-improving memory** for LLMs.

### Key Findings

- **Strategy space coverage > search sophistication**: UCB bandit (85%) matches Majority Vote at 1x budget. Beam Search (90%) needs 3x budget. PUCT variants plateau at 75-80%.
- **Playbook poisoning**: DC accuracy degrades over time (27%→17% on AIME), suggesting curated strategies accumulate noise.
- **Archetype sweet spot**: 4 curated archetypes = 3.6x improvement; 10 archetypes = back to baseline. Verification threshold was the bottleneck.
- **Verification adds little over DC alone**: +1.1% on AIME — verifier was over-engineered.

### Playbook Format

```
## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=0 :: Always verify data types before processing
```

Curator operations: ADD, UPDATE, MERGE, DELETE.

---

## Gotchas

- **vLLM/torch version chain**: vLLM pins exact torch versions (0.12.0→torch 2.9.0, 0.10.2→torch 2.8.0). Never pin `torch` independently — let vLLM's dependency pull the correct version.
- **`re.findall` with capturing groups**: `re.findall(r"\[(str|err)-\d{5}\]", text)` returns `['str']` not `['str-00001']`. Use non-capturing inner group: `r"\[((?:str|err)-\d{5})\]"`.
- **GRPO confidence-weighted rewards**: Soft rewards (0.0-1.0) dilute gradient signal. At low confidence (~1/N), reward std → tiny → advantage explosion. Use binary rewards.
- **flash-attn wheels**: No prebuilt wheels for torch >= 2.7. Omit flash-attn when using vLLM ≥ 0.12 (it handles attention internally).
- **TRL 0.27+ GRPO**: `generation_batch_size` must be divisible by `num_generations`. Set explicitly to `num_generations`. `max_prompt_length` is removed — use `max_completion_length` only.
- **vLLM ≥ 0.11.x perf regression**: 2.4x slower with TRL GRPOTrainer colocate mode (issue #4897). Only 0.10.2 has full performance.

See `KNOWLEDGE.md` for detailed technical notes on MCTS/PUCT, Thompson Sampling, vLLM GPU patterns, and async parallelization.

- **vLLM eval: Python API vs HTTP server**: For notebook/script evaluation, always use `vllm.LLM` + `llm.generate(prompts, SamplingParams)` (offline batch API) rather than launching an HTTP server + OpenAI client. Offline batch is ~10-30x faster than sequential HTTP calls because: (1) no HTTP/JSON serialization overhead, (2) all prompts enter the continuous batching scheduler at once, (3) no per-request network round-trip. The HTTP server pattern is only needed for multi-process or remote clients. See `eval_modal.py` and `lib/evaluation.py` for canonical patterns.
- **vLLM engine reuse for absorption tests**: When evaluating conditions that share the same LoRA adapter but differ only in system prompt (e.g., C vs C-abs, D vs D-abs), merge the adapter ONCE and load the vLLM engine ONCE, then run multiple `llm.generate()` passes with different prompts. Each vLLM engine load takes 2-8 min; grouping saves ~10 min per absorption pair. Use `destroy_model_parallel()` + `torch.cuda.empty_cache()` between groups to reclaim GPU memory.
- **vLLM LoRA hot-swap (0.10.2+)**: vLLM supports `--enable-lora` for multi-LoRA serving without merging. Adapters are served per-request via `model="adapter_name"`. For RL experiments, `load_inplace=True` enables adapter swap without server restart. This eliminates all LoRA merges but requires the base architecture to implement `SupportsLoRA` (Qwen2 does). Use for full-scale experiments; for pilots, grouped merge-once approach is simpler.
