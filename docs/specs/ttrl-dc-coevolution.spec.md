# Specification: TTRL + DC/ACE Co-Evolution on AIME 2024

> To implement this spec, clear context and run:
> `/duy-workflow:execute docs/specs/ttrl-dc-coevolution.spec.md`

## Goal

Test whether interleaved co-evolution of model weights (TTRL/GRPO) and strategy memory (DC/ACE playbook) produces synergistic improvement beyond either alone on AIME 2024, using Qwen-2.5-Math-7B on a single A100 40GB GPU.

## Requirements

### [REQ-1] Modular component architecture
Define ABC interfaces for 6 pluggable components: Generator, Evaluator, Curator, Trainer, PlaybookManager, CurriculumSelector. Each experiment condition is a configuration of component implementations. Adding a new component (e.g., GPT-4o-mini verifier) requires only a new class.
- Acceptance: Components can be swapped independently via condition config dict.

### [REQ-2] TTRL baseline replication
Implement GRPO training with majority-voting reward on Qwen-2.5-Math-7B using TRL's GRPOTrainer with vLLM colocate mode. LoRA rank=16 for A100 40GB feasibility. 16 generations per prompt. KL coefficient = 0.0. 30-60 episodes over 30 AIME problems.
- Acceptance: TTRL-only condition shows measurable improvement over baseline (>5% absolute) after 30 episodes. Per-episode pass@1 tracked.

### [REQ-3] DC/ACE playbook integration
Implement ActivePlaybook with reflect+curate pipeline reused from `experiments/dc_vs_verification/run.py`. Bullet dataclass with (id, section, content, helpful, harmful). Playbook context injected into system prompt. MAX_BULLETS=20. Curate operations: ADD/UPDATE/DELETE.
- Acceptance: DC-only condition maintains evolving playbook. Playbook size and bullet contents tracked per problem.

### [REQ-4] Interleaved co-evolution loop
For DC+TTRL: each episode processes all 30 problems in batches. Per batch: (1) generate 16 candidates with playbook context, (2) majority vote -> per-candidate reward, (3) GRPO weight update via vLLM sleep/wake, (4) reflect on best candidate -> tag playbook bullets, (5) curate playbook. DC operations happen in the reward function callback while vLLM is still awake.
- Acceptance: DC+TTRL runs end-to-end. Both playbook snapshots and model checkpoints saved per episode.

### [REQ-5] 4-condition ablation
| Condition | Playbook | Trainer | Evaluator |
|-----------|----------|---------|-----------|
| Baseline | Null | None | GroundTruth |
| DC-only | Active | None | MajVote |
| TTRL-only | Null | GRPO | MajVote |
| DC+TTRL | Active | GRPO | MajVote |

Run all on 30 AIME 2024 problems. Baseline uses frozen model, single generation, ground-truth check. DC-only uses frozen model with playbook evolution. TTRL-only evolves weights. DC+TTRL evolves both.
- Acceptance: All 4 conditions produce per-episode accuracy and final pass@1.

### [REQ-6] Analysis and decision framework
Bootstrap CI (B=10000) on accuracy differences. McNemar's test on per-problem binary outcomes (paired). Plots: (a) accuracy over episodes (4 conditions), (b) playbook size over time, (c) reward accuracy (maj vote label quality), (d) final accuracy bar chart with CIs. Decision verdict:
- DC+TTRL >> TTRL-only (>10%): co-evolution wins, publish
- DC+TTRL ~= TTRL-only (<5%): GRPO subsumes playbook
- DC+TTRL < TTRL-only: playbook interferes with RL
- Acceptance: 4 plots + statistical table + verdict printed.

### [REQ-7] AIME data loading and answer verification
Load 30 AIME 2024 problems from `ShinkaEvolve/examples/adas_aime/AIME_Dataset_1983_2025.csv`. Parse answers from `\boxed{}`, `####`, or last numeric. Check via `is_equiv()` (LaTeX normalization) + numeric int comparison. AIME answers are integers 0-999.
- Acceptance: 30 problems loaded. parse_answer + check_answer correct on: `\boxed{42}` -> "42", `#### 7` -> "7", `The answer is 100.` -> "100".

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RL library | TRL GRPOTrainer with vLLM colocate | veRL needs 8xA100 80GB; TRL colocate works on single A100 40GB |
| Fine-tuning | LoRA rank=16 (q_proj, v_proj, k_proj, o_proj) | Full FT needs ~56GB for optimizer states; LoRA ~8M params |
| Generations per prompt | 16 | Single GPU constraint; paper uses 64 but downsamples to 16 for training |
| Playbook style | ACE (Reflect+Curate) | 5.5x faster than DC, reuse existing code from run.py |
| Reflect/Curate model | Same training model | Single GPU; accept quality limitation |
| KL coefficient | 0.0 | Per TTRL paper + DeepSeek-R1 best practice |
| Playbook injection | System prompt | Fits Qwen2.5-Math chat template, ~500 tokens |
| vLLM mode | Sleep/wake colocate | Only viable approach for shared GPU inference+training |
| Base model | Qwen/Qwen2.5-Math-7B-Instruct | TTRL paper's primary model, strong math baseline |

## Completion Criteria

- [x] All 7 REQs implemented
- [ ] TTRL-only shows improvement over baseline (Phase 1 sanity check)
- [ ] All 4 conditions run on AIME 2024 without OOM or crash
- [ ] 4 analysis plots generated
- [ ] Statistical comparison printed
- [ ] Decision verdict output
- [ ] Notebook runs end-to-end on Colab A100

## Progress

| ID | Status | Notes |
|----|--------|-------|
| REQ-1 | COMPLETED | ABCs in cell-4, condition configs in cell-6 |
| REQ-2 | COMPLETED | GRPO config + GRPOTrainer setup in cell-8, training loop in cell-9 |
| REQ-3 | COMPLETED | Playbook, reflect/curate, evaluators in cell-7 |
| REQ-4 | COMPLETED | Co-evolution via dc_ttrl_reward_fn side-effect in cell-8, orchestration in cell-9 |
| REQ-5 | COMPLETED | 4 conditions orchestrated sequentially in cell-9 |
| REQ-6 | COMPLETED | Bootstrap CI, McNemar's, 4 plots, decision verdict in cell-10 |
| REQ-7 | COMPLETED | Data loading in cell-5, answer parsing in cell-3 |

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| OOM on A100 40GB | Reduce num_generations 16->8, or use QLoRA (4-bit) |
| Colab timeout mid-run | Resume from episode checkpoint (save after each episode) |
| Majority vote ties | Pick first answer lexicographically |
| Playbook exceeds 20 bullets | Curate prunes lowest helpful/harmful ratio |
| GRPO loss diverges | Clip gradients (max_grad_norm=1.0), reduce LR, log early |
| All 16 candidates wrong (reward=0 for all) | Skip GRPO step, curate playbook with is_correct=False |
| vLLM wake fails after training | Retry once; if fails, save checkpoint and stop |
| Reflect/curate returns unparseable JSON | Skip curate step, log warning, continue |

## Out of Scope

- Multi-GPU distributed training (veRL, Ray, DeepSpeed)
- Full fine-tuning (>40GB VRAM)
- Strong-model verifier (GPT-4o-mini as ranker) — future extension
- Archetype-anchored curriculum — future extension
- Benchmarks beyond AIME 2024 — future extension
- SFT from strong model DC traces (Project #1) — future work
- ACE online mode — future extension

## Technical Context

### Key Files to Reuse
- `experiments/dc_vs_verification/run.py:43-235` — Answer parsing (parse_answer, check_answer, is_equiv, extract_numeric_answer, last_boxed_only_string, strip_string, fix_fracs, fix_sqrt)
- `experiments/dc_vs_verification/run.py:244-510` — Bullet dataclass, Playbook dataclass, reflect pipeline, curate pipeline, make_initial_playbook
- `experiments/dc_vs_verification/run.py:515-677` — Condition implementation patterns
- `experiments/dc_vs_verification/analysis.py` — Analysis framework, decision logic template
- `dynamic-cheatsheet/dynamic_cheatsheet/language_model.py` — DC prompt integration patterns
- `dynamic-cheatsheet/run_benchmark.py` — DC benchmark runner patterns

### Patterns to Follow
- vLLM + AsyncOpenAI with Semaphore(64) for concurrent inference (from run.py)
- Playbook bullet format: `[id-NNNNN] helpful=N harmful=N :: content text`
- Three sections: STRATEGIES, COMMON_MISTAKES, SOLUTION_PATTERNS
- Curate operations: ADD/UPDATE/DELETE with JSON-parsed LLM output
- Checkpoint results as JSON after each condition/episode

### Dependencies
```
trl >= 0.15.0          # GRPOTrainer with vLLM colocate
vllm >= 0.7.0          # sleep/wake API
transformers >= 4.48    # Qwen2.5 support
peft >= 0.14           # LoRA
accelerate             # training primitives
datasets               # HF data loading
torch >= 2.5           # bf16
scipy                  # statistical tests
matplotlib             # plotting
nest_asyncio           # async in notebooks
```

### Files to Create
- `experiments/ttrl_dc_aime/ttrl_dc_aime.ipynb` — Main Colab notebook (primary deliverable)

## Execution Strategy

**Mode:** Subagent delegation

**REQ Groups (sequential):**
| Group | REQs | Description |
|-------|------|-------------|
| A | REQ-1, REQ-7 | Infrastructure: components ABCs + data loading + answer parsing |
| B | REQ-3 | Playbook: Bullet, ActivePlaybook, NullPlaybook, reflect, curate |
| C | REQ-2, REQ-4, REQ-5 | Training: GRPO config, co-evolution loop, 4 conditions |
| D | REQ-6 | Analysis: statistics, plots, decision framework |
