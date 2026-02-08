# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Stanford FarmShare shared cluster (rice-XX nodes). Home directories are NFS-mounted.

### Critical Rules

- **NEVER run tests interactively on the login node.** Always submit compute-heavy work (tests, training, builds) as SLURM batch jobs.
- After spawning any long-running process, track the PID so it can be cleaned up if the session dies.
- Use `timeout` when running tests to prevent zombie processes (e.g., `timeout 300 pytest ...`).
- **Always commit changes when task is complete.** Do not include `Co-Authored-By` lines.

## Repository Structure

```
grounded/
├── CLAUDE.md                 # This file
├── .gitignore                # Excludes: __pycache__, results/, *.sqlite, *.log, newton/, ShinkaEvolve/
├── src/                      # Main codebase: grounded-discovery experiment
│   ├── pyproject.toml        # Package: grounded-discovery v0.1.0 (Python >=3.10)
│   ├── conftest.py           # Pytest configuration
│   ├── run_experiment.py     # Main entry point (prelim + campaign)
│   ├── llm_client.py         # LLM abstraction (ShinkaEvolve fallback → raw OpenAI-compatible HTTP)
│   ├── slurm_prelim.sh       # SLURM job submission script
│   ├── environments/         # 9 counterfactual physics environments
│   ├── loop/                 # Discovery loop orchestrator + expression parser + prompts
│   ├── conditions/           # 7 experimental conditions (A-F + random baseline)
│   ├── gradient/             # Warp-based parameter fitting + SLURM wrapper
│   ├── ace_adapter/          # ACE framework adapter (data processor, prompts, DrSR)
│   ├── campaign/             # Campaign runner + config (162 runs)
│   ├── probes/               # Zero-shot and structural prior probes
│   ├── analysis/             # Statistics, plots, symbolic recovery, Levin extension
│   ├── tests/                # Unit tests (environments, kernels)
│   └── grounded_discovery/   # Parallel namespace package (mirrors src/)
├── ace/                      # Reserved for ACE framework imports (currently empty)
├── notebooks/                # Jupyter notebooks
│   ├── search_augmented_ace_poc.ipynb          # MCTS/bandit search over ACE playbooks on GSM8K
│   └── verified_archetype_discovery_poc.ipynb  # Game of 24 verification task
└── docs/
    ├── specs/
    │   ├── grounded-discovery.spec.md          # Full experiment requirements (15 REQ items)
    │   └── search-augmented-ace-poc.spec.md    # PoC spec (Greedy/PUCT/ES on GSM8K)
    ├── discoveries/
    │   └── grounded-discovery/
    │       ├── DISCOVERY.md   # Experiment design v3, landscape, hypothesis (iteration 6)
    │       └── LOG.md         # Append-only research log (6 iterations)
    └── research/
        └── grounded-abduction/
            ├── KNOWLEDGE.md   # Literature knowledge base
            └── MINDMAP.md     # Research connections map
```

### External Dependencies (gitignored, cloned separately)

- `ShinkaEvolve/` — LLM-guided evolutionary algorithm framework ([sakana.ai](https://sakana.ai/shinka-evolve/))
- `newton/` — GPU-accelerated physics simulation engine (NVIDIA Warp + MuJoCo Warp)
- `ace/` — Agentic Context Engineering framework (Generator/Reflector/Curator)

---

## Grounded Discovery — The Main Experiment

### What It Is

A controlled experiment testing whether **evolving context memory (ACE-style playbooks)** improves LLM-based equation recovery from interactive experiments on counterfactual physics environments. The core scientific question: *does structured context accumulation help when LLMs must reason from scratch on non-memorizable tasks?*

### Commands

```bash
cd src && uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .

# Run experiment
python3 src/run_experiment.py                  # prelim + full campaign
python3 src/run_experiment.py --prelim-only    # just the sanity check
python3 src/run_experiment.py --skip-prelim    # skip straight to campaign

# Tests (submit via SLURM, not login node)
timeout 300 pytest src/tests/

# Lint
black src/ && isort src/ && flake8 src/
```

### Architecture

**Discovery Loop** (`src/loop/orchestrator.py`):
```
For round r = 1..100:
  1. CHOOSE INPUTS   → LLM selects values for x_1..x_n
  2. OBSERVE         → Environment returns y = f(x_1..x_n) + noise
  3. HYPOTHESIZE     → LLM proposes y = g(x_1..x_n; θ) as Python expression
  4. FIT             → wp.Tape() optimizes θ (gradient conditions only)
  5. EVALUATE        → MSE on accumulated data + 50 held-out test points
  6. REFLECT         → ACE Reflector tags playbook bullets (ACE conditions)
  7. CURATE          → ACE Curator evolves playbook (every 10 rounds, ACE conditions)
```

**Key Interfaces**:
- `ConditionStrategy` (Protocol) — pluggable experimental conditions
- `BaseEnvironment` — abstract base for counterfactual physics (9 implementations)
- `DiscoveryLoop` — main orchestrator, accepts any `ConditionStrategy`
- `LLMClient` — tries ShinkaEvolve's client, falls back to raw OpenAI-compatible HTTP
- `CampaignRunner` — orchestrates 162 runs (9 envs × 6 conditions × 3 seeds)

### Environments (9 total)

**Tier 1 — Level 2 symmetry violations** (6 envs):
- `ExponentialDampedGravity` — modified gravitational coupling
- `AsymmetricDrag` — direction-dependent drag
- `NonReciprocalSpring` — Newton's 3rd law broken
- `VelocityDependentMass` — non-standard relativistic mass
- `CoupledNonlinearDamping` — coupled damped oscillator
- `FractionalDrag` — fractional-power drag

**Tier 2 — Level 3 structural novelty** (3 envs):
- `NonPolynomialConserved` — non-polynomial conserved quantity
- `CrossCoupledDynamics` — unknown cross-coupling
- `HistoryDependentForce` — delay-differential force

All environments present data as anonymous `x_1..x_n → y` with no physics labels.

### Experimental Conditions

| ID | Name | Context | Gradient | Description |
|----|------|---------|----------|-------------|
| A | Static | Fixed prompt | No | Baseline LLM |
| B | +ACE | Evolving playbook | No | ACE context engineering |
| C | +Gradient | Fixed prompt | wp.Tape() | SGA-style inner loop |
| D | +ACE+Gradient | Evolving playbook | wp.Tape() | Full system |
| E | DrSR-style | Idea library (P/N/I) | No | Direct comparison to DrSR |
| F | PySR | N/A | N/A | Symbolic regression baseline |
| R | Random | N/A | No | Random expression baseline |

### Configuration

`CampaignConfig` in `src/campaign/config.py`:
- Currently uses `hyperbolic/openai/gpt-oss-20b` (cheap PoC model, ~$6 for full campaign)
- Production: switch to `gpt-4o-mini` or `gpt-4.1-nano`
- 100 rounds per run, 5 points per round, 3 seeds
- Budget: $20 total, $1 per run (adjustable)
- Results stored in SQLite (`src/results/campaign.sqlite`)

### Evaluation Criteria

1. **Primary**: ACE (B,D) achieves lower test MSE@100 than static (A,C), p<0.05 paired t-test
2. **Secondary**: ACE matches or exceeds DrSR (E)
3. **Tertiary**: Gradient (C,D) improves parameter accuracy over non-gradient (A,B)
4. **Control**: Zero-shot probe success <20% (confirms non-memorizable)
5. **Levin Extension**: Strategy profiling, intervention experiment, transfer probe

---

## Notebooks

### `search_augmented_ace_poc.ipynb`

Self-contained Colab notebook comparing search strategies for evolving ACE playbooks on GSM8K with local Qwen2.5-7B via vLLM on A100.

**14 conditions** (expanded from original 3):
- Greedy ACE, Majority Vote, Best-of-N, Thompson Sampling, UCB Bandit
- PUCT-ACE (tree search), AB-MCTS, Discounted MCTS
- ES-ACE (evolutionary), Beam Search, and more

**Key implementation details**:
- vLLM serving Qwen2.5-7B-Instruct (`--dtype bfloat16 --gpu-memory-utilization 0.95 --enable-prefix-caching --max-model-len 8192`)
- Async GPU-optimized patterns via `AsyncOpenAI` + `asyncio.gather()` + `Semaphore(64)`
- Budget tracking per condition (~150 generation calls each)
- Bootstrap CI for statistical comparisons

### `verified_archetype_discovery_poc.ipynb`

Game of 24 verification task with vLLM integration and async parallel evaluation.

---

## Key Literature & References

### Core Papers (Experiment Design)

| Paper | Key Insight | Relevance |
|-------|------------|-----------|
| [PhysGym](https://arxiv.org/abs/2507.15550) (2025) | LLMs fail at interactive experiment design | Motivates our loop design |
| [SGA](https://arxiv.org/abs/2405.09783) (ICML 2024) | LLM+Warp bilevel optimization | Conditions C, D (gradient inner loop) |
| [NewtonBench](https://arxiv.org/abs/2510.07172) (2025) | Counterfactual physics taxonomy (Levels 1-3) | Our environment difficulty calibration |
| [DrSR](https://arxiv.org/abs/2506.04282) (2025) | Idea library for symbolic regression (99.94% vs 7.62%) | Condition E (direct comparison) |
| [LLM-SR](https://arxiv.org/abs/2404.18400) (ICLR 2025) | Equations as programs + evolutionary search | Background |
| [LLM-SRBench](https://arxiv.org/abs/2504.10415) (ICML 2025) | 239 SR problems, best 31.5% symbolic accuracy | Calibrates expectations |
| [ACE](https://arxiv.org/abs/2510.04618) (2025) | Generator/Reflector/Curator with playbook evolution | Core framework for conditions B, D |
| [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952) (2025) | Test-time evolving memory, Claude 3.5 23%→50% AIME | ACE precursor / parallel work |

### Verification & Small-Model-Big-Performance (Future Directions)

| Paper | Key Insight | How It Applies |
|-------|------------|----------------|
| [Weaver](https://arxiv.org/abs/2506.18203) (NeurIPS 2025) | Aggregate weak verifiers via Dawid-Skene, distill to 400M model retaining 98.7% accuracy at 0.03% compute | Makes verification essentially free — unlocks cheap generate→verify→curate loop |
| [REBASE](https://arxiv.org/abs/2408.00724) (ICLR 2025) | PRM-guided tree search, Pareto-optimal at all budgets; 7B beats 34B | Simpler alternative to PUCT for playbook tree search |
| [rStar-Math](https://arxiv.org/abs/2501.04519) (ICML 2025) | 3.8B model beats o1-preview via self-evolved MCTS+PRM | Proof that small model + good verification + tree search = big model performance |
| [ThinkPRM](https://arxiv.org/abs/2504.16828) (2025) | Process reward model using only 1% labels, 7B outperforms full-data discriminative PRMs | Cheap step-level verification |
| [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314) (ICLR 2025 Oral) | Adaptive compute allocation per-problem difficulty; 14B matches 4x larger model | Adaptive-N: easy problems get 1-2 samples, hard problems get tree search |
| [Adaptive Inference-Time Compute](https://arxiv.org/abs/2410.02725) (2024) | Model predicts mid-generation if restarting will help; 74% of 16-sample gain with 1.2 avg samples | Early stopping when confident |
| [Optimal Stopping vs Best-of-N](https://arxiv.org/abs/2510.01394) (2025) | Optimal stopping reduces generations by 15-35% | Adaptive N via stopping thresholds |
| [Inference Scaling fLaws](https://arxiv.org/abs/2411.17501) (2024) | Imperfect verifiers impose hard ceilings on resampling | Important negative result — motivates Weaver-style calibrated verification |
| [GenRM](https://arxiv.org/abs/2408.15240) (ICLR 2025) | Generative verifiers via next-token prediction; fine-tuned 9B surpasses GPT-4 | Chain-of-thought verification as generation |
| [AB-MCTS / TreeQuest](https://arxiv.org/abs/2503.04412) (Sakana 2025) | Adaptive wider-vs-deeper branching; GEN/CONT node types | Reference implementation for tree search over playbooks |
| [OPTS](https://arxiv.org/abs/2503.01163) (ACL 2025) | Thompson Sampling for prompt strategy selection; +7% on BBH | Directly applicable to bandit-based playbook strategy selection |
| [FoVer](https://arxiv.org/abs/2505.15960) (2025) | PRMs via formal verification data; generalizes across 12 benchmarks | Cross-task PRM generalization |
| [Process Advantage Verifiers](https://arxiv.org/abs/2410.08146) (ICLR 2025) | Reward = progress (advantage); weak prover improves strong policy | Maps to reflector's improvement tagging |

### Theoretical Foundations

| Paper | Key Insight |
|-------|------------|
| [Rate-Distortion for Prompt Compression](https://arxiv.org/abs/2407.15504) (NeurIPS 2024) | Fundamental limits of compressing prompts; large gap between current methods and optimum |
| [Certified Self-Consistency](https://arxiv.org/abs/2510.17472) (2025) | Unifies self-consistency and TTRL; Martingale stopping rule; builds on Condorcet's theorem |
| [Information-Theoretic ICL](https://arxiv.org/abs/2401.15530) (ICML 2024) | Decomposes ICL error into irreducible + meta-learning + intra-task |
| [Self-Verification Limitations](https://arxiv.org/abs/2402.08115) (2024) | LLM self-critique fails; external verification exploits actual NP gap |
| [Can 1B Surpass 405B?](https://arxiv.org/abs/2502.06703) (2025) | With compute-optimal TTS, 0.5B outperforms GPT-4o, 3B surpasses 405B on MATH |

---

## Future Direction: Cheap Verification Unlocks the Full Loop

The strategic direction: **small model with performance of big models** via cheap, high-quality verification.

### The Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Small Generator (7B)                                    │
│  Generates 8-16 candidates with evolving cheatsheet      │
│  in context (ACE playbook / Dynamic Cheatsheet)          │
└──────────────┬──────────────────────────────────────────┘
               │ candidates
               ▼
┌─────────────────────────────────────────────────────────┐
│  Distilled Verifier (400M, Weaver-style)                 │
│  Scores candidates for near-zero cost                    │
│  Trained via weak verifier ensemble + Dawid-Skene        │
│  Retains 98.7% of full ensemble accuracy                 │
└──────────────┬──────────────────────────────────────────┘
               │ high-confidence solutions only
               ▼
┌─────────────────────────────────────────────────────────┐
│  Cheatsheet Curator (ACE Curator)                        │
│  Compresses key insight from verified solution back      │
│  into the evolving playbook/cheatsheet                   │
│  20-50 strategies at the right abstraction level         │
└─────────────────────────────────────────────────────────┘
```

### Why This Is Near-Optimal

Each component operates at its information-theoretic boundary:

1. **Verification ensemble → Condorcet's jury theorem**: Many independent weak verifiers with p>0.5 accuracy → exponential error decay in number of verifiers. Weaver proves this works in practice with Dawid-Skene latent variable estimation.

2. **Distillation works because verification < generation** (co-NP vs NP argument): Checking a solution is fundamentally simpler than generating one. A 400M model can learn to verify what a 70B model generates. (Caveat: LLM self-verification does NOT exploit this gap — you need external/trained verifiers.)

3. **Cheatsheet converges on rate-distortion bound**: The playbook/cheatsheet compresses the problem distribution into 20-50 strategies at the right abstraction level. Rate-distortion theory ([arXiv:2407.15504](https://arxiv.org/abs/2407.15504)) formalizes this — steep initial gains from adding strategies, diminishing returns after saturation.

### The Two Gaps to True Optimality

**Gap 1: Step-level verification** — Current system verifies final answers only (generate-and-select). True optimality requires pruning at each reasoning step (tree search, not just best-of-N). Papers:
- [ThinkPRM](https://arxiv.org/abs/2504.16828): Process reward models using only 1% labels
- [REBASE](https://arxiv.org/abs/2408.00724): PRM-guided tree search, Pareto-optimal at all budgets
- [rStar-Math](https://arxiv.org/abs/2501.04519): 3.8B with self-evolved PRM beats o1-preview
- [Process Advantage Verifiers](https://arxiv.org/abs/2410.08146): Reward = progress (advantage function)

**Gap 2: Adaptive N** — Stop generating when confidence is high, flag as gap when confidence stays low. Easy problems cost nothing, impossible ones get caught early. Papers:
- [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314): 4x compute savings via difficulty-adaptive allocation
- [Adaptive Inference-Time Compute](https://arxiv.org/abs/2410.02725): 74% of 16-sample gain with 1.2 avg samples
- [Optimal Stopping vs Best-of-N](https://arxiv.org/abs/2510.01394): 15-35% fewer generations via stopping thresholds
- [Certified Self-Consistency](https://arxiv.org/abs/2510.17472): Martingale stopping rule with statistical guarantees

### Applying to Grounded Discovery

For the equation discovery experiment specifically:
1. **Generator**: Small model (7B) proposes equations with ACE playbook in context
2. **Verifier**: Distilled verifier scores equation quality (MSE prediction, structural plausibility). For equation discovery, MSE on held-out data IS the verifier — no need for a learned one. But for the LLM reasoning steps (experiment design, hypothesis generation), a trained PRM could help.
3. **Adaptive N**: Generate more equation candidates for hard environments (Tier 2), fewer for easy ones (Tier 1). Stop early when MSE is already low.
4. **Step-level verification**: Instead of scoring complete equations, score intermediate reasoning steps — "is this functional form choice promising?" before spending compute on parameter fitting.
5. **Cheatsheet**: ACE curator compresses verified insights back into playbook. Only high-confidence, MSE-improving strategies survive.

### Important Caveats

- [Inference Scaling fLaws](https://arxiv.org/abs/2411.17501): Imperfect verifiers impose hard ceilings. Weaver's calibrated ensemble addresses this, but ceiling still exists.
- [CJT Independence Violation](https://arxiv.org/abs/2409.00094): LLM verifiers sharing training data violate Condorcet independence assumption. Weaver addresses via filtering + weighted aggregation.
- [Self-Verification Limitations](https://arxiv.org/abs/2402.08115): LLM self-critique adds nearly no value. The verifier must be a separately trained model or use formal/automated checking.
- The co-NP argument has limits: verifying *absence* of errors (safety) is fundamentally harder than verifying *presence* of a solution. For equation discovery, we mostly need the NP version (does this equation fit?), so the gap is exploitable.

---

## Cross-Project Integration (ShinkaEvolve + ACE)

The two frameworks are complementary and composable:

1. **ACE playbook as ShinkaEvolve's task_sys_msg** — evolves mutation strategies based on which approaches improve fitness
2. **Program.text_feedback ↔ ACE Reflector** — structured helpful/harmful tagging
3. **Meta-recommendations ↔ ACE Curator** — replaces free-form meta-LLM with ADD/UPDATE/MERGE/DELETE operations
4. **ProgramDatabase ↔ playbook persistence** — store playbook snapshots alongside generations in SQLite

Newton serves as evaluation environment when evolving robot controllers or physics algorithms.

---

## Session Notes

### Notebook Implementation Notes

- **Thompson Sampling in search_augmented_ace_poc.ipynb**: Budget docstring was wrong (said 99, actual is 105 calls). Fixed. Beta-Bernoulli conjugate update and argmax-of-samples selection are textbook correct. Key limitation: fixed pool after seed phase means no online adaptation.
- **PUCT backprop bug (fixed)**: `backprop(child, reward)` gave the child credit for the parent's playbook evaluation. Fix: always backprop to the leaf whose playbook was actually evaluated.
- **PUCT Bayesian Q-estimator**: Uses `(s+1)/(n+2)` — correct Beta posterior mean for binary rewards, but reward_history contains fractional batch accuracies. Works as shrinkage estimator but is NOT a proper Beta posterior for continuous rewards. Document as "Beta-inspired shrinkage."
- **Progressive widening k=1**: Standard PW uses k=10. Our k=1 is deliberately conservative since each expansion costs a curate call. Justified design choice.
- **AB-MCTS deviation from Sakana paper**: Our implementation uses regret-based Beta updates; Sakana's original ([arXiv:2503.04412](https://arxiv.org/abs/2503.04412)) uses separate GEN/CONT node types with backed-up score distributions. Credit assignment issue: "deeper+regress → should have expanded" is a counterfactual assumption without evidence. Reference implementation: [github.com/SakanaAI/treequest](https://github.com/SakanaAI/treequest).
- **Discounted TS timing bug (fixed)**: Discount applied BEFORE posterior update; should be AFTER per arXiv:2305.10718. Floor reduced from 1.0 to 0.1.
- **14 conditions total**: Greedy, MajVote, Best-of-N, Thompson, UCB, PUCT, AB-MCTS, Discounted MCTS, ES, Beam Search, and more.

### Performance Optimization Notes

- **vLLM GPU config for Colab A100 (40GB)**: `--dtype bfloat16 --gpu-memory-utilization 0.95 --max-num-seqs 1024 --max-num-batched-tokens 16384 --enable-prefix-caching --max-model-len 8192`. Prefix caching gives ~13% throughput gain. AWQ quantization NOT recommended — model fits in 14GB, dequantization overhead hurts.
- **Async optimization (Feb 2026)**: Main bottleneck was synchronous `curate()` calls (~154 calls at ~0.8s each). Fix: `curate_async()` with `llm_call_async`, `max_tokens=256`. Parallelized seed-phase curate loops. Expected 3-5x speedup.
- **Parallelization by condition**: MajVote: all 100 via asyncio.gather. Thompson: all arms in parallel. PUCT: virtual loss for K=8 parallel leaves. Greedy: batch 5 gen+ref calls between curate intervals.
- **Virtual loss for parallel PUCT**: `virtual_loss_count` on MCTSNode. Effective Q = `(Q*N)/(N+virtual_loss_count)`. Scales to ~32x parallelism (AlphaGo-proven). See arXiv:1902.04522.

### Design Considerations

- **Beam Search budget**: Width=3 evaluates all K beams on every batch → ~327 calls (higher than other conditions). Intentional tradeoff: breadth vs budget.
- **UCB vs Thompson**: UCB1 uses c=sqrt(2) (Auer et al. 2002). Same seed/pool for fair comparison. UCB is deterministic, Thompson stochastic. With 50 problems and 6 arms, UCB may over-exploit.
- **Variance-aware PUCT** (arXiv:2512.21648, Dec 2025): Inverse-RPO incorporates variance estimates. Consider as 4th Q-estimator mode.
- **REBASE as alternative to PUCT**: Simpler, Pareto-optimal at all budgets, avoids exploration-exploitation complexity. Worth implementing as a condition.
