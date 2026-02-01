# Specification: Grounded Discovery Experiment

> Use `/duy-workflow:execute docs/specs/grounded-discovery.spec.md` to implement.

## Goal

Build infrastructure to test whether ACE-style evolving context memory improves LLM equation recovery on 9 counterfactual physics environments across 6 conditions.

## Requirements

### Environment Layer

1. **[REQ-ENV-1]** 9 counterfactual physics environments as Python classes
   - Tier 1 (6 envs): modified gravity coupling, asymmetric drag, non-reciprocal spring, velocity-dependent mass, anharmonic oscillator, modified projectile
   - Tier 2 (3 envs): non-polynomial conserved quantity, cross-coupled dynamics, history-dependent force
   - Each: `evaluate(x_1..x_n) -> y` with Gaussian noise (sigma=0.01*|y|)
   - Each: held-out test set of 50 points
   - Each: input range constraints
   - Acceptance: unit test verifying each environment produces correct output for known inputs

2. **[REQ-ENV-2]** Warp template kernel library for gradient computation
   - Templates: polynomial, trigonometric, power-law, coupled-ODE, delay-term
   - Each template: parameterized @wp.kernel with wp.Tape() autodiff
   - Expression matcher: map LLM Python string → best-fit template + initial parameters
   - Acceptance: gradient from Warp matches finite-difference numerical gradient (tol=5e-2)

3. **[REQ-ENV-3]** Zero-shot and structural prior probes
   - Zero-shot: give LLM 20 data points → ask for law → success must be <20%
   - Structural prior: ask LLM for functional forms without data → must not guess correct class
   - Acceptance: all 9 environments pass both probes

### Loop Layer

4. **[REQ-LOOP-1]** 100-round interactive discovery loop
   - Steps: CHOOSE INPUTS → OBSERVE → HYPOTHESIZE → FIT → EVALUATE → REFLECT → CURATE
   - LLM calls via ShinkaEvolve's LLMClient (import from `shinka.llm.llm`)
   - Expression parsing: Python string → callable function
   - Data accumulation: all past observations available to agent each round
   - Held-out MSE tracking at every round
   - Acceptance: 1 env x condition A x 10 rounds completes without error

5. **[REQ-LOOP-2]** Results storage via extended ProgramDatabase
   - Import from `shinka.database.dbase`
   - Additional fields: environment_id, condition, seed, round, mse_train, mse_test, expression_str, playbook_snapshot
   - Acceptance: can query "best expression for env X, condition Y, seed Z"

### Condition Layer

6. **[REQ-COND-A]** Condition A: Static baseline
   - Fixed system prompt, no playbook, no gradient
   - LLM proposes equation given accumulated data
   - Acceptance: produces MSE curve over 100 rounds

7. **[REQ-COND-B]** Condition B: +ACE
   - Import Generator, Reflector, Curator from `ace.core`
   - Custom prompts for equation discovery
   - Custom DataProcessor: `answer_is_correct` via MSE threshold
   - Playbook sections: EQUATION FORMS, PARAMETER STRATEGIES, EXPLORATION HEURISTICS, COMMON MISTAKES
   - Reflector tags based on MSE improvement/degradation
   - Curator every 10 rounds
   - Acceptance: playbook evolves (bullets added/removed over 100 rounds)

8. **[REQ-COND-C]** Condition C: +Gradient
   - After LLM proposes form, match to Warp template
   - Optimize parameters via wp.Tape() + Adam (20 steps)
   - Submit gradient fitting as SLURM GPU job
   - Acceptance: fitted parameters improve MSE vs unfitted

9. **[REQ-COND-D]** Condition D: +ACE+Gradient
   - Combines B and C
   - Acceptance: both playbook evolution and gradient fitting active

10. **[REQ-COND-E]** Condition E: DrSR-style
    - ACE with P/N/I sections (Positive/Negative/Invalid) instead of helpful/harmful
    - Positive: equations improving MSE; Negative: worsening; Invalid: parse failures
    - Acceptance: idea library accumulates P/N/I entries

11. **[REQ-COND-F]** Condition F: PySR baseline
    - Run PySR on accumulated data at rounds 25, 50, 75, 100
    - Report best symbolic expression
    - Acceptance: PySR produces valid symbolic expression

12. **[REQ-COND-R]** Random search baseline
    - Random expressions from grammar {+,-,*,/,^,sin,cos,exp,log,sqrt}
    - Same loop structure, no LLM
    - Acceptance: produces MSE curve (expected: poor)

### Campaign Layer

13. **[REQ-CAMP-1]** Campaign runner for 162 runs
    - 9 envs x 6 conditions x 3 seeds
    - Checkpoint/resume from SQLite
    - Cost tracking with budget enforcement
    - Paired seeds across conditions
    - Acceptance: can run subset (1 env x 2 conditions x 1 seed) end-to-end

### Analysis Layer

14. **[REQ-ANALYSIS-1]** Statistical analysis
    - Paired t-tests: ACE (B,D) vs Static (A,C)
    - Learning curves: MSE vs round for all conditions
    - Symbolic recovery: SymPy simplification match rate
    - Experiment design quality: input variance metric
    - Acceptance: produces summary table + plots for pilot run

15. **[REQ-LEVIN-1]** Levin extension measurements
    - L1: Strategy profile coding (taxonomy of playbook bullets)
    - L2: Intervention (remove zero-helpful bullets at round 50, measure degradation)
    - L3: Transfer probe (E1→E2 with static, trained, random, ablated controls)
    - L4: Learning curve shape visual comparison
    - Acceptance: intervention experiment runs on 1 environment

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Gradient framework | Warp @wp.kernel templates | GPU autodiff, compatible with Newton codebase |
| LLM client | ShinkaEvolve's LLMClient | Multi-model, cost tracking, retries, bandit selection |
| ACE integration | Import as package | No code duplication, changes propagate |
| DrSR implementation | ACE with P/N/I sections | Reuses 90% of ACE code |
| SR baseline | PySR library | State-of-the-art, proper comparison |
| Compute | SLURM for GPU jobs only | LLM API calls are HTTP, no GPU needed |
| Storage | Extended ProgramDatabase | SQLite, checkpoint/resume, proven |
| Expression matching | Template library | 5-10 templates cover most LLM outputs; fallback to numerical gradients |

## Project Structure

```
src/
  __init__.py
  pyproject.toml
  environments/
    __init__.py
    base.py              # BaseEnvironment ABC
    tier1.py             # 6 Level-2 counterfactual envs
    tier2.py             # 3 Level-3 counterfactual envs
    kernels.py           # Warp @wp.kernel templates
  loop/
    __init__.py
    orchestrator.py      # 100-round discovery loop
    expression_parser.py # Python string → callable + template match
    prompt_templates.py  # System/user prompts per condition
  conditions/
    __init__.py
    static.py            # Condition A
    ace_condition.py     # Condition B
    gradient.py          # Condition C
    ace_gradient.py      # Condition D
    drsr.py              # Condition E
    pysr_baseline.py     # Condition F
    random_search.py     # Random baseline
  gradient/
    __init__.py
    fitter.py            # Warp template parameter fitting
    slurm_wrapper.py     # SLURM job submission for GPU fitting
  ace_adapter/
    __init__.py
    data_processor.py    # EquationDataProcessor
    prompts.py           # Equation-domain prompts for Generator/Reflector/Curator
    drsr_adapter.py      # P/N/I section adapter
  campaign/
    __init__.py
    runner.py            # 162-run orchestrator
    config.py            # Hydra-style config
  probes/
    __init__.py
    zero_shot.py         # Anti-memorization probe
    structural_prior.py  # Structural prior probe
  analysis/
    __init__.py
    statistics.py        # Paired t-tests, summary tables
    plots.py             # Learning curves, comparison plots
    symbolic.py          # SymPy recovery checking
    levin/
      __init__.py
      strategy_coder.py  # L1: taxonomy coding
      intervention.py    # L2: bullet removal experiment
      transfer.py        # L3: cross-env transfer probe
  tests/
    test_environments.py
    test_kernels.py
    test_loop.py
    test_conditions.py
```

## Completion Criteria

- [ ] All 9 environments implemented with unit tests
- [ ] Warp template kernels match numerical gradients
- [ ] Zero-shot probes pass (<20% success) on all 9 environments
- [ ] End-to-end: 1 env x condition A x 10 rounds completes
- [ ] End-to-end: 1 env x condition B (ACE) x 10 rounds with playbook evolution
- [ ] End-to-end: 1 env x condition C (gradient) x 10 rounds with Warp fitting
- [ ] PySR baseline produces output on 1 environment
- [ ] Campaign runner handles checkpoint/resume
- [ ] Analysis produces paired t-test table + learning curve plots
- [ ] All tests pass as SLURM batch jobs (not on login node)

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| LLM produces unparseable expression | Log error, use previous round's expression, mark as Invalid in DrSR |
| Warp template match fails | Fall back to numerical gradient (finite differences) |
| LLM exceeds token budget | Truncate context to most recent N observations |
| PySR times out | Report best expression found so far |
| Environment returns NaN/Inf | Clip to valid range, add to noise budget |
| SLURM job fails | Retry once, then fall back to CPU gradient |
| Cost budget exceeded | Stop campaign, report partial results |
| Zero-shot probe passes (>20%) | Increase environment difficulty (adjust parameters) |

## Technical Context

### Key Files to Import From
- `ace/ace.py`: ACE orchestrator (modes, training loops)
- `ace/core/generator.py`: Generator.generate()
- `ace/core/reflector.py`: Reflector.reflect()
- `ace/core/curator.py`: Curator.curate()
- `ace/core/bulletpoint_analyzer.py`: BulletpointAnalyzer.analyze()
- `ShinkaEvolve/shinka/llm/llm.py`: LLMClient (query, batch_query, cost tracking)
- `ShinkaEvolve/shinka/database/dbase.py`: ProgramDatabase, Program
- `ShinkaEvolve/shinka/launch/scheduler.py`: JobScheduler, SlurmCondaJobConfig
- `newton/newton/examples/diffsim/example_diffsim_ball.py`: wp.Tape() pattern

### Patterns to Follow
- ShinkaEvolve's evaluate.py pattern for evaluation scripts
- ACE's DataProcessor interface for custom domains
- Newton's @wp.kernel pattern for differentiable computation
- Hydra configs from ShinkaEvolve for experiment configuration
- SQLite WAL mode from ProgramDatabase for concurrent access

### Reference Document
- `docs/discoveries/grounded-discovery/DISCOVERY.md` — complete experiment design (v3, iteration 6)
