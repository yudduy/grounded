# Discovery: Grounded Discovery — Experiment Design
> Status: Candidate | Iteration: 3 | Last updated: 2026-01-31

## Problem
Design a feasible, well-controlled experiment testing whether **evolving context memory (ACE)** improves LLM-based equation recovery from interactive experiments in a differentiable physics simulator. The experiment must resist memorization, include proper baselines, and produce interpretable results.

## Landscape (What Exists)

### Current SOTA
- **PhysGym** (2025): 97 interactive physics problems. Gemini-2.5-Pro: 74%→32% (anonymized). LLMs "fail to design informative experiments." [arXiv:2507.15550](https://arxiv.org/abs/2507.15550)
- **SGA** (ICML 2024): Bilevel LLM+Warp. MSE 5.2×10⁻⁵. Fixed experiments, no memory. [arXiv:2405.09783](https://arxiv.org/abs/2405.09783)
- **NewtonBench** (2025): 324 counterfactual tasks. Level 1 (exponent change) → Level 3 (symmetry break). GPT-5: 73%→30%. [arXiv:2510.07172](https://arxiv.org/abs/2510.07172)
- **DrSR** (2025): Idea Library for SR. 99.94% vs 7.62%. Closest analog to ACE for equations. [arXiv:2506.04282](https://arxiv.org/abs/2506.04282)
- **LLM-SR** (ICLR 2025 Oral): Equations as programs + evolutionary search. [arXiv:2404.18400](https://arxiv.org/abs/2404.18400)
- **LLM-SRBench** (ICML 2025 Oral): 239 SR problems. Best: 31.5% symbolic accuracy. [arXiv:2504.10415](https://arxiv.org/abs/2504.10415)

### The Actual Gap
Not "nobody has combined X+Y+Z" (that's an engineering claim). The gap is empirical:
- **Does structured context accumulation (ACE/DrSR-style) help when the LLM must reason from scratch** (no memorization possible)?
- **Does interactive experiment design (choosing what to test) provide an advantage** over passive data observation?
- **Does differentiable gradient feedback improve the inner loop** of hypothesis refinement?

These are testable questions that existing papers don't answer because:
- DrSR doesn't test on interactive environments
- PhysGym doesn't test context accumulation
- SGA doesn't test interactive experiment design
- NewtonBench doesn't test any adaptive method

## Current Hypothesis

- **Statement**: On Level 2+ counterfactual physics tasks, an LLM with ACE-style evolving context will recover the hidden law in fewer rounds than the same LLM with static prompting. The effect size will be comparable to DrSR's idea library gains on symbolic regression (~10-50% improvement in success rate), not DrSR's dramatic 99% vs 7% (which was on memorizable oscillation problems).
- **Confidence**: 65
- **Status**: Candidate (survived 2 adversarial rounds)

### Final Experiment Design (v3)

#### Counterfactual Environment Design

Use NewtonBench's taxonomy for principled difficulty control:

**Tier 1: 6 environments with Level 2 shifts** (symmetry violations)
- Modified gravitational coupling: F = α · (m₁ + m₂)^β / r^γ (operator change: × → +)
- Asymmetric drag: F_drag = -c · v · |v|^(a-1) where a varies per direction
- Non-reciprocal spring: F₁₂ ≠ -F₂₁ (Newton's 3rd law broken)
- Velocity-dependent mass: m_eff = m₀ · (1 + v²/c²)^α (non-standard α)
- Anharmonic oscillator: F = -k·x - β·x³ (Duffing oscillator with unknown β)
- Modified projectile: h(t) = v₀t - ½αt^β + γ·sin(ωt) (damped + oscillatory)

**Tier 2: 3 environments with Level 3 shifts** (structural novelty)
- Non-polynomial conserved quantity: E = m·v^a + k·x^b (non-standard a,b)
- Cross-coupled dynamics: ẍ = -αx + βy, ÿ = γx - δy (unknown coupling)
- History-dependent force: F(t) = f(x(t), x(t-τ)) (delay-differential)

**Implementation**: Pure Python analytical functions (not Newton physics engine) with Warp `wp.Tape()` for gradient computation on parameter fitting. This sidesteps the Newton-on-CPU performance issue entirely while still using Warp's autodiff.

**Why not Newton's actual physics?** Newton simulates real physics. Counterfactual physics requires custom dynamics. Using analytical wrappers with `wp.Tape()` gradients is honest and practical. Newton's value is the gradient infrastructure, not the specific physics.

#### Variables
- All presented anonymously: `x_1, x_2, ..., x_n` and `y` (target)
- No physics labels, no units, no domain hints
- Agent told: "You are interacting with a black-box system. Choose input values, observe outputs, discover the governing relationship."

#### The Loop (100 rounds)

```
For round r = 1..100:
  1. CHOOSE INPUTS: LLM selects values for x_1..x_n
     (constrained to valid ranges provided upfront)

  2. OBSERVE: Environment returns y = f(x_1..x_n) + ε
     (small Gaussian noise, σ=0.01·|y|)

  3. HYPOTHESIZE: LLM proposes: y = g(x_1..x_n; θ)
     as executable Python expression

  4. FIT (gradient conditions): wp.Tape() optimizes θ
     via L-BFGS or Adam (20 steps)

  5. EVALUATE: MSE on all r observations so far
     + MSE on 50 held-out test points (not revealed to agent)

  6. REFLECT (ACE conditions):
     ACE Reflector analyzes: expression, score trend, data patterns
     Tags playbook bullets helpful/harmful

  7. CURATE (every 10 rounds, ACE conditions):
     ACE Curator: ADD/UPDATE/MERGE/DELETE playbook entries
```

#### Conditions (6 total)

| Condition | Context | Gradient | Description |
|-----------|---------|----------|-------------|
| A: Static | Fixed prompt | No | Baseline LLM |
| B: +ACE | Evolving playbook | No | ACE context engineering |
| C: +Gradient | Fixed prompt | wp.Tape() | SGA-style inner loop |
| D: +ACE+∇ | Evolving playbook | wp.Tape() | Full system |
| E: DrSR-style | Idea library (P/N/I) | No | Direct comparison to DrSR |
| F: PySR | N/A | N/A | Symbolic regression baseline |

**Why include DrSR (condition E)?** It's the closest published method. If ACE can't beat DrSR's simpler idea library, ACE adds no value for this domain.

#### Controls

1. **Zero-shot probe**: Before any experiment, give LLM 20 data points and ask for the law. If success rate >20%, environment is too easy. Generate harder counterfactual.
2. **Structural prior probe**: Ask LLM "what functional forms might relate these variables?" without data. If it proposes the correct form class, add structural noise.
3. **Random experiment baseline**: Same loop but LLM replaced by random expression generator from grammar {+,-,*,/,^,sin,cos,exp,log,sqrt}.

#### Metrics

1. **Primary: Test MSE@100** — MSE on held-out 50 test points after 100 rounds
2. **Secondary: Symbolic recovery rate** — fraction where SymPy simplification matches true form (manual verification)
3. **Tertiary: Sample efficiency curves** — test MSE vs round number (learning curves)
4. **Quaternary: Experiment design quality** — variance of chosen inputs (exploration diversity)
5. **Qualitative: Playbook trace** — evolution of ACE bullets, coded for: mathematical insight, curve-fitting heuristic, noise

#### Statistical Design

- 9 environments × 6 conditions × 3 random seeds = 162 runs
- ~100 LLM calls per run × 162 = ~16,200 LLM calls
- At $0.01/call (GPT-4o-mini) or $0.05/call (GPT-4o): $162 – $810
- Use GPT-4o-mini as primary, GPT-4o for ablation on 3 environments
- Paired comparisons (same seed) for statistical power

### Evidence Supporting
- DrSR proved idea accumulation works for SR (99.94% vs 7.62%) [arXiv:2506.04282](https://arxiv.org/abs/2506.04282)
- SGA proved LLM+gradient bilevel works [arXiv:2405.09783](https://arxiv.org/abs/2405.09783)
- NewtonBench Level 2+ counterfactuals resist memorization [arXiv:2510.07172](https://arxiv.org/abs/2510.07172)
- ACE works on reasoning tasks (+8-10%) [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
- wp.Tape() provides differentiable gradient computation (confirmed in Newton code)

### Evidence Against
- PhysGym: LLMs fail at experiment design [arXiv:2507.15550](https://arxiv.org/abs/2507.15550)
- NewtonBench: code interpreter hurts strong models (gradient may hurt) [arXiv:2510.07172](https://arxiv.org/abs/2510.07172)
- ACE untested on symbolic math (LOG.md iteration 2, verifier critique)
- LLM-SRBench: best model achieves only 31.5% symbolic accuracy even with full context [arXiv:2504.10415](https://arxiv.org/abs/2504.10415)

### Survived Attacks
- **Memorization (verifier 1, 85%)**: Addressed by Level 2+ counterfactual + zero-shot control + structural prior probe
- **"PhysGym did this" (verifier 1, 90%)**: Differentiated by counterfactual laws, ACE/DrSR comparison, gradient ablation
- **"Power laws still pattern-matchable" (verifier 2, 80%)**: Addressed by Level 2+ (symmetry violations, non-reciprocal forces, cross-coupling) not just exponent changes
- **"30 rounds too few" (verifier 2)**: Increased to 100 rounds
- **"No DrSR comparison" (verifier 2)**: Added DrSR as condition E
- **"Gradient may hurt" (verifier 2)**: This is now a testable prediction — if gradient hurts, that's a result (replicates NewtonBench's code interpreter finding)

### Open Questions
1. Will ACE playbook format suit mathematical insights? (Empirical — the experiment answers this)
2. Will GPT-4o-mini have sufficient capacity for Level 2+ counterfactual laws? (Pilot with 1 environment)
3. Is Warp's wp.Tape() usable with analytical functions or only Newton simulation? (Need to verify)

## Cross-Domain Analogies
| Source Domain | Insight | Transfers? | Limitations |
|---------------|---------|------------|-------------|
| DrSR's Idea Library | P/N/I categorization dramatically improves SR | Yes — direct comparison | DrSR has no interactive experiments |
| NewtonBench Difficulty Levels | Principled taxonomy prevents false positive "discovery" | Yes — our environment design | NewtonBench is static, not interactive |
| SGA's Bilevel | Gradient inner loop for constant fitting | Yes — our conditions C, D | SGA doesn't accumulate strategy knowledge |
| Active Learning | Choose most informative samples to label | Yes — our experiment design step | Standard AL doesn't accumulate strategy |

## Rejected Hypotheses
| Hypothesis | Why Rejected | What We Learned |
|------------|--------------|-----------------|
| v1: Discover standard physics laws | Memorization confound (verifier 1) | Must use non-memorizable counterfactual laws |
| v2: Level 1 counterfactual (exponent changes only) | Still pattern-matchable (verifier 2) | Need Level 2+ with symmetry violations |

## Evaluation Criteria
1. **Primary**: ACE (B,D) achieves lower test MSE@100 than static (A,C) with p<0.05 (paired t-test)
2. **Secondary**: ACE matches or exceeds DrSR (E) — validates general-purpose context engineering for equation discovery
3. **Tertiary**: Gradient (C,D) improves parameter accuracy over non-gradient (A,B) — validates SGA's bilevel insight
4. **Control**: Zero-shot probe success <20% — confirms tasks aren't memorizable
5. **Interesting negative**: If gradient hurts (C < A), replicates NewtonBench's code interpreter finding in a new setting
6. **Qualitative**: Playbook trace shows emergent mathematical reasoning strategies

## Implementation Plan (4 weeks)

**Week 1: Environment + Infrastructure**
- Implement 9 counterfactual environments as Python functions
- Implement wp.Tape() gradient wrapper for parameter fitting
- Run zero-shot and structural prior probes to validate difficulty
- Pilot: 1 environment × conditions A,B × 1 seed

**Week 2: Full Baseline Runs**
- Conditions A (static) and C (+gradient) on all 9 environments
- PySR baseline (condition F) on all 9 environments
- Random search baseline on all 9 environments
- Analyze: is there signal? Adjust difficulty if needed.

**Week 3: ACE + DrSR Runs**
- Conditions B (+ACE), D (+ACE+∇), E (DrSR) on all 9 environments
- All 3 seeds for statistical power
- Collect playbook traces

**Week 4: Analysis + Write-up**
- Statistical comparisons (paired tests, learning curves)
- Playbook trace analysis (qualitative coding)
- Write CS224N paper: "Context Engineering for Interactive Equation Discovery"
