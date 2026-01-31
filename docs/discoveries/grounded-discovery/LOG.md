# Research Log: Grounded Discovery Experiment Design
> Append-only. Never edit past entries.

---
## Iteration 1 | 2026-01-31

**Goal**: Map the landscape of LLM physics discovery systems and identify the feasible experiment gap.

### Observations (raw — what I found)

1. **PhysGym** (arXiv:2507.15550): 97 physics problems, 6 domains. Best: Gemini-2.5-Pro at 74% (L1, full context) → 32% (L4, minimal knowledge). Uses equation-based simulation (not differentiable). LLM proposes experiments within 100-experiment budget. Key insight: performance is **non-monotonic** — some problems solved only with less prior knowledge.

2. **SGA** (Ma et al., ICML 2024): Bilevel optimization using Warp (Newton's backend!). LLM proposes symbolic expressions + parameterization, differentiable MPM optimizes constants. MSE 5.2×10⁻⁵ vs baseline 128.0 on constitutive laws. Uses top-K feedback + loss curves to guide LLM. Temperature-based exploration (0.5 exploit, 1.0 explore, 1:3 ratio).

3. **AI-Newton** (arXiv:2504.01538): Pure symbolic — no LLM. DSL with algebraic operations + derivatives. Discovers Newton's 2nd law, energy/momentum conservation from 46 experiments. Uses Gröbner basis for simplification. Handles noise. But doesn't scale — DSL is hand-designed.

4. **Gravity-Bench** (ICML 2025): Interactive binary star discovery. OOD physics (non-standard gravity). Current models struggle. Budget-constrained observation planning.

5. **NewtonBench** (arXiv:2510.07172): 324 tasks, "metaphysical shift" — mutated physics laws. GPT-5: 73% vanilla → 30% complex. Code interpreter paradox: helps weak models, hurts strong ones. 0.0001 noise → 13-15% drop.

6. **DrSR** (arXiv:2506.04282): Dual-reasoning SR with "Idea Library" — categorizes past outputs as Positive/Negative/Invalid, accumulates insights. 99.94% vs 7.62% (LLM-SR) on oscillations. This is essentially ACE's playbook applied to symbolic regression.

7. **Newton API** (local code): `ViewerNull` exists for headless. `wp.Tape()` for gradients. `state.particle_q.numpy()` for data extraction. `model.set_gravity()` to change gravity. Existing pendulum and diffsim_ball examples provide templates.

8. **FarmShare constraint**: No GPU on login nodes. Newton uses Warp which strongly prefers CUDA. CPU fallback exists (`self.graph = None` path in examples) but may be slow.

### Interpretation

The landscape reveals a clear gap: **nobody combines interactive experiment design + differentiable physics + evolving context memory**. Specifically:

- SGA has differentiable physics but the LLM doesn't design experiments (fixed scenarios)
- PhysGym has interactive experiment design but no differentiable physics
- DrSR has evolving memory (idea library ≈ ACE playbook) but no physics simulation
- Gravity-Bench has interactive + OOD physics but no gradients

Our experiment unifies all three: the LLM designs experiments (PhysGym-style), gets gradient signal (SGA-style), and accumulates insights (DrSR/ACE-style).

**Feasibility concerns**:
1. GPU: Newton on CPU is possible but slow. For toy problems (single pendulum, 2 particles), CPU should suffice. SLURM GPU jobs are available on FarmShare.
2. LLM costs: 50 calls per environment × 3 environments × 4 conditions = 600 LLM calls. At ~$0.01-0.10/call, manageable.
3. Anti-memorization: Critical. Must use obfuscated variables AND modified physics constants. PhysGym's L4 approach + NewtonBench's metaphysical shift.

**Key design decision**: Start with single-agent (not population-based ShinkaEvolve) to reduce complexity. The core comparison is ACE vs no-ACE, not evolutionary vs non-evolutionary. Population-based search is a follow-up.

### Decision
- **Action**: HYPOTHESIZE — formulate the concrete experiment design
- **Rationale**: Landscape is well-mapped, gap is clear, tools exist
- **Confidence delta**: +15 (from 60 → 75) — the gap is real and the tools are available

### What changed in DISCOVERY.md
- Created full document with landscape, hypothesis, experiment design, evaluation criteria
- Populated cross-domain analogies from DrSR, SGA, PhysGym, Gravity-Bench

---

## Iteration 2 | 2026-01-31

**Goal**: Stress-test experiment design v1 via adversarial verifier, then revise.

### Observations (raw — what I found)

1. **Adversarial verifier rated 7 flaws** (confidence 60-90%):
   - FATAL: Anti-memorization insufficient (85%) — changing g=9.81→5 doesn't prevent structural recall of T=2π√(L/g). LLMs pattern-match the *form*, not the constant.
   - FATAL: PhysGym already did the exact interactive-experiment-design paradigm (90%) — our v1 was essentially "PhysGym + ACE + Newton."
   - Weakness: Novel gap overstated (80%) — all components exist separately
   - Weakness: Newton CPU unvalidated (70%)
   - Weakness: 50 calls may be insufficient (75%)
   - Weakness: ACE not designed for physics (65%)
   - Weakness: Confounded comparison design (60%)

2. **SGA explicitly addressed memorization** by using imaginary constitutive laws rather than real physics.

3. **NewtonBench's counterfactual shifts** — F=Gm₁m₂/r^1.5 instead of r². Dimensionally coherent mutations. GPT-5 drops from 73% → 30%.

4. **DrSR's idea library** — Positive/Negative/Invalid categorization. 99.94% vs 7.62%.

5. **PhysGym failure mode**: LLMs "fail to design informative experiments" — doesn't know what to test next.

### Interpretation

The verifier's memorization critique is correct and devastating for v1. The fix: **counterfactual physics** where the functional form itself is novel. The reframe: not "discovery" but "interactive law recovery on non-memorizable tasks." The scientific question: *does ACE help when memorization fails?*

### Decision
- **Action**: REFINE — major revision of hypothesis and experiment design
- **Rationale**: Verifier found genuine fatal flaws. Counterfactual physics + reframed hypothesis addresses both.
- **Confidence delta**: -5 (from 75 → 70)

### What changed in DISCOVERY.md
- Switched all environments to counterfactual physics
- Added zero-shot memorization control
- Reframed hypothesis: "ACE improves sample efficiency on counterfactual physics"
- Added PySR and random search baselines
- Added anonymized variable names
- Reduced to 30-round budget
- Moved v1 to Rejected table

---

## Iteration 3 | 2026-01-31

**Goal**: Run domain skeptic (verifier 2) on revised v2 design, then produce final experiment design.

### Observations (raw — what I found)

1. **Domain skeptic (verifier 2) key critiques**:
   - Level 1 counterfactuals (exponent changes) are still pattern-matchable (80% confidence) — LLMs generalize power-law structures
   - LLM-SRBench (ICML 2025 Oral) already does iterative LLM+data for equation discovery — our "first context engineering for physics" claim is false
   - 30 rounds too few — PySR needs 1000s of evaluations, SRBench uses 30 independent trials
   - Need direct comparison to DrSR, not just PySR
   - Gradient may hurt (NewtonBench found code interpreter degrades strong models)
   - The unfair question: "Why would LLM+memory beat PySR which tries millions of forms?"

2. **NewtonBench Level taxonomy** (from research):
   - Level 1: Exponent modification (F∝r^1.5 instead of r^2) — still solvable by structural priors
   - Level 2: Symmetry violations (m₁×m₂ → m₁+m₂, non-reciprocal forces) — requires genuine reasoning
   - Level 3: Structural novelty (new terms, cross-coupling) — very hard

3. **LLM-SRBench** (ICML 2025 Oral): 239 problems, best system 31.5% symbolic accuracy. Creates novel equations by combining known terms with synthetic variations. This IS context engineering for equation discovery — contradicts our novelty claim.

### Interpretation

The v2 design needs two more fixes:
1. **Upgrade to Level 2+ counterfactuals** — not just exponent changes but symmetry violations
2. **Include DrSR as explicit condition** — it's the closest published method and the honest comparison
3. **Increase rounds to 100** — 30 is too few for symbolic regression
4. **Reframe the gap honestly**: not "first to combine X+Y+Z" but "does structured context accumulation help when LLMs must reason from scratch on interactive, non-memorizable tasks?"
5. **Use analytical functions with wp.Tape()** instead of Newton physics engine — counterfactual physics requires custom dynamics anyway, and this sidesteps GPU issues

The experiment is now a clean ablation study: Static vs ACE vs DrSR × with/without gradient, on 9 counterfactual environments of calibrated difficulty.

### Decision
- **Action**: REFINE → produce v3 (final candidate)
- **Rationale**: Both verifiers' concerns addressed. Hypothesis is modest and testable.
- **Confidence delta**: -5 (from 70 → 65) — more honest about expected effect size

### What changed in DISCOVERY.md
- Upgraded environments to Level 2+ (symmetry violations, non-reciprocal forces, cross-coupling)
- Added DrSR as condition E (direct comparison)
- Increased to 100 rounds
- Reframed gap as empirical questions, not engineering novelty
- Added structural prior probe control
- Added statistical design (162 runs, paired comparisons)
- Added 4-week implementation plan
- Moved v2 Level 1 counterfactual to Rejected table
- Added honest effect size prediction (~10-50%, not 99% vs 7%)
- Changed implementation to analytical functions + wp.Tape() (sidesteps Newton-on-CPU)

---
