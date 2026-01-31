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

## Iteration 4 | 2026-01-31

**Goal**: Map landscape for Levin's "Platonic space" hypothesis and determine if/how to integrate a "Levin test" into the v3 experiment design.

### Observations (raw — what I found)

1. **Levin's sorting algorithm paper** (Zhang, Goldstein, Levin 2025, Adaptive Behavior): Distributed sorting agents exhibit three unexpected competencies: (a) robustness to defective elements, (b) delayed gratification (temporarily accepting worse positions to navigate around defects), (c) algotype clustering — elements running the same sorting algorithm spontaneously cluster during sorting, despite nothing in the code specifying this. Published and peer-reviewed.

2. **Platonic Representation Hypothesis** (Huh et al., ICML 2024, arXiv:2405.07987): Neural networks trained with different objectives, data, and modalities converge toward shared representations as they scale. Experiments on 5 vision + 11 language models show increasing cross-modal alignment with scale. Covered in Quanta Magazine (Jan 2026).

3. **"Geometry of Thought"** (arXiv:2601.13358): Analyzed 25,000+ CoT trajectories across 4 domains. Found universal -0.4 step-to-step coherence constant (oscillatory "zig-zag"). Domain-specific phase transitions with scale: legal reasoning "crystallizes" (45% dimensionality collapse), code forms "lattice" of strategic modes, science/math remain "liquid."

4. **KPZ universality in ML**: NO published work connecting KPZ universality class to SGD training or ML optimization. Only ML used as solver for KPZ equations (e.g., KPZNet arXiv:2306.06952). This connection is entirely speculative.

5. **Yang-Baxter in ML**: NO published work showing YBE structures emerging in ML. Only ML used to solve YBE (R-matrix Net, 2024). This connection is entirely speculative.

6. **Exaptation in evolutionary computation**: Well-established concept. Pareto-based multi-objective EAs can create selection pressures for exaptation (IEEE 2009). Novelty search and quality-diversity algorithms explicitly create conditions for bonus competencies (GECCO 2011). Computational metabolic models show latent potential for new functions (Scientific American).

7. **Halting time universality in optimization**: Published work (arXiv:1511.06444, MIT ORC) identifies two universality classes for optimization convergence — Gumbel-like and Gaussian-like. SGD on MNIST shares halting time distributions with gradient descent in spin glasses. This IS universality-class thinking applied to ML.

8. **Spontaneous symmetry in protein models** (bioRxiv 2025): Transformer-based protein generator produces symmetric structures without symmetry in training. A single attention head responsible. Directly analogous to Levin's "side quests" but in a generative model.

9. **LLM latent space geometry**: Shared depth-parameterized geometric curriculum across 24 LLMs (arXiv:2511.21594). Persistent homology reveals multi-scale topological structure (arXiv:2505.20435). Rich tools exist (Riemannian geometry, persistent homology, CKA) but have NOT been applied to evolving knowledge artifacts like playbooks.

10. **Emergent structure in knowledge bases**: NO published work studies emergent algebraic/topological structure in ACE-style playbooks or DrSR-style idea libraries. This is a genuine research gap. The tools exist; the application doesn't.

### Interpretation

The landscape splits cleanly:
- **Grounded claims**: Sorting algorithm side quests, convergent representations, universal CoT dynamics, exaptation in evolution, halting time universality — all published with evidence.
- **Ungrounded claims**: KPZ universality in ML, Yang-Baxter in ML — no evidence. These should NOT be part of our experiment framing.
- **Testable gap**: Whether LLM knowledge artifacts (ACE playbooks) exhibit emergent structure. Tools exist (persistent homology, CKA, embedding geometry), published precedent exists at the activation level, but nobody has applied them to evolving strategy libraries. This is genuinely novel and feasible.

The "Levin test" extension is low-cost (analyzes artifacts already collected in v3), falsifiable, and fills a real gap. It transforms the paper from "ACE helps equation discovery" to "we can measure emergent structure in LLM knowledge artifacts."

Key risk: ACE playbooks are short text strings, not high-dimensional activations. Topological structure may be too simple to detect anything interesting. Mitigation: use the null result as evidence that text-level knowledge artifacts don't exhibit the same convergence as activation-level representations.

### Decision
- **Action**: REFINE — add Levin extension (L1-L5 measurements) to v3 experiment design
- **Rationale**: Zero additional LLM cost, genuine research gap, falsifiable, strengthens paper
- **Confidence delta**: +0 on primary hypothesis (65 stays 65), new exploratory hypothesis at 40

### What changed in DISCOVERY.md
- Updated iteration counter to 4
- Added Open Question 4 about playbook self-organization
- Added 4 new cross-domain analogies (Levin sorting, Platonic representations, exaptation, Geometry of Thought)
- Added full "Extension: Levin Test" section with L1-L5 measurements, published-vs-speculative table, falsifiability criteria, honest assessment
- Added Levin analysis to Week 4 implementation plan

---

## Iteration 5 | 2026-01-31

**Goal**: Adversarial verification of Levin extension, then revise based on critique. Also incorporate user clarification: interest is in universality classes broadly, not KPZ specifically.

### Observations (raw — what I found)

1. **Adversarial verifier (verifier 3) rated 2 FATAL flaws**:
   - FATAL (90%): Persistent homology on ~30 points in ~768D is catastrophically underpowered. Concentration of measure makes all pairwise distances converge, destroying the distance-scale information TDA relies on. Random point clouds of size 30 in R^768 produce non-trivial Betti numbers — indistinguishable from any "signal."
   - FATAL (85%): PRH analogy is a category error. PRH concerns convergence of learned internal representations. Playbooks are output text artifacts from a stochastic generation process. Same LLM + same task = similar outputs is trivially expected, not "Platonic convergence."

2. **Weaknesses identified**:
   - "Side quests" (helpful_count=0 bullets) have a trivial explanation: Curator inertia / token budget not exceeded (80%)
   - Transfer test conflates "ACE makes good prompts" with "playbooks have emergent structure" (75%)
   - CKA on ~30 bullets is statistically meaningless (80%)

3. **Devastating counterexample**: Generate 10 random sets of 30 sentences from GPT-4 on any topic, embed them, compute persistent homology → you'll find non-trivial Betti numbers and cross-set convergence. This reproduces ALL predictions without any evolutionary dynamics.

4. **TDA on NLP is a real field**: 118 papers surveyed (arXiv:2411.10298). TDA on sentence embeddings has been done (Holmes 2020, Wright State). But typical studies use hundreds-to-thousands of points, not 20-50.

5. **Universality classes in ML** (user clarification — not just KPZ):
   - Halting time universality (arXiv:1511.06444): SGD on MNIST shares convergence-time distribution with spin glasses → Gumbel vs Gaussian universality classes
   - Representation universality (Nature Machine Intelligence, Oct 2025): debate on whether representations are universal or idiosyncratic
   - Phase transition universality: Ising-class ML phase detection (PTEP 2023)
   - These are all behavioral/convergence-level measurements, not geometric/topological

### Interpretation

The verifier is right: the geometric-level measurements (L1-persistent homology, L2-CKA) are fundamentally broken for our sample sizes. But the *behavioral*-level question remains valid and testable: **does ACE playbook evolution converge to the same strategies regardless of initialization (a universality class)?**

The fix: abandon geometric/topological measurements entirely. Replace with:
1. **Strategy coding** — human-interpretable categorization of what strategies emerge
2. **Learning curve universality** — do ACE vs non-ACE show different convergence functional forms?
3. **Transfer probes with proper controls** — does the playbook generalize beyond what the LLM's prior would?
4. **Proper null models** — random curator, random strategy generation

This is honest, feasible, and still tests the core Levin intuition (do emergent "side quests" appear?) without the broken measurement methodology.

### Decision
- **Action**: REFINE — replace geometric measurements with behavioral ones
- **Rationale**: Verifier found genuine fatal flaws in measurement methodology. Core question survives with different measurement approach.
- **Confidence delta**: -5 (from 40 → 35 on Levin extension; primary hypothesis unchanged at 65)

### What changed in DISCOVERY.md
- Updated iteration to 5
- Replaced L1-L5 measurements entirely: dropped persistent homology (L1), CKA (L2), Betti number tracking (L5); replaced with strategy coding (L1), proper null model for side quests (L2), controlled transfer probe (L3), universality class identification via learning curves (L4), PCA-based diversity dynamics (L5)
- Updated falsifiability criteria to match new measurements
- Updated honest assessment: added "What was tried and rejected" section documenting the geometric approach failure
- Lowered confidence from 40→35
- Reframed from "Platonic space" to "universality" (less philosophically loaded, more testable)

---

## Iteration 6 | 2026-01-31

**Goal**: Run domain skeptic (verifier 4) on revised Levin extension v2, then produce final v3.

### Observations (raw — what I found)

1. **Domain skeptic key critiques**:
   - 3 seeds catastrophically underpowered for χ² test on 4 categories — need 20+ seeds for distributional claims (confidence: 95%)
   - "Universality class" via curve fitting misuses the physics term — Clauset et al. 2009 showed distinguishing power laws from alternatives needs hundreds of samples (confidence: 90%)
   - Side quest detection confounded: zero-helpful bullets persist due to curator inertia (can't distinguish from emergent retention without intervention) (confidence: 85%)
   - Diversity dynamics measurement is predetermined — selection pressure always reduces diversity vs random (confidence: 90%)
   - Transfer probe is the best measurement but needs ablated playbook control (same structure, scrambled content) (confidence: 80%)

2. **NFL theorem relevance**: the claim of "convergence regardless of initialization" runs into No Free Lunch — what we're really testing is whether ACE's curator has a strong inductive bias that dominates the initialization signal. This is weaker but more defensible.

3. **Critic quoted Levin's own critics**: "I do think you may be reading too much significance into many of your discoveries, particularly the bubble sort algorithm." — applicable to our extension too.

4. **Actionable suggestion**: intervention experiments (forcibly remove/inject strategy bullets, measure recovery) provide causal evidence that passive observation cannot.

### Interpretation

Both verifiers agree: the Levin extension's core intuition (search for emergent structure in the process, not just the output) is interesting but the measurements need to be much more conservative. The intervention experiment (L2) is the strongest contribution — it's causal and doesn't require large N. The transfer probe (L3) is second-strongest with proper ablated control. Strategy profiling (L1) can only be descriptive with 3 seeds. "Universality class" language should be dropped — we're testing for "consistent inductive biases."

The extension is now modest but honest: 3 measurements (intervention, transfer, descriptive profiling) that add ~150 LLM calls total and analyze existing artifacts. The theoretical framing connects to published work (exaptation, inductive bias, Levin sorting) without overclaiming.

### Decision
- **Action**: ACCEPT as Candidate — the Levin extension has survived 2 adversarial rounds with appropriate revisions
- **Rationale**: Measurements are honest, feasible, properly controlled, and falsifiable. The extension adds novel analysis at low cost.
- **Confidence delta**: -5 (from 35 → 30 on Levin extension). Primary hypothesis unchanged at 65.

### What changed in DISCOVERY.md
- Updated iteration to 6
- Replaced L1-L5 measurements with v3: descriptive strategy profiles (L1), causal intervention experiment (L2), transfer probe with ablated control (L3), qualitative learning curve comparison (L4), dropped L5 (diversity dynamics)
- Added "Survived Attacks (Levin Extension)" section documenting all 6 addressed critiques
- Updated falsifiability criteria
- Updated "tried and rejected" list with items 4-6
- Lowered confidence to 30

---
