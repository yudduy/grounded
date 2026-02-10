# Distillation Handbook

Index of distilled questions and their incompressible answers.

---

## Optimal In-Context RL for LLM Self-Improvement
- File: docs/distill/in-context-reinforcement-learning-llm.md
- Core: The playbook's value is not in its content but in the process that generates it -- optimize the search over playbook configurations, not any single playbook.
- Thinkers: Shannon, Kolmogorov, Deutsch, Polya, Lai & Robbins, Snell et al., Xie et al.
- Opens: How do you discover which strategy archetypes are missing before encountering the problems that need them?

## Discovering Missing Strategy Archetypes
- File: docs/distill/discovering-missing-strategy-archetypes.md
- Core: A system cannot find its own gaps from the inside. Gap discovery is Probe (adversarial + cross-domain exposure) → Detect (cluster failures into silhouettes) → Fill (recombine across domains, select for diversity). Fast-reactive, never proactive.
- Thinkers: Mendeleev, Altshuller (TRIZ), Tonegawa (immune V(D)J), Kuhn, Gould & Vrba, Mouret & Clune (MAP-Elites), Page, Nemhauser
- Opens: How do you build a critic better than the system it criticizes? Is there a bootstrapping path from weak mutual criticism to strong?

## Small Model, Big Performance: Verification (Feb 2026)
- File: docs/distill/small-model-big-performance-verification.md
- Core: Architecture sound, theory oversold. All 6 papers VERIFIED. "Near-optimal" is a composition fallacy; Condorcet fails with correlated verifiers; rate-distortion uncomputed. Cheatsheet is the load-bearing component; verification is optimization on top. Run baselines first.
- Papers: Weaver, rStar-Math, REBASE, ThinkPRM, Adaptive ITC, Inference Scaling fLaws — all verified
- Landscape: rStar2-Agent (agentic RL > MCTS), Dynamic Cheatsheet validated, verifier correlation now known, adaptive N being solved, step-level verification cost collapsing
- Opens: What is the actual gap between Dynamic Cheatsheet alone and DC + outcome verification on AIME 2024 with 7B?

## SDFT: Integrate or Sequence? (Feb 2026)
- File: docs/distill/sdft-integrate-or-sequence.md
- Core: Sequence, do not integrate. SDFT solves context overflow, which the ACE curator already prevents at 30-problem scale. The 3-stage loop (discover-cache-consolidate) has zero validated transitions. Run the SDFT MVE independently (hand-curated playbook + distillation, 2-3 hrs) before integrating.
- Thinkers: Polya, McClelland (CLS), Xie et al. (ICL as Bayesian inference)
- Opens: If ACE+TTRL shows context drift, is SDFT the right fix -- or should you simply re-curate the playbook against the updated model?

## Memory x RL Interaction: Publishability (Feb 2026)
- File: docs/distill/memory-rl-interaction-publishability.md
- Core: Publishable with 3 mandatory fixes; marginal without. MATH-500 is saturated (92.8% baseline) — paper-killer. Novelty gap (co-evolving context + weights) is real but closing fast (4 papers in 6 weeks). Absorption test has prior art (FT+RAG, GenPI) but co-evolution variant is novel. Reframe from "does memory help?" to "how do two learning channels interact?"
- Thinkers: Vygotsky (scaffolding), McClelland (CLS), Goodhart
- Opens: What determines the rate at which RL training absorbs explicit knowledge from context into weights?

## Highest-Leverage Move for Small-Model AIME (Feb 2026)
- File: docs/distill/small-model-aime-via-diversity-regularized-memory.md
- Core: Skip diversity-regularized memory; train RL directly. BAPO achieves 70.8% AIME24 at 7B via entropy-preserving GRPO on distilled bases. Small models have diversity DEFICIT, not diversity COLLAPSE — memory preserves diversity that doesn't exist. EVOL-RL's novelty reward IS the ACE/DC insight recast as RL reward shaping.
- Thinkers: Kolmogorov, Elman (curriculum), NVIDIA ADLR (AceMath-RL), BAPO authors
- Opens: What is the remaining 30% made of? Are unsolved AIME problems beyond 7B capacity, or solvable with better test-time search?
