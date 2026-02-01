# Research: Grounded Abduction — Learning Physical Laws Through Embodied Simulation
> Started: 2026-01-31

## Summary

This research maps the intersection of abductive reasoning, evolutionary LLM search (ShinkaEvolve/AlphaEvolve), context engineering (ACE/Dynamic Cheatsheet), test-time learning (TTT-Discover/TTRL), and differentiable physics simulation (Newton) toward building systems that discover physical laws through embodied experimentation. The core thesis: neither frozen LLM search, nor context engineering, nor test-time RL alone suffices for scientific invention — the missing piece is **manipulative abduction** grounded in physically consistent simulation.

The key paper "LLMs Can't Jump" (Zahavy, 2026) formalizes this gap using Peirce's inference taxonomy and Einstein's discovery of General Relativity as a case study. We aim to formalize the mathematical optimality of combining ShinkaEvolve's evolutionary search with ACE's playbook evolution for abductive discovery tasks.

## Core Concepts

- **Abduction (Peirce, 1934)**: Inference to the best explanation. Given surprising fact C and rule A→C, infer A is plausible. Structural permutation: Rule+Result→Case. Distinct from deduction (Rule+Case→Result, truth-preserving) and induction (Case+Result→Rule, statistical). (Stanford Encyclopedia of Philosophy; Zahavy 2026)
- **Manipulative Abduction (Magnani, 2009)**: Abduction grounded in embodied simulation — active interaction with mental/physical models to generate hypotheses through "thinking by doing." Einstein's elevator thought experiment is the paradigmatic example. Requires counterfactual intervention, not passive observation. (Magnani, "Abductive Cognition," Springer 2009)
- **Frankfurt's Problem**: Peirce's abduction schema assumes the hypothesis A already exists — it describes hypothesis *selection*, not hypothesis *invention*. The creative leap of generating genuinely novel hypotheses remains unformalized. (Stanford Encyclopedia; Zahavy 2026 Section 5)
- **Einstein's E-J-A Cycle**: Sense Experience (E) → Jump (J) → Axioms (A) → Deduction → Theorems (S) → Verification against E. The Jump is the abductive leap that cannot be reduced to induction or deduction. (Einstein's letter to Solovine; Norton 2020; Zahavy 2026)
- **Creativity as Compression (Schmidhuber, 2008)**: Discovery = finding simpler programs that explain observations. Zahavy argues this fails when there's no error signal (Newton's gravity had near-zero loss) and when hypothesis space must *expand* before it simplifies (non-Euclidean geometry). (Schmidhuber 2008; Zahavy 2026 Section 3)
- **Frozen LLM Problem**: In AlphaEvolve/FunSearch/ShinkaEvolve, the LLM never updates weights from evolutionary feedback. Pure search without learning — the model cannot internalize discovered patterns. Max-Q estimators create absorbing states with no policy improvement to escape local optima. (EvoTune, arXiv 2504.05108; user proposal)
- **Test-Time Learning**: Updating LLM weights during inference via RL (LoRA). TTT-Discover uses PUCT + entropic objective for exploration. TTRL boosts pass@1 by 211% on AIME 2024. Solves absorbing state problem via policy improvement. (TTT-Discover arXiv 2601.16175; TTRL arXiv 2504.16084)
- **Context Engineering**: External memory that evolves without weight updates. ACE's Generate→Reflect→Curate loop accumulates domain knowledge in structured playbooks. Dynamic Cheatsheet improved Claude 3.5 from 23%→50% on AIME 2024. (ACE arXiv 2510.04618; DC arXiv 2504.07952)
- **Bilevel Optimization for Discovery**: LLM proposes discrete hypotheses (outer loop), differentiable simulator optimizes continuous parameters (inner loop). Scientific Generative Agent (SGA) validated on constitutive law discovery. (Ma et al., ICML 2024, arXiv 2405.09783)
- **PUCT (Predictor + Upper Confidence bounds for Trees)**: Tree search algorithm used in AlphaZero/AlphaProof. Balances exploitation (best-known actions) with exploration (uncertain actions). When combined with LLM prior policy, enables structured hypothesis space exploration. (Hubert et al., Nature 2025; TTT-Discover 2026)

## Key Literature

- **"LLMs Can't Jump" (Zahavy, 2026)**: Position paper from Google DeepMind. LLMs master induction and deduction but lack abduction. Uses Einstein's GR as case study. Proposes physically consistent world models for counterfactual simulation. [philsci-archive.pitt.edu/28024/](https://philsci-archive.pitt.edu/28024/)
- **"AlphaEvolve" (Novikov et al., 2025)**: Gemini-powered evolutionary coding agent. Discovered faster matrix multiplication. Limited to problems with automated evaluation — no physical experimentation. [arXiv:2506.13131](https://arxiv.org/abs/2506.13131)
- **"FunSearch" (Google DeepMind, 2023)**: Frozen LLM + evaluator for mathematical discovery. Best-shot prompting. Addressed cap set problem. Nature publication. [nature.com/articles/s41586-023-06924-6](https://www.nature.com/articles/s41586-023-06924-6)
- **"ShinkaEvolve" (Sakana AI, 2025)**: Sample-efficient evolutionary program discovery (~150 evaluations for SOTA circle packing). Bandit-based LLM ensemble, novelty rejection sampling. [arXiv:2509.19349](https://arxiv.org/abs/2509.19349)
- **"ThetaEvolve" (2025)**: Simplifies AlphaEvolve with single LLM + optional test-time RL. Island-based evolution with lazy penalties. [arXiv:2511.23473](https://arxiv.org/abs/2511.23473)
- **"EvoTune" (CLAIRE-Labo, 2025)**: Directly addresses frozen LLM limitation — combines evolutionary search with DPO-based fine-tuning. Validates that unfrozen LLMs significantly outperform frozen baselines. [arXiv:2504.05108](https://arxiv.org/abs/2504.05108)
- **"TTT-Discover" (Yuksekgonul et al., 2026)**: RL at test-time for discovery. SOTA on Erdős minimum overlap, autocorrelation inequalities. Uses open models. [arXiv:2601.16175](https://arxiv.org/abs/2601.16175)
- **"TTRL" (2025)**: Test-time RL without ground-truth labels. 211% boost on AIME 2024 with Qwen-2.5-Math-7B. [arXiv:2504.16084](https://arxiv.org/abs/2504.16084)
- **"ACE" (Zhang et al., 2025)**: Agentic Context Engineering. +10.6% on agent tasks, +8.6% on finance. Generator/Reflector/Curator loop with playbook evolution. [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
- **"Dynamic Cheatsheet" (2025)**: Non-parametric memory for black-box LLMs. +27% on AIME, +9% on GPQA-Diamond. [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)
- **"SGA: LLM and Simulation as Bilevel Optimizers" (Ma et al., 2024)**: LLM proposes discrete hypotheses, differentiable sim optimizes continuous params. ICML 2024. [arXiv:2405.09783](https://arxiv.org/abs/2405.09783)
- **"AI Physicist" (Wu & Tegmark, 2018)**: Discovers physical laws in simulated universes using divide-and-conquer + Occam's razor. >90% accuracy on gravity, EM, harmonics. [arXiv:1810.10525](https://arxiv.org/abs/1810.10525)
- **"AI-Newton" (Fang et al., 2025)**: Discovers Newton's 2nd law, energy conservation from data without prior knowledge. Three-layer DSL architecture. [arXiv:2504.01538](https://arxiv.org/abs/2504.01538)
- **"EVA: Evolutionary Abduction" (Pietrantuono, 2022)**: Computational abduction via evolutionary operators. Outperforms causal structure discovery. [ICLR OpenReview](https://openreview.net/forum?id=PnraKzlFvp)
- **"PhysGym" (2025)**: Benchmark for LLM-based interactive physics discovery with sequential data gathering. [arXiv:2507.15550](https://arxiv.org/abs/2507.15550)
- **"Genie 3" (DeepMind, 2026)**: Action-controllable world model generating interactive 3D environments. Prerequisite for manipulative abduction. [deepmind.google/blog/genie-3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
- **"Abductive Cognition" (Magnani, 2009)**: Seminal book on manipulative abduction, embodied cognition, epistemic mediators. (Springer)
- **"Inducing Causal World Models in LLMs" (2025)**: Causal Physics Module augmenting LLMs with explicit physical dynamics. Causal Consistency Score metric. [arXiv:2507.19855](https://arxiv.org/abs/2507.19855)
- **"ARC-AGI" (Chollet, 2019/2024)**: Benchmark requiring abductive reasoning from sparse examples. 2024 prize improved SOTA from 33%→55.5%. Tests logical leap but not manipulative component. [arcprize.org](https://arcprize.org/)

## Methods & Techniques

- **Evolutionary Program Search (AlphaEvolve/ShinkaEvolve)**: Population of programs mutated by LLM, evaluated by automated fitness. Best-shot prompting, island model, bandit-based model selection. Use when: optimizing code/algorithms with clear metrics.
- **ACE Playbook Evolution**: Generate→Reflect→Curate loop. Structured bullets with helpful/harmful counters. ADD/UPDATE/MERGE/DELETE operations. Semantic dedup via FAISS. Use when: accumulating domain knowledge without weight updates.
- **Test-Time RL (TTT-Discover/TTRL)**: LoRA weight updates during inference. PUCT for tree search. Entropic objective for exploration diversity. Use when: single hard problem requiring adaptation beyond frozen model capacity.
- **DPO for Evolutionary Learning (EvoTune)**: Convert program rankings into preference pairs. Update LLM via Direct Preference Optimization. Avoids explicit Q-value learning. Use when: evolutionary search should improve the generator itself.
- **Bilevel LLM+Simulator (SGA)**: Outer loop = LLM proposes symbolic hypothesis. Inner loop = differentiable sim optimizes continuous params. Use when: hypothesis has both symbolic and continuous components.
- **Symbolic Regression (AI Feynman, PSE)**: Search for mathematical expressions fitting data. Physics-informed priors (dimensional analysis, symmetry). Use when: discovering interpretable equations from dense data.
- **MCTS/PUCT with LLM Prior**: LLM provides prior policy for tree search. UCB-style exploration bonus. Use when: hypothesis space is discrete and combinatorial.

## Implementations

- **ShinkaEvolve** (Python): Evolutionary LLM search framework with Hydra config, SQLite population DB, multi-island model. Local repo: `/home/users/duynguy/proj/grounded/ShinkaEvolve`. [sakana.ai/shinka-evolve](https://sakana.ai/shinka-evolve/)
- **ACE** (Python): Context engineering with Generator/Reflector/Curator. Local repo: `/home/users/duynguy/proj/grounded/ace`. [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
- **Newton** (Python/Warp): GPU-accelerated differentiable physics engine. Local repo: `/home/users/duynguy/proj/grounded/newton`. [github.com/newton-physics/newton](https://github.com/newton-physics/newton)
- **EvoTune** (Python): Evolutionary search + DPO fine-tuning. [github.com/CLAIRE-Labo/EvoTune](https://github.com/CLAIRE-Labo/EvoTune)
- **PySR** (Python/Julia): Symbolic regression with physics priors. [github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)
- **ARC-AGI** (Python): Abductive reasoning benchmark. [github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI)

## Open Gaps

1. **No system combines evolutionary LLM search + context engineering + differentiable physics for abductive discovery.** ShinkaEvolve does search, ACE does context, Newton does physics — nobody has unified them. Impact: HIGH. Approach: This project.
2. **Mathematical optimality of ShinkaEvolve+ACE hybrid is uncharacterized.** When does playbook-conditioned mutation outperform static prompting? What's the regret bound? Impact: HIGH. Approach: Formalize as multi-armed bandit over mutation strategies with ACE as adaptive policy.
3. **Frankfurt's Problem remains open computationally.** Abduction as formalized by Peirce assumes hypotheses exist. No system generates genuinely novel hypotheses — they recombine existing concepts. Impact: HIGH but possibly intractable. Approach: Manipulative abduction via interactive simulation may sidestep this by grounding hypothesis generation in physical variation.
4. **Q-estimator absorbing states in frozen LLM search.** Max-Q creates local optima that frozen LLMs cannot escape. EMA/median-Q proposed but not validated on physics tasks. Impact: MEDIUM. Approach: Compare max-Q vs EMA-Q vs PUCT-entropic on physics discovery benchmarks.
5. **No physics discovery benchmark with controlled difficulty.** PhysGym exists but is limited. Need pendulum→momentum→projectile progression with ground-truth verification. Impact: MEDIUM. Approach: Build in Newton.
6. **TTT-Discover hasn't been applied to physical law discovery.** Only tested on math/algorithms/biology. Impact: MEDIUM. Approach: Adapt to physics domain.
7. **ACE playbook evolution has no theoretical convergence guarantees.** Empirically works but no PAC-style bounds. Impact: MEDIUM. Approach: Analyze as online learning with structured hypothesis class.

## Connections

- [[Manipulative Abduction]] --requires--> [[Differentiable Physics (Newton)]]: Physical simulation provides the "embodied" substrate for thought experiments
- [[ACE Playbook]] --replaces--> [[ShinkaEvolve task_sys_msg]]: Playbook evolves mutation strategies based on which approaches improve fitness
- [[ShinkaEvolve EvolutionRunner]] --evolves--> [[ACE Prompt Templates]]: Evolutionary search can optimize ACE's generator/reflector/curator prompts
- [[PUCT/MCTS]] --explores--> [[Hypothesis Space]]: Tree search over discrete symbolic hypotheses (physical laws)
- [[TTT-Discover LoRA]] --updates--> [[Frozen LLM]]: Weight updates enable escaping absorbing states in evolutionary search
- [[EvoTune DPO]] --solves--> [[Frozen LLM Problem]]: Preference learning from program rankings
- [[Bilevel SGA]] --combines--> [[LLM (discrete) + Newton (continuous)]]: LLM proposes law form, Newton optimizes constants via gradients
- [[ACE Reflector]] --tags--> [[ShinkaEvolve Program.text_feedback]]: Structured helpful/harmful tagging of mutation strategies
- [[ACE Curator]] --replaces--> [[ShinkaEvolve meta_rec_interval]]: Structured playbook operations vs. free-form meta-LLM recommendations
- [[ARC-AGI]] --tests--> [[Abduction (logical)]]: But misses manipulative/embodied component
- [[Creativity as Compression]] --contradicted by--> [[Einstein's GR discovery]]: No error signal from Newtonian mechanics to drive inductive search (Zahavy 2026)
- [[EVA (Evolutionary Abduction)]] --precursor to--> [[ShinkaEvolve + Abduction]]: Evolutionary operators for causal hypothesis generation

## Formal Framework Sketch: ShinkaEvolve × ACE for Abductive Discovery

### Problem Setting
- **Environment**: Differentiable physics simulator (Newton) with state space S, action space A (interventions), observation function O
- **Goal**: Discover physical law f* : S → S (ground truth dynamics) expressed as symbolic program
- **Agent**: LLM M (frozen or learning) with context/playbook P

### The Abductive Loop (formalized)
```
For each generation g = 1, ..., G:
  1. EXPERIMENT DESIGN (ACE Generator + Playbook):
     intervention_g = Generator(playbook_P, history_H)

  2. SIMULATE (Newton):
     observation_g = Newton.step(state, intervention_g)

  3. HYPOTHESIZE (ShinkaEvolve mutation):
     hypothesis_g = LLM.mutate(parent_program, playbook_P, observation_g)

  4. EVALUATE (Newton differentiable):
     score_g = ||hypothesis_g(states) - Newton.ground_truth(states)||
     gradient_g = ∇_params score_g  (via Newton's differentiability)

  5. REFLECT (ACE Reflector):
     tags = Reflector(hypothesis_g, score_g, playbook_bullets_used)

  6. CURATE (ACE Curator, every K steps):
     playbook_P = Curator(playbook_P, recent_reflections, stats)

  7. LEARN (optional, TTT-Discover style):
     M.update_weights(LoRA, reward=score_improvement)
```

### Mathematical Structure
- **Outer loop** (ShinkaEvolve): Evolutionary search over program space P with population {p_i}
- **Middle loop** (ACE): Online learning over mutation strategy space via playbook evolution
- **Inner loop** (Newton): Gradient-based continuous parameter optimization within each hypothesis

This is a **three-level optimization**:
- Level 1: min_params ||f_hypothesis(x; params) - f*(x)|| (Newton gradient descent)
- Level 2: max_playbook E[fitness(mutate(parent, playbook))] (ACE bandit/online learning)
- Level 3: max_population E[best_fitness(population_g)] (ShinkaEvolve evolution)

### Regret Analysis (sketch)
If ACE's playbook is viewed as an adaptive policy over K mutation strategies, and each strategy has unknown expected fitness improvement μ_k, then:
- ACE's helpful/harmful counters implement an empirical mean estimator
- The Curator's ADD/DELETE implements an adaptive arm set (contextual bandit with growing action space)
- Regret bound: O(√(KT log T)) for K strategies over T mutations, assuming sub-Gaussian rewards

The key question: **does ACE's structured curation outperform ShinkaEvolve's bandit-based model selection alone?** Hypothesis: yes, because ACE preserves *why* a strategy worked (the bullet content), enabling transfer across similar problems, while bandits only track aggregate statistics.

## Progress

- [x] Read "LLMs Can't Jump" paper — full extraction complete
- [x] Research abductive reasoning foundations (Peirce, Magnani)
- [x] Research AlphaEvolve family (AlphaEvolve, FunSearch, ShinkaEvolve, ThetaEvolve, EvoTune)
- [x] Research test-time learning (TTT-Discover, TTRL, PUCT)
- [x] Research context engineering (ACE, Dynamic Cheatsheet)
- [x] Research physics discovery AI (AI Physicist, SGA, AI-Newton, symbolic regression)
- [x] Research world models and embodied simulation (Genie, Newton, causal world models)
- [x] Map ShinkaEvolve × ACE integration points
- [x] Sketch formal three-level optimization framework
- [ ] Deep dive: regret bounds for ACE playbook as contextual bandit
- [ ] Deep dive: convergence of evolutionary search with adaptive mutation distribution
- [ ] Deep dive: differentiable physics gradients through symbolic hypothesis
- [ ] Identify closest existing formal frameworks (PAC-Bayes, evolutionary game theory, online convex optimization)
- [ ] Research: how does SGA's bilevel optimization relate to our three-level structure?
- [ ] Research: causal discovery methods (PC algorithm, do-calculus) as formal grounding for abduction
