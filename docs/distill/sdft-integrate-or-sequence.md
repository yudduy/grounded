# SDFT: Integrate or Sequence?

## The Question You Asked
Should SDFT (Self-Distillation Fine-Tuning) be integrated into the current ACE+TTRL co-evolution experiment, or kept as a separate follow-up? Given limited compute on a single A100, we cannot afford to waste time on a combined experiment that fails for unclear reasons.

## The Real Question
When you have two unvalidated primitives that theoretically compose, does combining them risk undebuggable failure -- or does separating them risk redundant compute on an obviously incomplete experiment?

## The Root
The exploration-exploitation tradeoff applied to experiment design itself. Scientific rigor (isolate variables) vs. engineering pragmatism (finite resources, the whole may differ from parts).

## Assumptions to Challenge
1. SDFT actually works at 7B scale -- **CONTESTED** (55%). Works for Science QA (+4pts) but untested for math reasoning. OPSD at 8B shows +0.8% on AIME with 4-8x token efficiency, but SDFT paper's 3B *underperforms* SFT.
2. Context overflow is a real problem at this scale -- **REFUTED** (80%). 30 problems, 5 epochs, curator enforces ~80K budget. Playbook stabilizes after ~50-80 encounters. SDFT solves a problem that does not arise here.
3. ACE+TTRL is meaningful without consolidation -- **CONFIRMED** (85%). The combination is the novel experiment. Context drift (playbook becoming stale as weights update) is itself an informative finding worth observing.
4. The 3-stage loop is architecturally sound -- **CONTESTED** (40%). Zero of three transitions (TTRL->ACE, ACE->SDFT, SDFT->TTRL) validated. Three different optimization objectives (majority-vote rewards, reflector curation, KL divergence). Architecture fiction disguised as design.
5. On-policy distillation preserves reasoning quality at 7B -- **CONTESTED** (50%). OPSD at 8B works for AIME. But SDFT paper tests factual knowledge, not strategic/procedural knowledge. Mode-covering in KL minimization may smooth sharp reasoning chains.
6. Single A100 fits SDFT on top of GRPO -- **CONTESTED** (45%). Memory fits with LoRA. Wall-clock is the constraint: 2.5x FLOP overhead risks exceeding Colab session limits.

## First Principles
1. Three unvalidated primitives combined produce undecomposable failures. If ACE+TTRL+SDFT fails, three-way attribution is intractable.
2. ACE+TTRL is the higher-priority unknown -- no one has published this combination. Its result (positive or negative) determines whether SDFT integration is worth pursuing.
3. SDFT's value proposition (consolidate context into weights to prevent overflow) addresses a problem that does not exist at 30 problems and 5 epochs. The curator already compresses.
4. Expected information value per compute unit is higher for clean experiments. ACE+TTRL alone yields interpretable results; the combined experiment has higher ceiling but lower expected value.
5. A cheap MVE (minimum viable experiment) can isolate the SDFT hypothesis in 2-3 hours: hand-curated playbook + distillation test on base Qwen-2.5-Math-7B.
6. The 3-stage loop (discover-cache-consolidate) is a narrative, not a proof. Satisfying metaphors are where researchers waste compute.

## The Answer
**Sequence, do not integrate.** SDFT solves context overflow, which the ACE curator already prevents at this experiment's scale (30 problems, 5 epochs). The 3-stage loop has zero validated transitions -- combining three unvalidated primitives creates undebuggable failures. ACE+TTRL alone has higher expected information value per compute unit because its results are interpretable: a positive result proves context-weight co-evolution works; a negative result (context drift) tells you whether consolidation is even needed. Run the SDFT MVE independently (hand-curated playbook + distillation, 2-3 hours) to validate the primitive before integrating it into anything.

## Key Insights
1. **Context overflow is not your bottleneck.** 30 problems x 5 epochs = ~150 encounters. The curator compresses to ~50 entries. SDFT is a solution to a problem you do not have.
2. **The real bottleneck is coverage of strategy space** (per your own GSM8K finding). Neither SDFT nor the 3-stage loop addresses this.
3. **Zero of three loop transitions are validated.** TTRL->ACE (can playbooks steer GRPO-trained models?), ACE->SDFT (can distillation consolidate *strategic* knowledge, not just factual?), SDFT->TTRL (does consolidation improve next-round rollouts?). All untested.
4. **SDFT at 7B for math is contested.** The paper shows +4pts on Science QA (factual). OPSD at 8B shows +0.8% on AIME (strategic). Procedural knowledge distillation is a different distribution than factual.
5. **The MVE is cheap and decisive.** Hand-curated playbook + SDFT on base model, 2-3 hours. Tests: does the distilled model without playbook outperform the original with playbook? If no, the entire 3-stage framing collapses.
6. **Context drift is observable data, not a problem to prevent.** If ACE+TTRL shows playbook staleness as weights update, that finding *motivates* SDFT integration with evidence rather than narrative.
7. **Sequencing dominates for resource-constrained researchers.** Lower variance, decomposable failures, each result informs the next decision. The combined experiment's higher ceiling does not compensate for its higher probability of uninterpretable failure.

## Contrarian Truth
**The 3-stage loop is a premature abstraction.** It sounds inevitable -- discover, cache, consolidate -- but satisfying narratives are where researchers waste compute. SDFT is designed for cross-task continual learning (preventing catastrophic forgetting across domains), not within-task consolidation on 30 problems. It is a solution imported from a different problem. The popular view says "these complementary techniques logically compose." The truth: they are three separate research bets, and composing them before validating any single transition creates an experiment whose failure tells you nothing.

## What Changes Tomorrow
1. **Run the SDFT MVE** (2-3 hours): hand-curated AIME playbook (5-10 strategies) + SDFT distillation on Qwen-2.5-Math-7B. If the distilled model without playbook underperforms the original with playbook, drop the 3-stage framing entirely.
2. **Run ACE+TTRL alone** as the main experiment (8-12 hours). Observe whether context drift occurs. A clean result here determines the entire SDFT integration decision.
3. **If both succeed**: integrate in a follow-up with validated primitives. If either fails: you know exactly what and why.

## The Next Question
If ACE+TTRL shows context drift (playbook becomes stale as weights update), is SDFT the right consolidation mechanism -- or should you simply re-curate the playbook against the updated model? The cheaper intervention (re-curation) may dominate the expensive one (distillation).

## Sources (by Lindy-ness)
### Timeless (100+ years)
- Polya, *How to Solve It* (1945) -- the original playbook for mathematical problem solving
- Russo et al., Thompson Sampling Tutorial (2018; algorithm from 1933) -- optimal exploration-exploitation

### Enduring (10-100 years)
- Xie et al., "ICL as Implicit Bayesian Inference" (2021) -- theoretical basis for why ICL works
- Snell et al., "Scaling LLM Test-Time Compute" (2024) -- small model + compute > big model
- McClelland et al., Complementary Learning Systems (1995) -- hippocampal replay as discover-cache-consolidate

### Recent (< 10 years)
- [SDFT (arXiv 2601.19897)](https://arxiv.org/abs/2601.19897) -- 3B underperforms SFT, 7B +4pts Science QA, 2.5x FLOP overhead
- [OPSD (arXiv 2601.18734)](https://arxiv.org/abs/2601.18734) -- 8B: 77.5% AIME 2024, 4-8x token efficiency vs GRPO
- [TTRL (arXiv 2504.16084)](https://arxiv.org/abs/2504.16084) -- 211% improvement on Qwen-2.5-Math-7B for AIME
- [ACE (arXiv 2510.04618)](https://arxiv.org/abs/2510.04618) -- three-agent playbook evolution framework
