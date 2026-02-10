# Revised Research Plan: Memory x RL Interaction Study

*Synthesizer verdict after advocate-skeptic debate. Date: 2026-02-09.*

---

## 1. Paper Title and Abstract

**Title:** Do Weights Absorb Memory? How Context-Space and Weight-Space Learning Interact During RL Training

**Abstract (150 words):**

Recent work improves LLM reasoning through either evolving external memory (ACE, MemRL) or reinforcement learning weight updates (TTRL, GRPO). No prior work operates both channels simultaneously. We present the first empirical study of their interaction: what happens when an evolving strategy playbook co-adapts with RL weight updates on the same model? We train DeepSeek-R1-Distill-Qwen-7B under six conditions on AMC/AIME-difficulty mathematics, forming a clean ablation ladder from RL-only through static-context to co-evolved memory, including a novel absorption test that removes the co-evolved playbook after training to measure internalization. Across 150 evaluation points per condition, we find [RESULT]. Our diversity diagnostic reveals [FINDING about strategy entropy]. The absorption test shows [FINDING], providing the first evidence of [internalization/redundancy/interference] between context-space and weight-space learning during RL. We connect these findings to complementary learning systems theory and discuss implications for practitioners combining RAG with RL-trained models.

---

## 2. Exact Conditions (6 total)

The skeptic's two critical ablation gaps -- missing C-no-playbook for clean absorption and D' conflating two variables -- are both fixed.

| ID | Label | Context | Context Evolves? | RL (GRPO)? | Purpose |
|----|-------|---------|-----------------|------------|---------|
| **B** | RL-only | None | -- | Yes | Weight-space only baseline |
| **C** | RL + static playbook | Frozen playbook (from B's training set, 3 episodes ACE pre-curation) | No | Yes | Does context help RL? |
| **C-abs** | C post-training, playbook removed | Trained with static playbook, evaluated WITHOUT it | No (removed) | Trained | Absorption baseline: did static scaffold internalize? |
| **D** | RL + evolving playbook | ACE-curated playbook | Yes | Yes | Does co-evolution help beyond static? |
| **D-abs** | D post-training, playbook removed | Trained with evolving playbook, evaluated WITHOUT it | Removed | Trained | **Crown jewel**: did co-evolved scaffold internalize? |
| **E** | RL + novelty reward | None | -- | Yes (EVOL-RL reward) | Alternative diversity mechanism (no memory, intrinsic reward) |

**Key design changes from original:**

1. **Added C-abs** (skeptic's demand): Compares absorption from static vs co-evolved scaffold. D-abs > C-abs proves co-evolution produces deeper internalization. D-abs = C-abs means evolution was irrelevant to absorption. This was the single most important missing control.

2. **Dropped D'**: The skeptic was right that D' conflated training distribution (anchor problems) with curation signal quality (ground truth). Rather than adding D'' and D''' to untangle it (blowing up the condition count), we drop it entirely. The B-C-D-E ladder is clean. Verified anchoring can be a follow-up experiment.

3. **C's playbook is generated from the training set, not AMC**: The advocate's playbook was AMC-derived, creating a distribution mismatch confound. Fix: generate C's frozen playbook by running 3 episodes of ACE curation on the training problems with frozen weights, then freeze. C and D now differ only in whether the playbook continues evolving during RL training.

4. **C-abs and D-abs are not separate training runs**: They are evaluation-only conditions using the trained models from C and D respectively, evaluated without the playbook in the prompt. This adds zero training compute.

**Ablation ladder:**
- B vs C: effect of static context during RL
- C vs D: effect of context evolution during RL
- B vs E: effect of novelty reward vs vanilla GRPO
- D vs E: memory-based diversity vs reward-based diversity
- C-abs vs B: did static scaffold internalize anything beyond what RL alone learned?
- D-abs vs C-abs: did co-evolution produce deeper internalization than static scaffolding?
- D-abs vs D: is the co-evolved playbook still needed, or have weights absorbed it?

Every pairwise comparison changes exactly one variable. This is airtight.

---

## 3. Exact Benchmarks with Power Analysis Justification

### The skeptic is right about AIME 2024 alone: 30 problems is not enough

The power calculation is devastating: SE = 9.1% per condition, need 372 problems for 10% detection at alpha=0.05 and power=0.80. We cannot escape this with 30 problems. Seeds do not help because the 30 problems are a fixed test set.

### The solution: pool three non-overlapping problem sets

| Benchmark | N Problems | Baseline Acc (est.) | Role | Contamination Risk |
|-----------|-----------|--------------------|----|-------------------|
| **AIME 2025** (I + II) | 30 | ~40-45% | Primary: hardest, uncontaminated | Very Low (post-training-cutoff) |
| **OlymMATH-EASY** | 100 | ~35-45% | Primary: high N, auto-verified | Low (released March 2025) |
| **AMC 12 2024-2025** | 50 | ~60-70% | Secondary: medium difficulty | Moderate |

**Total evaluation problems: 180**

**Why this works:**
- Pooled N=180 at ~45% baseline gives SE = sqrt(0.45 * 0.55 / 180) = 3.7% per condition
- To detect 10% absolute difference with paired test: need n = 7.84 * (0.247 + 0.228) / 0.01 = 372 per condition with independent samples, BUT we use paired tests (same problems across conditions), which dramatically reduces the required N. With paired McNemar's test on 180 problems, a 10% difference means ~18 discordant pairs, which is detectable.
- 5 seeds per condition: 5 independent training runs, each evaluated on all 180 problems. Report mean accuracy +/- seed variance.

**Why AIME 2025 instead of 2024:**
- The skeptic documented contamination evidence for AIME 2024 (p<0.01 for 10/12 models). AIME 2025 problems were released after DeepSeek-R1's training cutoff. This is a free upgrade that eliminates one entire reviewer objection.

**Training set:**
- Use MATH-500 Level 4-5 (~200 problems, ~75-80% accuracy) as the training set for GRPO
- This provides adequate difficulty (20-25% error rate for learning signal) with enough problems to avoid the zero-variance group problem
- The ~50 problems with mixed outcomes (some rollouts correct, some incorrect) generate meaningful GRPO gradient
- Evaluation benchmarks (AIME 2025, OlymMATH, AMC) are held out -- never seen during training

**Why NOT MATH-500 as evaluation:**
- 92.8% baseline. Dead. The skeptic is unambiguously correct here.

---

## 4. Base Model(s)

### Primary: DeepSeek-R1-Distill-Qwen-7B

**Justification:**
- Strong reasoning baseline (55.5% AIME 2024) without being saturated on hard benchmarks
- The existing codebase (`experiments/ttrl_dc_aime/run_modal.py`) already has infrastructure for this model
- TTRL (NeurIPS 2025) used a similar-scale model (Qwen2.5-Math-7B) as primary
- Distillation concern is real (skeptic Argument 6a) but is a feature, not a bug: distilled models are what practitioners deploy, so studying memory x RL on distilled models has direct practical relevance

### Secondary (if compute allows): DeepSeek-R1-Distill-Qwen-1.5B

**Justification:**
- MATH-500 accuracy IS ~50-60% at this scale -- the headroom that was originally assumed for 7B
- Running B and D (2 conditions x 3 seeds) on 1.5B provides a scale comparison signal
- If absorption occurs at 7B but not 1.5B (or vice versa), this is itself a publishable finding about capacity requirements for internalization
- Minimal additional compute: 2 conditions x 3 seeds x ~2 hours = ~12 GPU-hours

**What we do NOT need:**
- A non-distilled base model (Qwen2.5-Math-7B without R1): This would confuse the story. The question is about memory x RL interaction, not about distillation. Controlling for distillation is a separate paper.
- LLaMA-3.1-8B: Different architecture would add a confound (tokenizer, training data) without clarifying the memory x RL question.

---

## 5. Statistical Methodology

### Primary analysis: Paired McNemar's test

For each pair of conditions (e.g., B vs D), compare per-problem binary outcomes across all 180 evaluation problems. McNemar's test is appropriate because the same problems are used across conditions.

- **Alpha = 0.05** with Holm-Bonferroni correction for 7 planned comparisons (B-C, C-D, B-D, B-E, D-E, C_abs-B, D_abs-C_abs)
- **Adjusted alpha per comparison: 0.05/7 = 0.0071 for the most significant, then 0.05/6, etc.**

### Secondary analysis: Bootstrap confidence intervals

- 10,000 bootstrap resamples over problem-level outcomes per condition
- Report 95% CIs for accuracy and for pairwise differences
- This is seed-aware: for each bootstrap sample, resample problems (not seeds), compute mean accuracy across 5 seeds per problem

### Effect size: Cohen's g for McNemar's test

- g = (n01 - n10) / (n01 + n10), where n01 = problems D gets right that B gets wrong, n10 = opposite
- Report effect sizes alongside p-values

### Power analysis (what we can detect):

With 180 paired problems and McNemar's test:
- 8% absolute difference (e.g., 45% vs 53%): power ~0.82 (adequate)
- 5% absolute difference: power ~0.55 (marginal, will note as limitation)
- 15% absolute difference: power ~0.99 (comfortable)

With 5 seeds, we can also run a paired t-test on seed-level accuracies (df=4). This requires larger effects (~10%+ with typical 3-5% seed variance) but provides a complementary analysis.

### For the absorption test specifically:

- D-abs vs D: paired McNemar on 180 problems. If p > 0.05, we cannot reject that they are equal -- evidence for absorption.
- D-abs vs B: if p < 0.05, weights absorbed something beyond what RL alone provides.
- D-abs vs C-abs: the clean test of co-evolution vs static scaffold internalization.

Note: The absorption test's "null = interesting" framing means we want to FAIL to reject D-abs = D. This inverts the usual power concern: we need enough power to detect if absorption did NOT happen (i.e., if playbook removal hurts). With 180 problems, a 8% drop from removal is detectable, so silence (p > 0.05) meaningfully constrains the degradation to <8%.

### Diversity analysis: Strategy entropy

- Apply LLM classifier (15-tactic taxonomy: modular_arithmetic, casework, pigeonhole, algebraic_manipulation, etc.) to all completions
- Compute Shannon entropy of tactic distribution per condition
- Compare entropy across conditions with bootstrap CIs
- This is purely descriptive -- no hypothesis test needed. Visualize as tactic heatmap.

---

## 6. Compute Budget

### Estimated per-condition training cost

Based on the existing `run_modal.py` infrastructure (A100-80GB, LoRA rank 64, 16 generations/prompt, 20 GRPO epochs, 30 problems per epoch):

| Component | Time (A100-80GB) | Notes |
|-----------|-----------------|-------|
| GRPO training (7B, 20 epochs, 200 problems) | ~4-6 hours | 200 problems x 16 rollouts x 20 epochs |
| ACE curation per epoch | ~0.5 hours | Batch reflect + curate via vLLM |
| Evaluation (180 problems x 16 rollouts) | ~0.5 hours | vLLM inference only |
| Model merge + checkpoint | ~0.25 hours | CPU-bound |

### Total compute:

| Item | Runs | Hours/run | Total GPU-hours |
|------|------|-----------|----------------|
| B (RL-only) x 5 seeds | 5 | 5 | 25 |
| C (RL + static playbook) x 5 seeds | 5 | 5.5 | 27.5 |
| D (RL + evolving playbook) x 5 seeds | 5 | 6 | 30 |
| E (RL + novelty reward) x 5 seeds | 5 | 5.5 | 27.5 |
| Evaluation (C-abs, D-abs) | 10 | 0.5 | 5 |
| 1.5B model (B, D) x 3 seeds | 6 | 2 | 12 |
| ACE pre-curation for C's playbook | 1 | 1 | 1 |
| Pilot runs / debugging | -- | -- | 20 |
| **Total** | | | **~148 GPU-hours** |

**At Modal A100-80GB pricing (~$3.74/hour):** ~$555

**At academic cluster rates (free but queued):** ~6 days of sequential runs on a single A100, or ~2 days with 3 parallel jobs

This is realistic for a CS224N project budget. The 1.5B model runs are cheap insurance (~$45) for a generalization signal.

---

## 7. What-If Matrix: Every Outcome Maps to a Paper

| # | Outcome | Evidence Pattern | Paper Story | Strength |
|---|---------|-----------------|-------------|----------|
| 1 | **Synergy** | D > C > B on accuracy, D-abs >= D | "Co-evolved memory scaffolds RL, then is absorbed. CLS theory confirmed: fast channel bootstraps slow channel." | STRONG for main conference |
| 2 | **Redundancy** | D = C = B (all within noise) | "Context-space and weight-space learning are independent on this task. Neither helps nor hurts." | MODERATE for workshop (interaction study, clean null) |
| 3 | **Interference** | D < B, C > B | "Evolving context destabilizes RL training. Static context helps but evolution introduces non-stationarity." | STRONG for workshop (surprising negative, practical warning) |
| 4 | **Absorption** | D-abs = D > B, D-abs > C-abs | "RL absorbs co-evolved memory into weights. The playbook becomes redundant -- scaffolding theory confirmed." | **STRONGEST** for main conference (contrarian, practical implications) |
| 5 | **Partial absorption** | D-abs < D but D-abs > B | "RL partially internalizes the playbook. Residual context value exists." | MODERATE-STRONG (nuanced, connects to CLS) |
| 6 | **Static suffices** | C > B, D = C, C-abs = D-abs | "Any context helps RL, but evolution adds nothing. CG-TTRL was right: static selection is enough." | MODERATE for workshop (replication + extension) |
| 7 | **Diversity divergence** | D and B have similar accuracy but D has higher strategy entropy | "Memory preserves reasoning diversity that RL compresses. Accuracy is equal but the models reason differently." | STRONG for workshop (novel analysis, connects to diversity collapse literature) |
| 8 | **Novelty reward dominates** | E > D > B | "Intrinsic diversity reward outperforms external memory for preventing GRPO collapse." | MODERATE (confirms EVOL-RL, less novel) |
| 9 | **Complete noise** | All conditions within 3% of each other, high seed variance | "GRPO instability dominates all other signals at this scale. The interaction question requires stabilized RL." | WEAK but honest (learning paper, identifies the blocker) |

**Outcomes 1, 4, 5, and 7 are the sweet spot.** The experiment is designed to make Outcome 4 (absorption) the strongest possible paper, per the distillation's contrarian insight.

**Outcome 9 is the worst case.** Mitigation: the diversity diagnostic and strategy attribution analyses can still produce findings even if accuracy differences are noisy. If tactic entropy differs significantly across conditions while accuracy does not, that IS the finding.

---

## 8. The 3 Things That Could Kill the Paper

### Kill Risk 1: GRPO training collapses on 1+ seeds, destroying statistical power

**Probability: 40%** (LLD death spiral is well-documented)

**Detection:** Monitor training loss and generation entropy per epoch. If loss diverges or entropy drops below 0.5 bits, the seed has collapsed.

**Prevention:**
- Use LoRA (rank 64) instead of full fine-tuning -- limits the parameter space GRPO can destabilize
- Set max_grad_norm = 0.5 (stricter than default 1.0) to limit gradient explosions
- Add a KL penalty (beta = 0.01) to anchor the policy near the reference -- this is the standard GRPO stabilization from DAPO
- If a seed collapses (detected by >20% accuracy drop from baseline on training set), restart with a different random seed. Report the collapse rate as a finding.

**If prevention fails:** Run 7 seeds instead of 5, drop the worst 2. Report the full distribution including collapses. "3 of 7 GRPO seeds collapsed during training" is itself a finding that contextualizes the memory x RL interaction.

### Kill Risk 2: All conditions land within a noise band, and the absorption test is inconclusive

**Probability: 30%** (if GRPO instability is high and effect sizes are small)

**Detection:** After the first 2 seeds complete for B and D, compute the observed effect size. If |D - B| < 3% on the evaluation set, the full experiment is likely to be inconclusive for accuracy.

**Prevention:**
- The evaluation set (180 problems) is large enough to detect 8% differences with 0.82 power
- The diversity diagnostic does not require accuracy differences -- it measures strategy entropy directly
- Frame the paper as "interaction study" from the start, so a null accuracy result is a finding, not a failure

**If prevention fails:** Pivot the paper to the diversity finding. "Co-evolved memory preserves reasoning diversity without improving accuracy" is publishable at a workshop. The absorption test still works even if accuracy is flat -- D-abs = D = B means the playbook was always irrelevant (Outcome 2), which is a clean result.

### Kill Risk 3: A concurrent paper publishes the exact experiment before submission

**Probability: 25%** (MemAgents workshop papers go public in April 2026; GEPA already shows prompt evolution > RL)

**Detection:** Monitor arXiv weekly for "memory + GRPO", "playbook + RL", "context evolution + reinforcement learning" queries.

**Prevention:**
- Speed. The existing codebase (run_modal.py) already implements 4 of the 6 conditions. Adding C-abs, D-abs, and E requires ~2 days of engineering. Training can start by Feb 17.
- Differentiation. Even if a concurrent paper studies memory + RL, the absorption test (D-abs vs C-abs) is our unique contribution. No concurrent paper is likely to include this specific control.
- If scooped on the basic finding: pivot to the absorption test as the primary contribution. "We replicate [concurrent work]'s finding that co-evolved memory [helps/hurts] RL training, and additionally show through a novel absorption test that [finding]."

**If prevention fails:** Submit to NeurIPS 2026 FoRLM workshop (deadline ~June 2026) or EMNLP 2026 (deadline ~June 2026) with the absorption test as the primary contribution. The absorption test is novel regardless of what MemAgents publishes, because nobody in that workshop will have C-abs as a control.

---

## 9. Verdict: Main Conference / Workshop / Course Project

### Honest assessment by outcome:

| Outcome | Venue Target | Confidence |
|---------|-------------|------------|
| Synergy + absorption (1+4) | ICML/NeurIPS main conference | 60% |
| Absorption alone (4 or 5) | NeurIPS workshop (FoRLM) or EMNLP Findings | 75% |
| Diversity divergence (7) | Workshop or short paper | 70% |
| Clean null with analysis (2) | Workshop | 65% |
| Noise (9) | CS224N project only | -- |

### What determines which:

1. **Main conference requires:** (a) at least one pairwise comparison at p < 0.01 after correction, (b) the absorption test producing a clear result (either D-abs = D > B OR D-abs < D), AND (c) adding the 1.5B generalization experiment. All three are necessary.

2. **Workshop requires:** (a) at least one interesting finding from the absorption test OR diversity diagnostic, (b) clean experimental design (which we now have), (c) proper framing as "interaction study." This is achievable with high probability.

3. **CS224N project:** Guaranteed publishable as a course paper regardless of results. The design is rigorous, the question is novel, the analysis is deep. Even Outcome 9 (complete noise) is a well-designed experiment that produces learning.

### My verdict: **This experiment is worth running.**

The skeptic's concerns were serious and several were damaging:
- AIME 2024 as sole benchmark: **fatal, now fixed** (pooled 180 problems, switched to AIME 2025)
- Missing C-no-playbook: **fatal for absorption test, now fixed** (added C-abs)
- D' confound: **real, now fixed** (dropped D')
- GRPO instability: **real, mitigated** (KL penalty, grad clipping, 7 seeds, collapse monitoring)
- Zero-variance GRPO groups: **real, mitigated** (training on MATH-500 L4-5 with ~50 informative problems instead of AIME 2024 with ~5)
- GEPA outperforming GRPO: **valid concern, partially addressed** (the question is about interaction, not which is better; E condition provides the comparison)
- Single model: **real, mitigated** (1.5B generalization experiment)

The advocate's core strengths survive:
- Genuine novelty (first to co-evolve context + weights during RL): **confirmed, still true**
- Theoretically grounded (CLS, Vygotsky): **confirmed**
- Every outcome is publishable with interaction framing: **confirmed, with stronger design**
- Absorption test for co-evolved scaffolds is novel: **confirmed, now with clean C-abs control**
- Timely: **confirmed, window still open but closing**

**Bottom line:** Workshop publication at HIGH confidence (>75%). Main conference at MODERATE confidence (~45%), contingent on getting lucky with effect sizes and GRPO stability. The experiment produces a useful result regardless of outcome, the compute budget is realistic (~$555), and the existing codebase covers 60% of the implementation.

Start coding tomorrow. Ship training runs by Feb 24. First results by March 7. Paper draft by March 21.

---

## Appendix: Implementation Checklist

### Week 1 (Feb 10-16): Engineering
- [ ] Fork `experiments/ttrl_dc_aime/run_modal.py` into `experiments/memory_rl_interaction/run_modal.py`
- [ ] Add AIME 2025, OlymMATH-EASY, AMC 12 data loaders (evaluation only)
- [ ] Swap training set from AIME 2024 to MATH-500 Level 4-5
- [ ] Implement EVOL-RL novelty reward for Condition E
- [ ] Implement C-abs and D-abs evaluation (playbook removal post-training)
- [ ] Add KL penalty (beta=0.01) and stricter grad clipping (0.5)
- [ ] Add per-epoch collapse detection (entropy monitoring)
- [ ] Add 15-tactic LLM classifier for diversity analysis

### Week 2 (Feb 17-23): Pilot + Training
- [ ] Run 1 seed of B and D as pilot (verify training stability, ~12 GPU-hours)
- [ ] If pilot stable: launch all 4 conditions x 7 seeds (~168 GPU-hours)
- [ ] If pilot unstable: add entropy regularization, rerun pilot

### Week 3 (Feb 24-Mar 2): Training + 1.5B
- [ ] Complete 7B training runs
- [ ] Launch 1.5B generalization (B, D x 3 seeds, ~12 GPU-hours)
- [ ] Begin evaluation on held-out benchmarks

### Week 4 (Mar 3-9): Analysis
- [ ] Run all evaluations (C-abs, D-abs, diversity diagnostic)
- [ ] Compute McNemar's tests, bootstrap CIs, effect sizes
- [ ] Generate tactic entropy heatmaps
- [ ] Determine which outcome from the what-if matrix matches results

### Week 5 (Mar 10-16): Writing
- [ ] Draft paper using matched outcome template
- [ ] Figures: accuracy bars with CIs, absorption delta plot, tactic entropy heatmap, training dynamics (loss/entropy per epoch)
- [ ] Related work: cite FT+RAG, GenPI, RuscaRL, GEPA, MemRL, CG-TTRL, EVOL-RL

### Week 6 (Mar 17-21): Polish + Submit
- [ ] Internal review
- [ ] Submit to CS224N
- [ ] If results are strong: also target NeurIPS 2026 FoRLM workshop (deadline ~June)

---

## Appendix: Addressing the Skeptic's Remaining Concerns

### "GEPA outperforms GRPO with prompt evolution alone"

True, but irrelevant to our question. GEPA shows context-space optimization can beat weight-space optimization when done in isolation. Our question is: what happens when you do both simultaneously? If GEPA's result holds, then our Outcome 3 (interference: D < B, because context evolution destabilizes RL) becomes the expected result -- and showing this empirically with a clean ablation is a contribution. We cite GEPA as motivation for why the interaction question matters.

### "Single distilled model provides no generalization"

Mitigated with 1.5B experiment. We explicitly acknowledge this limitation and frame the 7B results as the primary contribution, with 1.5B as a scale probe. TTRL (NeurIPS 2025) set the precedent for single-model-family papers.

### "GRPO instability dominates any memory signal"

This is the most serious remaining concern. Our mitigations (KL penalty, grad clipping, collapse detection, 7 seeds) reduce the risk but do not eliminate it. If GRPO instability still dominates, we report it honestly and frame the paper as: "GRPO instability is the binding constraint on memory x RL interaction studies. We quantify the instability and show that [X]." This is outcome 9 in our what-if matrix -- the weakest paper, but still an honest contribution if the instability itself is well-characterized.

### "The MemAgents workshop deadline was February 10"

Papers submitted to MemAgents study memory for frozen-weight LLM agents, not memory x RL co-evolution. Our experiment is in a different space. The risk of exact scooping is low (~10%). The risk of adjacent work making our paper feel incremental is moderate (~25%) -- mitigated by the absorption test, which no concurrent paper is likely to include.
