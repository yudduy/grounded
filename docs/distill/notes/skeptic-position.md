# Skeptic Position: The Case AGAINST Publishability

*A rigorous adversarial review of the 5-condition Memory x RL experiment.*

---

## Executive Summary

This experiment will not produce a publishable result at a main conference. The combination of (1) a statistically hopeless primary benchmark, (2) a contaminated and underpowered secondary benchmark, (3) known GRPO training instability that will dominate any memory-related signal, (4) a confounded key condition, and (5) concurrent work that is converging on the same question with better resources means the most likely outcome is 3 months of GPU-hours producing noisy, ambiguous results that cannot survive peer review. The individual problems are each survivable; together they are fatal.

---

## Argument 1: AIME 2024 Is a Statistical Graveyard

The distillation already established that MATH-500 is dead (92.8% baseline). The proposed fix is to use AIME 2024 as the primary benchmark. But AIME 2024 is arguably *worse*.

### 1a. 30 problems is not a benchmark; it is a coin flip

With 30 binary-outcome problems and a baseline of 55.5% (DeepSeek-R1-Distill-Qwen-7B pass@1), the standard error of a single accuracy estimate is:

```
SE = sqrt(0.555 * 0.445 / 30) = 0.0908 = 9.1%
```

A 95% confidence interval on a single condition's accuracy spans roughly +/-18 percentage points. To detect a 10% absolute improvement (55% -> 65%) between two conditions with a two-sample proportion test at alpha=0.05 and power=0.80:

```
n = (z_alpha + z_beta)^2 * [p1(1-p1) + p2(1-p2)] / (p1-p2)^2
n = (1.96 + 0.84)^2 * [0.555*0.445 + 0.65*0.35] / 0.10^2
n = 7.84 * [0.247 + 0.228] / 0.01
n = 7.84 * 47.5 = 372 problems per condition
```

You need ~372 problems per condition to detect a 10% difference with adequate power. You have 30. **You are underpowered by a factor of 12x.**

Even with 5 seeds (effectively 150 problem-seed pairs, but these are NOT independent -- same 30 problems repeated), the effective sample size for detecting condition differences is still driven by the 30 unique problems. Seeds reduce variance from training randomness, not from problem sampling. The 30 problems are a fixed test set, not resampled.

### 1b. AIME 2024 is contaminated

Evidence from paired permutation tests validates with p<0.01 for 10/12 models that AIME 2024 problems have contaminated pretraining or fine-tuning pipelines, inflating scores by 10-20 points over uncontaminated contests like AIME 2025. (Source: [MathArena analysis](https://arxiv.org/html/2503.21380v2); [AIME benchmark discussions](https://llm-stats.com/benchmarks/aime?tab=discussions))

DeepSeek-R1-Distill-Qwen-7B's 55.5% pass@1 on AIME 2024 may already be inflated. If the true uncontaminated accuracy is closer to 40-45%, then GRPO training on these problems may show artificial gains simply from memorization reinforcement, not genuine reasoning improvement. The D vs C comparison becomes meaningless if the model has already memorized some AIME 2024 answers during pretraining.

### 1c. TTRL's own instability confirms the problem

The TTRL paper (NeurIPS 2025) itself reports AIME 2024 instability: three independent runs on Qwen-2.5-Math-7B produced pass@1 scores of 43.3%, 43.3%, and 46.7% -- a 3.4 percentage point range. (Source: [TTRL GitHub](https://github.com/PRIME-RL/TTRL))

That is on a model with a *lower* baseline than DeepSeek-R1-Distill-Qwen-7B. With a higher baseline (55.5%), the variance-to-signal ratio is even worse because the remaining hard problems have inherently higher variance (they are the ones the model sometimes gets right, sometimes wrong). With 5 conditions and 5 seeds each, you will see overlapping confidence intervals everywhere.

### 1d. The power calculation a reviewer will do

A skeptical reviewer will compute: with 30 AIME problems and 5 seeds, can you distinguish a 5% accuracy difference between conditions? Using a paired bootstrap over problem-level accuracy (the correct approach, since all conditions see the same 30 problems):

- Each seed produces 30 binary outcomes per condition.
- 5 seeds produce 5 paired accuracy estimates.
- A paired t-test on 5 observations (df=4) requires t > 2.776 for p < 0.05.
- With typical GRPO training variance (sigma ~ 3-5% across seeds, per TTRL), a 5% difference yields t ~ 5/(4/sqrt(5)) = 2.8.

This is *barely* significant, assuming optimistic variance. If seed variance is 6-7% (plausible with GRPO instability), the test fails. **You are running a 5-condition experiment where the primary benchmark cannot reliably distinguish between conditions.**

---

## Argument 2: GRPO Instability Will Dominate the Signal

### 2a. The LLD Death Spiral is not hypothetical

Recent work (arXiv:2512.04220) identifies Lazy Likelihood Displacement (LLD) as a systematic failure mode in GRPO: a self-reinforcing spiral where declining likelihood leads to low-confidence responses, inflating gradients, and ultimately causing collapse. The paper reports this follows a consistent three-phase trajectory: early stagnation, steady decay, and accelerated collapse.

On Qwen2.5-7B (the same architecture family as DeepSeek-R1-Distill-Qwen-7B), the authors report up to 32% performance degradation from LLD collapse. This is not a rare edge case -- it is the default behavior of vanilla GRPO on reasoning tasks.

### 2b. Zero-variance groups kill the reward signal

GRPO computes advantages within groups of sampled responses. When all responses for a given prompt receive the same reward (all correct or all incorrect), the within-group variance is zero and no learning signal exists. At 55.5% baseline on AIME 2024, roughly 17 problems (~55%) are always correct and 8 problems (~27%) are always wrong across rollouts. Only ~5 problems (~17%) have mixed outcomes that generate learning signal.

**The model can only learn from ~5 of the 30 AIME problems.** The rest contribute zero gradient. This is not 150 problem-seed pairs; this is 25 informative problem-seed pairs (5 problems x 5 seeds). GRPO training on this is gradient starvation.

### 2c. Adding memory makes instability worse, not better

Condition D (evolving playbook) adds a changing context to an already unstable training process. Every time the curator modifies the playbook (ADD/UPDATE/MERGE/DELETE), the effective prompt distribution shifts. The policy was optimized for the *old* playbook; now it faces a new one. This is a non-stationary reward landscape layered on top of an already unstable optimizer.

The playbook poisoning evidence from this project's own experiments (DC accuracy 27% -> 17% on AIME) suggests the curator accumulates noise over time. In Condition D, this noise feeds into GRPO training, which amplifies it through the LLD mechanism. The expected outcome is higher variance in D than B, not higher accuracy.

### 2d. Scaf-GRPO and DAPO already address the instability

Recent work on Scaf-GRPO (arXiv:2510.19807), DAPO (arXiv:2503.14476), and M-GRPO (arXiv:2512.13070) all propose specific fixes for GRPO instability: entropy regularization, likelihood-preserving regularization, momentum anchoring. The proposed experiment uses vanilla GRPO (or TTRL-style GRPO) without any of these stabilization techniques.

A reviewer will ask: "You are studying the effect of memory on RL training, but you are using an RL algorithm known to be unstable. How do you distinguish memory effects from optimizer instability?" This is a confound that cannot be resolved without ablating against a stabilized GRPO baseline.

---

## Argument 3: The Absorption Test Cannot Produce Clean Results

### 3a. The test is underdetermined

The absorption test removes the playbook from Condition D after training and measures accuracy. There are three possible outcomes:

1. **D-no-playbook = D-with-playbook > B**: "Weights absorbed the playbook." Good story.
2. **D-no-playbook < D-with-playbook**: "Weights did not absorb." Less interesting but publishable.
3. **D-no-playbook = D-with-playbook = B**: "The playbook never helped." Null result.

But outcome 1 has an alternative explanation: **GRPO training improved the model regardless of the playbook, and the playbook was irrelevant scaffolding.** If D-no-playbook = B+epsilon (where epsilon is from GRPO training variance), and D-with-playbook = B+epsilon (because the playbook was noise), then 1 and 3 are statistically indistinguishable on 30 AIME problems.

To distinguish "absorption" from "irrelevant scaffolding," you need D-no-playbook > C-no-playbook (static playbook removal should show less absorption than evolving playbook removal). But C-no-playbook is not in the experimental design. The design tests D-no-playbook vs B, which conflates training improvement with absorption.

### 3b. FT+RAG and GenPI already own this space

The scaffold-removal paradigm is not novel. FT+RAG (ICLR 2026, arXiv:2510.01375) trains with retrieval hints and then removes them, forcing internalization. GenPI (arXiv:2411.15927) internalizes prompts into weights with 100% performance retention on some tasks. Learning by Distilling Context (arXiv:2209.15189) established the concept in 2022.

The claim that "co-evolved scaffolding" is different is technically true but practically thin. A reviewer familiar with FT+RAG will write: "The authors test whether a scaffold can be removed after training. This has been shown repeatedly (FT+RAG, GenPI, Distilling Context). The novelty is that the scaffold co-evolved, but the authors provide no evidence that co-evolution produces qualitatively different absorption dynamics than static scaffolding." Without a direct comparison of absorption rates between C and D (which requires C-no-playbook), the claim is unsupported.

### 3c. RuscaRL makes the test look incremental

RuscaRL (arXiv:2508.16949) does rubric-scaffolded RL with *gradual decay during training*. This is more sophisticated than the proposed binary remove-after-training test. RuscaRL measures absorption *dynamics* (how fast the model learns to do without the scaffold). The proposed test only measures the *endpoint* (can the model do without the scaffold after all training is done). A reviewer will say: "RuscaRL already does this, more elegantly, with a decay schedule that provides richer signal about internalization dynamics."

---

## Argument 4: D' Is Fatally Confounded

### 4a. Two variables, one comparison

D' changes both:
- **Training distribution**: adds 10 anchor problems with known ground truth
- **Curation signal**: curator gets ground-truth correctness feedback

If D' > D, is it the curriculum effect (easy problems stabilize training) or the signal quality effect (ground truth prevents poisoning)? Without D'' (anchors + majority-vote curation) or D''' (no anchors + ground-truth curation), the comparison is uninterpretable.

### 4b. 10 anchor problems out of how many?

If the training set is ~500 problems (MATH-500 size), 10 anchors are 2% of training data. If it is 30 problems (AIME 2024), 10 anchors are 33%. The effect of anchoring depends critically on the ratio, but the experimental design does not specify the training set size. If training on AIME 2024 directly (30 problems), adding 10 problems with ground truth means 33% of training has perfect signal -- this is a massive intervention, not a small anchor.

### 4c. The reviewer's demand is predictable

Every reviewer who reads this design will write: "Weakness: D' conflates two variables. The authors should add D'' (anchors without ground truth) to isolate the curriculum effect from the signal quality effect." This is not a speculative objection; it is the standard experimental design critique and will appear in 3/3 reviews.

---

## Argument 5: The Novelty Window Is Closing (or Already Closed)

### 5a. The MemAgents workshop deadline was February 5-10, 2026

The ICLR 2026 MemAgents workshop (submission deadline: February 10, 2026) explicitly lists "memory x RL" as a target topic. Papers are already submitted. At least some of them will study memory co-evolution with RL training. By the time this experiment runs (1-2 months for compute, 1 month for writing), the workshop papers will be public.

### 5b. MEM-alpha already combines memory + RL

MEM-alpha (arXiv:2509.25911, submitted to MemAgents workshop) uses reinforcement learning to train agents to construct and manage complex memory systems. While the specific framing differs (MEM-alpha optimizes memory construction, not playbook co-evolution), a reviewer will see both as "using RL to improve memory for reasoning" and demand explicit differentiation.

### 5c. GEPA suggests prompt evolution may already dominate RL

GEPA (arXiv:2507.19457) demonstrates that reflective prompt evolution *outperforms* GRPO by 10% on average and up to 20%, while using 35x fewer rollouts. If context-space evolution is already better than weight-space RL, the interaction question becomes moot: you should just do prompt evolution and skip RL entirely.

This is the most devastating concurrent result. If GEPA's results hold, the framing "how do context-space and weight-space learning interact?" has an answer: "context-space dominates, and adding weight-space learning adds noise." The proposed experiment could end up confirming GEPA's conclusion rather than producing novel insights.

### 5d. Prompt Augmentation Scales GRPO (arXiv:2602.03190) suggests static context is enough

This February 2026 paper shows that simply augmenting prompts with diverse templates during GRPO training (no evolution, no curation) enables stable scaling and SOTA results on AIME24. If static prompt augmentation is sufficient, the elaborate ACE curation machinery in Conditions C and D may be unnecessary complexity. A reviewer will ask: "Have you compared against simple prompt augmentation as a baseline?"

---

## Argument 6: The Base Model Choice Creates Confounds

### 6a. Distilled models are not base models

DeepSeek-R1-Distill-Qwen-7B was created by distilling reasoning traces from DeepSeek-R1 (a much larger model). The reasoning capabilities are already "baked in" via distillation, not learned from scratch via RL. When you run GRPO on this model, you are applying RL on top of distillation on top of pretraining. The three learning channels (pretraining -> distillation -> RL + memory) make it impossible to attribute effects cleanly.

If the playbook contains strategies that the model already learned during distillation, Condition D's playbook becomes redundant (not because RL absorbed it, but because distillation already did). The absorption test would show D-no-playbook = D-with-playbook, but for the wrong reason.

### 6b. A single model family provides no generalization signal

The distillation already flagged this: a reviewer will demand at least one additional model. But the problem is deeper than reviewer preferences. If the result depends on DeepSeek-R1-Distill-Qwen-7B's specific distillation-trained reasoning patterns, it may not transfer to:
- Qwen2.5-Math-7B (trained with RL from scratch, different reasoning style)
- LLaMA-3.1-8B (different architecture, different training data)
- Phi-3-Mini-4K (different scale and training approach)

A result on one distilled model is a data point, not a finding.

---

## Argument 7: The Expected Outcomes Are All Problematic

### 7a. The "memory helps" outcome is incremental

If D > B by 5-10% on AIME 2024, this is consistent with CG-TTRL's finding that context helps RL (+7% relative improvement) and with Prompt Augmentation Scales GRPO's finding that prompt diversity helps. It does not demonstrate that *evolved* context is better than *static* context or *random* context augmentation. Without the C baseline being competitive and D significantly beating C, the result is "context helps" which is already known.

### 7b. The "memory hurts" outcome blames the wrong thing

If D < B, is it because:
- Memory genuinely interferes with RL training?
- ACE's curation mechanism is noisy and poisons the playbook?
- GRPO is unstable and adding changing context destabilizes it further?
- The playbook is irrelevant to AIME problems (distribution mismatch)?

All four explanations are plausible. Without isolating them, the result is uninterpretable. A reviewer will write: "The authors show that Condition D underperforms, but the confound between memory quality, memory relevance, and training stability makes it impossible to conclude that 'memory hurts RL training'."

### 7c. The "absorption" outcome is confounded

As argued in Argument 3, D-no-playbook = D-with-playbook > B cannot be cleanly attributed to absorption without a C-no-playbook comparison.

### 7d. The null result is the most likely outcome

Given:
- 30 problems with 9% SE per condition
- GRPO instability adding 3-7% seed variance
- Playbook evolution adding non-stationary noise
- Contamination inflating baseline scores

The most likely outcome is that all 5 conditions cluster within a 50-62% band on AIME 2024, with overlapping confidence intervals, and no pairwise comparison reaches p < 0.05. The qualitative analysis experiments (diversity diagnostic, strategy attribution) cannot rescue the paper because they depend on the *quantitative* differences being real. If D and B have similar accuracy, you cannot meaningfully analyze "what strategies D absorbed from the playbook" because there is no evidence it absorbed anything.

---

## Summary: The Fatal Combination

No single argument above is individually fatal. But together they form a compound threat:

| Threat | Severity | Survivable Alone? | In Combination? |
|--------|----------|-------------------|-----------------|
| AIME 2024 statistical power (30 problems) | Critical | Maybe with effect size >15% | No |
| AIME 2024 contamination | High | Maybe if using AIME 2025 | Compounds power problem |
| GRPO instability (LLD, zero-variance) | High | With stabilized GRPO | Compounds with memory non-stationarity |
| D' confound (two variables) | Medium-High | With D'' ablation | D'' adds compute, delays timeline |
| Absorption test confounds | Medium-High | With C-no-playbook | Adds 5 more seeds, compute |
| Novelty window closing (MemAgents, GEPA) | Medium | If fast enough | Not controllable |
| Single distilled model | Medium | With second model | More compute, delays |

The experiment needs to simultaneously: use a larger benchmark, use uncontaminated data, stabilize GRPO, add D'' and C-no-playbook conditions, test a second model, and finish before the MemAgents workshop papers go public. This is not a "fix three things" situation; it is a fundamental mismatch between the ambition of the question and the resources available to answer it.

---

## The Bottom Line

**For a CS224N course project:** The experiment is fine. Run it, write it up, learn from it. The question is interesting, the conditions are well-structured, and the analysis experiments are creative. Course projects are graded on design and insight, not statistical power.

**For a publishable paper:** The experiment will almost certainly produce noisy, ambiguous results on AIME 2024 that cannot distinguish between conditions. The absorption test is confounded. The D' comparison is confounded. GRPO instability will dominate any memory signal. And by the time results are ready, concurrent work (MemAgents workshop, GEPA, Prompt Augmentation) will have answered adjacent questions with better resources. The recommended path is to treat this as a pilot study, identify the most promising condition, and design a follow-up experiment with (a) 100+ problem benchmark, (b) stabilized GRPO, (c) two model families, and (d) clean ablations.

---

## Sources

- [How Many Random Seeds? Statistical Power Analysis in Deep RL](https://arxiv.org/abs/1806.08295) - Henderson et al. 2018
- [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) - Agarwal et al. 2021
- [On GRPO Collapse: The LLD Death Spiral](https://arxiv.org/abs/2512.04220) - December 2025
- [DAPO: Open-Source LLM RL at Scale](https://arxiv.org/abs/2503.14476) - ByteDance 2025
- [Scaf-GRPO: Scaffolded GRPO](https://arxiv.org/abs/2510.19807) - October 2025
- [M-GRPO: Momentum-Anchored Policy Optimization](https://arxiv.org/abs/2512.13070) - December 2025
- [TTRL: Test-Time Reinforcement Learning (NeurIPS 2025)](https://arxiv.org/abs/2504.16084)
- [GEPA: Reflective Prompt Evolution Outperforms RL](https://arxiv.org/abs/2507.19457) - July 2025
- [Prompt Augmentation Scales GRPO](https://arxiv.org/abs/2602.03190) - February 2026
- [MEM-alpha: Learning Memory Construction via RL](https://arxiv.org/abs/2509.25911) - September 2025
- [ICLR 2026 MemAgents Workshop](https://sites.google.com/view/memagent-iclr26/)
- [FT+RAG: Scaffold-then-remove (ICLR 2026)](https://arxiv.org/abs/2510.01375)
- [GenPI: Generative Prompt Internalization](https://arxiv.org/abs/2411.15927)
- [RuscaRL: Rubric-Scaffolded RL](https://arxiv.org/abs/2508.16949)
- [CG-TTRL: Context-Guided Test-Time RL](https://arxiv.org/abs/2511.06430)
- [AIME 2024 Contamination Evidence](https://arxiv.org/html/2503.21380v2)
- [Evaluating LLMs via Comparative Signals](https://arxiv.org/abs/2602.03061)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - Table 5: 92.8% MATH-500, 55.5% AIME 2024
- [AIME 2024 Leaderboard](https://llm-stats.com/benchmarks/aime-2024)
