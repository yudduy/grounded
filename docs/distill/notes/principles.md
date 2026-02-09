# First Principles: Should SDFT Be Integrated or Sequenced?

*Philosopher's analysis of the integrate-vs-sequence decision for SDFT in the ACE+TTRL experiment.*

---

## The Decision Structure

This is not primarily a technical question. It is an **experiment design** question under uncertainty, and the correct framework is **information theory applied to resource-constrained sequential decisions**.

The decision: given one A100, one unvalidated system (ACE+TTRL), and one promising consolidation method (SDFT), should you:
- (A) Run ACE+TTRL alone, validate it, then add SDFT
- (B) Run the combined 3-stage loop (TTRL->ACE->SDFT) from the start

---

## Assumption Challenges

### Assumption 1: SDFT actually works at 7B scale

**Statement**: SDFT's self-distillation mechanism (ICL-conditioned teacher -> on-policy student) produces meaningful gains at 7B parameter scale.

**Trace to bedrock**: SDFT's mechanism depends on ICL quality. ICL performs Bayesian posterior updating over the hypothesis space encoded in weights (Xie et al., 2021). The teacher's advantage over the student is the *mutual information between the demonstrations and the correct output*, mediated by the model's ICL capacity. At smaller scales, ICL capacity is weaker, so the teacher-student gap narrows and the distillation signal degrades.

**Evidence AGAINST**:
- The SDFT paper itself reports that at 3B, SDFT *underperforms* SFT. The ICL capacity is insufficient to provide a useful teacher signal. The crossover happens somewhere between 3B and 7B.
- The 7B improvement is only +4 points over SFT on Science QA. This is modest and task-specific. Science QA is a relatively straightforward knowledge task, not competition math.
- OPSD (Self-Distilled Reasoner, arXiv 2601.18734) reports that at 1.7B, on-policy self-distillation provides "limited gains" over GRPO on AIME, with consistent benefits only emerging at 4B+.
- Mathematical reasoning requires deeper compositional chains than Science QA. Per-step error compounding means ICL quality must be substantially higher for math than for QA.

**Evidence FOR**:
- SDFT at 7B shows +4 points, and at 14B the gap widens to +7 points. The trend is clearly positive and the 7B result is above SFT, not below.
- Qwen-2.5-7B specifically has strong math-oriented pretraining. Its ICL capacity for math may be higher than a generic 7B model's ICL for Science QA.
- OPSD at 8B (close to 7B) achieves 77.5% on AIME 2024 vs GRPO's 76.7% while being 4-8x more token-efficient. This proves on-policy self-distillation works for math at this scale.
- SDFT's compute overhead is only 2.5x FLOPs over SFT. It is not computationally prohibitive.

**Verdict**: CONTESTED
**Confidence**: 55% (works, but gains may be marginal for competition math at exactly 7B)

**Critical nuance**: "Works" is ambiguous. SDFT works *for what it claims* (reducing catastrophic forgetting in continual learning). The question is whether its mechanism translates to the ACE+TTRL context, where the goal is consolidating *evolved playbook strategies* into weights, not learning new factual skills. This is a different distribution of knowledge -- procedural/strategic rather than declarative/factual.

---

### Assumption 2: Context overflow is a real problem in a short AIME experiment

**Statement**: With only 30 problems over ~5 epochs, the evolving playbook will overflow the context window, creating a genuine need for SDFT-style consolidation.

**Trace to bedrock**: Context overflow is a function of (playbook growth rate) * (epochs) vs (context window size). The playbook grows when new strategies are ADDed and shrinks when strategies are DELETEd or MERGEd. The question is whether net growth exceeds the context budget within 5 epochs on 30 problems.

**Evidence AGAINST**:
- 30 problems * 5 epochs = 150 problem encounters. With aggressive curation (which ACE already does), the playbook stabilizes after ~50-80 encounters. The project's own GSM8K results show 50 entries as roughly optimal.
- The ACE curator enforces a token budget (~80K default). It actively compresses the playbook to fit. Overflow is *by design prevented* by the curator.
- AIME has ~15-20 core technique families. A well-compressed playbook needs 20-50 entries. This fits comfortably in a 32K context window, let alone 128K.
- Context overflow is a problem for *long-horizon continual learning* (hundreds of tasks). 30 problems over 5 epochs is an extremely short horizon.

**Evidence FOR**:
- ACE playbooks in practice tend to *grow before they compress*. Early epochs accumulate strategies liberally. If the first 2-3 epochs add faster than they curate, temporary overflow is possible.
- AIME problems are diverse (algebra, number theory, combinatorics, geometry). Strategy diversity needed may be higher than GSM8K (arithmetic word problems with more structural similarity).
- The model generates long chain-of-thought solutions for AIME (potentially 1000+ tokens each). Combined with playbook, total context pressure is real even if the playbook itself is moderate.

**Verdict**: REFUTED for this experiment scale
**Confidence**: 80%

**The honest assessment**: At 30 problems and 5 epochs, context overflow is a theoretical concern, not a practical bottleneck. The ACE curator's built-in compression handles this scale. SDFT's value proposition -- consolidating context knowledge into weights to free up context -- solves a problem that does not arise in this experiment. It *would* arise in a 1000-problem, 50-epoch continual learning experiment. But that is not what is proposed.

---

### Assumption 3: ACE+TTRL co-evolution is a meaningful primitive without consolidation

**Statement**: Running ACE (playbook evolution) simultaneously with TTRL (test-time RL with majority-vote rewards) produces results that are interpretable and valuable even without the SDFT consolidation step.

**Trace to bedrock**: A primitive is "meaningful" if its output contains information that (a) answers the research question and (b) cannot be obtained from simpler experiments. The research question is whether structured context evolution (ACE) complements weight-updating (TTRL). This question is answerable without SDFT.

**Evidence FOR**:
- ACE alone (without any weight updates) has been validated on GSM8K. TTRL alone has been validated on AIME 2024 (211% improvement on Qwen-2.5-Math-7B). Neither has been combined. The combination is the novel experiment.
- The interaction between evolving context and evolving weights is the core research question. SDFT is an *optimization* of this interaction, not a *prerequisite* for it.
- If ACE+TTRL shows mutual reinforcement (each making the other more effective), that is a publishable finding regardless of whether SDFT is added later.
- If ACE+TTRL shows no interaction or negative interaction, that is equally informative and tells you whether SDFT integration is even worth pursuing.

**Evidence AGAINST**:
- Without consolidation, the playbook exists only in context. If TTRL updates the weights, the model may "outgrow" the playbook -- strategies that were helpful for the pre-TTRL model may become irrelevant or harmful for the post-TTRL model. This creates a distribution shift between context and weights.
- The 3-stage loop argument says that without consolidation, you get "context drift" -- the playbook and weights diverge rather than converge. This could make ACE+TTRL *worse* than TTRL alone if the playbook provides outdated guidance.

**Verdict**: CONFIRMED
**Confidence**: 85%

**The key insight**: The potential for "context drift" (playbook becoming stale as weights update) is itself an interesting finding. Observing whether and how this happens is more informative than preventing it preemptively with SDFT. You need to see the failure mode to understand if consolidation is actually needed.

---

### Assumption 4: The "3-stage loop" (TTRL -> ACE -> SDFT = discover -> cache -> consolidate) is architecturally sound

**Statement**: The three stages form a coherent, mutually reinforcing loop where each stage's output feeds productively into the next.

**Trace to bedrock**: Architecturally sound means the loop has a fixed point -- iterating it converges toward better performance rather than oscillating or diverging. This requires each stage to be *compatible* with the others' inputs and outputs.

**Evidence AGAINST**:
- TTRL operates on weights. ACE operates on context. SDFT operates on both (distilling context into weights). The data types are different across stages. There is no guarantee that the output of one stage is well-formed input for the next.
- TTRL uses majority-vote rewards. ACE uses reflector-based curation. SDFT uses KL divergence between teacher and student. These are three fundamentally different optimization objectives. Multi-objective optimization often creates Pareto conflicts rather than synergy.
- The loop has not been tested even once. Calling it "architecturally sound" based on the logical narrative (discover-cache-consolidate) is reasoning from metaphor, not from evidence. Many architectures that sound logical fail in practice due to unforeseen interactions.
- The SDFT paper's use case is continual learning (preventing forgetting when learning new tasks). The 3-stage loop repurposes it as a consolidation step within a single-task training loop. This is a category error -- SDFT is designed for cross-task preservation, not within-task consolidation.

**Evidence FOR**:
- The discover-cache-consolidate pattern is well-established in cognitive science. Hippocampal replay (Complementary Learning Systems theory, McClelland et al., 1995) describes exactly this: fast learning in hippocampus (context), slow consolidation into cortex (weights), with replay as the bridge (distillation).
- OPSD demonstrates that on-policy self-distillation can improve on GRPO for math reasoning. If SDFT functions like OPSD in the consolidation role, the architecture is plausible.
- Each stage addresses a clear weakness of the others: TTRL has no memory across problems, ACE has no weight updates, SDFT bridges them.

**Verdict**: CONTESTED
**Confidence**: 40% (plausible narrative, no empirical support, high risk of unforeseen interactions)

**The dangerous thing about this assumption**: It is an architectural *story*, not an architectural *proof*. The human tendency to find narrative coherence (discover-cache-consolidate sounds inevitable, like a natural law) masks the fact that each transition is an engineering challenge with multiple failure modes. The transitions are where this breaks, not the stages.

---

### Assumption 5: On-policy distillation from ICL teacher preserves reasoning quality at 7B

**Statement**: When a 7B model distills from its own ICL-conditioned version (teacher = model + demonstrations), the resulting weight updates preserve the quality of mathematical reasoning.

**Trace to bedrock**: Distillation preserves quality when the teacher distribution is a faithful representation of the target task's solution distribution. The teacher here is the *same model* conditioned on demonstrations. The question is whether conditioning on demonstrations produces a sufficiently different (and better) distribution to create a useful gradient signal, while being close enough to not introduce distributional noise.

**Evidence AGAINST**:
- At 7B, the teacher-student gap is small (+4 points on Science QA). A small gap means a weak distillation signal, which means slow learning and vulnerability to noise.
- Mathematical reasoning requires precise multi-step chains. KL divergence minimization between teacher and student may "smooth out" the sharp reasoning steps, replacing precise chains with averaged probability mass across multiple paths. This is the "mode covering" problem in KL minimization.
- The SDFT paper explicitly notes it "cannot transform non-reasoning models to produce chain-of-thought." If the base model's chain-of-thought quality is marginal (as it may be at 7B for AIME), SDFT cannot improve it.
- OPSD at 1.7B shows "limited gains" -- suggesting that for smaller models, the teacher simply is not good enough to provide useful supervision beyond what GRPO already provides.

**Evidence FOR**:
- OPSD at 8B achieves 77.5% on AIME 2024, outperforming GRPO (76.7%). This directly demonstrates that on-policy self-distillation preserves (and slightly improves) reasoning quality at near-7B scale.
- The 4-8x token efficiency of OPSD vs GRPO suggests that the dense token-level supervision from the teacher is more informative per token than GRPO's sparse outcome-level reward. Quality is preserved because the supervision is richer, not despite it.
- Qwen-2.5-Math-7B has been specifically trained for mathematical reasoning. Its ICL capacity for math demonstrations is likely stronger than a generic 7B model's ICL for other tasks.

**Verdict**: CONTESTED
**Confidence**: 50%

**The crux**: The evidence from OPSD (which is very close to SDFT in mechanism) suggests it works at 8B for math. But the evidence from SDFT itself is on Science QA, not math, and the improvement at 7B is modest. Whether the OPSD evidence transfers to the specific SDFT implementation is uncertain.

---

### Assumption 6: A single A100 has enough compute for SDFT on top of GRPO training

**Statement**: Adding SDFT's distillation step to the existing ACE+TTRL pipeline fits within the compute budget of a single A100 (Google Colab).

**Trace to bedrock**: Compute is a hard physical constraint. An A100 has 80GB HBM, ~312 TFLOPS FP16. Training a 7B model with LoRA requires ~20-30GB for model + optimizer states. Generation (for GRPO rollouts) dominates wall time. The question is whether SDFT's additional forward passes (teacher + student KL computation) fit in the remaining budget.

**Evidence AGAINST**:
- SDFT requires 2.5x the FLOPs and 4x the wall-clock time of SFT. Added on top of GRPO (which already requires multiple rollouts per problem), the total compute multiplier may be 3-5x over GRPO alone.
- GRPO on Qwen-2.5-7B with 8 rollouts per problem is already compute-intensive. Each AIME solution is 1000+ tokens. 30 problems * 8 rollouts * 5 epochs = 1200 long generations + reward computation + gradient updates. Adding SDFT's teacher-student passes doubles the forward pass cost.
- Google Colab has a time limit (~12 hours for Colab Pro). A combined ACE+TTRL+SDFT pipeline may exceed this.
- OPSD experiments used 8xA100 GPUs. Scaling down to 1xA100 means either much longer training or smaller batch sizes (which can hurt GRPO's group-relative estimation quality).

**Evidence FOR**:
- LoRA reduces the trainable parameter count dramatically (typically 0.1-1% of total parameters). This makes the gradient computation for SDFT cheap relative to full fine-tuning.
- SDFT uses the same model as teacher and student (no separate model to load). Memory overhead is minimal -- just the KL divergence computation on top of existing forward passes.
- If SDFT is run as a *separate phase* after GRPO (not interleaved), the peak compute is the same as GRPO -- just the total training time increases.
- With flash attention and mixed precision, a single A100 can handle 7B model inference and training efficiently.

**Verdict**: CONTESTED
**Confidence**: 45%

**The practical reality**: It *probably* fits on a single A100 in terms of memory. The question is wall-clock time. If the Colab session has a 12-hour limit, adding SDFT may push the experiment from "tight but feasible" to "requires multiple sessions with checkpointing." This is an engineering annoyance, not a fundamental blocker, but it increases the probability of wasted compute from session interruptions.

---

## Meta-Questions

### When is combining unvalidated primitives justified?

**First principles**: Combining unvalidated primitives is justified when:

1. **The interaction IS the hypothesis.** If your research question is "does A+B produce emergent behavior that neither A nor B exhibits alone?", running them separately by definition cannot answer it. But this is NOT the current question. The question is "does SDFT consolidation improve ACE+TTRL?" -- which is a *modular* question (does adding C to A+B help?) not an *interaction* question.

2. **The primitives are cheap relative to the interaction.** If validating A and B separately costs nearly as much as validating A+B together, there is no savings from sequencing. In this case, ACE+TTRL is already a substantial experiment. Adding SDFT is not free -- it adds complexity, compute, and debugging surface.

3. **Failure diagnosis is straightforward.** If you can attribute a combined failure to one specific component (through ablation, logging, or monotonicity arguments), combining is safe. ACE+TTRL+SDFT has THREE interacting components. If the combined system fails, was it because ACE's playbook was bad? TTRL's rewards were noisy? SDFT's distillation corrupted the weights? Three-way attribution is notoriously hard.

4. **The opportunity cost of sequencing exceeds the debugging cost of combining.** If you can only run one experiment ever, combining makes sense. If you can run two, sequencing is safer.

**Verdict for this case**: Combining is NOT justified. The question is modular (does adding SDFT help?), the failure diagnosis is hard (3-way interaction), and sequencing is affordable (you can run ACE+TTRL first, then add SDFT in a follow-up).

### Expected information value: ACE+TTRL alone vs. combined

**Alone (ACE+TTRL)**: You learn whether context evolution and weight updates are complementary. This answers the core research question. The information value is HIGH because:
- No one has published this combination
- The result (positive or negative) informs whether the 3-stage loop is worth pursuing
- A negative result (ACE+TTRL worse than TTRL alone) would *prevent* you from wasting compute on SDFT integration

**Combined (ACE+TTRL+SDFT)**: You learn whether the full 3-stage loop works. But if it fails, you do not know WHY (was it the ACE-TTRL interaction or the SDFT step?). If it succeeds, you do not know which components were essential. The information is LESS DECOMPOSABLE.

**Expected information value comparison**: ACE+TTRL alone has HIGHER expected information value per compute unit because its results are more interpretable. The combined experiment has higher *ceiling* value (if it works, it is a bigger result) but lower *expected* value (more likely to produce uninterpretable failures).

This is a classic risk-return tradeoff. For a resource-constrained researcher (single A100), the lower-variance option (sequence) dominates.

### Minimum viable experiment for the SDFT hypothesis

The minimum viable test for "SDFT consolidation helps" does NOT require the full 3-stage loop. Instead:

1. **Take a pre-trained Qwen-2.5-Math-7B**
2. **Manually construct a playbook** (5-10 high-quality AIME strategies, hand-curated)
3. **Run SDFT**: use the playbook as demonstrations, distill the ICL-conditioned model into base weights
4. **Test**: does the distilled model perform better WITHOUT the playbook than the original model WITH it?

This test isolates the SDFT hypothesis (can ICL knowledge be consolidated into weights?) from the ACE and TTRL machinery. It takes ~2-3 hours on a single A100 and produces a clean signal.

If this MVE shows SDFT cannot consolidate even hand-curated playbook knowledge at 7B, there is no point integrating it into the full loop. If it succeeds, you have a validated primitive worth integrating.

---

## The Contrarian Truth

**The contrarian truth is this: the 3-stage loop is a premature abstraction.**

It sounds elegant: TTRL discovers, ACE caches, SDFT consolidates. Discover-cache-consolidate. Three stages, complete cycle, closed loop. The narrative is satisfying.

But satisfying narratives are where researchers waste compute. The question is not "does this narrative make sense?" but "does each transition actually work?" And right now, NONE of the transitions have been validated:

- TTRL -> ACE: Can evolving playbook context actually steer GRPO-trained models? Never tested.
- ACE -> SDFT: Can ICL-conditioned self-distillation consolidate *playbook strategies* (not just factual knowledge)? Never tested. The SDFT paper tests factual knowledge (Science QA, medical reasoning), not strategic/procedural knowledge.
- SDFT -> TTRL: Does the consolidated model's improved base performance translate to better GRPO rollouts in the next iteration? Never tested.

**Zero of three transitions validated. The loop is pure architecture fiction.**

The popular view: "These are complementary techniques that logically compose into a powerful system." The contrarian truth: "These are three separate research bets that MIGHT compose, and composing them before validating any transition creates an undebuggable experiment."

The even deeper contrarian truth: **SDFT may be solving the wrong problem entirely.** SDFT's value proposition is consolidating context knowledge into weights to prevent context overflow and catastrophic forgetting. But in a 30-problem, 5-epoch experiment:
- Context overflow does not occur (the curator compresses the playbook)
- Catastrophic forgetting is not the risk (the model is learning one task, not sequencing across tasks)
- The real bottleneck is coverage of strategy space (per the project's own key finding)

SDFT addresses none of these. It is a solution looking for a problem in this experimental context. It WOULD be valuable in a long-horizon, multi-domain continual learning setting. But that is not what is proposed.

---

## Synthesis: The Recommendation from First Principles

**Sequence, do not integrate. And run the MVE first.**

The reasoning chain:
1. ACE+TTRL has never been validated -> it is the higher-priority unknown
2. SDFT addresses context overflow -> which is not a problem at this experiment's scale
3. Combining three unvalidated primitives -> creates undebuggable failure modes
4. The minimum viable experiment for SDFT -> is cheap and independent
5. Sequencing preserves information value -> a clean ACE+TTRL result informs SDFT integration

**The protocol**:
1. Run the MVE for SDFT (2-3 hours, single A100): hand-curated playbook + distillation test
2. Run ACE+TTRL alone (main experiment, 8-12 hours)
3. If both succeed: integrate in a follow-up experiment with the validated primitives
4. If either fails: you know exactly what failed and why

**The cost of this protocol**: one additional small experiment (the MVE) and possibly one additional full experiment (if you decide to integrate). Total: ~20 hours of A100, spread across 2-3 sessions.

**The cost of integrating now**: if the combined experiment fails (probable, given three unvalidated transitions), you have spent 12+ hours with no interpretable result and must decompose retroactively. The debugging alone may take longer than running the experiments sequentially.

---

## What Changes Tomorrow

If the MVE shows SDFT cannot consolidate strategic/procedural playbook knowledge at 7B (which I estimate at ~40% probability), the entire 3-stage loop framing collapses and you should invest in other consolidation mechanisms (e.g., LoRA fine-tuning on successful solution traces, or simply keeping the playbook in context permanently).

If ACE+TTRL shows negative interaction (playbook becomes stale as weights update), SDFT becomes MORE interesting as a potential fix -- but you would not have discovered this interaction without running ACE+TTRL alone first.

---

## Sources

- [SDFT: Self-Distillation Enables Continual Learning (arXiv 2601.19897)](https://arxiv.org/abs/2601.19897) -- 3B underperforms SFT, 7B +4pts, 14B +7pts, 2.5x FLOP overhead
- [OPSD: Self-Distilled Reasoner (arXiv 2601.18734)](https://arxiv.org/abs/2601.18734) -- 8B achieves 77.5% AIME 2024, 4-8x token efficiency vs GRPO, 1.7B shows limited gains
- [TTRL: Test-Time Reinforcement Learning (arXiv 2504.16084)](https://arxiv.org/abs/2504.16084) -- 211% improvement on Qwen-2.5-Math-7B for AIME 2024
- [GRPO compute discussion (huggingface/open-r1#100)](https://github.com/huggingface/open-r1/issues/100)
- [ICL Distillation (arXiv 2412.13243)](https://arxiv.org/abs/2412.13243) -- hardware constraints for distillation
- [On-Policy Distillation (Thinking Machines Lab)](https://thinkingmachines.ai/blog/on-policy-distillation/)
