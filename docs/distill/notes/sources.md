# Source Notes: SDFT Integrate or Sequence

Research mined for the distillation on whether SDFT should be integrated into ACE+TTRL or sequenced separately.

---

## 1. PRIMARY PAPERS (Direct Decision Inputs)

### SDFT: Self-Distillation Enables Continual Learning (arXiv:2601.19897)
- **Authors**: Idan Shenfeld, Mehul Damani et al. MIT / Improbable AI Lab.
- **Key insight**: On-policy self-distillation from ICL teacher preserves prior capabilities; fails at 3B scale.
- **Lindy**: Recent
- **Critical details for our decision**:
  - **Model sizes**: Qwen 2.5 at 3B, 7B, 14B. Primary experiments on 7B-Instruct.
  - **Hardware**: Single NVIDIA H200 GPU. No multi-GPU needed.
  - **Compute overhead**: 2.5x FLOPs and 4x wall-clock time vs SFT (on-policy rollouts, single trajectory per prompt).
  - **Training**: 2 epochs for skill learning, 4 epochs for knowledge acquisition.
  - **3B failure**: "At small scales (3B), ICL is too weak to provide meaningful teacher guidance."
  - **7B results**: Science Q&A 70.2% vs SFT 66.2%; Tool Use 70.6% vs 63.2%. Consistent 4-7pt gains.
  - **Sequential learning**: 3 tasks learned sequentially without catastrophic forgetting.
  - **NOT tested on math reasoning**. Domains: Science QA, Tool Use, Medical, Knowledge Acquisition.
  - **No LoRA comparison**. Full fine-tuning only.
  - **Loss function**: Reverse KL divergence, student (input only) vs teacher (input + demo). Single trajectory.
  - **Complementary to RL**: "SDFT addresses a complementary learning regime" -- demos without rewards vs RL with rewards.
  - **Limitations**: Requires strong ICL (fails 3B); spurious artifacts; cannot handle behavioral shifts (non-reasoning to reasoning); struggles with noisy demos.

### SDPO: Reinforcement Learning via Self-Distillation (arXiv:2601.20802)
- **Authors**: Hubotter, Lubeck, Shenfeld et al. Same research group as SDFT.
- **Key insight**: Self-distillation for RL reaches GRPO accuracy 10x faster; 7x shorter reasoning traces.
- **Lindy**: Recent
- **Critical details**:
  - Same self-distillation principle applied to RL with tokenized feedback.
  - Tested: Olmo3-7B-Instruct (Chemistry), Qwen3 0.6B-8B.
  - SDPO reaches GRPO accuracy in 4x fewer generations.
  - Reasoning traces 7x shorter (avoids logical loops GRPO produces).
  - Evaluated: scientific reasoning, tool use, competitive programming (LiveCodeBench v6).
  - **Implication**: Could replace GRPO as the RL algorithm in TTRL for better sample efficiency.

### TTRL: Test-Time Reinforcement Learning (arXiv:2504.16084, NeurIPS 2025)
- **Authors**: Zuo, Zhou et al.
- **Key insight**: Majority vote as pseudo-reward enables RL on unlabeled test data; 211% boost on AIME for 7B.
- **Lindy**: Recent
- **Critical details**:
  - **Hardware**: 8x NVIDIA A100 80GB (or 40GB in some configs).
  - **Training budget**: AIME 60 episodes, AMC 50, MATH 40.
  - **GRPO hyperparameters**: LR 5e-7 constant, AdamW, KL coefficient 0, max gen 3072 tokens.
  - **Sampling**: N=64 for majority vote, downsampled to 16 for RL rollouts. Temperature 1.0.
  - **Qwen-2.5-Math-7B on AIME 2024**: 16.7% -> 43.3% (+159.3%).
  - **Framework**: verl (not TRL). Full fine-tuning.
  - **Single A100 gap**: Paper uses 8xA100. Adaptation to 1xA100 requires LoRA, smaller batches, gradient accumulation.

### CG-TTRL: Context-Guided Test-Time RL (arXiv:2511.06430)
- **Authors**: (2025).
- **Key insight**: TF-IDF context selection adds 7% relative improvement over vanilla TTRL; converges faster.
- **Lindy**: Recent
- **Critical details**:
  - Primarily 1.5B models; also tested 7B.
  - Context selection via TF-IDF cosine similarity (simple, not learned).
  - Context in BOTH sampling phases (exploitation + exploration).
  - Training: 40 epochs/120 steps on MATH-500; strong performance after 1-5 epochs.
  - Hardware: 2x A100 80GB. ~39 hours convergence.
  - DeepSeek-R1-Distill: baseline 37.2% -> TTRL 56.8% -> CG-TTRL 59.4%.
  - **Key question**: CG-TTRL is static-context TTRL. Does ACE curation add enough over TF-IDF to justify complexity?

---

## 2. ICL-TO-WEIGHTS DISTILLATION (The Core Mechanism)

### System-2 Fine-tuning / "Contextual Shadowing" (arXiv:2505.01812)
- **Authors**: (2025).
- **Key insight**: Training with context prefix catastrophically suppresses learning signal ("contextual shadowing").
- **Lindy**: Recent (but the mechanism may be enduring)
- **Critical details**:
  - Tested Qwen 2.5: 0.5B through 32B, plus Llama 3.1 8B.
  - **"FT-ICL gap"**: Naive fine-tuning consistently underperforms ICL, especially outside math/coding.
  - **Contextual shadowing**: When training data includes context before QA, answer tokens become "unsurprising" given context, suppressing learning signal. This is EXACTLY what happens if you naively distill ACE playbook into weights.
  - **Best approach**: Self-QA protocol -- generate questions using ICL, answer them referencing knowledge, fine-tune on QA pairs WITHOUT original context prefix.
  - Math and coding showed strongest Sys2-FT performance (near ICL parity for large models).
  - **"Curse of overexposure"**: Fine-tuning can degrade ICL ability on the same knowledge.

### Brewing Knowledge in Context (arXiv:2506.11516)
- **Authors**: (2025).
- **Key insight**: ICL is mathematically equivalent to implicit KD; attention simulates gradient descent toward a reference model.
- **Lindy**: Enduring (theoretical framework)
- **Implication**: If playbook context is well-curated, ICL is already performing effective distillation at inference time. Explicit weight distillation adds value only if the context window overflows or if you need to free up context budget for new knowledge.

### On-Policy Distillation of Language Models (arXiv:2306.13649)
- **Authors**: (2023).
- **Key insight**: On-policy distillation addresses distribution mismatch by training on self-generated sequences.
- **Lindy**: Enduring
- **Details**: GKD (Generalized Knowledge Distillation) uses reverse KL to prevent overestimating low-probability regions. Scales from 120M to 13B consistently. This is the theoretical foundation SDFT builds on.

### MiniLLM: On-Policy Distillation (arXiv:2306.08543)
- **Authors**: (2023).
- **Key insight**: Reverse KL more suitable than forward KL for generative LM distillation -- prevents mode-covering.
- **Lindy**: Enduring
- **Relevance**: Confirms SDFT's choice of reverse KL. But note: at 7B with self-distillation (no external teacher), the student IS the teacher -- the question is whether there's enough signal.

---

## 3. CATASTROPHIC FORGETTING (Is This Even Our Problem?)

### Mechanistic Analysis of Catastrophic Forgetting (arXiv:2601.18699, 2025)
- **Key insight**: Three coupled mechanisms at different timescales: attention disruption (epochs 1-2), representational drift (epochs 3-5), loss landscape flattening (epoch 4+).
- **Lindy**: Enduring (mechanistic understanding)
- **For our experiment**: At 5 epochs on 30 AIME problems, all three mechanisms are theoretically active. But dataset is tiny (30 problems) -- unclear if there's enough gradient to trigger meaningful forgetting.

### Self-Synthesized Rehearsal / SSR (ACL 2024 Long)
- **Key insight**: LLM generates synthetic rehearsal instances; LoRA does NOT prevent forgetting.
- **Lindy**: Recent
- **Critical**: LoRA being ineffective for forgetting prevention contradicts common wisdom. If our TTRL uses LoRA on single A100 (as we'd need to), forgetting may still occur.

### Continual Learning as Context Budget (Jessy Lin blog, 2025)
- **Key insight**: Forgetting = context budget problem; adding new ICL examples pushes out old ones.
- **Lindy**: Enduring (framing)
- **Relevance**: This IS the ACE problem. Context overflow drives the need for SDFT. But with only 30 AIME problems and a 128K context window, do we actually overflow?

---

## 4. EXPERIMENT DESIGN PRINCIPLES

### Controlled Ablation Study Methodology (various sources)
- **Key insight**: LOCO (Leave-One-Component-Out) is gold standard for causal attribution.
- **Lindy**: Timeless (scientific method)
- **Relevance**: If SDFT is untested on math reasoning, and ACE+TTRL co-evolution is untested, combining them risks undebuggable failure. LOCO says: validate each component first. BUT this assumes independence -- SDFT+TTRL may have synergistic interactions LOCO would miss.

---

## 5. PREVIOUSLY COLLECTED (from prior distillation)

### Retained from prior sources.md (most relevant subset)

- **Xie et al. (2021)**: ICL as implicit Bayesian inference -- Enduring
- **von Oswald et al. (2023)**: Transformers learn ICL by gradient descent -- Enduring
- **Min et al. (2022)**: Demonstration format > labels -- Enduring
- **Snell et al. (2024)**: Test-time compute scaling; small model + compute > big model -- Enduring
- **STaR / Zelikman (2022)**: Reasoning bootstrapping loop -- Enduring
- **V-STaR / Hosseini (2024)**: Learn from failures via DPO-trained verifier -- Recent
- **Self-Consistency / Wang (2023)**: Majority voting over reasoning paths -- Enduring
- **Reflexion / Shinn (2023)**: Verbal reinforcement learning -- Enduring
- **DeepSeek-R1 (2025)**: Pure RL produces emergent reasoning; distilled 1.5B at 45.7% AIME -- Recent
- **ACE Framework (2025)**: Three-agent playbook evolution -- Recent
- **LeaP / Luo (2025)**: Peer reasoning, 7B > 671B -- Recent
- **DeepMind (2023)**: LLMs cannot self-correct without external feedback -- Enduring
- **Self-Improvement Paradox / Sun (2025)**: Bootstrapping works conditionally with verification -- Recent
- **Polya "How to Solve It" (1945)**: Original playbook for problem-solving -- Timeless
- **Thompson Sampling / Russo (2018)**: Optimal exploration-exploitation -- Timeless

---

## KEY GAPS IN THE LITERATURE

1. **No one has tested SDFT on math reasoning.** The paper tests tool use, science QA, medical, knowledge acquisition. Sys2-FT suggests math is one of the EASIER domains for FT-ICL transfer, but unvalidated for SDFT specifically.

2. **SDFT compute on A100 vs H200**: Paper uses H200. The 4x wall-clock overhead means a 1-hr SFT job becomes 4 hrs. On A100 (slower), possibly 6-8 hrs. For 30 AIME problems, 5 epochs -- feasible but not cheap.

3. **SDFT + GRPO interaction is untested.** SDPO exists and outperforms GRPO, but nobody has done SDFT-then-GRPO or GRPO-then-SDFT. Contextual shadowing from Sys2-FT suggests ordering matters enormously.

4. **TTRL on single A100 is unvalidated.** Paper uses 8xA100. Adapting requires LoRA, smaller batches, gradient accumulation -- each could change dynamics.

5. **CG-TTRL already partially solves the problem.** If the goal is "use context to improve TTRL," CG-TTRL does this with TF-IDF. Question: does ACE curation add enough over TF-IDF to justify complexity?

6. **Context overflow may not occur.** 30 AIME problems with 128K context window -- the playbook may never actually grow large enough to warrant consolidation via SDFT.

7. **SDPO may be the better integration point than SDFT.** If the goal is sample-efficient RL that avoids GRPO's logical loops, SDPO (same authors) is a drop-in GRPO replacement, not a separate stage.
