# Advocate Position: The Case FOR the Memory x RL Experiment

**Thesis: This experiment occupies a genuinely unexplored intersection in the literature, asks a question the field urgently needs answered, and is publishable with the benchmark fix the distillation already identified. The ceiling-effect problem is solvable; the novelty is real and time-sensitive.**

---

## 1. The Novelty Gap Is Real and Empirically Verified

No published or preprint paper simultaneously evolves a curated strategy playbook while running RL weight updates on the same model. This is not an incremental claim -- it is a verifiable gap confirmed by exhaustive search across both "camps" of the literature:

**Camp A (evolving context, frozen weights):** ACE (arXiv:2510.04618), MemRL (arXiv:2601.03192), Live-Evo (arXiv:2602.02369), MemSkill (arXiv:2602.02474), TAME (arXiv:2602.03224), Dynamic Cheatsheet (arXiv:2504.07952). Every single one of these papers explicitly freezes the LLM weights and evolves only the context.

**Camp B (RL weight updates, static/no context):** TTRL (NeurIPS 2025, arXiv:2504.16084), DeepSeek-R1 (arXiv:2501.12948), standard GRPO/RLVR. No external evolving memory.

**The closest competitor, CG-TTRL (arXiv:2511.06430), uses static context selection (TF-IDF retrieval) -- it is precisely our Condition C, not Condition D.** Memory-R1 (arXiv:2508.19828) uses GRPO to train memory CRUD operations, but the RL optimizes memory management behavior, not mathematical reasoning with memory in context. Prompt Augmentation Scales GRPO (arXiv:2602.03190) augments prompts statically via paraphrasing, not through outcome-driven curation.

The experiment bridges these two camps for the first time. This is not "yet another memory paper" -- it is the first to study what happens when context-space learning and weight-space learning operate simultaneously on the same model.

---

## 2. The Question Has Deep Theoretical Grounding

The experiment is not ad hoc benchmark chasing. It instantiates a precise theoretical question from complementary learning systems theory (McClelland et al., 1995): **what happens when a fast, episodic learning channel (the evolving playbook, analogous to the hippocampus) operates simultaneously with a slow, statistical learning channel (RL weight updates, analogous to the cortex)?**

Four outcomes are possible, each theoretically interesting:
1. **Synergy** (D > B and D > C): The channels are complementary -- context scaffolds exploration, weights consolidate.
2. **Interference** (D < B): The channels compete -- context distracts from RL's gradient signal.
3. **Redundancy** (D = C after training): Static scaffolding is sufficient -- evolution adds nothing.
4. **Absorption** (D without playbook = D with playbook > B): RL absorbed the playbook -- the scaffolding became unnecessary.

Outcome 4 connects directly to Vygotsky's Zone of Proximal Development (1930s): the playbook serves as temporary scaffolding that the learner internalizes and eventually no longer needs. This is the first empirical test of this educational theory in the context of RL-trained LLMs.

**Crucially, the reframing as "how do two learning channels interact?" makes EVERY outcome publishable.** A null result (D = B) means the channels are independent. Interference (D < B) means they compete. This is fundamentally different from a binary "does memory help?" framing where null = unpublishable.

---

## 3. The MATH-500 Problem Is Solvable -- And the Fix Strengthens the Paper

Yes, MATH-500 is saturated at 92.8% for DeepSeek-R1-Distill-Qwen-7B. The distillation correctly identified this as a critical flaw. But this is a benchmark choice problem, not a design problem. The fix is straightforward and well-precedented:

**Option A: AIME 2024 as primary benchmark.** Baseline is 55.5% for this model, with 7B SOTA at 70.8% (BP-Math-7B, arXiv:2503.21380). This gives ~15% improvement headroom -- more than enough to detect condition differences. AIME is 30 problems, which reduces statistical power, but 5 seeds x 30 problems = 150 observations per condition, adequate for paired tests.

**Option B: MATH-500 Level 4-5 stratification.** Filter to the ~200 hardest problems where baseline accuracy is ~75-80%. This preserves the standard benchmark while giving 20-25% headroom.

**Option C: Add OlymMATH-EASY (100 problems, AIME difficulty).** Non-contaminated, manually curated, automated verification. Top 7B models reach ~60-70% -- excellent headroom with good problem count. MathArena provides similar properties.

**Option D: Add a weaker model.** DeepSeek-R1-Distill-Qwen-1.5B scores ~50-60% on MATH-500 -- the original assumed headroom. Running conditions B and D on 1.5B with 3 seeds provides both the headroom AND a scale-comparison signal.

Any of these fixes is a one-line change in the experimental config. The 5-condition design, the analysis experiments, the theoretical framing -- all remain unchanged. The benchmark fix is engineering, not redesign.

---

## 4. The Absorption Test Is the Crown Jewel

While the general concept of scaffold-then-remove exists (FT+RAG at ICLR 2026, GenPI arXiv:2411.15927, Learning by Distilling Context arXiv:2209.15189), none of these test the specific case of a **co-evolved** scaffold. The distinction matters:

- **GenPI/FT+RAG:** A *static* scaffold (retrieval hints, fixed prompts) is used during training, then removed. The scaffold never changes based on what the model learns.
- **Our absorption test:** A *co-evolved* scaffold (ACE-curated playbook that updates based on RL training outcomes) is used during training, then removed. The scaffold adapted as the model adapted.

This is a fundamentally harder question. In the static case, the model learns to perform without a fixed crutch. In the co-evolved case, the scaffold and the model co-adapted -- the playbook contains strategies that emerged specifically because the model was learning to use them. Removing a co-evolved scaffold tests whether the *dynamic interaction* produced internalization, not just whether the model can cope without a static input.

**RuscaRL (arXiv:2508.16949) is the closest analogue:** it decays rubric scaffolding during training. But RuscaRL's rubrics are predefined checklists that decay on a schedule, not curated strategy playbooks that evolve based on the model's actual performance. Our absorption test measures *cumulative* internalization after full co-evolution, not gradual weaning during training.

If the absorption test shows D-without-playbook = D-with-playbook > B, this would be a genuine contribution: **proof that RL training absorbed co-evolved context into weights.** This result has never been demonstrated.

---

## 5. The Timing Window Is Real But Still Open

The distillation correctly notes that four memory papers appeared in Jan-Feb 2026 (MemRL, MemSkill, Live-Evo, TAME) and the ICLR 2026 MemAgents workshop signals community convergence. But every one of these papers freezes weights. The co-evolution experiment remains unpublished.

**For a CS224N project, the timing is actually perfect.** The field is converging on memory for LLM agents, making the experiment maximally relevant to reviewers and readers. The question is live and important. But the specific answer -- what happens when you combine context evolution with weight evolution -- has not been reported.

The February 2026 arXiv burst (4 papers in 6 weeks) shows independent teams are exploring the memory space. This validates the direction. Being the first to cross the bridge between Camp A and Camp B is worth doing quickly.

---

## 6. The Experimental Design Is Mostly Clean

The B-C-D ladder is a textbook ablation:

| Condition | Context | Evolves? | RL? | What it tests |
|-----------|---------|----------|-----|--------------|
| B | None | -- | Yes | RL-only baseline |
| C | Static playbook | No | Yes | Does context help RL? |
| D | Evolving playbook | Yes | Yes | Does co-evolution help beyond static? |
| E | None (novelty reward) | -- | Yes | Alternative to memory (diversity via reward) |

B vs C isolates the effect of providing context. C vs D isolates the effect of context evolution. B vs E isolates the effect of novelty reward. D vs E compares two approaches to the diversity problem (external memory vs intrinsic reward).

**The D' confound (anchor problems + ground-truth curation):** Yes, D' changes two variables. But this is defensible as a "practical best-case" condition -- in deployment, if you have any verified examples, you would use them for both training and curation. Frame D' as "verified anchoring" (a practical bundle), not as a mechanistic ablation. The B-C-D ladder provides the mechanistic story; D' provides the pragmatic ceiling.

---

## 7. The Diversity Diagnostic Leverages a Known Problem

GRPO suffers from diversity collapse -- this is established by DRA-GRPO (arXiv:2505.09655), DiverseGRPO (arXiv:2512.21514), ProGRPO (arXiv:2602.05281), and GAPO (EMNLP 2025). The standard scalar correctness reward creates winner-takes-all dynamics where one reasoning path dominates.

**The experiment provides a natural test:** Does an evolving playbook (which explicitly curates diverse strategies via ADD/UPDATE/MERGE/DELETE) prevent the diversity collapse that plagues vanilla GRPO? If Condition D shows higher strategy entropy than Condition B after RL training, this alone is a publishable finding -- the playbook acts as an implicit diversity regularizer.

This connects to EVOL-RL's novelty reward (Condition E, arXiv:2509.15194), which addresses diversity collapse through reward shaping. Comparing D (diversity via external memory) with E (diversity via intrinsic reward) is a novel and informative comparison that no prior work has made.

---

## 8. Single Model Is Defensible for CS224N -- and Even for Workshop

**TTRL was accepted at NeurIPS 2025 with primarily Qwen-2.5-Math-7B experiments.** ACE published with a single model family. For a CS224N project paper, a single 7B model with 5 conditions x 5 seeds = 25 training runs is substantial. The expectation for a course project is original research contribution, not scaling generalization.

If the paper targets a workshop (MemAgents at ICLR 2026, or FoRLM at NeurIPS 2026), single-model is standard. For main conference, adding B and D on DeepSeek-R1-Distill-Qwen-1.5B (where MATH-500 IS ~50%) would address both the generalization concern and the benchmark saturation concern simultaneously -- two problems, one fix.

---

## 9. The Paper Is Stronger If the Result Is Negative

This is the contrarian insight from the distillation and it bears emphasis: **"Weights absorb memory: explicit knowledge becomes redundant during RL training"** (outcome 4) is a cleaner, more surprising, and more citable result than "memory helps a little" (outcome 1).

Most practitioners assume context and weights are complementary -- that RAG + fine-tuning is better than either alone. Showing that RL training makes explicit memory redundant would challenge this assumption and have immediate practical implications: stop spending tokens on memory systems for RL-trained models.

The absorption test is designed to detect this. If it works, the paper writes itself. If it doesn't, the diversity diagnostic and strategy attribution still provide novel empirical evidence about the memory x RL interaction.

---

## 10. Summary: Why This Experiment Is Worth Doing

| Criterion | Assessment |
|-----------|-----------|
| **Novelty** | STRONG. First to co-evolve context + weights during RL training. |
| **Timeliness** | STRONG. Community converging (MemAgents workshop, 4 papers in 6 weeks), window open. |
| **Theoretical grounding** | STRONG. Complementary learning systems, Vygotsky scaffolding, dual-channel interaction. |
| **Experimental design** | GOOD. B-C-D ladder is clean. D' has a known confound but is defensible. |
| **Benchmark (with fix)** | GOOD. AIME 2024 primary (55.5% baseline, 15% headroom) solves ceiling effect. |
| **Analysis depth** | STRONG. Absorption test + diversity diagnostic + strategy attribution go beyond accuracy-chasing. |
| **Publishability** | Workshop: HIGH confidence. Main conference: MODERATE confidence with fixes. |
| **Downside risk** | LOW. Every outcome is theoretically interesting with the "two learning channels" framing. |

**The minimum viable experiment:** Fix the benchmark (AIME 2024 primary, MATH-500 Level 4-5 secondary), cite scaffold-removal prior art (FT+RAG, GenPI) while distinguishing co-evolution, reframe as "interaction study" not "does memory help?", and run 5 conditions x 5 seeds. This is doable, novel, timely, and publishable.

---

## Recommended Benchmark Configuration

| Benchmark | Role | Problems | Baseline | Headroom | Purpose |
|-----------|------|----------|----------|----------|---------|
| AIME 2024 | Primary | 30 | 55.5% | ~15% | High sensitivity, adequate difficulty |
| MATH-500 L4-5 | Secondary | ~200 | ~75-80% | ~20% | Standard benchmark, stratified |
| OlymMATH-EASY | Tertiary (if compute allows) | 100 | ~60-70% | ~25% | Non-contaminated, AIME-level |

---

## Key Citations Supporting the Case

- **Novelty gap confirmed:** ACE (2510.04618), MemRL (2601.03192), Live-Evo (2602.02369), MemSkill (2602.02474), TAME (2602.03224) -- all freeze weights. CG-TTRL (2511.06430) -- static context only. Memory-R1 (2508.19828) -- memory CRUD, not reasoning + memory.
- **Absorption test novelty:** FT+RAG (2510.01375, ICLR 2026) and GenPI (2411.15927) test static scaffolds. Co-evolved scaffold removal is novel. RuscaRL (2508.16949) decays rubrics during training, not after.
- **Diversity collapse is real:** DRA-GRPO (2505.09655), EVOL-RL (2509.15194), DiverseGRPO (2512.21514), GAPO (EMNLP 2025).
- **Prompt augmentation works for GRPO:** arXiv:2602.03190 -- but uses static augmentation, not outcome-driven evolution.
- **Single model accepted at top venues:** TTRL (NeurIPS 2025), ACE (well-cited preprint).
- **Benchmarks:** AIME 2024 baseline 55.5%, SOTA 70.8% for 7B (BP-Math-7B, 2503.21380). OlymMATH (2503.21380) and MathArena (2505.23281) provide non-saturated alternatives.
- **Theoretical foundation:** McClelland et al. (1995), Complementary Learning Systems. Vygotsky ZPD/scaffolding theory (1930s).
