# Sources: Publishability Assessment for Memory × RL Experiment

**Research Question**: Is the proposed 5-condition experiment studying memory × RL interaction for LLM reasoning publishable?

**Experiment Overview**: 5 conditions on DeepSeek-R1-Distill-Qwen-7B (B: GRPO-only, C: GRPO + static playbook, D: GRPO + evolving playbook, D': D + 10 anchor problems, E: GRPO + EVOL-RL novelty). Primary: MATH-500. Secondary: AIME 2024. 5 seeds each.

---

## 1. Novelty Gap: Context-Space × Weight-Space Interaction

### CG-TTRL: Context-Guided Test-Time RL
**Insight**: Efficiently adds context to TTRL, but test-time only, not training-time
**Citation**: CG-TTRL: Context-guided Test-time Reinforcement Learning for, arXiv:2511.06430
**Lindy**: ★★★★☆ (Published 2025, directly relevant to context+RL)
**Relevance**: CLOSE COMPETITOR. Studies context + RL but only at test-time, not during training. Proposed experiment studies training-time interaction, which is novel.

### Memory-R1: Memory Management via RL
**Insight**: Two agents (Manager + Answer) fine-tuned with PPO/GRPO for memory operations
**Citation**: Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning, arXiv:2508.19828
**Lindy**: ★★★★☆ (Published August 2025, strong empirical results)
**Relevance**: VERY CLOSE. Memory-R1 uses GRPO to learn memory operations (ADD/UPDATE/DELETE), achieving +28.5% F1 on LLaMA-3.1-8B. Proposed experiment differs by studying playbook evolution during reasoning training, not just memory CRUD operations.

### Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory
**Insight**: LLMs with persistent, evolving memory at inference via retrieval & synthesis
**Citation**: Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory, arXiv:2504.07952
**Lindy**: ★★★★★ (April 2025, foundational for inference-time memory)
**Relevance**: BASELINE COMPARISON. Dynamic Cheatsheet is inference-only. Proposed experiment trains with memory, not just uses it at test-time.

### ACE: Agentic Context Engineering
**Insight**: Evolving playbooks via Generator/Reflector/Curator prevent context collapse
**Citation**: Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models, arXiv:2510.04618
**Lindy**: ★★★★★ (October 2025, January 2026 v2, strong theoretical grounding)
**Relevance**: CORE MECHANISM. ACE provides the playbook evolution framework but does NOT study interaction with RL training. Proposed experiment is first to combine ACE with GRPO.

### Gap Assessment
**NOVELTY: STRONG.** No prior work studies playbook evolution DURING RL training. Existing work:
- TTRL/CG-TTRL: test-time only
- Memory-R1: memory CRUD ops, not strategy playbooks
- ACE: no RL, only inference-time evolution
- Dynamic Cheatsheet: no weight updates

**The proposed experiment bridges context-space learning (ACE) and weight-space learning (GRPO) during training, which is unexplored.**

---

## 2. The Absorption Test: Scaffolding Removal

### RuscaRL: Rubric-Scaffolded RL with Decay
**Insight**: Gradual scaffolding removal via inter-step decay prevents overfitting
**Citation**: Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning, arXiv:2508.16949
**Lindy**: ★★★★☆ (August 2025, strong results on AIME/MATH-500)
**Relevance**: VERY CLOSE. RuscaRL removes rubric scaffolding during training via decay schedule. Proposed absorption test removes playbook AFTER training to measure internalization, which is conceptually similar but empirically distinct.

### SEAL: Self-Adapting LLMs
**Insight**: LLMs generate self-edits to update weights, +15% accuracy on QA
**Citation**: Self-Adapting Language Models, arXiv:2506.10943, MIT 2025
**Lindy**: ★★★★☆ (June 2025, MIT research, strong empirical)
**Relevance**: PARALLEL APPROACH. SEAL studies context-to-weight transfer via self-edits, but uses RL to learn edit generation, not to study absorption of external memory.

### ICL-to-Weights Research
**Insight**: LLMs can internalize ICL examples into weights via synthetic data
**Citation**: Teaching large language models how to absorb new knowledge, MIT News 2025-11-12
**Lindy**: ★★★☆☆ (MIT news article, not peer-reviewed paper)
**Relevance**: CONCEPTUAL SUPPORT. SEAL framework shows ICL-to-weight transfer is possible, but no prior work measures playbook absorption via post-training removal test.

### TTT-E2E: Context Compression into Weights
**Insight**: Test-time training compresses context into weights via next-token prediction
**Citation**: Reimagining LLM Memory: Using Context as Training Data Unlocks Models That Learn at Test-Time, NVIDIA Technical Blog
**Lindy**: ★★★☆☆ (NVIDIA blog, not peer-reviewed)
**Relevance**: MECHANISM VALIDATION. TTT-E2E shows context can be compressed into weights, supporting absorption hypothesis.

### Gap Assessment
**NOVELTY: MODERATE-STRONG.** RuscaRL does gradual scaffolding removal DURING training. Proposed absorption test removes playbook AFTER training to measure what was internalized. This is a distinct empirical question: "Does RL training with evolving playbooks internalize strategies into weights?"

**No prior work directly measures this via post-training ablation.**

---

## 3. Venue Fit: Acceptance Standards

### ICML 2026
**Standards**: Original + rigorous research, significant interest, all claims supported by reproducible experiments or sound theory. 8 pages main paper + unlimited appendix.
**Citation**: ICML 2026 Call for Papers, https://icml.cc/Conferences/2026/CallForPapers
**Lindy**: ★★★★★ (Official venue)
**Relevance**: VENUE TARGET. ICML accepts experimental papers with well-designed experiments and adequate evidence for central claims.

### NeurIPS 2026
**Standards**: Reproducibility emphasized via checklist. Code/data/instructions recommended but not required unless central to contribution.
**Citation**: NeurIPS Paper Checklist Guidelines, https://neurips.cc/public/guides/PaperChecklist
**Lindy**: ★★★★★ (Official venue)
**Relevance**: VENUE TARGET. NeurIPS values reproducibility and soundness over scale.

### EMNLP 2025: 7B Models Accepted
**Example**: ThinkSLM 7B surpassed GPT-3.5-Turbo on logical reasoning benchmarks
**Citation**: THINKSLM: Towards Reasoning in Small Language Models, EMNLP 2025, aclanthology.org/2025.emnlp-main.1659.pdf
**Lindy**: ★★★★☆ (Accepted EMNLP 2025)
**Relevance**: SCALE VALIDATION. 7B models are acceptable at top-tier NLP venues if contributions are strong.

### NeurIPS 2025 Workshop: 7B Models Accepted
**Example**: Faithful Reasoning via Intervention Training (FRIT) applied DPO to Qwen3-8B and Mistral-7B-v0.1
**Citation**: Accepted papers, FoRLM @ NeurIPS'25, https://reasoning-workshop.github.io/accepted/
**Lindy**: ★★★★☆ (NeurIPS 2025 workshop)
**Relevance**: SCALE VALIDATION. 7B models are common at reasoning workshops.

### Gap Assessment
**VENUE FIT: STRONG.**
- ICML/NeurIPS accept experimental papers with well-designed experiments
- 7B models are acceptable if contribution is strong
- 5 conditions × 5 seeds = 25 runs per benchmark is respectable
- **CONCERN**: MATH-500 is saturated (92%+ baseline). AIME 2024 is more challenging but proposed experiment needs BOTH benchmarks to be convincing.

---

## 4. Closest Competitors

### 1. Memory-R1 (arXiv:2508.19828)
**How it differs**: Memory-R1 uses GRPO to learn memory CRUD operations (ADD/UPDATE/DELETE/NOOP), not to study playbook evolution during reasoning training. No absorption test.

### 2. ACE (arXiv:2510.04618)
**How it differs**: ACE evolves playbooks at inference only, no RL training. No study of weight-space × context-space interaction.

### 3. RuscaRL (arXiv:2508.16949)
**How it differs**: RuscaRL uses rubric scaffolding with gradual decay DURING training. Proposed experiment removes playbook AFTER training to measure absorption.

### 4. EVOL-RL (arXiv:2509.15194)
**How it differs**: EVOL-RL uses novelty reward to prevent cognitive collapse, achieving +11.8% on AIME25 pass@1 (4.6%→16.4% for Qwen3-4B). No memory/playbook mechanism. Proposed Condition E is direct comparison.

### 5. SEAL (arXiv:2506.10943)
**How it differs**: SEAL uses RL to learn self-edit generation for weight updates, not to study playbook absorption during reasoning training.

**COMPETITIVE LANDSCAPE**: Proposed experiment sits at intersection of Memory-R1 (GRPO + memory), ACE (playbook evolution), RuscaRL (scaffolding removal), and EVOL-RL (novelty reward). No single prior work combines all elements.

---

## 5. Fatal Flaw Search

### Flaw 1: MATH-500 Saturation
**Evidence**: DeepSeek-R1-Distill-Qwen-7B achieves 92.8% on MATH-500
**Citation**: DeepSeek-R1-Distill-Qwen-7B, HuggingFace Model Card, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
**Lindy**: ★★★★★ (Official model card, January 2025)
**Severity**: HIGH. Baseline is already 92.8%, leaving only 7.2% headroom. Improvements may be noise.
**Mitigation**: Focus on AIME 2024 as primary benchmark (55.5% baseline, more headroom).

### Flaw 2: 5 Seeds Insufficient
**Evidence**: Research using only 5 seeds may not be sufficient for broader generalizations
**Citation**: Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models, arXiv:2503.07329
**Lindy**: ★★★☆☆ (March 2025)
**Severity**: MODERATE. 5 seeds is common but not ideal. 10+ seeds would be better.
**Mitigation**: Use paired t-tests, bootstrap confidence intervals, or increase to 10 seeds if compute allows.

### Flaw 3: Playbook Degradation/Poisoning
**Evidence**: ACE and DC may degrade without reliable feedback; memory poisoning attacks exist
**Citation 1**: Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models, arXiv:2510.04618 (degradation without ground truth)
**Citation 2**: Memory Poisoning Attack and Defense on Memory Based LLM-Agents, arXiv:2601.05504
**Lindy**: ★★★★☆ (January 2025)
**Severity**: MODERATE. Condition D' (10 anchor problems) addresses this by providing ground truth, but may not be sufficient.
**Mitigation**: Report playbook quality metrics (e.g., diversity, coherence) over training. Include de-duplication analysis.

### Flaw 4: ACE Mechanism Definition
**Evidence**: ACE paper defines Generator/Reflector/Curator but implementation details vary
**Citation**: Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models, arXiv:2510.04618
**Lindy**: ★★★★★ (Official ACE paper)
**Severity**: LOW. ACE is well-defined in paper and has GitHub implementation.
**Mitigation**: Use official ACE implementation from https://github.com/ace-agent/ace

### Flaw 5: DeepSeek-R1-Distill-Qwen-7B Not Standard Base
**Evidence**: Most research uses LLaMA, Qwen, or Mistral base models, not distilled reasoning models
**Citation**: N/A (general observation)
**Severity**: LOW-MODERATE. DeepSeek-R1-Distill is already reasoning-trained via distillation from R1, which may confound results.
**Mitigation**: Justify choice as "strong reasoning baseline" or consider adding ablation with standard Qwen-7B base.

### Flaw 6: MATH-500 Saturation at Top Venues
**Evidence**: MATH-500 no longer run on new model releases due to saturation
**Citation**: Reinforcement Learning for Reasoning in Large [Language Models], arXiv:2504.20571
**Lindy**: ★★★★☆ (April 2025)
**Severity**: HIGH. Reviewers may flag MATH-500 as insufficient benchmark.
**Mitigation**: Make AIME 2024 primary benchmark, add OlymMATH or MathArena if possible.

---

## 6. BAPO/AceMath Context: SOTA on MATH-500

### AceMath-72B-Instruct
**Result**: Greatly outperforms Qwen2.5-Math-72B-Instruct, GPT-4o, Claude-3.5 Sonnet on math benchmarks
**Citation**: AceMath: Advancing Frontier Math Reasoning with Post-Training and Reward Modeling, arXiv:2412.15084
**Lindy**: ★★★★☆ (December 2024)
**Relevance**: SOTA CONTEXT. AceMath is 72B model, not 7B, so not directly comparable.

### BAPO: Boundary-Aware Policy Optimization
**Result**: Outperforms mainstream training/prompt-based methods in reliability across 4 benchmarks
**Citation**: BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search, arXiv:2601.11037
**Lindy**: ★★★★☆ (January 2025)
**Relevance**: ORTHOGONAL. BAPO focuses on reliability (IDK detection), not accuracy. Different research question.

### DeepSeek-R1-Distill-Qwen-7B Baseline
**MATH-500**: 92.8%
**AIME 2024**: 55.5%
**Citation**: DeepSeek-R1-Distill-Qwen-7B, HuggingFace Model Card
**Lindy**: ★★★★★ (Official)
**Relevance**: CRITICAL. This is the baseline. Expected improvements:
- MATH-500: 92.8% → ~95%? (only 7.2% headroom, gains may be <3%)
- AIME 2024: 55.5% → ~65-70%? (more headroom, gains could be 10-15%)

### AIME 2024 SOTA (7B models)
**BP-Math-7B**: 70.8 (AIME 2024)
**Citation**: Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models, arXiv:2503.21380
**Lindy**: ★★★★☆ (March 2025)
**Relevance**: COMPETITIVE TARGET. Proposed experiment needs to reach ~65-70% on AIME 2024 to be competitive with 7B SOTA.

### Gap Assessment
**SOTA CONTEXT**:
- MATH-500 is saturated; baseline already 92.8%
- AIME 2024 has more headroom (55.5% baseline, 70.8% SOTA for 7B)
- **Proposed experiment is NOT trivial if it shows clear absorption effect and reaches ~65-70% on AIME 2024**
- BAPO/AceMath target different research questions (reliability, 72B scale)

---

## Summary: Publishability Assessment

### Strengths
1. **Novelty**: First to study playbook evolution DURING RL training (context × weight interaction)
2. **Absorption test**: Novel empirical method (post-training playbook removal)
3. **Clean experimental design**: 5 conditions isolate factors (RL-only, static, evolving, anchor, novelty)
4. **Timely**: Builds on recent work (ACE, Memory-R1, EVOL-RL, RuscaRL) from 2025
5. **Scale acceptable**: 7B models are accepted at top venues if contribution is strong

### Weaknesses
1. **MATH-500 saturated**: 92.8% baseline, only 7.2% headroom. Reviewers will flag this.
2. **5 seeds may be insufficient**: 10+ seeds would strengthen reproducibility claims
3. **Playbook degradation**: Condition D may suffer from poisoning without reliable feedback (D' addresses this but needs analysis)
4. **DeepSeek-R1-Distill base**: Already reasoning-trained via distillation, may confound results
5. **Missing SOTA benchmark**: Should add OlymMATH or MathArena to avoid saturation critique

### Recommended Actions
1. **Make AIME 2024 primary benchmark** (not MATH-500)
2. **Add OlymMATH or MathArena** as third benchmark
3. **Increase to 10 seeds** if compute allows (or bootstrap CIs)
4. **Report playbook quality metrics** (diversity, de-duplication, coherence) to address poisoning concern
5. **Add ablation with standard Qwen-7B base** to show effect is not DeepSeek-R1-Distill-specific
6. **Target**: ICML 2026 main conference or NeurIPS 2026 workshop (if results are mixed)

### Verdict
**PUBLISHABLE at workshop (HIGH confidence), main conference (MODERATE confidence).**

**Workshop-level** (NeurIPS/ICML 2026 workshops): Very strong fit. Novel question, clean design, 7B scale acceptable.

**Main conference** (ICML/NeurIPS 2026): Publishable IF:
- Results show clear absorption effect (e.g., Condition D drops >10% after playbook removal)
- AIME 2024 improvements are significant (e.g., 55%→65%+)
- Playbook quality metrics show D' prevents degradation
- Add third non-saturated benchmark

**Risk**: If MATH-500 improvements are <3% and AIME 2024 improvements are <5%, reviewers may reject as "incremental" or "noisy."

---

## Key Citations for Paper

### Foundational Work
- ACE (arXiv:2510.04618): Playbook evolution mechanism
- Memory-R1 (arXiv:2508.19828): GRPO + memory management
- EVOL-RL (arXiv:2509.15194): Novelty reward to prevent collapse
- RuscaRL (arXiv:2508.16949): Scaffolding removal during training

### Comparison Baselines
- Dynamic Cheatsheet (arXiv:2504.07952): Test-time adaptive memory
- CG-TTRL (arXiv:2511.06430): Context-guided test-time RL
- SEAL (arXiv:2506.10943): Context-to-weight transfer

### Benchmarks
- AIME Benchmarks (AIME 2024/2025): High-difficulty math reasoning
- MATH-500 (saturated): Standard math benchmark
- OlymMATH (recommended): Non-saturated alternative

### Statistical Rigor
- Random Seeds in Fine-Tuning (arXiv:2503.07329): 5 seeds may be insufficient

---

**Lindy Rating Scale**:
- ★★★★★ = Official publication, widely cited, foundational
- ★★★★☆ = Peer-reviewed or strong venue, recent, high impact
- ★★★☆☆ = Preprint or blog, credible but not peer-reviewed
- ★★☆☆☆ = Early preprint or low citation count
- ★☆☆☆☆ = Unverified or speculative

**Total sources reviewed**: 38
**Date compiled**: 2026-02-09
