# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Environment

Stanford FarmShare shared cluster (rice-XX nodes). Home directories are NFS-mounted.

### Critical Rules

- **NEVER run tests interactively on the login node.** Always submit compute-heavy work (tests, training, builds) as SLURM batch jobs.
- After spawning any long-running process, track the PID so it can be cleaned up if the session dies.
- Use `timeout` when running tests to prevent zombie processes (e.g., `timeout 300 pytest ...`).
- **Always commit changes when task is complete.** Do not include `Co-Authored-By` lines.

---

## Research Focus: Search-Augmented ACE

This repository investigates **verification-driven reasoning** and **self-improving memory mechanisms** for LLMs. The core framework is ACE (Agentic Context Engineering) enhanced with search algorithms for playbook evolution.

### Key Research Questions

1. **Verified Archetype Discovery**: Can we identify problems that are both generalizable (strategies transfer broadly) and verifiable (correctness can be confidently assessed)?
2. **Confidence-Guided Curriculum**: Can contextual bandits balance exploitation (apply high-confidence strategies) with exploration (expand coverage)?
3. **Search vs. Greedy Curation**: Do bandit/tree search algorithms outperform greedy generate-reflect-curate?

### Current Experiments

| Experiment | Domain | Status |
|------------|--------|--------|
| `notebooks/search_augmented_ace_poc.ipynb` | GSM8K (50 problems) | Complete - 14 conditions ablated |
| `notebooks/verified_archetype_discovery_poc.ipynb` | Game of 24 | Complete - archetype pipeline verified |
| `experiments/dc_vs_verification/` | AIME 2024 (30 problems) | Complete - 4 conditions × 3 seeds |

### Preliminary Results (Search-Augmented ACE on GSM8K)

| Strategy | Accuracy (Last 20) | Budget |
|----------|-------------------|--------|
| Beam Search | 90.00% | 310 |
| UCB Bandit | 85.00% | 105 |
| Majority Vote | 85.00% | 100 |
| PUCT-Bayesian | 80.00% | 109 |
| Greedy ACE | 75.00% | 110 |
| Thompson | 70.00% | 105 |

**Key Finding**: Simple baselines (UCB, Majority Vote) achieve competitive performance at standard budget. Sophisticated search (PUCT variants) plateau at 75-80%, suggesting the bottleneck is coverage of strategy space, not search sophistication.

### DC vs Verification Results (AIME 2024)

| Condition | Accuracy | Budget |
|-----------|----------|--------|
| DC + Outcome Verification | 17.8% | N=8 + verify |
| Dynamic Cheatsheet (DC) | 16.7% | N=8 |
| Self-Consistency (N=16) | 14.4% | N=16 |
| Baseline (CoT Pass@1) | 13.3% | N=1 |

**Key Finding**: Verification adds only +1.1% over DC alone (Path A: verifier over-engineered). DC vs Self-Consistency gap is +2.2% (compute-neutral). Both DC variants degrade over time (27%→17%), indicating playbook poisoning. Oracle analysis shows ~10% selection gap — correct answers exist among candidates but aren't selected.

### Verified Archetype Discovery Results (Game of 24)

**Key Finding**: 4 curated archetypes = 3.6x improvement (3%→11%). 10 archetypes = back to baseline. The verifier was the bottleneck (rejected all archetypes at threshold 0.3). Concept validated but automated verification pipeline needs calibration.

---

## ACE Framework

Framework for LLM self-improvement via evolving "playbooks" — structured knowledge bases of strategies/formulas/insights. Three-agent architecture:

- **Generator**: Produces chain-of-thought answers using playbook context
- **Reflector**: Analyzes outputs and tags playbook bullets as helpful/harmful
- **Curator**: Evolves playbook via ADD/UPDATE/MERGE/DELETE operations

### Commands

```bash
cd ace && pip install -r requirements.txt
python -m eval.finance.run --task_name finer --mode offline --save_path results
```

### Playbook Format

```
## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=0 :: Always verify data types before processing

## FORMULAS & CALCULATIONS
[calc-00003] helpful=8 harmful=0 :: NPV = Σ(Cash Flow / (1+r)^t)
```

Curator operations: ADD, UPDATE, MERGE, DELETE. Token budget enforced (~80K default).

### Extending to New Domains

Implement a `DataProcessor` with 3 methods:
```python
class DataProcessor:
    def process_task_data(self, raw_data) -> List[Dict]:  # → {"context","question","target"}
    def answer_is_correct(self, predicted, ground_truth) -> bool
    def evaluate_accuracy(self, predictions, ground_truths) -> float
```

---

## Repository Structure

```
grounded/
├── notebooks/           # Primary research notebooks (PoC experiments)
├── ace/                 # ACE framework (dependency)
├── src/                 # Experimental infrastructure
├── docs/
│   ├── specs/           # Experiment specifications
│   ├── discoveries/     # Design docs and logs
│   └── research/        # Knowledge bases
├── KNOWLEDGE.md         # Technical insights and learnings
├── DIRECTIONS.md        # Viable research directions
└── PROPOSAL.md          # Top research proposals
```

---

## Technical Notes

See `KNOWLEDGE.md` for detailed technical insights on:
- MCTS/PUCT implementation details
- Thompson Sampling and bandit algorithms
- vLLM GPU optimization patterns
- Async parallelization strategies

### Gotchas Discovered

- **vLLM/torch version chain**: vLLM pins exact torch versions (0.12.0→torch 2.9.0, 0.10.2→torch 2.8.0). Never pin `torch` independently when using vLLM — let vLLM's dependency pull the correct version.
- **`re.findall` with capturing groups**: `re.findall(r"\[(str|err)-\d{5}\]", text)` returns `['str']` not `['str-00001']`. Use non-capturing inner group: `r"\[((?:str|err)-\d{5})\]"` to capture the full match minus brackets.
- **GRPO confidence-weighted rewards**: Soft rewards (0.0-1.0 from majority vote confidence) dilute gradient signal vs binary 1.0/0.0. At low confidence (~1/N), reward std becomes tiny → advantage explosion. TTRL paper uses binary rewards; stick with binary.
- **flash-attn wheels**: No prebuilt wheels for torch >= 2.7. Omit flash-attn when using vLLM ≥ 0.12 (it handles attention internally). If needed, build from source with `--no-build-isolation`.
