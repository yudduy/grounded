# Specification: Search-Augmented ACE PoC (MCTS/ES for Math Reasoning)

> Use `/duy-workflow:execute docs/specs/search-augmented-ace-poc.spec.md` to implement.

## Goal

Build a fully self-contained Colab notebook that compares three search strategies for evolving ACE-style playbooks on GSM8K math reasoning: **Greedy ACE** (sequential baseline), **PUCT-guided ACE** (tree search), and **ES-ACE** (evolutionary strategy with population), using a local Qwen2.5-7B model via vLLM on A100.

## Requirements

### Infrastructure

1. **[REQ-INFRA-1]** Single self-contained `.ipynb` notebook. No imports from this repo. All code in cells.
   - Acceptance: Copy notebook to fresh Colab A100 runtime → runs end-to-end.

2. **[REQ-INFRA-2]** Local LLM via vLLM serving Qwen2.5-7B-Instruct.
   - Setup cell: `pip install vllm`, launch vLLM server as background process, OpenAI-compatible client.
   - Acceptance: `client.chat.completions.create(...)` returns valid response.

3. **[REQ-INFRA-3]** GSM8K dataset loaded from HuggingFace `datasets` library.
   - Use `gsm8k` test split. Parse final numeric answer from `#### <number>` format.
   - Acceptance: 50 problems loaded with extracted ground truth answers.

### Core Algorithm

4. **[REQ-CORE-1]** **Playbook representation**: Structured text with numbered bullets, each having `helpful`/`harmful` counts. Sections: `STRATEGIES`, `COMMON MISTAKES`, `SOLUTION PATTERNS`.
   - Acceptance: Playbook can be serialized to string, parsed back, bullets updated.

5. **[REQ-CORE-2]** **Generator**: Given a math problem + playbook, produce a chain-of-thought solution and extract numeric answer. Track which playbook bullets were referenced.
   - Acceptance: Returns `(answer: str, bullets_used: List[str], raw_response: str)`.

6. **[REQ-CORE-3]** **Reflector**: Given problem, solution trace, predicted/actual answer, and bullets used → produce reflection text + tag each bullet as helpful/harmful/neutral.
   - Acceptance: Returns `(reflection: str, bullet_tags: Dict[str, str])`.

7. **[REQ-CORE-4]** **Curator**: Given current playbook + reflection + bullet stats → return updated playbook via ADD/UPDATE/DELETE operations.
   - Acceptance: Playbook changes are applied; bullet count stays within token budget (max 20 bullets for PoC).

### Search Strategies (3 conditions)

8. **[REQ-SEARCH-1]** **Greedy ACE** (baseline): Sequential loop. For each problem, generate → evaluate → reflect → curate. Playbook evolves linearly.
   - Budget: 1 LLM generation attempt per problem. Curate every 5 problems.
   - Acceptance: Processes N problems sequentially, tracks accuracy over time.

9. **[REQ-SEARCH-2]** **PUCT-ACE** (tree search over playbook states): Maintain a tree of playbook versions. Each node = playbook state. PUCT selects which playbook to expand next.
   - For each expansion: pick playbook via PUCT → solve K=3 problems → score = accuracy on those problems → reflect+curate → create child node with new playbook.
   - PUCT formula: `Q(s) + c_puct * P(s) * sqrt(N_parent) / (1 + N(s))` where Q = mean accuracy of descendants, P = 1/num_children (uniform prior), c_puct = 1.0.
   - Budget: Same total LLM calls as Greedy (controlled by total problems solved).
   - Acceptance: Tree has >1 branch; best leaf playbook selected for final eval.

10. **[REQ-SEARCH-3]** **ES-ACE** (evolutionary strategy): Population of K=4 playbooks. Each generation: evaluate each playbook on a batch of M=3 problems, select top-2 (tournament), curate/mutate to produce next generation.
    - Budget: Same total LLM calls as Greedy.
    - Acceptance: Population diversity maintained (playbooks differ); best-of-population selected.

### Evaluation & Analysis

11. **[REQ-EVAL-1]** **Controlled comparison**: All 3 strategies use the same 50 GSM8K problems, same random seed, same total LLM generation budget (~150 generation calls each, ~500 total across 3 conditions + overhead for reflect/curate).
    - Acceptance: Total LLM calls per condition documented and approximately equal.

12. **[REQ-EVAL-2]** **Metrics tracked per condition**:
    - Running accuracy (correct/total so far) plotted over problems solved.
    - Final accuracy on last 20 problems (after playbook has evolved).
    - Playbook size over time (number of bullets).
    - Total LLM calls breakdown (generate vs reflect vs curate).
    - Acceptance: All 4 metrics computed and stored.

13. **[REQ-EVAL-3]** **Plots** (matplotlib):
    - (a) Running accuracy curves for all 3 conditions on one plot.
    - (b) Bar chart: final accuracy comparison with error bars (bootstrap CI).
    - (c) Playbook evolution: bullet count over time per condition.
    - (d) Cost breakdown: stacked bar of LLM calls by type per condition.
    - Acceptance: 4 publication-quality plots generated in notebook.

14. **[REQ-EVAL-4]** **Statistical test**: Bootstrap confidence intervals on final accuracy difference between conditions. Report if PUCT or ES significantly beats Greedy.
    - Acceptance: 95% CI printed for each pairwise comparison.

### Notebook Structure

15. **[REQ-STRUCT-1]** Notebook cells organized as:
    ```
    1. Setup & Dependencies (vLLM install, model download, imports)
    2. GSM8K Data Loading & Parsing
    3. Core Components (Playbook, Generator, Reflector, Curator)
    4. Search Strategies (Greedy, PUCT, ES) — each as a class
    5. Experiment Runner (runs all 3 conditions)
    6. Analysis & Plotting
    7. Results Summary (markdown cell with findings)
    ```
    - Acceptance: Section headers in markdown cells match this structure.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM | Qwen2.5-7B-Instruct via vLLM | Strong math, runs on A100, no API costs |
| Dataset | GSM8K test (50 problems) | Fast iteration, clear correctness signal |
| Playbook max bullets | 20 | Fits in context window, fast to parse |
| PUCT c_puct | 1.0 | Standard default, tunable |
| ES population | 4 | Minimal but shows diversity benefit |
| Budget control | Count generation calls (not reflect/curate) | Generation is the "work"; reflect/curate is "overhead" |
| Answer extraction | Regex for `#### <number>` pattern | GSM8K standard format |
| Playbook prior P(s) | Uniform over children | Simplest; LLM prior is future work |

## Completion Criteria

- [ ] Notebook runs end-to-end on Colab A100 without errors
- [ ] All 3 conditions produce accuracy curves
- [ ] 4 plots generated with clear labels and legend
- [ ] Statistical comparison printed
- [ ] Markdown summary cell interprets results

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| vLLM OOM on A100 | Use `--max-model-len 4096` and `--gpu-memory-utilization 0.85` |
| LLM returns unparseable answer | Count as incorrect, log warning |
| Curator produces empty playbook | Reset to initial template playbook |
| GSM8K answer has units/formatting | Strip non-numeric chars, compare floats with tolerance |
| vLLM server takes long to start | Retry loop with 60s timeout, clear status messages |

## Technical Context

### Key References
- Dynamic Cheatsheet: sequential playbook evolution (Generate → Curate loop)
- ACE: structured playbook with Generator/Reflector/Curator + bullet tagging
- TTT-Discover: PUCT with max-Q (not mean-Q), entropic objective
- AlphaEvolve: island-based evolutionary search with LLM mutations

### Patterns to Follow
- Playbook bullet format: `[id-NNNNN] helpful=N harmful=N :: content text`
- Answer extraction: match `#### (\d+)` or final numeric in response
- vLLM OpenAI-compatible: `openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")`

### Files to Create
- `notebooks/search_augmented_ace_poc.ipynb` — the single deliverable
