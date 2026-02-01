# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Stanford FarmShare shared cluster (rice-XX nodes). Home directories are NFS-mounted.

### Critical Rules

- **NEVER run tests interactively on the login node.** Always submit compute-heavy work (tests, training, builds) as SLURM batch jobs.
- After spawning any long-running process, track the PID so it can be cleaned up if the session dies.
- Use `timeout` when running tests to prevent zombie processes (e.g., `timeout 300 pytest ...`).
- **Always commit changes when task is complete.** Do not include `Co-Authored-By` lines.

## Repository Structure

This repo contains three independent projects under `/home/users/duynguy/proj/grounded/`:

---

## ShinkaEvolve — LLM-Guided Evolutionary Algorithm Framework

Evolves populations of programs using LLMs as intelligent mutation operators. Multi-island evolutionary model with archive-based knowledge transfer.

### Commands

```bash
cd ShinkaEvolve && uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
shinka_launch variant=circle_packing_example          # Run evolution (Hydra)
shinka_visualize --port 8888 --open                   # WebUI
pytest tests/                                          # Tests
black shinka/ && isort shinka/ && flake8 shinka/       # Lint
```

**Configuration:** Hydra hierarchy under `configs/` — layers: task, evolution, database, cluster, variant. Override via CLI: `shinka_launch task=circle_packing evolution=large_budget cluster=remote`.

### Programmatic API

**EvolutionRunner** — main orchestrator (`shinka/core/runner.py`):
```python
from shinka.core.runner import EvolutionRunner, EvolutionConfig
from shinka.database.dbase import DatabaseConfig
from shinka.launch.scheduler import LocalJobConfig

runner = EvolutionRunner(
    evo_config=EvolutionConfig(
        num_generations=10, llm_models=["gpt-4o"], patch_types=["diff", "full"],
        task_sys_msg="Optimize circle packing...", init_program_path="initial.py",
    ),
    job_config=LocalJobConfig(eval_program_path="evaluate.py"),
    db_config=DatabaseConfig(num_islands=4, parent_selection_strategy="power_law"),
)
runner.run()  # Blocks until complete; auto-resumes if db_path exists
```

**LLMClient** — standalone LLM queries (`shinka/llm/llm.py`):
```python
from shinka.llm.llm import LLMClient
llm = LLMClient(model_names=["claude-3-sonnet"], temperatures=0.75, max_tokens=4096)
result = llm.query(msg="Write factorial function", system_msg="Python expert")
results = llm.batch_query(num_samples=5, msg="...", system_msg="...")
```
Supports: OpenAI, Anthropic, Azure (`"azure-{model}"`), DeepSeek, Gemini, Bedrock, OpenRouter (`"{provider}/{model}"`). Dynamic model selection via bandits (`llm_dynamic_selection="ucb"`).

**ProgramDatabase** — population storage (`shinka/database/dbase.py`):
```python
from shinka.database.dbase import ProgramDatabase, DatabaseConfig, Program
db = ProgramDatabase(DatabaseConfig(db_path="evolution.sqlite"), read_only=True)
best = db.get_best_program()                     # Best correct program
top = db.get_top_programs(n=10, correct_only=True)
parent, archive_insps, topk_insps = db.sample()  # Sample parent + inspirations
db.add(Program(id="p1", code="...", combined_score=0.95, correct=True))
```

**PromptSampler** — mutation prompt generation (`shinka/core/sampler.py`):
```python
from shinka.core.sampler import PromptSampler
sampler = PromptSampler(task_sys_msg="...", patch_types=["diff","full","cross"])
sys_msg, user_msg = sampler.initial_program_prompt()
sys_msg, user_msg, patch_type = sampler.sample(parent, archive_insps, topk_insps)
```

**Patch application** — pure functions (`shinka/edit/`):
```python
from shinka.edit.apply_diff import apply_diff_patch
from shinka.edit.apply_full import apply_full_patch
updated_code, n_applied, path, err, txt, diff = apply_diff_patch(patch_str, original_str=code)
```
Both only edit code inside `# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END` markers.

**JobScheduler** — execute evaluations (`shinka/launch/scheduler.py`):
```python
from shinka.launch.scheduler import JobScheduler, LocalJobConfig, SlurmCondaJobConfig
sched = JobScheduler(job_type="local", config=LocalJobConfig(), max_workers=4)
results, runtime = sched.run(exec_fname, results_dir)         # Sync
job = sched.submit_async(exec_fname, results_dir)              # Async
```
Job types: `"local"`, `"slurm_docker"` (SlurmDockerJobConfig), `"slurm_conda"` (SlurmCondaJobConfig).

### Composability

Independently usable: `LLMClient`, `PromptSampler`, `EmbeddingClient`, `apply_diff_patch`/`apply_full_patch`, `JobScheduler`, `ProgramDatabase` (read-only). Only `EvolutionRunner` is tightly coupled as the orchestrator.

---

## ACE — Agentic Context Engineering

Framework for LLM self-improvement via evolving "playbooks" — structured knowledge bases of strategies/formulas/insights. Three-agent architecture that preserves domain knowledge across iterations.

### Commands

```bash
cd ace && pip install -r requirements.txt
python -m eval.finance.run --task_name finer --mode offline --save_path results
python -m eval.finance.run --task_name finer --mode online --save_path results
python -m eval.finance.run --task_name finer --mode eval_only \
    --initial_playbook_path results/best_playbook.txt --save_path test_results
```

### Programmatic API

**ACE orchestrator** (`ace/ace.py`):
```python
from ace import ACE
ace = ACE(
    api_provider="openai",           # "sambanova", "together", "openai"
    generator_model="gpt-4",
    reflector_model="gpt-4",
    curator_model="gpt-4",
    max_tokens=4096,
    initial_playbook=None,           # Optional starting playbook string
    use_bulletpoint_analyzer=True,
    bulletpoint_analyzer_threshold=0.90,
)
results = ace.run(
    mode="offline",                  # "offline", "online", "eval_only"
    train_samples=train_data,        # List[Dict] with context/question/target
    val_samples=val_data,
    test_samples=test_data,
    data_processor=processor,        # Your DataProcessor instance
    config={
        'num_epochs': 2, 'max_num_rounds': 3, 'curator_frequency': 1,
        'eval_steps': 100, 'save_steps': 50, 'token_budget': 80000,
        'test_workers': 20, 'save_dir': './results',
    },
)
# results['training_results']['best_validation_accuracy']
# results['final_test_results']['accuracy']
```

**Generator** — answer production (`ace/core/generator.py`):
```python
from ace.core import Generator
gen = Generator(api_client, "openai", "gpt-4", max_tokens=4096)
response, bullet_ids, call_info = gen.generate(
    question="...", playbook=playbook_str, context="", reflection="(empty)"
)
# bullet_ids: list of playbook IDs used (e.g., ["str-00001", "calc-00003"])
```

**Reflector** — output analysis (`ace/core/reflector.py`):
```python
from ace.core import Reflector
ref = Reflector(api_client, "openai", "gpt-4")
reflection, bullet_tags, call_info = ref.reflect(
    question="...", reasoning_trace=response, predicted_answer="4",
    ground_truth="4", environment_feedback="Correct",
    bullets_used=formatted_bullets, use_ground_truth=True,
)
# bullet_tags: [{"id": "calc-00001", "tag": "helpful"}]
```

**Curator** — playbook evolution (`ace/core/curator.py`):
```python
from ace.core import Curator
cur = Curator(api_client, "openai", "gpt-4")
updated_playbook, next_id, operations, call_info = cur.curate(
    current_playbook=playbook_str, recent_reflection=reflection,
    question_context="...", current_step=1, total_samples=100,
    token_budget=80000, playbook_stats=stats, next_global_id=42,
)
# operations: [{"type": "ADD", "section": "...", "content": "...", "reason": "..."}]
```

**BulletpointAnalyzer** — semantic dedup (`ace/core/bulletpoint_analyzer.py`):
```python
from ace.core import BulletpointAnalyzer
analyzer = BulletpointAnalyzer(client, "gpt-4", embedding_model_name='all-mpnet-base-v2')
processed = analyzer.analyze(playbook_str, threshold=0.90, merge=True)
```

**Playbook utilities** (`playbook_utils.py`):
```python
from playbook_utils import parse_playbook_line, update_bullet_counts, apply_curator_operations, get_playbook_stats, extract_playbook_bullets
```

**LLM wrapper** (`llm.py`):
```python
from llm import timed_llm_call
response, call_info = timed_llm_call(client, "openai", "gpt-4", prompt, role="generator", call_id="gen-1")
```

### Extending to New Domains

Implement a `DataProcessor` with 3 methods (see `EXTENDING_ACE.md`):
```python
class DataProcessor:
    def process_task_data(self, raw_data) -> List[Dict]:  # → {"context","question","target"}
    def answer_is_correct(self, predicted, ground_truth) -> bool
    def evaluate_accuracy(self, predictions, ground_truths) -> float
```

### Playbook Format

```
## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=0 :: Always verify data types before processing

## FORMULAS & CALCULATIONS
[calc-00003] helpful=8 harmful=0 :: NPV = Σ(Cash Flow / (1+r)^t)
```

Curator operations: ADD, UPDATE, MERGE, DELETE. Token budget enforced (~80K default).

---

## Newton — GPU-Accelerated Physics Simulation Engine

Built on NVIDIA Warp + MuJoCo Warp. Linux Foundation project (v0.2.0 beta).

### Commands

```bash
cd newton && uv sync --extra examples
uv run -m newton.examples basic_pendulum
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests.test_examples -k test_basic.example_basic_shapes
uvx pre-commit run -a                                  # Lint (Ruff + Typos)
```

### Programmatic API

**Core simulation loop pattern:**
```python
import newton

# Build scene
builder = newton.ModelBuilder()
body = builder.add_body(xform=..., key="box")
builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
joint = builder.add_joint_revolute(parent=-1, child=body, axis=(0,0,1))
builder.add_articulation([joint])
builder.add_ground_plane()
model = builder.finalize()

# Create dynamics objects
state_0, state_1 = model.state(), model.state()
control = model.control()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

# Solver + collision
solver = newton.solvers.SolverXPBD(model, iterations=10)
collision_pipeline = newton.CollisionPipelineUnified()

# Step loop
for frame in range(num_frames):
    state_0.clear_forces()
    contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
    for _ in range(substeps):
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
```

**Key classes:**
- `ModelBuilder` — scene construction: `add_body()`, `add_shape_*()` (sphere/box/capsule/cylinder/mesh/plane), `add_joint_*()` (revolute/prismatic/ball/fixed/free/d6/distance/cable), `add_articulation()`, `add_particle()`, `add_spring()`, `finalize()`
- `Model` — immutable scene: `state()`, `control()`, `collide()`, `set_gravity()`
- `State` — time-varying: `particle_q/qd/f`, `body_q/qd/f`, `joint_q/qd`, `clear_forces()`
- `Control` — inputs: `joint_f`, `joint_target_pos/vel`, `tri_activations`, `clear()`
- `Contacts` — collision data: `rigid_contact_count`, `rigid_contact_point0/1`, `rigid_contact_normal`

**Solvers** (`newton.solvers`): `SolverXPBD` (general), `SolverVBD` (cloth), `SolverFeatherstone` (articulated), `SolverMuJoCo` (MuJoCo backend), `SolverSemiImplicit`, `SolverStyle3D`, `SolverImplicitMPM`. All share `solver.step(state_0, state_1, control, contacts, dt)`.

**Other APIs:**
- `newton.geometry` — collision detection (`BroadPhaseSAP`, `collide_*` functions), mesh utilities, SDF, terrain generation
- `newton.sensors` — `SensorContact`, `SensorIMU`, `SensorRaycast`, `SensorTiledCamera`, `SensorFrameTransform`
- `newton.ik` — `IKSolver` with `IKPositionObjective`, `IKRotationObjective`, `IKJointLimitObjective`
- `newton.viewer` — `ViewerGL`, `ViewerRerun`, `ViewerUSD`, `ViewerViser`, `ViewerNull`
- `newton.usd` — USD file import/export utilities

### Key Rules

- `newton/_src/` is **internal only** — never import from it in user code or examples
- Any user-facing class in `_src` must be re-exported via public modules
- Prefix-first naming: `ActuatorPD` not `PDActuator`, `add_shape_sphere()` not `add_sphere_shape()`
- `snake_case` methods, `kebab-case` CLI args, Google-style docstrings
- Run `uvx pre-commit run -a` before committing
- New examples: follow `Example` class pattern, implement `test_final()`, register in `README.md`

---

## Cross-Project Integration: ShinkaEvolve + ACE Hybrid

The two frameworks are complementary and composable at specific interfaces:

### Where ACE fits inside ShinkaEvolve

**ACE's playbook as ShinkaEvolve's task system message.** ShinkaEvolve's `EvolutionConfig.task_sys_msg` is the prompt that guides LLM mutations. Instead of a static string, use an ACE playbook that evolves based on which mutation strategies produce better programs:

```python
# ACE's Reflector tags which mutation strategies (playbook bullets) led to
# score improvements; Curator evolves the task_sys_msg between generations
reflector.reflect(
    question=f"Mutate program for {task}",
    reasoning_trace=llm_patch_output,
    predicted_answer=str(child_score),
    ground_truth=str(parent_score),  # or best-known score
    environment_feedback="Improved" if child_score > parent_score else "Regressed",
    bullets_used=extracted_strategy_bullets,
)
```

**Integration points:**
1. **PromptSampler ↔ ACE Generator**: Replace static prompt templates with playbook-conditioned generation. ACE's Generator already formats prompts with playbook context.
2. **Program.text_feedback ↔ ACE Reflector**: ShinkaEvolve already stores `text_feedback` per program. Feed this to ACE's Reflector for structured tagging.
3. **Meta-recommendations ↔ ACE Curator**: ShinkaEvolve has `meta_rec_interval` for periodic strategy updates via a meta-LLM. ACE's Curator is a more structured version of this — replaces the meta-LLM with ADD/UPDATE/MERGE/DELETE operations on a playbook.
4. **ProgramDatabase ↔ ACE playbook persistence**: Store playbook snapshots alongside generations in SQLite. Track playbook version that produced each program.

### Where ShinkaEvolve fits inside ACE

**Evolve ACE's prompts/playbook structure.** Use ShinkaEvolve to optimize the prompt templates in `ace/prompts/` or the playbook section structure itself:

```python
# The "program" being evolved is the ACE prompt template or playbook schema
# The "evaluation" is ACE's validation accuracy on a held-out set
evo_config = EvolutionConfig(
    task_sys_msg="Improve the ACE generator prompt template...",
    init_program_path="ace/prompts/generator.py",
)
```

### Newton as evaluation environment

Newton can serve as the evaluation function for either framework when evolving robot controllers or physics algorithms:

```python
# ShinkaEvolve evaluates evolved programs by running them in Newton
# JobScheduler.run() executes a script that: builds scene → runs policy → returns score
```

## Session Notes

- **Thompson Sampling in search_augmented_ace_poc.ipynb**: Budget docstring was wrong (said 99, actual is 105 calls). Fixed. The Beta-Bernoulli conjugate update and argmax-of-samples selection are textbook correct. Key limitation: fixed pool after seed phase means no online adaptation. Consider discounted TS (gamma=0.95) if bullet tag drift matters, and dynamic arm addition for longer runs.
- **PUCT backprop bug**: When expanding a node, `backprop(child, reward)` gave the child credit for reward earned by the parent's playbook evaluation. Fixed: always backprop to the leaf (the node whose playbook was actually evaluated). New children start with optimistic Q=0.5 prior.
- **PUCT Bayesian Q-estimator**: Uses `(s+1)/(n+2)` which is correct Beta posterior mean for binary rewards, but reward_history contains fractional batch accuracies (e.g., 0.67). Still works as a shrinkage estimator but isn't a proper Beta posterior. Document this if publishing.
- **Progressive widening k=1**: Standard PW uses `k * N^alpha` with k=10. Our k=1 is deliberately conservative since each expansion costs a curate call. This is a justified design choice, not a bug.
- **Frontier strategies to consider**: AB-MCTS (adaptive wider-vs-deeper branching, Sakana AI 2025), Process Reward Models for compute-optimal tree search (ICLR 2025), Fetch state-merging for dedup of semantically similar playbook nodes, OPTS bandit-based prompt strategy selection (March 2025).
