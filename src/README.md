# Grounded Discovery Experiment

Framework for LLM-guided equation discovery using ACE-style context memory.

## Project Structure

```
src/
├── pyproject.toml              # Project metadata and dependencies
├── conftest.py                 # Pytest configuration + import paths
├── __init__.py                 # Package initialization
├── README.md                   # This file
│
├── environments/               # Dynamical systems and counterfactual physics environments
│   ├── base.py                # BaseEnvironment + EnvironmentSpec
│   ├── tier1.py               # Single-system modifications (6 environments)
│   └── tier2.py               # Multi-system interactions (3 environments)
│
├── loop/                       # Main experiment orchestration
├── conditions/                 # Initial conditions and trajectory sampling
├── gradient/                   # JAX-based gradient discovery methods
├── ace_adapter/                # ACE integration: playbook generation
├── campaign/                   # Multi-run coordination
├── probes/                     # Symbolic equality and equation verification
├── analysis/                   # Results analysis and metrics
│   └── levin/                 # Levin extension synthesis analysis
└── tests/                      # Test suite
```

## Environments

### Tier 1: Single-System Modifications (6 environments)

Each modifies one aspect of classical physics:

1. **ModifiedGravityCoupling** — gravity with polynomial correction
2. **AsymmetricDrag** — asymmetric velocity-dependent damping
3. **NonReciprocalSpring** — spring with state-dependent coupling
4. **VelocityDependentMass** — inertia that changes with velocity
5. **AnharmonicOscillator** — cubic restoring force nonlinearity
6. **ModifiedProjectile** — 2D projectile with quadratic air resistance

### Tier 2: Multi-System Interactions (3 environments)

Require understanding across multiple coupled systems:

1. **NonPolynomialConserved** — transcendental coupling with implicit conservation
2. **CrossCoupledDynamics** — predator-prey with multiplicative interaction
3. **HistoryDependentForce** — three-variable system with implicit temporal coupling

## Usage

### Basic Environment Interaction

```python
from environments import ModifiedGravityCoupling

# Create environment
env = ModifiedGravityCoupling(seed=42, noise=False)

# Get specification
spec = env.spec
print(f"Name: {spec.name}")
print(f"Inputs: {spec.input_names}")
print(f"Ground truth: {spec.ground_truth_expr}")

# Sample random inputs
inputs = env.sample_inputs(n=10)  # shape: (10, 2)

# Evaluate at inputs
outputs = env.evaluate(inputs)    # shape: (10,)

# Get held-out test set MSE
mse = env.test_mse(lambda x: x[:, 0]**2)
```

### All Environments at Once

```python
from environments import ALL_ENVIRONMENTS

for env_cls in ALL_ENVIRONMENTS:
    env = env_cls(seed=42)
    print(f"{env.name} (Tier {env.spec.tier}): {env.spec.ground_truth_expr}")
```

## Integration Points

### With ACE

The grounded discovery experiment integrates with ACE:
- Environment specifications feed into ACE's discovery prompts
- ACE's playbook evolves strategies for successful equation discovery
- Curator tags which discovery strategies (playbook bullets) work best for each environment tier

### With ShinkaEvolve

ShinkaEvolve may be used for:
- Hyperparameter optimization of discovery algorithms
- Automated prompt engineering for equation recovery

## Dependencies

- `numpy>=1.24` — numerical computing
- `sympy>=1.12` — symbolic mathematics
- `scipy>=1.11` — scientific computing
- `pytest>=7.0` — testing (dev)
- `matplotlib>=3.7` — visualization (dev)
- Optional: `jax` for gradient-based methods
- Optional: `pysr` for PySR-based discovery

## Running Tests

```bash
cd /home/users/duynguy/proj/grounded
python -m pytest src/tests/ -v
```

## Notes

- All environments support optional noise via the `noise=True` parameter
- Test sets are deterministically generated based on `seed + 99999`
- Input ranges are environment-specific; see `spec.input_ranges`
- All experiments should use `conftest.py` for proper import paths
