"""Base class for counterfactual physics environments."""
import abc
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class EnvironmentSpec:
    """Specification for an environment's input space."""
    name: str
    description: str
    input_names: List[str]
    input_ranges: Dict[str, Tuple[float, float]]
    ground_truth_expr: str  # Human-readable, not exposed to LLM
    tier: int  # 1 or 2
    noise_sigma_rel: float = 0.01  # Relative noise level


class BaseEnvironment(abc.ABC):
    """Abstract base for counterfactual physics environments.

    Each environment defines a hidden physical law y = f(x1, ..., xn)
    with counterfactual (non-standard) parameters or functional forms.
    The agent must recover the law through interactive experimentation.
    """

    def __init__(self, seed: int = 0, noise: bool = True):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.noise = noise
        self._spec: Optional[EnvironmentSpec] = None
        self._test_inputs: Optional[np.ndarray] = None
        self._test_outputs: Optional[np.ndarray] = None
        self._generate_test_set()

    @property
    def spec(self) -> EnvironmentSpec:
        if self._spec is None:
            self._spec = self._make_spec()
        return self._spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def input_names(self) -> List[str]:
        return self.spec.input_names

    @property
    def input_ranges(self) -> Dict[str, Tuple[float, float]]:
        return self.spec.input_ranges

    @abc.abstractmethod
    def _make_spec(self) -> EnvironmentSpec:
        """Return the environment specification."""
        ...

    @abc.abstractmethod
    def _ground_truth(self, inputs: np.ndarray) -> np.ndarray:
        """Compute ground truth output. inputs shape: (n, d) -> (n,)"""
        ...

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the environment at given inputs with optional noise.

        Args:
            inputs: shape (n, d) where d = len(input_names)
        Returns:
            outputs: shape (n,)
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        y = self._ground_truth(inputs)
        if self.noise:
            # Hybrid noise: proportional + small absolute floor for scale-independence
            y_scale = np.std(y) if len(y) > 1 else max(np.abs(y).max(), 1.0)
            sigma = np.sqrt((self.spec.noise_sigma_rel * y) ** 2
                            + (self.spec.noise_sigma_rel * y_scale * 0.01) ** 2)
            y = y + self.rng.normal(0, np.maximum(sigma, 1e-10))
        return y

    def _generate_test_set(self):
        """Generate held-out test set of 200 points."""
        test_rng = np.random.RandomState(self.seed + 99999)
        spec = self.spec
        n_test = 200
        d = len(spec.input_names)
        test_inputs = np.zeros((n_test, d))
        for i, name in enumerate(spec.input_names):
            lo, hi = spec.input_ranges[name]
            test_inputs[:, i] = test_rng.uniform(lo, hi, n_test)
        self._test_inputs = test_inputs
        self._test_outputs = self._ground_truth(test_inputs)  # No noise on test set

    def test_mse(self, predict_fn) -> float:
        """Compute MSE on held-out test set.

        Args:
            predict_fn: callable that takes (n, d) array -> (n,) predictions
        Returns:
            Mean squared error
        """
        preds = predict_fn(self._test_inputs)
        return float(np.mean((preds - self._test_outputs) ** 2))

    def sample_inputs(self, n: int) -> np.ndarray:
        """Sample n random input points within the valid ranges.

        Args:
            n: number of points to sample
        Returns:
            inputs: shape (n, d)
        """
        spec = self.spec
        d = len(spec.input_names)
        inputs = np.zeros((n, d))
        for i, name in enumerate(spec.input_names):
            lo, hi = spec.input_ranges[name]
            inputs[:, i] = self.rng.uniform(lo, hi, n)
        return inputs

    def choose_inputs(self, requested: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Agent requests specific inputs, gets observations back.

        Args:
            requested: shape (n, d) input points chosen by agent
        Returns:
            (inputs, outputs) tuple
        """
        inputs = np.clip(requested,
                         [self.input_ranges[name][0] for name in self.input_names],
                         [self.input_ranges[name][1] for name in self.input_names])
        outputs = self.evaluate(inputs)
        return inputs, outputs
