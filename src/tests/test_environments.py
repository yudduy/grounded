"""Tests for counterfactual physics environments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from environments import ALL_ENVIRONMENTS
from environments.base import BaseEnvironment


class TestBaseEnvironment:
    """Test the BaseEnvironment interface across all environments."""

    @pytest.fixture(params=ALL_ENVIRONMENTS)
    def env(self, request):
        return request.param(seed=42, noise=False)

    def test_instantiation(self, env):
        assert isinstance(env, BaseEnvironment)
        assert env.name is not None
        assert len(env.input_names) > 0
        assert len(env.input_ranges) == len(env.input_names)

    def test_evaluate_shape(self, env):
        n = 10
        inputs = env.sample_inputs(n)
        assert inputs.shape == (n, len(env.input_names))
        outputs = env.evaluate(inputs)
        assert outputs.shape == (n,)
        assert np.all(np.isfinite(outputs))

    def test_evaluate_deterministic_no_noise(self, env):
        inputs = env.sample_inputs(5)
        out1 = env.evaluate(inputs)
        out2 = env.evaluate(inputs)
        np.testing.assert_array_equal(out1, out2)

    def test_evaluate_with_noise(self):
        env = ALL_ENVIRONMENTS[0](seed=42, noise=True)
        inputs = env.sample_inputs(100)
        out1 = env.evaluate(inputs)
        # Reset RNG
        env2 = ALL_ENVIRONMENTS[0](seed=42, noise=True)
        out2 = env2.evaluate(inputs)
        # Same seed should give same noisy output
        np.testing.assert_array_almost_equal(out1, out2)

    def test_test_set_exists(self, env):
        assert env._test_inputs is not None
        assert env._test_outputs is not None
        assert env._test_inputs.shape[0] == 50
        assert env._test_outputs.shape[0] == 50

    def test_test_mse(self, env):
        # Perfect predictor should have MSE ~0
        def perfect(x):
            return env._ground_truth(x)
        mse = env.test_mse(perfect)
        assert mse < 1e-10

    def test_choose_inputs_clips(self, env):
        # Create out-of-range inputs
        n = 5
        d = len(env.input_names)
        big = np.ones((n, d)) * 1e6
        clipped, _ = env.choose_inputs(big)
        for i, name in enumerate(env.input_names):
            lo, hi = env.input_ranges[name]
            assert np.all(clipped[:, i] <= hi + 1e-10)
            assert np.all(clipped[:, i] >= lo - 1e-10)

    def test_sample_inputs_in_range(self, env):
        inputs = env.sample_inputs(100)
        for i, name in enumerate(env.input_names):
            lo, hi = env.input_ranges[name]
            assert np.all(inputs[:, i] >= lo)
            assert np.all(inputs[:, i] <= hi)


class TestSpecificEnvironments:
    """Test specific known-input-output pairs."""

    def test_exponential_gravity_zero(self):
        from environments.tier1 import ExponentialDampedGravity
        env = ExponentialDampedGravity(seed=0, noise=False)
        inputs = np.array([[0.0, 0.0]])
        out = env.evaluate(inputs)
        assert abs(out[0]) < 1e-10

    def test_coupled_nonlinear_zero(self):
        from environments.tier1 import CoupledNonlinearDamping
        env = CoupledNonlinearDamping(seed=0, noise=False)
        inputs = np.array([[0.0, 0.0]])
        out = env.evaluate(inputs)
        assert abs(out[0]) < 1e-10

    def test_asymmetric_drag_sign(self):
        from environments.tier1 import AsymmetricDrag
        env = AsymmetricDrag(seed=0, noise=False)
        # At position=0, velocity=1: force = -sign(1)*1^2 = -1
        inputs = np.array([[1.0, 0.0]])
        out = env.evaluate(inputs)
        assert abs(out[0] - (-1.0)) < 1e-10
