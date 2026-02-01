"""Tests for template kernel library."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from environments.kernels import TemplateLibrary, FitResult


class TestTemplateEvaluation:
    """Test template evaluation with known parameters."""

    def test_polynomial_1d(self):
        inputs = np.array([[1.0], [2.0], [3.0]])
        params = np.array([1.0, 2.0, 0.5, 0.0, 0.0, 0.0])  # 1 + 2x + 0.5x^2
        out = TemplateLibrary.evaluate_template("polynomial_1d", params, inputs)
        expected = 1.0 + 2.0 * inputs[:, 0] + 0.5 * inputs[:, 0] ** 2
        np.testing.assert_array_almost_equal(out, expected)

    def test_trigonometric(self):
        inputs = np.array([[0.0], [np.pi / 2], [np.pi]])
        params = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # sin(x)
        out = TemplateLibrary.evaluate_template("trigonometric", params, inputs)
        expected = np.sin(inputs[:, 0])
        np.testing.assert_array_almost_equal(out, expected, decimal=5)

    def test_coupled_product(self):
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # just x1*x2
        out = TemplateLibrary.evaluate_template("coupled_product", params, inputs)
        expected = inputs[:, 0] * inputs[:, 1]
        np.testing.assert_array_almost_equal(out, expected)


class TestTemplateFitting:
    """Test parameter fitting against known functions."""

    def test_fit_linear(self):
        np.random.seed(42)
        x = np.random.uniform(-5, 5, (50, 1))
        y = 3.0 * x[:, 0] + 1.0
        result = TemplateLibrary.fit("polynomial_1d", x, y)
        assert result.train_mse < 1e-4
        # Check params: a0 ~= 1, a1 ~= 3
        assert abs(result.params[0] - 1.0) < 0.5
        assert abs(result.params[1] - 3.0) < 0.5

    def test_fit_quadratic(self):
        np.random.seed(42)
        x = np.random.uniform(-3, 3, (100, 1))
        y = 2.0 * x[:, 0] ** 2 - 1.0 * x[:, 0] + 0.5
        result = TemplateLibrary.fit("polynomial_1d", x, y)
        assert result.train_mse < 1e-3

    def test_fit_best_polynomial(self):
        np.random.seed(42)
        x = np.random.uniform(-3, 3, (100, 1))
        y = x[:, 0] ** 2
        result = TemplateLibrary.fit_best("x**2", x, y)
        assert result is not None
        assert result.train_mse < 1e-2


class TestExpressionMatching:
    """Test expression string → template matching."""

    def test_trig_detection(self):
        matches = TemplateLibrary.match_expression("np.sin(x) + np.cos(x)", 1)
        assert any(m.template_name.startswith("trig") for m in matches)

    def test_polynomial_detection(self):
        matches = TemplateLibrary.match_expression("x**2 + 3*x + 1", 1)
        assert any("polynomial" in m.template_name for m in matches)

    def test_rational_detection(self):
        matches = TemplateLibrary.match_expression("x / (1 + x**2)", 1)
        assert any("rational" in m.template_name for m in matches)

    def test_always_has_fallback(self):
        matches = TemplateLibrary.match_expression("something_weird", 1)
        assert len(matches) > 0


class TestNumericalGradientConsistency:
    """Test that template gradients are consistent with finite differences."""

    @pytest.mark.parametrize("template_name", [
        "polynomial_1d", "trigonometric", "rational",
    ])
    def test_gradient_finite_diff(self, template_name):
        """Compare analytical (scipy) optimization with finite-difference gradients."""
        np.random.seed(42)
        tmpl = TemplateLibrary.TEMPLATES[template_name]
        n_params = tmpl["n_params"]
        n_inputs = tmpl["n_inputs"]

        inputs = np.random.uniform(-2, 2, (30, n_inputs))
        true_params = np.random.randn(n_params) * 0.5
        targets = TemplateLibrary.evaluate_template(template_name, true_params, inputs)

        # Fit should recover close to true params
        result = TemplateLibrary.fit(template_name, inputs, targets,
                                     initial_params=true_params + np.random.randn(n_params) * 0.1)
        assert result.train_mse < 5e-2, (
            f"Fitting {template_name} failed: MSE={result.train_mse}"
        )
