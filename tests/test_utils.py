"""Tests for kalmax.utils module."""

import jax
import jax.numpy as jnp
import pytest

from kalmax.utils import (
    gaussian_pdf,
    log_gaussian_pdf,
    gaussian_norm_const,
    fit_gaussian,
    fit_gaussian_vmap,
    _wrap_minuspi_pi,
    _bin_indices_minuspi_pi,
    _circular_conv_fft_1d,
)


# ============================================================================
# Gaussian PDF tests
# ============================================================================


class TestGaussianPDF:
    def test_integration_to_1_1d(self):
        """1D Gaussian PDF should integrate to ~1."""
        mu = jnp.array([0.0])
        sigma = jnp.array([[1.0]])
        x_grid = jnp.linspace(-6, 6, 2000)[:, None]
        vals = jax.vmap(gaussian_pdf, in_axes=(0, None, None))(x_grid, mu, sigma)
        dx = x_grid[1, 0] - x_grid[0, 0]
        integral = jnp.sum(vals) * dx
        assert jnp.allclose(integral, 1.0, atol=0.01)

    def test_integration_to_1_2d(self):
        """2D Gaussian PDF should integrate to ~1."""
        mu = jnp.array([0.0, 0.0])
        sigma = jnp.eye(2)
        x = jnp.linspace(-4, 4, 150)
        xx, yy = jnp.meshgrid(x, x)
        grid = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        vals = jax.vmap(gaussian_pdf, in_axes=(0, None, None))(grid, mu, sigma)
        dx = x[1] - x[0]
        integral = jnp.sum(vals) * dx * dx
        assert jnp.allclose(integral, 1.0, atol=0.05)

    def test_peak_at_mean(self):
        mu = jnp.array([1.0, 2.0])
        sigma = jnp.eye(2) * 0.5
        val_at_mean = gaussian_pdf(mu, mu, sigma)
        val_away = gaussian_pdf(mu + 1.0, mu, sigma)
        assert val_at_mean > val_away

    def test_symmetry(self):
        mu = jnp.array([0.0])
        sigma = jnp.array([[1.0]])
        x_pos = jnp.array([1.0])
        x_neg = jnp.array([-1.0])
        assert jnp.allclose(gaussian_pdf(x_pos, mu, sigma), gaussian_pdf(x_neg, mu, sigma))

    def test_known_value_1d_standard(self):
        """N(0,1) evaluated at 0 should be 1/sqrt(2*pi)."""
        x = jnp.array([0.0])
        mu = jnp.array([0.0])
        sigma = jnp.array([[1.0]])
        expected = 1.0 / jnp.sqrt(2 * jnp.pi)
        assert jnp.allclose(gaussian_pdf(x, mu, sigma), expected, atol=1e-6)


# ============================================================================
# Log Gaussian PDF tests
# ============================================================================


class TestLogGaussianPDF:
    def test_matches_log_of_pdf(self):
        x = jnp.array([1.0, 0.5])
        mu = jnp.array([0.0, 0.0])
        sigma = jnp.eye(2) * 2.0
        log_val = log_gaussian_pdf(x, mu, sigma)
        pdf_val = gaussian_pdf(x, mu, sigma)
        assert jnp.allclose(log_val, jnp.log(pdf_val), atol=1e-5)


# ============================================================================
# Gaussian norm const tests
# ============================================================================


class TestGaussianNormConst:
    def test_1d_formula(self):
        sigma = jnp.array([[4.0]])  # variance = 4, std = 2
        nc = gaussian_norm_const(sigma)
        expected = 1.0 / jnp.sqrt(2 * jnp.pi * 4.0)
        assert jnp.allclose(nc, expected, atol=1e-6)

    def test_2d_formula(self):
        sigma = jnp.eye(2) * 2.0
        nc = gaussian_norm_const(sigma)
        expected = 1.0 / (2 * jnp.pi * jnp.sqrt(jnp.linalg.det(sigma)))
        assert jnp.allclose(nc, expected, atol=1e-6)


# ============================================================================
# fit_gaussian tests
# ============================================================================


class TestFitGaussian:
    def test_recovers_known_gaussian(self):
        """Should recover mean and mode of a known Gaussian-shaped likelihood."""
        N_bins = 200
        x = jnp.linspace(-5, 5, N_bins)[:, None]
        true_mean = 1.0
        true_std = 0.5
        likelihood = jnp.exp(-0.5 * ((x[:, 0] - true_mean) / true_std) ** 2)

        mu, mode, cov = fit_gaussian(x, likelihood)
        assert jnp.allclose(mu[0], true_mean, atol=0.1)
        assert jnp.allclose(mode[0], true_mean, atol=0.1)

    def test_mode_matches_argmax(self):
        N_bins = 100
        x = jnp.linspace(0, 1, N_bins)[:, None]
        likelihood = jnp.exp(-((x[:, 0] - 0.7) ** 2) / 0.01)
        _, mode, _ = fit_gaussian(x, likelihood)
        argmax_pos = x[jnp.argmax(likelihood), 0]
        assert jnp.allclose(mode[0], argmax_pos)


# ============================================================================
# fit_gaussian_vmap tests
# ============================================================================


class TestFitGaussianVmap:
    def test_output_shapes(self):
        T = 10
        N_bins = 50
        D = 2
        x = jnp.zeros((N_bins, D))
        x_vals = jnp.linspace(0, 1, N_bins)
        x = x.at[:, 0].set(x_vals)
        x = x.at[:, 1].set(x_vals)
        likelihoods = jnp.ones((T, N_bins))

        means, modes, covs = fit_gaussian_vmap(x, likelihoods)
        assert means.shape == (T, D)
        assert modes.shape == (T, D)
        assert covs.shape == (T, D, D)


# ============================================================================
# Circular helper tests
# ============================================================================


class TestCircularHelpers:
    def test_wrap_correctness(self):
        angles = jnp.array([0.0, jnp.pi, -jnp.pi, 2 * jnp.pi, -2 * jnp.pi, 3 * jnp.pi])
        wrapped = _wrap_minuspi_pi(angles)
        # All should be in [-pi, pi)
        assert jnp.all(wrapped >= -jnp.pi)
        assert jnp.all(wrapped < jnp.pi)
        # 0 should stay 0
        assert jnp.allclose(wrapped[0], 0.0)
        # pi should wrap to -pi
        assert jnp.allclose(wrapped[2], -jnp.pi)

    def test_bin_indices(self):
        n_bins = 8
        # Angle at -pi should map to bin 0
        theta = jnp.array([-jnp.pi])
        idx = _bin_indices_minuspi_pi(theta, n_bins)
        assert idx[0] == 0

        # Angle at 0 should map to middle bin
        theta = jnp.array([0.0])
        idx = _bin_indices_minuspi_pi(theta, n_bins)
        assert idx[0] == n_bins // 2

    def test_fft_convolution_delta(self):
        """Convolving with a delta should return the original signal."""
        N = 16
        x = jnp.sin(jnp.linspace(0, 2 * jnp.pi, N))
        delta = jnp.zeros(N)
        delta = delta.at[0].set(1.0)
        result = _circular_conv_fft_1d(x, delta)
        assert jnp.allclose(result, x, atol=1e-5)

    def test_fft_convolution_commutativity(self):
        N = 16
        x = jnp.sin(jnp.linspace(0, 2 * jnp.pi, N))
        k = jnp.cos(jnp.linspace(0, 2 * jnp.pi, N))
        assert jnp.allclose(_circular_conv_fft_1d(x, k), _circular_conv_fft_1d(k, x), atol=1e-5)


# ============================================================================
# make_simulated_dataset smoke test
# ============================================================================


class TestMakeSimulatedDataset:
    def test_smoke(self):
        """Smoke test: skip if ratinabox not installed."""
        try:
            import ratinabox  # noqa: F401
        except ImportError:
            pytest.skip("ratinabox not installed")

        from kalmax.utils import make_simulated_dataset

        time, position, spikes = make_simulated_dataset(time_mins=0.1, n_cells=5, firing_rate=5, random_seed=42)
        assert time.ndim == 1
        assert position.ndim == 2
        assert spikes.ndim == 2
        assert time.shape[0] == position.shape[0] == spikes.shape[0]
        assert spikes.shape[1] == 5
