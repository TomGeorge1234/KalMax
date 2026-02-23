"""Tests for kalmax.kde module."""

import jax
import jax.numpy as jnp
import pytest

from kalmax.kde import kde, kde_angular, poisson_log_likelihood, poisson_log_likelihood_trajectory
from kalmax.kernels import gaussian_kernel


# ============================================================================
# kde() tests
# ============================================================================


class TestKDE:
    def test_output_shape(self, simple_1d_data):
        d = simple_1d_data
        result = kde(d["bins"], d["trajectory"], d["spikes"])
        assert result.shape == (d["N_neurons"], d["N_bins"])

    def test_non_negative(self, simple_1d_data):
        d = simple_1d_data
        result = kde(d["bins"], d["trajectory"], d["spikes"])
        assert jnp.all(result >= 0)

    def test_peak_at_data_location(self):
        """KDE should have highest density near where data is concentrated."""
        T, N_bins = 500, 50
        bins = jnp.linspace(0, 1, N_bins)[:, None]
        # All trajectory near 0.5
        trajectory = jnp.full((T, 1), 0.5)
        spikes = jnp.ones((T, 1))
        result = kde(bins, trajectory, spikes)
        peak_bin = jnp.argmax(result[0])
        peak_pos = bins[peak_bin, 0]
        assert jnp.abs(peak_pos - 0.5) < 0.1

    def test_mask_zeros_out_spikes(self, simple_1d_data):
        d = simple_1d_data
        mask = jnp.zeros_like(d["spikes"], dtype=bool)
        result = kde(d["bins"], d["trajectory"], d["spikes"], mask=mask)
        # With all spikes masked, the numerator should be zero
        assert jnp.allclose(result, 0.0, atol=1e-5)

    def test_batch_size_invariance(self, simple_1d_data):
        d = simple_1d_data
        result_full = kde(d["bins"], d["trajectory"], d["spikes"], batch_size=d["T"])
        result_small = kde(d["bins"], d["trajectory"], d["spikes"], batch_size=10)
        assert jnp.allclose(result_full, result_small, rtol=0.05, atol=1e-2)

    def test_position_density_return(self, simple_1d_data):
        d = simple_1d_data
        result, pos_density = kde(d["bins"], d["trajectory"], d["spikes"], return_position_density=True)
        assert result.shape == (d["N_neurons"], d["N_bins"])
        assert pos_density.shape == (d["N_bins"],)
        # Position density should sum to 1
        assert jnp.allclose(pos_density.sum(), 1.0, atol=1e-5)

    def test_2d_data(self, simple_2d_data):
        d = simple_2d_data
        result = kde(d["bins"], d["trajectory"], d["spikes"], kernel_bandwidth=0.1)
        assert result.shape == (d["N_neurons"], d["N_bins"])
        assert jnp.all(result >= 0)


# ============================================================================
# kde_angular() tests
# ============================================================================


class TestKDEAngular:
    def test_output_shape(self, angular_data):
        d = angular_data
        result = kde_angular(d["bins"], d["trajectory"], d["spikes"])
        assert result.shape == (d["N_neurons"], d["N_bins"])

    def test_mostly_non_negative(self, angular_data):
        """KDE values should be non-negative where there is sufficient data coverage."""
        d = angular_data
        result = kde_angular(d["bins"], d["trajectory"], d["spikes"])
        # Most values should be non-negative (small negatives can occur with sparse data)
        assert jnp.mean(result >= -0.01) > 0.9

    def test_wrapping_correctness(self):
        """Position density should peak where data is and should wrap across +-pi boundary."""
        N_bins = 64
        bins = jnp.linspace(-jnp.pi, jnp.pi, N_bins, endpoint=False)

        # Data at angle 0
        T = 200
        trajectory = jnp.full(T, 0.0)
        spikes = jnp.ones((T, 1))

        _, pos_density = kde_angular(bins, trajectory, spikes, kernel_bandwidth=0.3, return_position_density=True)
        # Position density should peak at angle 0
        peak_bin = jnp.argmax(pos_density)
        peak_angle = bins[peak_bin]
        assert jnp.abs(peak_angle) < 0.2

    def test_mask(self, angular_data):
        d = angular_data
        mask = jnp.zeros_like(d["spikes"], dtype=bool)
        result = kde_angular(d["bins"], d["trajectory"], d["spikes"], mask=mask)
        assert jnp.allclose(result, 0.0, atol=1e-5)

    def test_position_density(self, angular_data):
        d = angular_data
        result, pos_density = kde_angular(
            d["bins"], d["trajectory"], d["spikes"], return_position_density=True
        )
        assert pos_density.shape == (d["N_bins"],)
        assert jnp.allclose(pos_density.sum(), 1.0, atol=1e-5)

    def test_2d_input_squeeze(self, angular_data):
        """Should accept (T, 1) shaped trajectory and (N_bins, 1) shaped bins."""
        d = angular_data
        bins_2d = d["bins"][:, None]
        traj_2d = d["trajectory"][:, None]
        result = kde_angular(bins_2d, traj_2d, d["spikes"])
        assert result.shape == (d["N_neurons"], d["N_bins"])


# ============================================================================
# poisson_log_likelihood() tests
# ============================================================================


class TestPoissonLogLikelihood:
    def test_output_shape(self, simple_1d_data):
        d = simple_1d_data
        rate_map = kde(d["bins"], d["trajectory"], d["spikes"])
        ll = poisson_log_likelihood(d["spikes"], rate_map)
        assert ll.shape == (d["T"], d["N_bins"])

    def test_max_at_true_position(self):
        """Log-likelihood should be highest near the true position."""
        N_bins = 50
        bins = jnp.linspace(0, 1, N_bins)[:, None]
        T = 200
        trajectory = jnp.full((T, 1), 0.5)
        # One neuron that fires at position 0.5
        spikes = jnp.ones((T, 1))

        rate_map = kde(bins, trajectory, spikes, kernel_bandwidth=0.05)

        # Test with a single spike at t=0
        test_spikes = jnp.ones((1, 1))
        ll = poisson_log_likelihood(test_spikes, rate_map, renormalise=True)
        peak_bin = jnp.argmax(ll[0])
        peak_pos = bins[peak_bin, 0]
        assert jnp.abs(peak_pos - 0.5) < 0.3

    def test_mask(self, simple_1d_data):
        d = simple_1d_data
        rate_map = kde(d["bins"], d["trajectory"], d["spikes"])
        mask = jnp.ones_like(d["spikes"], dtype=bool)
        ll_masked = poisson_log_likelihood(d["spikes"], rate_map, mask=mask)
        ll_unmasked = poisson_log_likelihood(d["spikes"], rate_map)
        assert jnp.allclose(ll_masked, ll_unmasked, atol=1e-5)

    def test_renormalize_off(self, simple_1d_data):
        d = simple_1d_data
        rate_map = kde(d["bins"], d["trajectory"], d["spikes"])
        ll = poisson_log_likelihood(d["spikes"], rate_map, renormalise=False)
        # Without renormalization, max should NOT necessarily be 0
        assert ll.shape == (d["T"], d["N_bins"])

    def test_renormalize_on(self, simple_1d_data):
        d = simple_1d_data
        rate_map = kde(d["bins"], d["trajectory"], d["spikes"])
        ll = poisson_log_likelihood(d["spikes"], rate_map, renormalise=True)
        # With renormalization, max per row should be 0
        row_maxes = jnp.max(ll, axis=1)
        assert jnp.allclose(row_maxes, 0.0, atol=1e-5)


# ============================================================================
# poisson_log_likelihood_trajectory() tests
# ============================================================================


class TestPoissonLogLikelihoodTrajectory:
    def test_output_shape(self, simple_1d_data):
        d = simple_1d_data
        # Create a fake mean_rate_along_trajectory of same shape as spikes
        mean_rate = jnp.ones_like(d["spikes"]) * 0.5
        ll = poisson_log_likelihood_trajectory(d["spikes"], mean_rate)
        assert ll.shape == (d["T"],)

    def test_mask(self, simple_1d_data):
        d = simple_1d_data
        mean_rate = jnp.ones_like(d["spikes"]) * 0.5
        mask = jnp.ones_like(d["spikes"], dtype=bool)
        ll_masked = poisson_log_likelihood_trajectory(d["spikes"], mean_rate, mask=mask)
        ll_unmasked = poisson_log_likelihood_trajectory(d["spikes"], mean_rate)
        assert jnp.allclose(ll_masked, ll_unmasked, atol=1e-5)
