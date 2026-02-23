"""Tests for kalmax.kalman module."""

import jax
import jax.numpy as jnp
import pytest

from kalmax.kalman import (
    KalmanFilter,
    kalman_filter,
    kalman_smoother,
    kalman_predict,
    kalman_update,
    kalman_likelihoods,
    calculate_S_matrix,
    calculate_K_matrix,
    fit_parameters,
    fit_mu0,
    fit_sigma0,
    fit_F,
    fit_Q,
    fit_H,
    fit_R,
)


# ============================================================================
# KalmanFilter class tests
# ============================================================================


class TestKalmanFilterClass:
    def test_init(self, kalman_system):
        ks = kalman_system
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )
        assert kf.dim_Z == ks["dim_Z"]
        assert kf.dim_Y == ks["dim_Y"]

    def test_filter_output_shapes(self, kalman_system, rng_key):
        ks = kalman_system
        T = 50
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        mus, sigmas = kf.filter(Y)
        assert mus.shape == (T, ks["dim_Z"])
        assert sigmas.shape == (T, ks["dim_Z"], ks["dim_Z"])

    def test_filter_reduces_uncertainty(self, kalman_system):
        """Filtering should reduce uncertainty compared to the prior."""
        ks = kalman_system
        T = 100
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )
        # Constant observation at 1.0
        Y = jnp.ones((T, ks["dim_Y"]))
        mus, sigmas = kf.filter(Y)

        # Variance should decrease compared to initial
        initial_var = jnp.trace(ks["sigma0"])
        final_var = jnp.trace(sigmas[-1])
        assert final_var < initial_var

    def test_filter_tracks_constant_signal(self, kalman_system):
        """Filter should converge to a constant signal."""
        ks = kalman_system
        T = 200
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )
        # Constant observation at 5.0
        Y = jnp.full((T, ks["dim_Y"]), 5.0)
        mus, sigmas = kf.filter(Y)

        # Position component should converge near 5.0
        assert jnp.abs(mus[-1, 0] - 5.0) < 0.5

    def test_control_input(self):
        """Filter with control input should differ from without."""
        dim_Z, dim_Y, dim_U = 2, 2, 2
        T = 50
        F = jnp.eye(dim_Z)
        H = jnp.eye(dim_Y)
        Q = jnp.eye(dim_Z) * 0.01
        R = jnp.eye(dim_Y) * 0.1
        B = jnp.eye(dim_Z) * 0.5
        mu0 = jnp.zeros(dim_Z)
        sigma0 = jnp.eye(dim_Z)

        kf = KalmanFilter(dim_Z=dim_Z, dim_Y=dim_Y, dim_U=dim_U, F=F, H=H, Q=Q, R=R, B=B, mu0=mu0, sigma0=sigma0)

        Y = jnp.zeros((T, dim_Y))
        U = jnp.ones((T, dim_U))

        mus_with_u, _ = kf.filter(Y, U=U)
        mus_without_u, _ = kf.filter(Y)

        # With control input pushing, means should differ
        assert not jnp.allclose(mus_with_u, mus_without_u, atol=0.01)

    def test_time_varying_params(self, kalman_system, rng_key):
        """Time-varying R should be accepted."""
        ks = kalman_system
        T = 30
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        R_tv = jnp.tile(ks["R"], (T, 1, 1))
        mus, sigmas = kf.filter(Y, R=R_tv)
        assert mus.shape == (T, ks["dim_Z"])

    def test_batch_size_invariance(self, kalman_system, rng_key):
        """Different batch sizes should produce same results."""
        ks = kalman_system
        T = 50
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))

        kf1 = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
            batch_size=T,
        )
        kf2 = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
            batch_size=10,
        )

        mus1, sigmas1 = kf1.filter(Y)
        mus2, sigmas2 = kf2.filter(Y)
        assert jnp.allclose(mus1, mus2, atol=1e-4)
        assert jnp.allclose(sigmas1, sigmas2, atol=1e-4)

    def test_smooth_output_shapes(self, kalman_system, rng_key):
        ks = kalman_system
        T = 50
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        mus_f, sigmas_f = kf.filter(Y)
        mus_s, sigmas_s = kf.smooth(mus_f, sigmas_f)
        assert mus_s.shape == (T, ks["dim_Z"])
        assert sigmas_s.shape == (T, ks["dim_Z"], ks["dim_Z"])

    def test_smooth_reduces_variance(self, kalman_system, rng_key):
        """Smoothed variances should be <= filtered variances."""
        ks = kalman_system
        T = 100
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        mus_f, sigmas_f = kf.filter(Y)
        mus_s, sigmas_s = kf.smooth(mus_f, sigmas_f)

        # Average trace of smoothed covariance should be <= filtered (ignoring boundary)
        filter_traces = jax.vmap(jnp.trace)(sigmas_f[10:-1])
        smooth_traces = jax.vmap(jnp.trace)(sigmas_s[10:-1])
        assert smooth_traces.mean() <= filter_traces.mean() + 1e-4

    def test_loglikelihood_shapes(self, kalman_system, rng_key):
        ks = kalman_system
        T = 30
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        mus_f, sigmas_f = kf.filter(Y)
        logP = kf.loglikelihood(Y, mus_f, sigmas_f)
        assert logP.shape == (T,)

    def test_loglikelihood_prefers_true_data(self, kalman_system, rng_key):
        """Log-likelihood should be higher for data consistent with the model."""
        ks = kalman_system
        T = 100
        kf = KalmanFilter(
            dim_Z=ks["dim_Z"],
            dim_Y=ks["dim_Y"],
            F=ks["F"],
            H=ks["H"],
            Q=ks["Q"],
            R=ks["R"],
            mu0=ks["mu0"],
            sigma0=ks["sigma0"],
        )

        # Good data: constant signal
        Y_good = jnp.zeros((T, ks["dim_Y"]))
        mus_f, sigmas_f = kf.filter(Y_good)
        ll_good = kf.loglikelihood(Y_good, mus_f, sigmas_f)

        # Bad data: random noise far from filtered estimates
        key = jax.random.PRNGKey(99)
        Y_bad = jax.random.normal(key, (T, ks["dim_Y"])) * 100
        ll_bad = kf.loglikelihood(Y_bad, mus_f, sigmas_f)

        assert ll_good.mean() > ll_bad.mean()


# ============================================================================
# Module-level function tests
# ============================================================================


class TestModuleFunctions:
    def test_predict_mean_formula(self):
        mu = jnp.array([1.0, 2.0])
        F = jnp.array([[1.0, 0.1], [0.0, 1.0]])
        Q = jnp.eye(2) * 0.01
        B = jnp.zeros((2, 0))
        u = jnp.zeros(0)
        mu_next, sigma_next = kalman_predict(mu, jnp.eye(2), F, Q, B, u)
        expected_mu = F @ mu
        assert jnp.allclose(mu_next, expected_mu)

    def test_predict_cov_formula(self):
        sigma = jnp.eye(2) * 0.5
        F = jnp.array([[1.0, 0.1], [0.0, 1.0]])
        Q = jnp.eye(2) * 0.01
        B = jnp.zeros((2, 0))
        u = jnp.zeros(0)
        _, sigma_next = kalman_predict(jnp.zeros(2), sigma, F, Q, B, u)
        expected = F @ sigma @ F.T + Q
        assert jnp.allclose(sigma_next, expected)

    def test_update_with_zero_noise(self):
        """With zero observation noise, posterior should equal observation."""
        mu = jnp.array([0.0, 0.0])
        sigma = jnp.eye(2)
        H = jnp.eye(2)
        R = jnp.eye(2) * 1e-10  # near-zero noise
        y = jnp.array([5.0, 3.0])
        mu_post, sigma_post = kalman_update(mu, sigma, H, R, y)
        # Should be very close to observation
        assert jnp.allclose(mu_post, y, atol=1e-3)

    def test_update_with_infinite_noise(self):
        """With very high observation noise, posterior should stay near prior."""
        mu = jnp.array([1.0, 2.0])
        sigma = jnp.eye(2) * 0.01
        H = jnp.eye(2)
        R = jnp.eye(2) * 1e6  # huge noise
        y = jnp.array([100.0, 100.0])
        mu_post, sigma_post = kalman_update(mu, sigma, H, R, y)
        # Should barely move from prior
        assert jnp.allclose(mu_post, mu, atol=0.1)

    def test_S_matrix_formula(self):
        sigma = jnp.eye(2) * 0.5
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.1]])
        S = calculate_S_matrix(sigma, H, R)
        expected = H @ sigma @ H.T + R
        assert jnp.allclose(S, expected)

    def test_K_matrix_formula(self):
        sigma = jnp.eye(2)
        H = jnp.array([[1.0, 0.0]])
        S = jnp.array([[1.5]])
        K = calculate_K_matrix(sigma, H, S)
        expected = sigma @ H.T @ jnp.linalg.inv(S)
        assert jnp.allclose(K, expected)

    def test_kalman_filter_jit(self, kalman_system, rng_key):
        ks = kalman_system
        T = 20
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        U = jnp.zeros((T, 0))
        F_t = jnp.tile(ks["F"], (T, 1, 1))
        B_t = jnp.tile(ks["B"], (T, 1, 1))
        Q_t = jnp.tile(ks["Q"], (T, 1, 1))
        H_t = jnp.tile(ks["H"], (T, 1, 1))
        R_t = jnp.tile(ks["R"], (T, 1, 1))

        # Should run without error (function is @jit decorated)
        mus, sigmas = kalman_filter(Y, U, ks["mu0"], ks["sigma0"], F_t, B_t, Q_t, H_t, R_t)
        assert mus.shape == (T, ks["dim_Z"])

    def test_kalman_smoother_jit(self, kalman_system, rng_key):
        ks = kalman_system
        T = 20
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        U = jnp.zeros((T, 0))
        F_t = jnp.tile(ks["F"], (T, 1, 1))
        B_t = jnp.tile(ks["B"], (T, 1, 1))
        Q_t = jnp.tile(ks["Q"], (T, 1, 1))
        H_t = jnp.tile(ks["H"], (T, 1, 1))
        R_t = jnp.tile(ks["R"], (T, 1, 1))

        mus_f, sigmas_f = kalman_filter(Y, U, ks["mu0"], ks["sigma0"], F_t, B_t, Q_t, H_t, R_t)
        mus_s, sigmas_s = kalman_smoother(mus_f, sigmas_f, U, mus_f[-1], sigmas_f[-1], F_t, B_t, Q_t)
        assert mus_s.shape == (T, ks["dim_Z"])

    def test_likelihoods_shapes(self, kalman_system, rng_key):
        ks = kalman_system
        T = 20
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        U = jnp.zeros((T, 0))
        F_t = jnp.tile(ks["F"], (T, 1, 1))
        B_t = jnp.tile(ks["B"], (T, 1, 1))
        Q_t = jnp.tile(ks["Q"], (T, 1, 1))
        H_t = jnp.tile(ks["H"], (T, 1, 1))
        R_t = jnp.tile(ks["R"], (T, 1, 1))

        mus_f, sigmas_f = kalman_filter(Y, U, ks["mu0"], ks["sigma0"], F_t, B_t, Q_t, H_t, R_t)
        Z = mus_f  # use filtered means as trajectory
        PZ, PZXF, PXZF = kalman_likelihoods(Z, Y, mus_f, sigmas_f, F_t, Q_t, H_t, R_t, B=B_t, U=U)
        assert PZ.shape == (T,)
        assert PZXF.shape == (T,)
        assert PXZF.shape == (T,)

    def test_likelihoods_positive(self, kalman_system, rng_key):
        ks = kalman_system
        T = 20
        Y = jax.random.normal(rng_key, (T, ks["dim_Y"]))
        U = jnp.zeros((T, 0))
        F_t = jnp.tile(ks["F"], (T, 1, 1))
        B_t = jnp.tile(ks["B"], (T, 1, 1))
        Q_t = jnp.tile(ks["Q"], (T, 1, 1))
        H_t = jnp.tile(ks["H"], (T, 1, 1))
        R_t = jnp.tile(ks["R"], (T, 1, 1))

        mus_f, sigmas_f = kalman_filter(Y, U, ks["mu0"], ks["sigma0"], F_t, B_t, Q_t, H_t, R_t)
        Z = mus_f
        PZ, PZXF, PXZF = kalman_likelihoods(Z, Y, mus_f, sigmas_f, F_t, Q_t, H_t, R_t, B=B_t, U=U)
        assert jnp.all(PZ >= 0)
        assert jnp.all(PZXF >= 0)
        assert jnp.all(PXZF >= 0)


# ============================================================================
# Parameter fitting tests
# ============================================================================


class TestParameterFitting:
    def _generate_data(self, rng_key, T=500):
        """Generate synthetic Kalman data for fitting tests."""
        dim_Z, dim_Y = 2, 2
        F_true = jnp.array([[0.99, 0.05], [0.0, 0.99]])
        H_true = jnp.eye(2)
        Q_true = jnp.eye(2) * 0.01
        R_true = jnp.eye(2) * 0.1

        key1, key2 = jax.random.split(rng_key)
        Z = jnp.zeros((T, dim_Z))
        z = jnp.zeros(dim_Z)
        zs = []
        for t in range(T):
            key1, subkey = jax.random.split(key1)
            z = F_true @ z + jax.random.normal(subkey, (dim_Z,)) * 0.1
            zs.append(z)
        Z = jnp.stack(zs)
        Y = Z @ H_true.T + jax.random.normal(key2, (T, dim_Y)) * jnp.sqrt(0.1)

        return Z, Y, F_true, Q_true, H_true, R_true

    def test_fit_parameters_output_shapes(self, rng_key):
        Z, Y, _, _, _, _ = self._generate_data(rng_key)
        mu0, sigma0, F, Q, H, R = fit_parameters(Z, Y)
        dim_Z = Z.shape[1]
        dim_Y = Y.shape[1]
        assert mu0.shape == (dim_Z,)
        assert sigma0.shape == (dim_Z, dim_Z)
        assert F.shape == (dim_Z, dim_Z)
        assert Q.shape == (dim_Z, dim_Z)
        assert H.shape == (dim_Y, dim_Z)
        assert R.shape == (dim_Y, dim_Y)

    def test_fit_roundtrip_recovery(self, rng_key):
        """Fitted parameters should approximately recover true ones."""
        Z, Y, F_true, Q_true, H_true, R_true = self._generate_data(rng_key, T=2000)
        _, _, F_fit, Q_fit, H_fit, R_fit = fit_parameters(Z, Y)

        # F should be close to identity-like matrix
        assert jnp.allclose(F_fit, F_true, atol=0.1)
        # H should be close to identity
        assert jnp.allclose(H_fit, H_true, atol=0.1)

    def test_individual_fits_match_joint(self, rng_key):
        Z, Y, _, _, _, _ = self._generate_data(rng_key)
        mu0_j, sigma0_j, F_j, Q_j, H_j, R_j = fit_parameters(Z, Y)

        assert jnp.allclose(fit_mu0(Z), mu0_j, atol=1e-5)
        assert jnp.allclose(fit_sigma0(Z), sigma0_j, atol=1e-5)
        assert jnp.allclose(fit_F(Z), F_j, atol=1e-5)
        assert jnp.allclose(fit_Q(Z), Q_j, atol=1e-5)
        assert jnp.allclose(fit_H(Z, Y), H_j, atol=1e-5)
        assert jnp.allclose(fit_R(Z, Y), R_j, atol=1e-5)

    def test_mu0_is_mean(self, rng_key):
        Z, _, _, _, _, _ = self._generate_data(rng_key)
        mu0 = fit_mu0(Z)
        assert jnp.allclose(mu0, Z.mean(axis=0))

    def test_Q_positive_definite(self, rng_key):
        Z, _, _, _, _, _ = self._generate_data(rng_key)
        Q = fit_Q(Z)
        eigenvalues = jnp.linalg.eigvalsh(Q)
        assert jnp.all(eigenvalues > 0)

    def test_R_positive_definite(self, rng_key):
        Z, Y, _, _, _, _ = self._generate_data(rng_key)
        R = fit_R(Z, Y)
        eigenvalues = jnp.linalg.eigvalsh(R)
        assert jnp.all(eigenvalues > 0)
