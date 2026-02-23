"""Shared fixtures for KalMax test suite."""

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture(autouse=True)
def jax_cpu():
    """Force JAX to use CPU for deterministic, reproducible tests."""
    jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixed PRNG key for reproducibility."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def simple_1d_data(rng_key):
    """1D dataset: 100 timesteps, 5 neurons, 20 bins, sine wave trajectory."""
    T, N_neurons, N_bins = 100, 5, 20
    bins = jnp.linspace(0, 1, N_bins)[:, None]  # (N_bins, 1)
    t = jnp.linspace(0, 2 * jnp.pi, T)
    trajectory = (0.5 + 0.4 * jnp.sin(t))[:, None]  # (T, 1)

    # Generate spikes: higher near trajectory position
    key1, key2 = jax.random.split(rng_key)
    spikes = jax.random.poisson(key1, lam=0.5, shape=(T, N_neurons)).astype(jnp.float32)

    return {
        "bins": bins,
        "trajectory": trajectory,
        "spikes": spikes,
        "T": T,
        "N_neurons": N_neurons,
        "N_bins": N_bins,
    }


@pytest.fixture
def simple_2d_data(rng_key):
    """2D dataset: 200 timesteps, 4 neurons, 5x5 bin grid, circular trajectory."""
    T, N_neurons = 200, 4
    n_per_dim = 5
    x = jnp.linspace(0, 1, n_per_dim)
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    bins = jnp.stack([xx.ravel(), yy.ravel()], axis=1)  # (25, 2)
    N_bins = bins.shape[0]

    t = jnp.linspace(0, 2 * jnp.pi, T)
    trajectory = jnp.stack([0.5 + 0.3 * jnp.cos(t), 0.5 + 0.3 * jnp.sin(t)], axis=1)  # (T, 2)

    key1, key2 = jax.random.split(rng_key)
    spikes = jax.random.poisson(key1, lam=0.3, shape=(T, N_neurons)).astype(jnp.float32)

    return {
        "bins": bins,
        "trajectory": trajectory,
        "spikes": spikes,
        "T": T,
        "N_neurons": N_neurons,
        "N_bins": N_bins,
    }


@pytest.fixture
def angular_data(rng_key):
    """Angular dataset: 500 timesteps, 3 neurons, 64 bins, random walk in [-pi, pi)."""
    T, N_neurons, N_bins = 500, 3, 64
    bins = jnp.linspace(-jnp.pi, jnp.pi, N_bins, endpoint=False)  # (N_bins,)

    key1, key2 = jax.random.split(rng_key)
    steps = jax.random.normal(key1, shape=(T,)) * 0.1
    trajectory = jnp.cumsum(steps)
    trajectory = jnp.mod(trajectory + jnp.pi, 2 * jnp.pi) - jnp.pi  # wrap to [-pi, pi)

    spikes = jax.random.poisson(key2, lam=0.5, shape=(T, N_neurons)).astype(jnp.float32)

    return {
        "bins": bins,
        "trajectory": trajectory,
        "spikes": spikes,
        "T": T,
        "N_neurons": N_neurons,
        "N_bins": N_bins,
    }


@pytest.fixture
def kalman_system():
    """2D constant-velocity system: state=[x, vx], obs=[x]."""
    dt = 0.1
    dim_Z, dim_Y = 2, 1
    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])
    Q = jnp.eye(dim_Z) * 0.01
    R = jnp.eye(dim_Y) * 0.1
    mu0 = jnp.zeros(dim_Z)
    sigma0 = jnp.eye(dim_Z)
    B = jnp.zeros((dim_Z, 0))

    return {
        "dim_Z": dim_Z,
        "dim_Y": dim_Y,
        "dt": dt,
        "F": F,
        "H": H,
        "Q": Q,
        "R": R,
        "mu0": mu0,
        "sigma0": sigma0,
        "B": B,
    }
