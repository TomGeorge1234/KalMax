"""Tests for kalmax.kernels module."""

import jax
import jax.numpy as jnp
import pytest

from kalmax.kernels import (
    gaussian_kernel,
    laplacian_kernel,
    uniform_kernel,
    epanechnikov_kernel,
    triangular_kernel,
)

ALL_KERNELS = [gaussian_kernel, laplacian_kernel, uniform_kernel, epanechnikov_kernel, triangular_kernel]
KERNEL_IDS = ["gaussian", "laplacian", "uniform", "epanechnikov", "triangular"]


# --- Symmetry ---
@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
def test_kernel_symmetry(kernel):
    x1 = jnp.array([0.3, 0.5])
    x2 = jnp.array([0.7, 0.2])
    bw = 0.1
    assert jnp.allclose(kernel(x1, x2, bw), kernel(x2, x1, bw), atol=1e-6)


# --- Peak at zero ---
@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
def test_kernel_peak_at_zero(kernel):
    x = jnp.array([0.5, 0.5])
    x_near = jnp.array([0.51, 0.5])
    bw = 0.2
    assert kernel(x, x, bw) >= kernel(x, x_near, bw)


# --- Non-negativity ---
@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
def test_kernel_non_negative(kernel):
    x1 = jnp.array([0.1, 0.9])
    x2 = jnp.array([0.8, 0.2])
    assert kernel(x1, x2, 0.5) >= 0


# --- Bandwidth scaling: larger bandwidth -> broader kernel ---
@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
def test_kernel_bandwidth_scaling(kernel):
    x1 = jnp.array([0.0])
    x2 = jnp.array([0.5])
    val_narrow = kernel(x1, x2, 0.1)
    val_wide = kernel(x1, x2, 1.0)
    # With wider bandwidth, the kernel value at distance 0.5 should be higher
    assert val_wide >= val_narrow - 1e-6


# --- 2D input support ---
@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
def test_kernel_2d_input(kernel):
    x1 = jnp.array([0.3, 0.4])
    x2 = jnp.array([0.6, 0.7])
    result = kernel(x1, x2, 0.5)
    assert result.shape == ()  # scalar output


# --- 1D input support ---
@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
def test_kernel_1d_input(kernel):
    x1 = jnp.array([0.3])
    x2 = jnp.array([0.6])
    result = kernel(x1, x2, 0.5)
    assert result.shape == ()


# --- JIT compatibility ---
@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
def test_kernel_jit(kernel):
    x1 = jnp.array([0.5])
    x2 = jnp.array([0.6])
    jitted = jax.jit(kernel, static_argnames=("bandwidth",))
    result = jitted(x1, x2, bandwidth=0.1)
    expected = kernel(x1, x2, 0.1)
    assert jnp.allclose(result, expected, atol=1e-6)


# --- vmap compatibility ---
@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=KERNEL_IDS)
def test_kernel_vmap(kernel):
    from functools import partial

    x1s = jnp.array([[0.1], [0.2], [0.3]])
    x2 = jnp.array([0.5])
    fn = partial(kernel, bandwidth=0.3)
    results = jax.vmap(fn, in_axes=(0, None))(x1s, x2)
    assert results.shape == (3,)


# --- Gaussian kernel known values ---
def test_gaussian_kernel_at_zero_distance():
    x = jnp.array([0.0])
    bw = 1.0
    val = gaussian_kernel(x, x, bw)
    # Should be 1 / sqrt(2*pi) for 1D unit variance
    expected = 1.0 / jnp.sqrt(2 * jnp.pi)
    assert jnp.allclose(val, expected, atol=1e-5)


def test_gaussian_kernel_numerical_normalization_1d():
    """Gaussian kernel should integrate to ~1 over the real line (approximated by a wide grid)."""
    bw = 0.5
    x_grid = jnp.linspace(-5, 5, 1000)[:, None]
    center = jnp.array([0.0])
    vals = jax.vmap(gaussian_kernel, in_axes=(0, None, None))(x_grid, center, bw)
    dx = x_grid[1, 0] - x_grid[0, 0]
    integral = jnp.sum(vals) * dx
    assert jnp.allclose(integral, 1.0, atol=0.01)


def test_gaussian_kernel_numerical_normalization_2d():
    """Gaussian kernel should integrate to ~1 over 2D."""
    bw = 0.5
    x = jnp.linspace(-3, 3, 100)
    xx, yy = jnp.meshgrid(x, x)
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=1)  # (10000, 2)
    center = jnp.array([0.0, 0.0])
    vals = jax.vmap(gaussian_kernel, in_axes=(0, None, None))(grid, center, bw)
    dx = x[1] - x[0]
    integral = jnp.sum(vals) * dx * dx
    assert jnp.allclose(integral, 1.0, atol=0.05)


# --- Known values for specific kernels ---
def test_uniform_kernel_inside():
    x1 = jnp.array([0.0])
    x2 = jnp.array([0.05])
    bw = 0.1
    val = uniform_kernel(x1, x2, bw)
    assert jnp.allclose(val, 1.0 / bw)


def test_uniform_kernel_outside():
    x1 = jnp.array([0.0])
    x2 = jnp.array([0.2])
    bw = 0.1
    val = uniform_kernel(x1, x2, bw)
    assert jnp.allclose(val, 0.0)


def test_epanechnikov_kernel_at_zero():
    x = jnp.array([0.0])
    bw = 1.0
    val = epanechnikov_kernel(x, x, bw)
    assert jnp.allclose(val, 3.0 / 4.0)


def test_triangular_kernel_at_zero():
    x = jnp.array([0.0])
    bw = 1.0
    val = triangular_kernel(x, x, bw)
    assert jnp.allclose(val, 1.0)


def test_laplacian_kernel_at_zero():
    x = jnp.array([0.0])
    bw = 1.0
    val = laplacian_kernel(x, x, bw)
    assert jnp.allclose(val, 1.0)
