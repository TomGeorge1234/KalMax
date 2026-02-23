"""Kernel functions for density estimation.

All kernels have the signature ``kernel(x1, x2, bandwidth) -> float`` where
``x1`` and ``x2`` are position vectors and ``bandwidth`` controls smoothing
width. Kernels must be pure JAX functions compatible with ``jax.vmap``.
"""

import jax.numpy as jnp
from kalmax.utils import gaussian_norm_const

__all__ = [
    "gaussian_kernel",
    "laplacian_kernel",
    "uniform_kernel",
    "epanechnikov_kernel",
    "triangular_kernel",
]


def gaussian_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    bandwidth: float,
) -> jnp.ndarray:
    """Calculates the gaussian kernel between two points x1 and x2 with covariance

    Parameters
    ----------
    x1: (D,) jnp.ndarray
        The first position
    x2: (D,) jnp.ndarray
        The second position
    bandwidth: float
        The bandwidth of the kernel

    Returns
    -------
    kernel: float
        The probability density at x
    """
    assert x1.ndim == 1
    assert x2.ndim == 1
    assert x1.shape[0] == x2.shape[0]
    D = x1.shape[0]

    covariance = jnp.eye(D) * bandwidth**2
    x = x1 - x2
    norm_const = gaussian_norm_const(covariance)
    kernel = norm_const * jnp.exp(-0.5 * jnp.sum(x @ jnp.linalg.inv(covariance) * x))
    return kernel


def laplacian_kernel(x1: jnp.ndarray, x2: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """Calculates the laplacian kernel between two points x1 and x2 with bandwidth h

    Parameters
    ----------
    x1: (D,) jnp.ndarray
        The first position
    x2: (D,) jnp.ndarray
        The second position
    bandwidth: float
        The bandwidth of the kernel

    Returns
    -------
    kernel: float
        The probability density at x
    """
    assert x1.ndim == 1
    assert x2.ndim == 1
    assert x1.shape[0] == x2.shape[0]

    x = jnp.linalg.norm(x1 - x2)
    kernel = jnp.exp(-x / bandwidth)
    return kernel


def uniform_kernel(x1: jnp.ndarray, x2: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """Calculates the uniform kernel between two points x1 and x2 with bandwidth h

    Parameters
    ----------
    x1: (D,) jnp.ndarray
        The first position
    x2: (D,) jnp.ndarray
        The second position
    bandwidth: float
        The bandwidth of the kernel

    Returns
    -------
    kernel: float
        The probability density at x
    """
    assert x1.ndim == 1
    assert x2.ndim == 1
    assert x1.shape[0] == x2.shape[0]

    x = jnp.linalg.norm(x1 - x2)
    kernel = jnp.where(x < bandwidth, 1 / bandwidth, 0)
    return kernel


def epanechnikov_kernel(x1: jnp.ndarray, x2: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """Calculates the epanechnikov kernel between two points x1 and x2 with bandwidth h

    Parameters
    ----------
    x1: (D,) jnp.ndarray
        The first position
    x2: (D,) jnp.ndarray
        The second position
    bandwidth: float
        The bandwidth of the kernel

    Returns
    -------
    kernel: float
        The probability density at x
    """
    assert x1.ndim == 1
    assert x2.ndim == 1
    assert x1.shape[0] == x2.shape[0]

    x = jnp.linalg.norm(x1 - x2)
    kernel = jnp.where(x < bandwidth, 3 / (4 * bandwidth) * (1 - (x / bandwidth) ** 2), 0)
    return kernel


def triangular_kernel(x1: jnp.ndarray, x2: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """Calculates the triangular kernel between two points x1 and x2 with bandwidth h

    Parameters
    ----------
    x1: (D,) jnp.ndarray
        The first position
    x2: (D,) jnp.ndarray
        The second position
    bandwidth: float
        The bandwidth of the kernel

    Returns
    -------
    kernel: float
        The probability density at x
    """
    assert x1.ndim == 1
    assert x2.ndim == 1
    assert x1.shape[0] == x2.shape[0]

    x = jnp.linalg.norm(x1 - x2)
    kernel = jnp.where(x < bandwidth, 1 - x / bandwidth, 0)
    return kernel
