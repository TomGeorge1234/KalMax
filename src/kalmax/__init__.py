"""KalMax: Kalman-based neural decoding in JAX."""

__version__ = "0.3.0"

from kalmax import kalman, kde, kernels, utils
from kalmax.kalman import KalmanFilter

__all__ = ["__version__", "kernels", "kde", "kalman", "utils", "KalmanFilter"]
