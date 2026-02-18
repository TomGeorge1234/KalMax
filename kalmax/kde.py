from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap, jit
from kalmax.utils import gaussian_pdf, _wrap_minuspi_pi, _bin_indices_minuspi_pi, _circular_conv_fft_1d
from kalmax.kernels import gaussian_kernel

from functools import partial

def kde(
        bins: jnp.ndarray,
        trajectory: jnp.ndarray,
        spikes: jnp.ndarray,
        kernel : Callable = gaussian_kernel,
        kernel_bandwidth: float = 0.01,
        mask: jnp.ndarray = None,
        batch_size: int = 36000,
        return_position_density: bool = False,
        ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Performs KDE to estimate the expected number of spikes each neuron will fire at each position in `bins` given past `trajectory` and `spikes` data. This estimate is an expected-spike-count-per-timebin, in order to get firing rate in Hz, divide this by dt.

    Kernel Density Estimation goes as follows (the denominator corrects for for non-uniform position density): 

              # spikes observed at x     sum_{spike_times} K(x, x(ts))     Ks
      mu(x) = ---------------------- ==> ----------------------------- :=  --
                  # visits to x            sum_{all_times} K(x, x(t))      Kx
              = exp[log(Ks) - log(Kx)]

    Optionally, a boolean mask same shape as spikes can be passed to ignore certain spikes. This restricts the KDE calculation to only the spikes where mask is True.
    
    Parameters
    ----------
    bins : jnp.ndarray, shape (N_bins, D,)
        The position bins at which to estimate the firing rate
    trajectory : jnp.ndarray, shape (T, D)
        The position of the agent at each time step
    spikes : jnp.ndarray, shape (T, N_neurons)
        The spike counts of the neuron at each time step (integer array, can be > 1)
    kernel : function
        The kernel function to use for density estimation. See `kernels.py` for signature and examples.
    kernel_bandwidth : float
        The bandwidth of the kernel
    mask : jnp.ndarray, shape (T, N_neurons), optional
        A boolean mask to apply to the spikes. If None, no mask is applied. Default is None.
    batch_size : int
        The time axis is split into batches of this size to avoid memory errors, each batch is then processed in series. Default is 36000 (chosen to be 1 hr at 10 and an amount which doesn't crash CPU)
    return_position_density : bool
        If True, this function also returns the position density (the denominator of the KDE) at each bin.

    
    Returns
    -------
    kernel_density_estimate : jnp.ndarray, shape (N_neurons, N_bins)
    position_density : jnp.ndarray, shape (N_neurons, N_bins) (optional)
    """
    assert bins.ndim == 2
    assert trajectory.ndim == 2
    assert spikes.ndim == 2

    N_neurons = spikes.shape[1]
    N_bins = bins.shape[0]
    T = trajectory.shape[0]

    # If not passed make a trivial mask (all True)
    if mask is None: mask = jnp.ones_like(spikes, dtype=bool)
    
    # vmap the kernel K(x,mu,sigma) so it takes in a vector of positions and a vector of means
    kernel_fn = partial(kernel, bandwidth=kernel_bandwidth)
    vmapped_kernel = vmap(vmap(kernel_fn, in_axes=(0, None)), in_axes=(None, 0))

    spike_density = jnp.zeros((N_bins, N_neurons))
    position_density = jnp.zeros((N_bins, N_neurons))

    N_batchs = int(jnp.ceil(T / batch_size))
    for i in range(N_batchs):
        start = i * batch_size
        end = min((i+1) * batch_size, T)
        
        # Get the batch of trajectory, spikes and mask
        trajectory_batch = trajectory[start:end]
        spikes_batch = spikes[start:end]
        mask_batch = mask[start:end]

        # Pairwise kernel values for each trajectory-bin position pair. The bulk of the computation is done here. 
        kernel_values = vmapped_kernel(trajectory_batch, bins)
        # Calculate normalisation position density (the +epsilon is means unvisited positions should approach 0 density and avoid nans)
        position_density_batch = kernel_values @ mask_batch + 1e-6
        # Calculate spike density, replace nans from no-spikes with 0
        spike_density_batch = kernel_values @ (mask_batch*spikes_batch)
        spike_density_batch = jnp.where(jnp.isnan(spike_density_batch), 0, spike_density_batch)

        # Add these to the running total
        spike_density += spike_density_batch
        position_density += position_density_batch

    # calculate kde at each bin position 
    kernel_density_estimate = jnp.exp(jnp.log(spike_density) - jnp.log(position_density)).T

    if return_position_density:
        return kernel_density_estimate, position_density.T
    else:
        return kernel_density_estimate


def poisson_log_likelihood(spikes : jnp.ndarray,
                           mean_rate : jnp.ndarray,
                           mask : jnp.ndarray = None,
                           renormalise=True):
        """Takes an array of spike counts and an array of mean rates and returns the log-likelihood of the spikes given the mean rate of the neuron (it's receptive field). 

        P(X|mu) = (mu^X * e^-mu) / X!
        log(P(X|mu)) = sum_{neurons} [ X * log(mu) - mu - log(X!) ]
        where 
        log(X!) = log(sqrt(2*pi)) + (X+0.5) * log(X) - X    (manually correcting for when X=0) #this stirlings approximation IS necessary as it avoids going through n! which can be enormous and give nans for large spike counts 

        Optionally, a boolean mask same shape as spikes can be passed to ignore certain spikes. This restricts the likelihood calculation to only the spikes where mask is True.
        
        Parameters
        ----------
        spikes : jnp.ndarray, shape (T, N_neurons,)
            How many spikes the neuron actually fired at each bin (int, can be > 1)
        mean_rate : jnp.ndarray, shape (N_neurons, N_bins,)
            The mean rate of the neuron (it's receptive field) at each bin. This is how many spikes you would _expect_ in at this position in a time dt.
        mask : jnp.ndarray, shape (T, N_neurons,), optional
            A boolean mask to apply to the spikes. If None, no mask is applied. Default is None.
        renormalise : bool, optional
            If True this renormalises so the maximum log-likelihood is always 0 (max likelihood is 1). Recommended to avoid nan errors when likelihoods are small. Default is True.
            
        Returns
        -------
        log_likelihood : jnp.ndarray, shape (T, N_bins,)
            The log-likelihood (summed over neurons) of the spikes given the mean rate of the neuron
        """
        # If not passed make a no-mask mask (all True)
        if mask is None: mask = jnp.ones_like(spikes, dtype=bool)

        # Calculate log factorial of spike counts NOTE this could be removed if you dont care about absolute likelihoods
        spikes_ = jnp.where(spikes == 0, 1, spikes) # replace 0 spikes with 1s because 0! = 1
        log_spikecount_factorial = jnp.log(jnp.sqrt(2*jnp.pi)) + (spikes_ + 0.5) * jnp.log(spikes_) - spikes_ # manually correcting for when X=0
        
        # Sum over neurons (which are unmasked) 
        logPXmu = (mask*spikes) @ jnp.log(mean_rate+1e-3) - mask @ mean_rate - jnp.sum(log_spikecount_factorial * mask, axis=1)[:,None]
        
        # Renormalise so max likelihood is 1
        if renormalise: logPXmu = logPXmu - jnp.max(logPXmu, axis=1)[:,None]
        return logPXmu


def poisson_log_likelihood_trajectory(spikes : jnp.ndarray,
                                      mean_rate_along_trajectory : jnp.ndarray,
                                      mask : jnp.ndarray = None,):
    """Takes an array of spike counts and an _equally shaped_ array of mean rates and returns the log-likelihood of the spikes given the mean rate of the neuron (it's receptive field). This is different from `poisson_log_likelihood` in that it takes in a trajectory of mean rates and spikes and returns the log-likelihood of the spikes given the trajectory of mean rates. 

    Parameters
    ----------
    spikes : jnp.ndarray, shape (T, N_neurons,)
        How many spikes the neuron actually fired at each bin (int, can be > 1)
    mean_rate_along_trajectory : jnp.ndarray, shape (T, N_neurons,)
        The mean rate of the neurons as calculated at each time step along the trajectory. This is how many spikes you would _expect_ in at this position in a time dt.
    mask : jnp.ndarray, shape (T, N_neurons,), optional
        A boolean mask to apply to the spikes. If None, no mask is applied. Default is None.
    renormalise : bool, optional
        If True this renormalises so the maximum log-likelihood is always 0 (max likelihood is 1). Recommended to avoid nan errors when likelihoods are small. Default is True.

    Returns
    -------
    log_likelihood : jnp.ndarray, shape (T,)
        The log-likelihood (summed over neurons) of the spikes given the mean rate of the neuron
    """
        
    # If not passed make a no-mask mask (all True)
    if mask is None: mask = jnp.ones_like(spikes, dtype=bool)
      
    # Calculate log factorial of spike counts NOTE this could be removed, its just a constant factor
    spikes_ = jnp.where(spikes == 0, 1, spikes) # replace 0 spikes with 1s because 0! = 1
    log_spikecount_factorial = jnp.log(jnp.sqrt(2*jnp.pi)) + (spikes_ + 0.5) * jnp.log(spikes_) - spikes_ # manually correcting for when X=0
      
    # Calculate log-likelihood and sum over (unmasked) neurons
    logPXmu = (spikes*jnp.log(mean_rate_along_trajectory+1e-3)) - (mean_rate_along_trajectory) -  (log_spikecount_factorial)
    logPXmu = jnp.sum(mask * logPXmu, axis=1)

    return logPXmu


@partial(jax.jit, static_argnames=("kernel", "return_position_density"))
def circular_kde(
    bins: jnp.ndarray,                       # (N_bins,) bin centers in [-pi, pi)
    trajectory: jnp.ndarray,                 # (T,) angles in radians
    spikes: jnp.ndarray,                     # (T, N_neurons) spike counts
    kernel=None,                             # unused placeholder
    kernel_bandwidth: float = 10.0,          # von Mises kappa
    mask: jnp.ndarray = None,                # (T, N_neurons) boolean
    return_position_density: bool = False,
    eps: float = 1e-6,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Circular KDE on [-pi, pi) with FFT circular convolution and von Mises kernel.

    Output is expected spikes per time-bin per angle-bin (divide by dt for Hz),
    shape (N_neurons, N_bins). If return_position_density=True, also returns
    the smoothed occupancy per neuron, shape (N_neurons, N_bins).

    IMPORTANT: bins are assumed to be uniformly spaced in [-pi, pi).
    """

    bins = jnp.asarray(bins).flatten()
    trajectory = jnp.asarray(trajectory).flatten()
    spikes = jnp.asarray(spikes)

    n_bins = bins.shape[0]
    T = trajectory.shape[0]
    n_neurons = spikes.shape[1]

    if mask is None:
        mask = jnp.ones((T, n_neurons), dtype=bool)
    mask_f = mask.astype(jnp.float32)

    # 1) bin indices consistent with [-pi, pi)
    idx = _bin_indices_minuspi_pi(trajectory, n_bins)  # (T,)

    # 2) von Mises kernel over offsets Δθ in [-pi, pi)
    # Build on symmetric grid => Δθ=0 sits at index n_bins//2
    dtheta = jnp.linspace(-jnp.pi, jnp.pi, n_bins, endpoint=False)
    kappa = kernel_bandwidth
    vm = jnp.exp(kappa * jnp.cos(dtheta))
    vm = vm / jnp.sum(vm)

    # Align for FFT: put Δθ=0 at index 0
    vm = jnp.roll(vm, -n_bins // 2)

    # 3) histogram per neuron using bincount (vmap over neurons)
    def hist_for_neuron(weights_t: jnp.ndarray) -> jnp.ndarray:
        return jnp.bincount(idx, weights=weights_t, length=n_bins)

    # occupancy: mask only
    pos_hist = vmap(hist_for_neuron, in_axes=1, out_axes=0)(mask_f)  # (N, B)

    # spikes: spikes * mask
    spike_w = spikes.astype(jnp.float32) * mask_f                    # (T, N)
    spike_hist = vmap(hist_for_neuron, in_axes=1, out_axes=0)(spike_w)  # (N, B)

    # 4) smooth via circular convolution
    pos_smooth = vmap(_circular_conv_fft_1d, in_axes=(0, None), out_axes=0)(pos_hist, vm)
    spike_smooth = vmap(_circular_conv_fft_1d, in_axes=(0, None), out_axes=0)(spike_hist, vm)

    # 5) ratio
    kde = spike_smooth / (pos_smooth + eps)
    # (optional alternative) kde = (spike_smooth + eps) / (pos_smooth + eps)

    if return_position_density:
        return kde, pos_smooth
    return kde