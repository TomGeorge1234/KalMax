# **KalMax**:  Kalman based neural decoding in Jax 
**KalMax** = **Kal**man smoothing of **Max**imum likelihood estimates in Jax.

You provide $\mathbf{S} \in \mathbb{N}^{T \times N}$ (spike counts) and $\mathbf{X} \in \mathbb{R}^{T \times D}$ (a continuous variable, e.g. position) and `KalMax` provides jax-optimised functions and classes for:

1. **Fitting rate maps** using kernel density estimation (KDE)
2. **Calculating likelihood** maps $p(\mathbf{s}_t|\mathbf{x})$
3. **Kalman filter / smoother**

<img src="figures/display_figures/input_data.png" width=350>




#### Why are these functionalities combined into one package?...

Because Likelihood Estimation + Kalman filtering = Powerful neural decoding. By Kalman filtering/smoothing the maximum likelihood estimates (as opposed to the spikes themselves) we bypass the issues of naive Kalman filters (spikes are rarely linearly related to position) and maximum likelihood decoding (which does not account for temporal continuity in the trajectory), outperforming both for no extra computational cost.
<img src="figures/display_figures/filter_comparisons.gif" width=850>


Core `KalMax` functions are optimised and jit-compiled in jax making them **very fast**. For example `KalMax` kalman filtering is >13 times faster than an equivalent numpy implementation by the popular [`pykalman`](https://github.com/pykalman/pykalman/tree/master) library (see [demo](./kalmax_demo.ipynb)).

<img src="figures/display_figures/kalman_speed_comparison.png" width=150>


# Install
```
pip install kalmax
```

# Usage  

A full demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomGeorge1234/KalMax/blob/main/kalmax_demo.ipynb) is provided in the [`kalmax_demo.ipynb`](./kalmax_demo.ipynb). Sudo-code is provided below. 

```python
import kalmax 
import jax.numpy as jnp 
```

```python
# 0. PREPARE DATA IN JAX ARRAYS
S_train = jnp.array(...) # (T, N_CELLS)      train spike counts
Z_train = jnp.array(...) # (T, DIMS)         train continuous variable
S_test  = jnp.array(...) # (T_TEST, N_CELLS) test spike counts
bins    = jnp.array(...) # (N_BINS, DIMS)    coordinates at which to estimate receptive fields / likelihoods)
```
<img src="figures/display_figures/data.png" width=850>

```python
# 1. FIT RECEPTIVE FIELDS using kalmax.kde
firing_rate = kalmax.kde.kde(
    bins = bins,
    trajectory = Z_train,
    spikes = S_train,
    kernel = kalmax.kernels.gaussian_kernel,
    kernel_kwargs = {'covariance':0.01**2*np.eye(DIMS)}, # kernel bandwidth
    ) # --> (N_CELLS, N_BINS)
```
<img src="figures/display_figures/receptive_fields.png" width=850>


```python
# 2.1 CALCULATE LIKELIHOODS using kalmax.poisson_log_likelihood
log_likelihoods = kalmax.kde.poisson_log_likelihood(
    spikes = S_test,                       
    mean_rate = firing_rate,
    ) # --> (T_TEST, N_CELLS)

# 2.2 FIT GAUSSIAN TO LIKELIHOODS using kalmax.utils.fit_gaussian
MLE_means, MLE_modes, MLE_covs = kalmax.utils.fit_gaussian_vmap(
    x = bins, 
    likelihoods = jnp.exp(log_likelihoods),
    ) # --> (T_TEST, DIMS), (T_TEST, DIMS, DIMS)
```
<img src="figures/display_figures/likelihood_maps_fitted.png" width=850>

```python
# 3. KALMAN FILTER / SMOOTH using kalmax.KalmanFilter.KalmanFilter
kalman_filter = kalmax.kalman.KalmanFilter(
    dim_Z = DIMS, 
    dim_Y = N_CELLS,
    # SEE DEMO FOR HOW TO FIT/SET THESE
    F=F, # state transition matrix
    Q=Q, # state noise covariance
    H=H, # observation matrix
    R=R, # observation noise covariance
    ) 

# [FILTER]
mus_f, sigmas_f = kalman_filter.filter(
    Y = Y, 
    mu0 = mu0,
    sigma0 = sigma0,
    ) # --> (T, DIMS), (T, DIMS, DIMS)

# [SMOOTH]
mus_s, sigmas_s = kalman_filter.smooth(
    mus_f = mus_f, 
    sigmas_f = sigmas_f,
    ) # --> (T, DIMS), (T, DIMS, DIMS)
```
