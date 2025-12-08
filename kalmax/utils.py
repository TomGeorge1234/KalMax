import jax 
import jax.numpy as jnp
import tqdm as tqdm

def gaussian_pdf(x : jnp.ndarray,
                 mu : jnp.ndarray, 
                 sigma : jnp.ndarray,) -> jnp.ndarray:
    """ Calculates the gaussian pdf of a multivariate normal distribution of mean mu and covariance sigma at x

    Parameters
    ----------

    x: (D,) array
        The position at which to evaluate the pdf
    mu: (D,) array
        The mean of the distribution
    sigma: (D, D) array
        The covariance of the distribution
    
    Returns
    -------
    pdf: float
        The probability density at x
    """
    assert x.ndim == 1
    assert mu.ndim == 1
    assert sigma.ndim == 2
    assert x.shape[0] == mu.shape[0]
    assert x.shape[0] == sigma.shape[0]
    assert sigma.shape[0] == sigma.shape[1]

    x = x - mu
    norm_const = gaussian_norm_const(sigma)
    return norm_const * jnp.exp(-0.5 * jnp.sum(x @ jnp.linalg.inv(sigma) * x, axis=-1))

def log_gaussian_pdf(x : jnp.ndarray,
                     mu : jnp.ndarray,
                     sigma : jnp.ndarray,) -> jnp.ndarray:
    """ Calculates the log of the gaussian pdf of a multivariate normal distribution of mean mu and covariance sigma at x

    Parameters
    ----------
    x: (D,) array
        The position at which to evaluate the pdf
    mu: (D,) array
        The mean of the distribution
    sigma: (D, D) array
        The covariance of the distribution

    Returns 
    -------
    log_pdf: float
        The log probability density at x
    """
    assert x.ndim == 1
    assert mu.ndim == 1
    assert sigma.ndim == 2
    assert x.shape[0] == mu.shape[0]
    assert x.shape[0] == sigma.shape[0]
    assert sigma.shape[0] == sigma.shape[1]

    x = x - mu
    norm_const = gaussian_norm_const(sigma)
    return jnp.log(norm_const) - 0.5 * jnp.sum(x @ jnp.linalg.inv(sigma) * x)



def gaussian_norm_const(sigma : jnp.ndarray) -> jnp.ndarray:
    """Calculates the normalizing constant of a multivariate normal distribution with covariance sigma

    Parameters
    ----------
    sigma: jnp.ndarray, shape (D, D)
        The covariance matrix of the distribution

    Returns
    -------
    norm_const: jnp.ndarray, shape (1,)
        The normalizing constant
    """
    assert sigma.ndim == 2
    D = sigma.shape[0]
    return 1 / jnp.sqrt((2 * jnp.pi) ** D * jnp.linalg.det(sigma))

def fit_gaussian(x, likelihood):
    """Fits a multivariate-Gaussian to the likelihood function P(spikes | x) in x-space.
    
    Parameters
    ----------
    x : jnp.ndarray, shape (N_bins,D)
        The position bins in which the likelihood is calculated
    likelihood : jnp.ndarray, shape (N_bins,)
        The combined likelihood (not log-likelihood) of the neurons firing at each position bin
        
    Returns
    -------
    mu : jnp.ndarray, shape (D,)
        The mean of the Gaussian
    mode : jnp.ndarray, shape (D,)
        The mode of the Gaussian
    covariance : jnp.ndarray, shape (D, D)
        The covariance of the Gaussian    
    """
    assert x.ndim == 2
    assert likelihood.ndim == 1
    assert x.shape[0] == likelihood.shape[0]
    
    mu = (x.T @ likelihood) / likelihood.sum()
    mode = x[jnp.argmax(likelihood)]
    covariance = ((x - mu) * likelihood[:, None]).T @ (x - mu) / likelihood.sum()
    return mu, mode, covariance


# Like fit_gaussian, but accepts likelihoods of shape (T, N_bins)
# returns means, modes and covariances of shape (T, D), (T, D), (T, D, D)
fit_gaussian_vmap = jax.vmap(fit_gaussian, in_axes=(None, 0)) 



def make_simulated_dataset(time_mins = 60, n_cells = 100, firing_rate = 10, random_seed=None, **kwargs):
    """Makes a simulated dataset for an agent randomly foraging a 1 m square box. Data generated with the RatInABox package and defaults ot 50 place cells. Returns the data as jax arrays.
    
    Parameters
    ----------
    time_mins: int
        The number of minutes to simulate the agent for
    kwargs: dict
        Additional arguments to pass to the RatInABox simulation
    
    Returns
    -------
    time: jnp.ndarray, shape (N,)
        The time points of the simulation
    position: jnp.ndarray, shape (N, dims)
        The position of the agent at each time point
    spikes: jnp.ndarray, shape (N, N_cells)
        The spikes of the 50 place cells at each time point
    """

    from ratinabox.Environment import Environment 
    from ratinabox.Agent import Agent 
    from ratinabox.Neurons import PlaceCells

    if random_seed is not None:
        import numpy as np
        np.random.seed(random_seed)
    
    env_params = kwargs.get('env_params', {})
    agent_params = kwargs.get('agent_params', {'dt':0.1})
    place_cell_params = kwargs.get('place_cell_params', {'n':n_cells, 'max_fr':firing_rate,'widths':0.1})

    env = Environment(params=env_params)
    agent = Agent(env, params=agent_params)
    place_cells = PlaceCells(agent, params=place_cell_params)
    

    for i in tqdm.tqdm(range(int(60 * time_mins / agent.dt))):
        agent.update()
        place_cells.update()

    time  = jnp.array(agent.history['t'])
    position = jnp.array(agent.history['pos'])
    spikes = jnp.array(place_cells.history['spikes'])

    return time, position, spikes

