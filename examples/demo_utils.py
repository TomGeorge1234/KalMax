# Some (mostly plotting and poorly written) utilities for the Kalman filter demo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def animate_over_time(
            plot_function,
            t_start,
            t_end,
            speed_up=10, #relative to real time
            fps=20,):
    """Animates the plot function over time_range.

    Params
    ------
    plot_function: function, it must act as it's own init function when ax=None
        Function that takes ax, time and time_start as input and returns ax.
        Must have the following signature 
        def plot_function(ax,
                            time,
                            time_start=0 # time to start (relevant if plotting a trajectory) 
                            ):
            # do something
            return ax
        _type_: _description_
    """
    # plot function should take ax and time as input
    time_per_frame = speed_up/fps
    times = np.arange(t_start, t_end, time_per_frame)
    
    def update(time, ax, fig):
        for ax_ in fig.get_axes():
            ax_.clear()
        ax = plot_function(ax=ax, t_start=t_start, t_end=time)
        return ax
    
    ax = plot_function()
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(plt.gcf(), update, frames=times, fargs=(ax, plt.gcf(),), interval=1000/fps)        
    plt.close()
    return anim

def plot_trajectory(
        trajectory,
        time_stamps,
        ax=None, 
        t_start=None, 
        t_end=None,
        **plot_kwargs):
    """Plots a trajectory over a given time range

    Params
    ------
    ax: matplotlib axis, optional
        Axis to plot on
    t_start: float, optional
        Start time
    t_end: float, optional
        End time
    trajectory: np.ndarray, optional
        Trajectory to plot, shape (T, 2)
    time_stamps: np.ndarray, optional
        Time stamps for the trajectory, shape (T,)
    plot_kwargs: dict, optional
        Additional plotting arguments like 'color', 'alpha', 'label', 'scatter_points', 'show_line', 'linewidth', 'title', 'xlabel', 'ylabel'

    Returns
    -------
    ax: matplotlib axis
    """
    if t_start is None: t_start = time_stamps[0]
    if t_end is None: t_end = time_stamps[-1]
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(6, 6)) 
    
    id_start, id_end = np.argmin(np.abs(time_stamps - t_start)), np.argmin(np.abs(time_stamps-t_end))
    trajectory_ = trajectory[id_start:id_end]

    # Get plot kwargs
    color = plot_kwargs.get('color', 'k')
    scatter_points = plot_kwargs.get('scatter_points', True)
    show_line = plot_kwargs.get('show_line', True)
    linewidth = plot_kwargs.get('linewidth',1)
    title = plot_kwargs.get('title',None)
    xlabel = plot_kwargs.get('xlabel','x [m]')
    ylabel = plot_kwargs.get('ylabel','y [m]')
    alpha = plot_kwargs.get('alpha',1)
    label = plot_kwargs.get('label',None)
    min_x = plot_kwargs.get('min_x',trajectory[:,0].min().round(1))
    max_x = plot_kwargs.get('max_x',trajectory[:,0].max().round(1))
    min_y = plot_kwargs.get('min_y',trajectory[:,1].min().round(1))
    max_y = plot_kwargs.get('max_y',trajectory[:,1].max().round(1))
    
    if show_line:
        ax.plot(trajectory_[:,0],trajectory_[:,1],color=color, linewidth=linewidth, alpha=alpha, label=label)
    if scatter_points:
        ax.scatter(trajectory_[:,0],trajectory_[:,1],color=color, linewidth=0, s=6, alpha=alpha)
    ax.set_xlim(min_x, max_x); ax.set_ylim(min_y, max_y); ax.set_aspect('equal', 'box')
    ax.set_xticks([min_x, max_x]); ax.set_yticks([min_y, max_y])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    return ax 

def plot_ellipse(ax, mean, cov, color):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_) # convert from variance to standard deviation along the eigenvectors
    ell = matplotlib.patches.Ellipse(xy=mean,
                                     width=lambda_[0]*2, 
                                     height=lambda_[1]*2,
                                     angle=np.rad2deg(np.arctan(v[:, 0][1] / v[:, 0][0])),
                                     lw=1, 
                                     fill=True, 
                                     edgecolor=color,
                                     facecolor=color,
                                     alpha=0.5,)
    ax.add_artist(ell)
    return ax 