"""
This file provides basic functions for the Jupyter notebooks used in the Introductory Biophysics module
at Ghent University in the academic year 2025/2026.
"""

__author__ = "Schäfer, Torben"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Global state flags
_hint_accuracy = False


# Euler's method


def create_statespace(size: tuple[tuple[float, float], tuple[float, float]],
                      resolution: int,
                      model,
                      title: None | str = None,
                      labels: None | tuple[str, str] = None,
                      figsize: tuple[int, int] = (6, 4),
                      color = plt.cm.plasma,
                      show: bool=True
                      ) -> None:
    """
    Creates the state space for a given two-dimensional model to visualise dynamical change.

    Parameters:
    - size: Sets borders for plot for the x and y axis.
    - resolution: Determines number of direction arrows.
    - model: Function for two-dimensional model of differential equations.
    """

    # Setup space
    x_axis, y_axis = size
    x_values = np.linspace(*x_axis, resolution)
    y_values = np.linspace(*y_axis, resolution)

    # Create grid
    x, y = np.meshgrid(x_values, y_values)
    u = np.zeros_like(x)
    v = np.zeros_like(y)

    # Compute direction vectors
    dx, dy = model((x, y))
    u, v = dx, dy

    # Estimates vector lengths
    lengths = np.hypot(u, v)

    # Create colormap
    colormap = color
    vmin = np.percentile(lengths, 5)
    vmax = np.percentile(lengths, 95)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Normalisation
    lengths[lengths == 0] = 1
    u /= lengths
    v /= lengths

    # Visualisation
    plt.figure(figsize=figsize)
    quiver = plt.quiver(x, y, u, v, lengths, cmap=colormap, norm=norm)
    plt.colorbar(quiver, label="Change Rate")
    plt.xlim(*x_axis)
    plt.ylim(*y_axis)
    plt.title(title if title else "Two-dimensional State Space")
    plt.xlabel(labels[0] if labels else "x-Axis")
    plt.ylabel(labels[1] if labels else "y-Axis")
    plt.tight_layout()
    if show:
        plt.show()
    return


def estimate_delta(start: np.array, model, accuracy: int = 2, stop: int = 8, steps: int = 10) -> float:
    """
    Dynamically estimate suitable delta values for Euler's method.

    Parameters:
    - start: Current state vector x_n of the system.
    - model: Callable function representing the two-dimensional system of differential equations.
    - accuracy: Number of decimal places used for the comparison between trajectories.
    - stop: Maximum negative exponent defining the minimum step size (delta threshold).
    - steps: Number of substeps used to estimate accuracy.

    Returns:
    - delta: Adaptive step size for the numerical calculation.
    """

    global _hint_accuracy
    delta = 0.1

    # Iteratively refine delta
    while True:
        projection = np.zeros((steps, 2))
        projection[0] = start

        # Calculate fine trajectory
        for step in range(1, steps):
            previous = projection[step - 1]
            change = np.array(model(previous))
            projection[step] = previous + (delta / steps) * change
        fine = projection[-1]

        # Calculate coarse trajectory
        coarse = start + delta * np.array(model(start))

        # Return delta if changes are within threshold
        if np.allclose(fine, coarse, atol=10**(-accuracy)):
            return delta

        # Stop iteration if below threshold
        elif delta <= 10 ** (-stop):
            if not _hint_accuracy:
                print("Hint: Accuracy of calculation cannot be guaranteed.")
                _hint_accuracy = True
            return delta

        # Reduce delta for next iteration
        delta /= 10
    return


def calculate_trajectory(start: np.array,
                           model,
                           delta: float = 0.1,
                           steps: int = 10_000,
                           dynamic=False) -> np.ndarray:
    """
    Calculate a trajectory for given start values in a two-dimensional model
    using Euler's method.

    Parameters:
    - start: Starting point for the trajectory calculation.
    - model: Callable function representing the two-dimensional system of differential equations.
    - delta: Step size for the numerical integration.
    - steps: Number of integration steps.

    Returns:
        Array of (x, y) trajectory points.
    """

    # Setup array
    trajectory = np.zeros((steps, 2))
    trajectory[0] = start
    deltas = np.zeros(steps - 1)

    # Calculate trajectory
    for step in range(1, steps):
        previous = trajectory[step - 1]
        change = np.array(model(previous))
        if dynamic:
            delta = estimate_delta(previous, model)
            deltas[step - 1] = delta
        trajectory[step] = previous + delta * change
    if dynamic:
        print("Step size (δt) statistics:")
        print(f"Min: {np.min(deltas)}, Max: {np.max(deltas)}, Average: {round(np.mean(deltas), 8)}")
    return trajectory


def plot_trajectory(trajectory: np.array,
                    size: tuple[tuple[float, float], tuple[float, float]],
                    resolution: int,
                    model,
                    title: None | str = None,
                    labels: None | tuple[str, str] = None,
                    figsize: tuple[int, int] = (6, 4),
                    color_field = plt.cm.plasma,
                    color_traj="red") -> None:
    """Plot the trajectory in an existing state space."""
    create_statespace(size, resolution, model, title, labels, figsize, color_field, show=False)
    plt.plot(*trajectory.T, color=color_traj)
    plt.show()
    return


def plot_time_series(trajectory: np.array,
                     delta: float,
                     title: None | str = None,
                     labels: None | tuple[str, str] = None,
                     yaxis: None | str = None,
                     figsize: tuple[int, int] = (6, 4),
                     colors: tuple[str, str] = ("red", "blue")
                     ) -> None:
    """
    Plot the time evolution of variables in a trajectory array.

    Parameters:
    - trajectory: 2D array containing the values of variables at each time step.
    - delta: Time step size used in the previous Euler integration.
    - title: Optional title of the plot.
    - labels: Optional labels for each variable.
    - yaxis: Optional label for the y-axis.
    - figsize: Size of the figure in inches.
    - colors: Line colors for each variable.
    """
    plt.figure(figsize=figsize)

    time, plots = np.shape(trajectory)

    for plot in range(plots):
        plt.plot(np.arange(time) * delta,
                 trajectory[:, plot],
                 label=labels[plot] if labels else f"Var {plot + 1}",
                 color=colors[plot] if colors else None)

    plt.xlabel("Time")
    plt.ylabel(yaxis if yaxis else "Value")
    plt.title(title if title else "Time Series")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return None
