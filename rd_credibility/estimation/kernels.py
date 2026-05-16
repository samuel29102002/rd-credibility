"""Kernel functions for local polynomial regression."""

import numpy as np


def triangular(u):
    """
    Triangular kernel.

    Parameters
    ----------
    u : np.ndarray
        Scaled distances, typically (x - cutoff) / bandwidth.

    Returns
    -------
    np.ndarray
        Kernel weights, zero outside [-1, 1].
    """
    u = np.asarray(u, dtype=np.float64)
    return np.maximum(1.0 - np.abs(u), 0.0)


def uniform(u):
    """
    Uniform (rectangular) kernel.

    Parameters
    ----------
    u : np.ndarray
        Scaled distances.

    Returns
    -------
    np.ndarray
        Kernel weights, 0.5 inside [-1, 1], zero outside.
    """
    u = np.asarray(u, dtype=np.float64)
    return 0.5 * (np.abs(u) <= 1.0).astype(np.float64)


def epanechnikov(u):
    """
    Epanechnikov kernel.

    Parameters
    ----------
    u : np.ndarray
        Scaled distances.

    Returns
    -------
    np.ndarray
        Kernel weights, zero outside [-1, 1].
    """
    u = np.asarray(u, dtype=np.float64)
    return 0.75 * np.maximum(1.0 - u**2, 0.0)


KERNELS = {
    "triangular": triangular,
    "uniform": uniform,
    "epanechnikov": epanechnikov,
}


def get_kernel(name):
    """
    Retrieve a kernel function by name.

    Parameters
    ----------
    name : str
        One of 'triangular', 'uniform', 'epanechnikov'.

    Returns
    -------
    callable
        The kernel function.

    Raises
    ------
    ValueError
        If the kernel name is not recognized.
    """
    if name not in KERNELS:
        raise ValueError(f"Unknown kernel '{name}'. Choose from {list(KERNELS.keys())}")
    return KERNELS[name]
