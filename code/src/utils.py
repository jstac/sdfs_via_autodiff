import numpy as np
import jax.numpy as jnp
import jax
from numba import njit
import jaxopt


# == Fixed point solvers == #

def fwd_solver(f,
               x_init,
               tol=1e-7,
               max_iter=1_000_000,
               print_skip=10,
               verbose=True):
    "Uses successive approximation on f."

    if verbose:
        print("Beginning iteration\n\n")

    current_iter = 0
    x = x_init
    error = tol + 1
    while error > tol and current_iter < max_iter:
        x_new = f(x)
        error = jnp.max(jnp.abs(x_new - x))
        if verbose and current_iter % print_skip == 0:
            print("iter = {}, error = {}".format(current_iter, error))
        current_iter += 1
        x = x_new

    if current_iter == max_iter:
        print(f"Warning: Hit maximum iteration number {max_iter}")
    else:
        if verbose:
            print(f"Iteration converged after {current_iter} iterations")

    return x, current_iter


def AA_solver(f, x_init, tol=1e-7, max_iter=10000, verbose=True,
              print_skip=10):
    # hard coded parameters for now
    AA = jaxopt.AndersonAcceleration(f, verbose=verbose, mixing_frequency=5,
                                     tol=tol, maxiter=max_iter, history_size=2,
                                     beta=8.0, implicit_diff=False,
                                     ridge=1e-5, jit=True, unroll=True)
    out = AA.run(x_init)
    w_out = out[0]
    current_iter = int(out[1][0])

    if current_iter == max_iter:
        print(f"Warning: Hit maximum iteration number {max_iter}")
    else:
        if verbose:
            print(f"Iteration converged after {current_iter} iterations")

    return w_out, current_iter


def newton_solver(f, x_init):
    "Apply Newton's algorithm."
    f_root = lambda x: f(x) - x
    g = lambda x: x - jnp.linalg.solve(jax.jacobian(f_root)(x), f_root(x))
    return fwd_solver(g, x_init)


# == Fixed point interface function == #

def fixed_point_interface(solver, f, params, x_init, tol=1e-7, verbose=True,
                          print_skip=10):
    """
    This function marginalizes f to operate on x alone and then calls the
    solver.
    """
    x_star, num_iter = solver(lambda x: f(x, params), x_init, tol=tol,
                              verbose=verbose, print_skip=print_skip)
    return x_star, num_iter


# == Misc. utility function == #

@njit
def draw_from_cdf(F, U):
    " Draws from F when U is uniform on (0, 1) "
    return np.searchsorted(F, U)


def compute_spec_rad(Q):
    """
    Function to compute spectral radius of a matrix.

    """
    return np.max(np.abs(np.linalg.eigvals(Q)))


# == Interpolation related utilities == #

@jax.jit
def jit_map_coordinates(vals, coords):
    return jax.scipy.ndimage.map_coordinates(vals, coords, order=1,
                                             mode='nearest')


def vals_to_coords(grids, x_vals):
    # jax.jit doesn't allow dynamic shapes
    dim = 4

    intervals = jnp.asarray([grid[1] - grid[0] for grid in grids])
    low_bounds = jnp.asarray([grid[0] for grid in grids])

    intervals = intervals.reshape(dim, 1)
    low_bounds = low_bounds.reshape(dim, 1)

    return (x_vals - low_bounds) / intervals
