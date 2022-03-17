import numpy as np
import jax.numpy as jnp
import jax
from numba import njit

# == Fixed point solvers == #

def fwd_solver(f, 
               x_init, 
               tol=1e-7, 
               max_iter=1_000_000,
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
        current_iter += 1
        x = x_new

    if current_iter == max_iter:
        print(f"Warning: Hit maximum iteration number {max_iter}")
    else:
        if verbose:
            print(f"Iteration converged after {current_iter} iterations") 

    return x, current_iter

def newton_solver(f, x_init):
    "Apply Newton's algorithm."
    f_root = lambda x: f(x) - x
    g = lambda x: x - jnp.linalg.solve(jax.jacobian(f_root)(x), f_root(x))
    return fwd_solver(g, x_init)


# == Fixed point interface function == #

def fixed_point_interface(solver, f, params, x_init):
    """
    This function marginalizes f to operate on x alone and then calls the
    solver.
    """
    x_star, num_iter = solver(lambda x: f(x, params), x_init)
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

