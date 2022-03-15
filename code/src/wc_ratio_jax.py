"""

Single-index routines for computing the wealth consumption ratio of the SSY
model.

CPU and JAX based code.

"""

import numpy as np
from ssy_model import *
import jax
import jax.numpy as jnp
from jax import jit
from jax.config import config

# Tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)


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
            print(f"Iteration converged after {iter} iterations") 

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


# == Operator == #

def T(w, params):
    "T via JAX operations."
    H, β, θ = params
    Tw = 1 + β * (jnp.dot(H, (w**θ)))**(1/θ)
    return Tw

T = jit(T)


def wc_ratio(model, 
             algorithm="newton",
             init_val=np.exp(5), 
             single_index_output=True,   # output as w[m] or w[l, k, i, j]
             verbose=True):
    """
    Iterate to convergence on the Koopmans operator associated with the SSY
    model and then return the wealth consumption ratio.

    - model is an instance of SSY or GCY

    """

    # Unpack 
    β = model.β
    θ = model.θ
    N = model.N
    H = model.H

    w_init = np.ones(N) * init_val

    # Convert arrays to jnp
    w_init = jnp.array(w_init)
    H = jnp.array(H)

    # Choose the solver
    solver = newton_solver if algorithm == "newton" else fwd_solver

    # Call the solver
    params = H, β, θ 
    w_star, iter = fixed_point_interface(solver, T, params, w_init)

    # Return output in desired shape
    if single_index_output:
        w_out = w_star
    else:
        w_out = np.empty((L, K, I, J))
        for m in range(M):
            l, k, i, j = single_to_multi(m, K, I, J)
            w_out[l, k, i, j] = w_star[m]

    return w_out

