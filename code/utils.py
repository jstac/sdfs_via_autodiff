import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
from numba import njit
from textwrap import dedent


# == Fixed point solvers == #

default_tolerance = 1e-7
default_max_iter = int(1e6)

def fwd_solver(f,
               x_init,
               tol=default_tolerance,
               max_iter=default_max_iter,
               verbose=True,
               print_skip=1000):
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


def newton_solver(f, 
                  x_init, 
                  tol=default_tolerance, 
                  max_iter=default_max_iter,
                  bicgstab_atol=1e-4,
                  verbose=True,
                  print_skip=1):
    """
    Apply Newton's algorithm to find a fixed point of f. The routine defines 
    g via g(x) = f(x) - x and then searches for a root of g via Newton's
    method, which iterates on 

        x_{n+1} = x_n - J(x_n)^{-1} g(x_n)

    until convergence, where J(x) is the Jacobian of g at x. The implementation 
    below defines 

        q(x) := x - J(x)^{-1} g(x)

    and passes this function to fwd_solver.

    To compute J(x)^{-1} g(x) we can in principle use
    `jnp.linalg.solve(jax.jacobian(g)(x), g(x))`. However, this operation is
    very memory intensive when x is high-dimensional. It also requires that g
    is a regular 2D array (matrix), which necessitates conversion to a single
    index. 

    To avoid instantiating the large matrix J(x), we use jax.jvp to define the
    linear map v -> J(x) v. This map is computed on demand for any given v,
    which avoids instantiating J(x).  We then pass this to a solver that can
    invert arbitrary linear maps.
    """
    g = lambda x: f(x) - x
    def q(x):
        # First we define the map v -> J(x) v from x and g
        jac_x_prod = lambda v: jax.jvp(g, (x,), (v,))[1]
        # Next we compute J(x)^{-1} g(x).  Currently we use 
        # sparse.linalg.bicgstab. Another option is sparse.linalg.bc
        # but this operation seems to be less stable.
        b = jax.scipy.sparse.linalg.bicgstab(
                jac_x_prod, g(x), 
                atol=bicgstab_atol)[0]
        return x - b
    return fwd_solver(q, x_init, tol, max_iter, verbose, print_skip)


def anderson_solver(f, 
                    x_init, 
                    tol=default_tolerance, 
                    max_iter=default_max_iter,
                    verbose=True):
    # hard coded parameters for now
    jax_a = jaxopt.AndersonAcceleration(f, 
                                        verbose=verbose, 
                                        mixing_frequency=5,
                                        tol=tol, 
                                        maxiter=max_iter, 
                                        history_size=2,
                                        beta=8.0, 
                                        implicit_diff=False,
                                        ridge=1e-5, 
                                        jit=True, 
                                        unroll=True)
    out = jax_a.run(x_init)
    w_out = out[0]
    current_iter = int(out[1][0])

    if current_iter == max_iter:
        print(f"Warning: Hit maximum iteration number {max_iter}")
    else:
        if verbose:
            print(f"Iteration converged after {current_iter} iterations")

    return w_out, current_iter


def fixed_point_via_gradient_decent(f, x_init):

    def loss(x):
        v = f(x) - x
        return jnp.dot(v.flatten(), v.flatten())

    gd = jaxopt.GradientDescent(fun=loss, 
                                maxiter=1000, 
                                tol=0.0001,
                                stepsize=0.0)
    res = gd.run(init_params=x_init)
    solution, state = res

    return solution


# == List solvers for simple access == #

# A dictionary of available solvers.
solvers = dict((("newton", newton_solver),
                ("anderson", anderson_solver),
                ("gd", fixed_point_via_gradient_decent),
                ("successive_approx", fwd_solver)))





