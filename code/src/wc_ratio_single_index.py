"""

Single-index routines for computing the wealth consumption ratio of the SSY
model.

CPU and JAX based code.

"""

import numpy as np
from ssy_model import *
from utils import *
import jax
import jax.numpy as jnp
from jax.config import config

# Tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)


# == Operator == #

@jax.jit
def T(w, params):
    "T via JAX operations."
    H, β, θ = params
    Tw = 1 + β * (jnp.dot(H, (w**θ)))**(1/θ)
    return Tw


def wc_ratio_single_index(model, 
                 algorithm="newton",
                 init_val=800, 
                 single_index_output=False,   # output as w[m] or w[l, k, i, j]
                 verbose=True):
    """
    Iterate to convergence on the Koopmans operator associated with the SSY
    model and then return the wealth consumption ratio.

    - model is an instance of SSY or GCY

    """

    # Unpack 
    β, θ, N, H = model.β, model.θ, model.N, model.H
    H = jax.device_put(H)  # Put H on the device (GPU)
    params = H, β, θ 

    w_init = jnp.ones(N) * init_val

    try:
        solver = solvers[algorithm]
    except KeyError:
        msg = f"""\
                  Algorithm {algorithm} not found.  
                  Falling back to successive approximation.
               """
        print(dedent(msg))
        solver = fwd_solver

    # Marginalize T given the parameters
    T_operator = lambda x: T(x, params)
    # Call the solver
    w_star, num_iter = solver(T_operator, w_init)

    # Return output in desired shape
    if single_index_output:
        w_out = w_star
    else:
        L, K, I, J = model.L, model.K, model.I, model.J
        M = L * K * I * J
        w_out = np.empty((L, K, I, J))
        for m in range(M):
            l, k, i, j = single_to_multi(m, K, I, J)
            w_out[l, k, i, j] = w_star[m]

    return w_out
