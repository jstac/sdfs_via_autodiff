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

def T(w, params):
    "T via JAX operations."
    H, β, θ = params
    Tw = 1 + β * (jnp.dot(H, (w**θ)))**(1/θ)
    return Tw

T = jax.jit(T)


def wc_ratio_single_index(model, 
                 algorithm="newton",
                 init_val=np.exp(5), 
                 single_index_output=False,   # output as w[m] or w[l, k, i, j]
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
    w_init = jax.device_put(w_init)
    H = jax.device_put(H)

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

