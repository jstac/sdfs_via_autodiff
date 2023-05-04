"""
Multi-index routines for computing the wealth consumption ratio of the SSY
model.

"""

import jax
import jax.numpy as jnp
from jax.config import config

# Import local modules
import sys
sys.path.append('..')
from ssy_model import *
from utils import *


# Optionally tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)

# == Operator == #

def wc_ratio(T, 
             w_init,
             algorithm="newton",
             verbose=True):
    """
    Iterate solve for the fixed point of the Koopmans operator 
    associated with the model and return the wealth consumption ratio.

    - model is an instance of SSY or GCY

    """
    # Unpack 
    # shapes, params, arrays = model.discretize_multi_index()

    # Marginalize T given the parameters
    # T_operator = lambda w: T(w, shapes, params, arrays)
    # Set up the initial condition for the fixed point search
    # w_init = jnp.ones(shapes) * init_val

    try:
        solver = solvers[algorithm]
    except KeyError:
        msg = f"""\
                  Algorithm {algorithm} not found.  
                  Falling back to successive approximation.
               """
        print(dedent(msg))
        solver = fwd_solver

    # Call the solver
    w_star, num_iter = solver(T, w_init)

    return w_star


def test_wc_ratio_discrete_ssy():
    init_val = 800.0 
    ssy = SSY()
    L, K, I, J = 3, 3, 3, 3
    T = ssy.discrete_T_operator_factory(L, K, I, J)
    w_init = jnp.ones((L, K, I, J)) * init_val
    w_star = wc_ratio(T, w_init, algorithm="newton")
    print(w_star)

