"""

=== Not currently working, probably will be discarded ===

Multi-index routines for computing the wealth consumption ratio of the SSY
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

    # Unpack
    (L, K, I, J,
     β, θ, γ, μ_c,
     h_λ_states, h_λ_P,              
     h_c_states, h_c_P,
     h_z_states, h_z_P,
     z_states,   z_Q,
     σ_c_states, σ_z_states) = params

    # Create intermediate arrays
    Z1 = jnp.reshape(jnp.exp(θ * h_λ_states), (1, L))
    Pλ = h_λ_P * Z1
    Z2 = jnp.reshape(jnp.exp(0.5*((1-γ)*σ_c_states)**2), (K, 1))
    Pc = h_c_P * Z2
    Z3 = jnp.reshape(jnp.exp((1-γ)*(μ_c+z_states)), (I, J, 1))
    Pz = z_Q * Z3

    # Reshape prior to summing
    Pλ = jnp.reshape(Pλ, (L, 1, 1, 1, L, 1, 1, 1))
    Pc = jnp.reshape(Pc, (1, K, 1, 1, 1, K, 1, 1))
    h_z_P = jnp.reshape(h_z_P, (1, 1, I, 1, 1, 1, I, 1))
    Pz = jnp.reshape(Pz, (1, 1, I, J, 1, 1, 1, J))
    w = jnp.reshape(w, (1, 1, 1, 1, L, K, I, J))

    # Sum along last four axes
    return jnp.sum(Pλ * Pc * h_z_P * Pz * w, axis=(4, 5, 6, 7))

#T = jax.jit(T)


def wc_ratio_multi_index(model, 
                          algorithm="newton",
                          init_val=np.exp(5), 
                          verbose=True):
    """
    Iterate to convergence on the Koopmans operator associated with the SSY
    model and then return the wealth consumption ratio.

    - model is an instance of SSY or GCY

    """


    # Unpack 
    m = model
    params = (m.L, m.K, m.I, m.J,
         m.β, m.θ, m.γ, m.μ_c,
         m.h_λ_states, m.h_λ_P,              
         m.h_c_states, m.h_c_P,
         m.h_z_states, m.h_z_P,
         m.z_states,   m.z_Q,
         m.σ_c_states, m.σ_z_states) 

    w_init = jnp.ones((m.L, m.K, m.I, m.J)) * init_val

    # Choose the solver
    solver = newton_solver if algorithm == "newton" else fwd_solver

    # Call the solver
    w_star, iter = fixed_point_interface(solver, T, params, w_init)

    return w_star

