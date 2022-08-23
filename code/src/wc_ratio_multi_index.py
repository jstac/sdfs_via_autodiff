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

def _T(w, shapes, constants, arrays):
    "T via JAX operations."

    # Unpack
    L, K, I, J = shapes
    β, θ, γ, μ_c = constants
    (h_λ_states, h_λ_P,              
     h_c_states, h_c_P,
     h_z_states, h_z_P,
     z_states,   z_Q,
     σ_c_states, σ_z_states) = arrays

    # Create intermediate arrays
    B1 = jnp.exp(θ * h_λ_states)
    B1 = jnp.reshape(B1, (1, 1, 1, 1, L, 1, 1, 1))

    B2 = jnp.exp(0.5*((1-γ)*σ_c_states)**2)
    B2 = jnp.reshape(B2, (1, K, 1, 1, 1, 1, 1, 1))

    B3 = jnp.exp((1-γ)*(μ_c+z_states))
    B3 = jnp.reshape(B3, (1, 1, I, J, 1, 1, 1, 1))

    # Reshape existing matrices prior to summing
    Phλ = jnp.reshape(h_λ_P, (L, 1, 1, 1, L, 1, 1, 1))
    Phc = jnp.reshape(h_c_P, (1, K, 1, 1, 1, K, 1, 1))
    Phz = jnp.reshape(h_z_P, (1, 1, I, 1, 1, 1, I, 1))
    Pz = jnp.reshape(z_Q, (1, 1, I, J, 1, 1, 1, J))
    w = jnp.reshape(w, (1, 1, 1, 1, L, K, I, J))

    # Take product and sum along last four axes
    A = w**θ * B1 * B2 * B3 * Phλ * Phc * Phz * Pz
    B = np.sum(A, axis=(4, 5, 6, 7))

    Tw = 1 + β * B**(1/θ)

    return Tw

_T = jax.jit(_T, static_argnums=(1, 2))

def T(w, params):

    (L, K, I, J,
    β, θ, γ, μ_c,
    h_λ_states, h_λ_P,              
              h_c_states, h_c_P,
              h_z_states, h_z_P,
              z_states,   z_Q,
              σ_c_states, σ_z_states) = params

    shapes = L, K, I, J
    constants = β, θ, γ, μ_c,
    arrays = (h_λ_states, h_λ_P,              
              h_c_states, h_c_P,
              h_z_states, h_z_P,
              z_states,   z_Q,
              σ_c_states, σ_z_states) 
    return _T(w, shapes, constants, arrays)

    


def wc_ratio_multi_index(model, 
                          algorithm="newton",
                          init_val=800, 
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

