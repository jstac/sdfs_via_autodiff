
"""

Multi-index discretized routines for computing the wealth consumption ratio of
the GCY model.

"""

import numpy as np
from numba import njit

import jax
import jax.numpy as jnp

# Import local modules
import sys
sys.path.append('..')
from gcy_model import *
sys.path.append('../..')
from solvers import solver

from quantecon import MarkovChain, rouwenhorst

# Optionally tell JAX to use 64 bit floats
from jax.config import config
config.update("jax_enable_x64", True)


def discretize_gcy(gcy, shapes):
    """
    Discretizes the GCY model using a multi-index and returns a discrete
    representation of the model.  The states 

        (z, z_π, h_z, h_c, h_zπ, h_λ)

    are indexed with 

        (i_z, i_z_π, i_h_z, i_h_c, i_h_zπ, i_h_λ)
        
    and the sizes are 

        (n_z, n_z_π, n_h_z, n_h_c, n_h_zπ, n_h_λ)

    The discretizations for z and z_π use iterations of the Rouwenhorst method. 

        z_π_states[i_h_zπ, i_z_π] is the i_z_π-th z_π state 

        z_π_Q[i_h_zπ, i_z_π, j_z_π] = trans prob for z_π 

        z_states[i_z_π, i_h_z, i_h_zπ, i_z] is the i_z-th z state 

        z_Q[i_z_π, i_h_z, i_h_zπ, i_z, i_z] = trans prob for z 
    """

    # Organize and unpack
    n_z, n_z_π, n_h_z, n_h_c, n_h_zπ, n_h_λ = shapes
    params = gcy.params
    (β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, 
         ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z, 
         ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = params

    # Discretize the h processes
    h_z_mc = rouwenhorst(n_h_z, ρ_z, s_z)
    h_c_mc = rouwenhorst(n_h_c, ρ_c, s_c)
    h_zπ_mc = rouwenhorst(n_h_zπ, ρ_zπ, s_zπ)
    h_λ_mc = rouwenhorst(n_h_λ, ρ_λ, s_λ)

    # Build h states
    h_z_states = h_z_mc.state_values
    h_c_states = h_c_mc.state_values
    h_zπ_states = h_zπ_mc.state_values
    h_λ_states = h_λ_mc.state_values

    # Build h transition probabilities
    h_z_Q = h_z_mc.P
    h_c_Q = h_c_mc.P
    h_zπ_Q = h_zπ_mc.P
    h_λ_Q = h_λ_mc.P

    # For convenience, store the sigma states as well
    σ_z_states = φ_z * np.exp(h_z_states)
    σ_c_states = φ_c * np.exp(h_c_states)
    σ_zπ_states = φ_zπ * np.exp(h_zπ_states)

    # Next we build the states and transitions for 
    #
    #   z_π' = ρ_ππ * z_π + σ_zπ * η1
    #
    # Note that z_π will have a one-dimensional grid for each
    # value of σ_zπ, and hence h_zπ.
    z_π_states = np.zeros((n_h_zπ, n_z_π))
    # z_π_Q holds transition probs over i_z_π for each i_h_zπ
    z_π_Q = np.zeros((n_h_zπ, n_z_π, n_z_π))
    for i_h_zπ, σ_zπ in enumerate(σ_zπ_states):
        mc = rouwenhorst(n_z_π, ρ_ππ, σ_zπ)
        for i_z_π in range(n_z_π):
            z_π_states[i_h_zπ, i_z_π] = mc.state_values[i_z_π]
            z_π_Q[i_h_zπ, i_z_π, :] = mc.P[i_z_π, :]

    # Finally, we build the states and transitions for 
    #
    #   z' = ρ * z + ρ_π * z_π + σ_z * η0
    #
    # Note that z will have a one-dimensional grid for each
    # pair (z_π, σ_z, h_zπ), and hence each (z_π, h_z, h_zπ).
    z_states = np.zeros((n_z_π, n_h_z, n_h_zπ, n_z))
    # z_Q holds transition probs over i_z for each (i_z_π, i_h_z, i_h_zπ)
    z_Q = np.zeros((n_z_π, n_h_z, n_h_zπ, n_z, n_z))
    for i_h_zπ in range(n_h_zπ):
        for i_h_z, σ_z in enumerate(σ_z_states):
            for i_z_π, z_π in enumerate(z_π_states[i_h_zπ, :]):
                mc = rouwenhorst(n_z, ρ, σ_z, ρ_π * z_π)
                for i_z in range(n_z):
                    z_states[i_z_π, i_h_z, i_h_zπ, i_z] = mc.state_values[i_z]
                    z_Q[i_z_π, i_h_z, i_h_zπ, i_z, :] = mc.P[i_z, :]

    # Set up all arrays and put them on the device
    arrays =  (z_states,    z_Q,
               z_π_states,  z_π_Q,
               h_z_states,  h_z_Q,  σ_z_states,
               h_c_states,  h_c_Q,  σ_c_states, 
               h_zπ_states, h_zπ_Q, σ_zπ_states,
               h_λ_states,  h_λ_Q)


    return arrays


def T_gcy(w, shapes, params, arrays):
    """
    Implement a discrete version of the operator T for the GCY model via
    JAX operations.  In reading the following, it is helpful to remember
    that the state is 
        
        (z, z_π, h_z, h_c, h_zπ, h_λ)

    """

    # Unpack and set up
    n_z, n_z_π, n_h_z, n_h_c, n_h_zπ, n_h_λ = shapes

    (β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, 
         ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z, 
         ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = params

    (z_states,      z_Q,
     z_π_states,  z_π_Q,
     h_z_states,  h_z_Q,  σ_z_states,
     h_c_states,  h_c_Q,  σ_c_states, 
     h_zπ_states, h_zπ_Q, σ_zπ_states,
     h_λ_states,  h_λ_Q)   = arrays

    θ = (1 - γ) / (1 - 1/ψ)
    n = len(shapes)

    # Number the states (z, z_π, h_z, h_c, h_zπ, h_λ)
    state_numbers = { 'z'      : 0,
                      'z_π'    : 1,
                      'h_z'    : 2,
                      'h_c'    : 3, 
                      'h_zπ'   : 4, 
                      'h_λ'    : 5}

    w = jnp.reshape(w, (1, 1, 1, 1, 1, 1, n_z, n_z_π, n_h_z, n_h_c, n_h_zπ, n_h_λ))

    # Create intermediate arrays
    A1 = jnp.exp(θ * h_λ_states)
    indices_a = [1] * n 
    indices_b = [1] * n 
    indices_b[state_numbers['h_λ']] = n_h_λ
    indices = indices_a + indices_b
    A1 = jnp.reshape(A1, indices)

    A2 = jnp.exp(0.5 * ((1 - γ) * σ_c_states)**2)
    indices_a = [1] * n 
    indices_b = [1] * n 
    indices_a[state_numbers['h_c']] = n_h_c
    indices = indices_a + indices_b
    A2 = jnp.reshape(A2, indices)

    A3 = jnp.exp((1 - γ) * (μ_c + z_states))
    # z_states have the form [i_z_π, i_h_z, i_h_zπ, i_z]
    indices_a = [1] * n 
    indices_b = [1] * n 
    indices_a[state_numbers['z_π']] = n_z_π
    indices_a[state_numbers['h_z']] = n_h_z
    indices_a[state_numbers['h_zπ']] = n_h_zπ
    indices_a[state_numbers['z']] = n_z
    indices = indices_a + indices_b
    A3 = jnp.reshape(A3, indices)

    # Reshape h_z_Q
    indices = [1] * n 
    indices[state_numbers['h_z']] = n_h_z
    h_z_Q = jnp.reshape(h_z_Q, indices + indices)

    # Reshape h_zπ_Q
    indices = [1] * n 
    indices[state_numbers['h_zπ']] = n_h_zπ
    h_zπ_Q = jnp.reshape(h_zπ_Q, indices + indices)

    # Reshape h_λ_Q
    indices = [1] * n 
    indices[state_numbers['h_λ']] = n_h_λ
    h_λ_Q = jnp.reshape(h_λ_Q, indices + indices)

    # Reshape h_c_Q
    indices = [1] * n 
    indices[state_numbers['h_c']] = n_h_c
    h_c_Q = jnp.reshape(h_c_Q, indices + indices)

    # Reshape z_π_Q[i_h_zπ, i_z_π, j_z_π]
    indices_a = [1] * n 
    indices_a[state_numbers['h_zπ']] = n_h_zπ
    indices_a[state_numbers['z_π']] = n_z_π
    indices_b = [1] * n 
    indices_b[state_numbers['z_π']] = n_z_π
    z_π_Q = jnp.reshape(z_π_Q, indices_a + indices_b)

    breakpoint()

    # Reshape z_Q[i_z_π, i_h_z, i_h_zπ, i_z, j_z]
    indices_a = [1] * n 
    indices_a[state_numbers['z_π']] = n_z_π
    indices_a[state_numbers['h_z']] = n_h_z
    indices_a[state_numbers['h_zπ']] = n_h_zπ
    indices_a[state_numbers['z']] = n_z
    indices_b = [1] * n 
    indices_b[state_numbers['z']] = n_z
    z_Q = jnp.reshape(z_Q, indices_a + indices_b)

    # Take product and sum along last six axes
    H = A1 * A2 * A3 * h_z_Q * h_zπ_Q * h_λ_Q * h_c_Q * z_π_Q * z_Q
    Hwθ = jnp.sum(w**θ * H, axis=(6, 7, 8, 9, 10, 11))

    # Define and return Tw
    Tw = 1 + β * Hwθ**(1/θ)
    return Tw

T_gcy = jax.jit(T_gcy, static_argnums=(1, ))



# ============= Tests ====================== #

@njit
def T_gcy_loops(w, shapes, params, arrays):
    """
    This function replicates T_gcy but with for loops.  It's only
    purpose is to test the vectorized code in T_gcy.  It should only be
    used with very small shapes!
    """
    
    # Unpack and set up
    n_z, n_z_π, n_h_z, n_h_c, n_h_zπ, n_h_λ = shapes

    (β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, 
         ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z, 
         ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = params

    (z_states,      z_Q,
     z_π_states,  z_π_Q,
     h_z_states,  h_z_Q,  σ_z_states,
     h_c_states,  h_c_Q,  σ_c_states, 
     h_zπ_states, h_zπ_Q, σ_zπ_states,
     h_λ_states,  h_λ_Q)   = arrays

    θ = (1 - γ) / (1 - 1/ψ)
    Hwθ = np.empty(shapes)

    for i_h_z, h_z in enumerate(h_z_states):
        for i_h_c, h_c in enumerate(h_c_states):
            for i_h_zπ, h_zπ in enumerate(h_zπ_states):
                for i_h_λ, h_λ in enumerate(h_λ_states):
                    for i_z_π, z_π in enumerate(z_π_states[i_h_zπ, :]):
                        for i_z, z in enumerate(z_states[i_z_π, i_h_z, i_h_zπ, :]):
                            Hwθ_sum = 0.0
                            σ_c = σ_c_states[i_h_c]
                            a2 = np.exp(0.5 * ((1 - γ) * σ_c)**2)
                            a3 = np.exp((1 - γ) * (μ_c + z))
                            for j_z, zp in enumerate(z_states[i_z_π, i_h_z, i_h_zπ, :]):
                                for j_z_π, z_πp in enumerate(z_π_states[i_h_zπ, :]):
                                    for j_h_z, h_zp in enumerate(h_z_states):
                                        for j_h_c, h_cp in enumerate(h_c_states):
                                            for j_h_zπ, h_zπp in enumerate(h_zπ_states):
                                                for j_h_λ, h_λp in enumerate(h_λ_states):
                                                        a1 = np.exp(θ * h_λp)
                                                        p0 = z_Q[i_z_π, i_h_z, i_h_zπ, i_z, j_z] 
                                                        p1 = z_π_Q[i_h_zπ, i_z_π, j_z_π] 
                                                        p2 = h_z_Q[i_h_z, j_h_z] 
                                                        p3 = h_c_Q[i_h_c, j_h_c] 
                                                        p4 = h_zπ_Q[i_h_zπ, j_h_zπ] 
                                                        p5 = h_λ_Q[i_h_λ, j_h_λ] 
                                                        a = a1 * a2 * a3
                                                        p = p0 * p1 * p2 * p3 * p4 * p5 
                                                        Hwθ_sum +=  \
                                                            w[j_z, j_z_π, j_h_z, j_h_c, j_h_zπ, j_h_λ]**θ * \
                                                             p * a
                            Hwθ[i_z, i_z_π, i_h_z, i_h_c, i_h_zπ, i_h_λ] = Hwθ_sum

    # Define and return Tw
    Tw = 1 + β * Hwθ**(1/θ)
    return Tw


def test_vectorized_equals_loops(shapes=(2, 3, 4, 5, 6, 7)):
    """
    Test that T generated by vectorized logic produces the same result as T
    generated by loops.  Evaluate at a random choice of w.
    """
    gcy = GCY()
    params = gcy.params
    arrays = discretize_gcy(gcy, shapes)
    w = np.exp(np.random.randn(*shapes))  # Test operator at w
    w1 = T_gcy(w, shapes, params, arrays)
    w2 = T_gcy_loops(w, shapes, params, arrays)
    print(np.allclose(w1, w2))


def test_compute_wc_ratio_gcy(shapes=(3, 3, 3, 3, 3, 3), algo="successive_approx"):
    """
    Solve a small version of the model using T_gcy.
    """
    gcy = GCY()

    # Build discrete rep of GCY
    params = gcy.params
    arrays = discretize_gcy(gcy, shapes)

    # Shift arrays to the device
    arrays = [jax.device_put(array) for array in arrays]

    # Marginalize T
    T = lambda w : T_gcy(w, shapes, params, arrays)

    # Call the solver
    init_val = 800.0 
    w_init = jnp.ones(shapes) * init_val
    w_star = solver(T, w_init, algorithm=algo)

    return w_star


