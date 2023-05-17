"""
Multi-index discretized routines for computing the wealth consumption ratio of
the GCY model.

"""

import jax
import jax.numpy as jnp

# Import local modules
import sys
sys.path.append('..')
from ssy_model import *
sys.path.append('../..')
from solvers import solver

# Optionally tell JAX to use 64 bit floats
from jax.config import config
config.update("jax_enable_x64", True)


def discretize_gcy(gcy, shapes):
    """
    Discretizes the GCY model using a multi-index and returns a discrete
    representation of the model.  The states 

        (z, z_π, h_z, h_c, h_zπ, h_λ)

    are indexed with 

        (i_z, i_zπ, i_hz, i_hc, i_hzπ, i_hλ)
        
    and the sizes are 

        (n_z, n_zπ, n_hz, n_hc, n_hzπ, n_hλ)

    The discretizations for z and z_π use iterations of the Rouwenhorst method. 

        z[i_hz, i_z] is the i_z-th z state when h_z = h_z[i_hz] 

        z_π[i_hzπ, i_zπ] is the i_zπ-th z_π state when h_zπ = h_zπ[i_hzπ] 

        z_Q[i_zπ, i_hz, i_z, j_z] = trans prob for z given i_zπ, i_hz

        z_π_Q[i_hzπ, i_zπ, j_zπ] = trans prob for zπ when h_zπ index = i_hzπ

    """

    # Organize and unpack
    n_z, n_zπ, n_hz, n_hc, n_hzπ, n_hλ = shapes
    params = gcy.params
    β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, 
         ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z, 
         ρ_ππ, φ_zπ, ρ_zπ, s_zπ = params

    # Discretize
    h_λ_mc = rouwenhorst(n_hλ, ρ_λ, s_λ)
    h_c_mc = rouwenhorst(n_hc, ρ_c, s_c)
    h_z_mc = rouwenhorst(n_hz, ρ_z, s_z)
    h_zπ_mc = rouwenhorst(n_hzπ, ρ_zπ, s_zπ)

    # Build h states
    h_λ_states = h_λ_mc.state_values
    h_c_states = h_c_mc.state_values
    h_z_states = h_z_mc.state_values
    h_zπ_states = h_zπ_mc.state_values

    # For convenience, store the sigma states as well
    σ_c_states = φ_c * np.exp(h_c_states)
    σ_z_states = φ_z * np.exp(h_z_states)
    σ_zπ_states = φ_zπ * np.exp(h_zπ_states)


    # Next we build the states and transitions for 
    #
    #   z_π' = ρ_ππ * z_π + σ_zπ * η1
    #
    # Note that z_π will have a one-dimensional grid for each
    # value of σ_zπ, and hence h_zπ.
    z_π_states = np.zeros((n_hzπ, n_zπ))
    # z_π_Q holds transition probs over i_zπ for each i_hzπ
    z_π_Q = np.zeros((n_hzπ, n_zπ, n_zπ))

    for i, σ_zπ in enumerate(σ_zπ_states):
        mc = rouwenhorst(n_zπ, ρ_ππ, σ_zπ)
        for j in range(n_zπ):
            z_π_states[i_hzπ, i_zπ] = mc.state_values[j]
            z_π_Q[i_hzπ, i_zπ, :] = mc.P[i_zπ, :]

    # Finally, we build the states and transitions for 
    #
    #   z' = ρ * z + ρ_π * z_π + σ_z * η0
    #
    # Note that z will have a one-dimensional grid for each
    # pair (z_π, σ_z), and hence each (z_π, h_z).
    z_states = np.zeros((n_zπ, n_hz, n_z))
    # z_Q holds transition probs over i_z for each (i_zπ, i_hz)
    z_Q = np.zeros((n_zπ, n_hz, n_z, n_z))
    for i_zπ, z_π in enumerate(z_π_states):
        for i_hz, σ_z in enumerate(σ_z_states):
            mc = rouwenhorst(n_z, ρ, σ_z, ρ_π * z_π)
            for i_z in range(n_z):
                z_states[i_zπ, i_hz] = mc.state_values[i_z]
                z_Q[i_zπ, i_hz, i_z, :] = mc.P[i_z, :]

    # Set up all arrays and put them on the device
    arrays =  z_states,    z_Q,
              z_π_states,  z_π_Q,
              h_z_states,  h_z_mc.P,  σ_z_states
              h_c_states,  h_c_mc.P,  σ_c_states, 
              h_zπ_states, h_zπ_mc.P, σ_zπ_states,
              h_λ_states,  h_λ_mc.P)

    return arrays


def T_gcy(w, shapes, params, arrays):
    """
    Implement a discrete version of the operator T for the GCY model via
    JAX operations.  In reading the following, it is helpful to remember
    that the state is 
        
        (z, z_π, h_z, h_c, h_zπ, h_λ)

    """

    # Unpack
    n_z, n_zπ, n_hz, n_hc, n_hzπ, n_hλ = shapes

    β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, 
         ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z, 
         ρ_ππ, φ_zπ, ρ_zπ, s_zπ = params

    z_states,      z_Q,
      z_π_states,  z_π_Q,
      h_z_states,  h_z_mc.P,  σ_z_states
      h_c_states,  h_c_mc.P,  σ_c_states, 
      h_zπ_states, h_zπ_mc.P, σ_zπ_states,
      h_λ_states,  h_λ_mc.P = arrays

    θ = (1 - γ) / (1 - 1/ψ)
    w = jnp.reshape(w, (1, 1, 1, 1, 1, 1, n_z, n_zπ, n_hz, n_hc, n_hzπ, n_hλ))


    # Create intermediate arrays
    A1 = jnp.exp(θ * h_λ_states)
    A1 = jnp.reshape(A1, (1, 1, 1, 1, 1, 1, L, 1, 1, 1))
    A2 = jnp.exp(0.5 * ((1 - γ) * σ_c_states)**2)
    A2 = jnp.reshape(A2, (1, K, 1, 1, 1, 1, 1, 1))
    A3 = jnp.exp((1 - γ) * (μ_c + z_states))
    A3 = jnp.reshape(A3, (1, 1, I, J, 1, 1, 1, 1))

    # Reshape existing matrices prior to reduction
    Phλ = jnp.reshape(h_λ_P, (L, 1, 1, 1, L, 1, 1, 1))
    Phc = jnp.reshape(h_c_P, (1, K, 1, 1, 1, K, 1, 1))
    Phz = jnp.reshape(h_z_P, (1, 1, I, 1, 1, 1, I, 1))
    Pz = jnp.reshape(z_Q, (1, 1, I, J, 1, 1, 1, J))

    # Take product and sum along last four axes
    H = A1 * A2 * A3 * Phλ * Phc * Phz * Pz

    Hwθ = jnp.sum(w**θ * H, axis=(6, 7, 8, 9, 10, 11))

    # Define and return Tw
    Tw = 1 + β * Hwθ**(1/θ)
    return Tw

T_gcy = jax.jit(T_gcy, static_argnums=(1, ))




# ============= Tests ====================== #

#@njit
def T_ssy_loops(w, shapes, params, arrays):
    """
    This function replicates T_ssy but with for loops.  It's only
    purpose is to test the vectorized code in T_ssy.  It should only be
    used with very small shapes!
    """
    L, K, I, J = shapes
    β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ = params
    (h_λ_states, h_λ_P,              
     h_c_states, h_c_P,
     h_z_states, h_z_P,
     z_states,   z_Q,
     σ_c_states, σ_z_states) = arrays
    θ = (1 - γ) / (1 - 1/ψ)
    Hwθ = np.empty((L, K, I, J))

    for l in range(L):
        for k in range(K):
            for i in range(I):
                for j in range(J):
                    σ_c, z = σ_c_states[k], z_states[i, j]
                    a2 = np.exp(0.5 * ((1 - γ) * σ_c)**2)
                    a3 = np.exp((1 - γ) * (μ_c + z))
                    Hwθ_sum = 0.0
                    for lp in range(L):
                        h_λp = h_λ_states[lp]
                        a1 = np.exp(θ * h_λp)
                        for kp in range(K):
                            for ip in range(I):
                                for jp in range(J):
                                    Hwθ_sum +=  \
                                            w[lp, kp, ip, jp]**θ * \
                                            a1 * a2 * a3 * \
                                            h_λ_P[l, lp] * h_c_P[k, kp] * \
                                            h_z_P[i, ip] * z_Q[i, j, jp]
                    Hwθ[l, k, i, j] = Hwθ_sum

    # Define and return Tw
    Tw = 1 + β * Hwθ**(1/θ)
    return Tw


def test_vectorized_equals_loops(shapes=(2, 3, 4, 5)):
    """
    Test that T generated by vectorized logic produces the same result as T
    generated by loops.  Evaluate at a random choice of w.
    """
    ssy = SSY()
    params = ssy.params
    arrays = discretize_ssy(ssy, shapes)
    w = np.exp(np.random.randn(*shapes))  # Test operator at w
    w1 = T_ssy(w, shapes, params, arrays)
    w2 = T_ssy_loops(w, shapes, params, arrays)
    print(w1)
    print(w2)
    print(np.allclose(w1, w2))


def test_compute_wc_ratio_ssy(shapes=(2, 3, 4, 5), algo="successive_approx"):
    """
    Solve a small version of the model using T_ssy.
    """
    ssy = SSY()

    # Build discrete rep of SSY
    params = ssy.params
    arrays = discretize_ssy(ssy, shapes)

    # Shift arrays to the device
    arrays = [jax.device_put(array) for array in arrays]

    # Marginalize T
    T = lambda w : T_ssy(w, shapes, params, arrays)

    # Call the solver
    init_val = 800.0 
    w_init = jnp.ones(shapes) * init_val
    w_star = solver(T, w_init, algorithm=algo)

    return w_star


