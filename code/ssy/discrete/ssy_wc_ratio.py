"""
Multi-index discretized routines for computing the wealth consumption ratio of
the SSY model.

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


def discretize_ssy(ssy, shapes):
    """
    Discretizes the SSY model using a multi-index and returns a discrete
    representation of the model.

    This discrete representation is then passed to the Koopmans operator.

    The discretization uses iterations of the Rouwenhorst method.  The
    indices are

        h_λ[l] for l in range(L)
        h_c[k] for k in range(K)
        h_z[i] for i in range(I)
        z[i, j] is the j-th z state when σ_z = σ_z[i] for j in range(J)

        z_Q[i, j, jp] is trans prob from j to jp when σ_z index = i

    """

    # Organize and unpack
    params = ssy.params
    L, K, I, J = shapes
    β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ = params

    # Discretize
    h_λ_mc = rouwenhorst(L, ρ_λ, s_λ, 0)
    h_c_mc = rouwenhorst(K, ρ_c, s_c, 0)
    h_z_mc = rouwenhorst(I, ρ_z, s_z, 0)

    # Build states
    h_λ_states = h_λ_mc.state_values
    h_c_states = h_c_mc.state_values
    h_z_states = h_z_mc.state_values
    σ_z_states = ϕ_z * np.exp(h_z_states)
    z_states = np.zeros((I, J))

    # Build transition probabilities
    z_Q = np.zeros((I, J, J))

    for i, σ_z in enumerate(σ_z_states):
        mc_z = rouwenhorst(J, ρ, σ_z, 0)
        for j in range(J):
            z_states[i, j] = mc_z.state_values[j]
            z_Q[i, j, :] = mc_z.P[j, :]
             
    # For convenience, store the sigma states as well
    σ_c_states = ϕ_c * np.exp(h_c_states)
    σ_z_states = ϕ_z * np.exp(h_z_states)

    # Set up all arrays and put them on the device
    arrays = (h_λ_states,  h_λ_mc.P,
              h_c_states,  h_c_mc.P,
              h_z_states,  h_z_mc.P,
              z_states,    z_Q,
              σ_c_states, σ_z_states)

    return arrays


def T_ssy(w, shapes, params, arrays):
    """
    Implement a discrete version of the operator T for the SSY model via
    JAX operations.  
    """

    # Unpack
    L, K, I, J = shapes
    (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = params
    (h_λ_states, h_λ_P,              
     h_c_states, h_c_P,
     h_z_states, h_z_P,
     z_states,   z_Q,
     σ_c_states, σ_z_states) = arrays
    θ = (1 - γ) / (1 - 1/ψ)
    w = jnp.reshape(w, (1, 1, 1, 1, L, K, I, J))


    # Create intermediate arrays
    A1 = jnp.exp(θ * h_λ_states)
    A1 = jnp.reshape(A1, (1, 1, 1, 1, L, 1, 1, 1))
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

    Hwθ = jnp.sum(w**θ * H, axis=(4, 5, 6, 7))

    # Define and return Tw
    Tw = 1 + β * Hwθ**(1/θ)
    return Tw

T_ssy = jax.jit(T_ssy, static_argnums=(1, ))




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


