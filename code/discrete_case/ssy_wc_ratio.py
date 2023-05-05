"""
Multi-index discretized routines for computing the wealth consumption ratio of
the SSY model.

"""

import jax
import jax.numpy as jnp
from jax.config import config

# Import local modules
import sys
sys.path.append('..')
from ssy_model import *
from solvers import solver

# Optionally tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)


# ================ Discretize SSY multi-index =============== #
#
# Code to discretize the SSY model using a multi-index, and provide the
# Koomaps operator.
#
#

def discretize_ssy_multi_index(ssy, L=4, K=4, I=4, J=4):
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
    shapes = L, K, I, J
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
        mc_z = rouwenhorst(J, 0, σ_z, ρ)
        for j in range(J):
            z_states[i, j] = mc_z.state_values[j]
            z_Q[i, j, :] = mc_z.P[j, :]
             
    # For convenience, store the sigma states as well
    σ_c_states = ϕ_c * jnp.exp(h_c_states)
    σ_z_states = ϕ_z * jnp.exp(h_z_states)

    # Set up all arrays and put them on the device
    arrays = (h_λ_states,  h_λ_mc.P,
              h_c_states,  h_c_mc.P,
              h_z_states,  h_z_mc.P,
              z_states,    z_Q,
              σ_c_states, σ_z_states)

    return shapes, params, arrays



def multi_index_ssy_T(w, shapes, params, arrays):
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

    # Create intermediate arrays
    B1 = jnp.exp(θ * h_λ_states)
    B1 = jnp.reshape(B1, (1, 1, 1, 1, L, 1, 1, 1))
    B2 = jnp.exp(0.5*((1-γ)*σ_c_states)**2)
    B2 = jnp.reshape(B2, (1, K, 1, 1, 1, 1, 1, 1))
    B3 = jnp.exp((1-γ)*(μ_c+z_states))
    B3 = jnp.reshape(B3, (1, 1, I, J, 1, 1, 1, 1))

    # Reshape existing matrices prior to reduction
    Phλ = jnp.reshape(h_λ_P, (L, 1, 1, 1, L, 1, 1, 1))
    Phc = jnp.reshape(h_c_P, (1, K, 1, 1, 1, K, 1, 1))
    Phz = jnp.reshape(h_z_P, (1, 1, I, 1, 1, 1, I, 1))
    Pz = jnp.reshape(z_Q, (1, 1, I, J, 1, 1, 1, J))
    w = jnp.reshape(w, (1, 1, 1, 1, L, K, I, J))

    # Take product and sum along last four axes
    A = w**θ * B1 * B2 * B3 * Phλ * Phc * Phz * Pz
    Hwθ = jnp.sum(A, axis=(4, 5, 6, 7))

    # Define and return Tw
    Tw = 1 + β * Hwθ**(1/θ)
    return Tw

# JIT compile the intermediate function _T
multi_index_ssy_T = jax.jit(multi_index_ssy_T, static_argnums=(1, ))




# ================ Discretize SSY single index =============== #
#
# Single-index routines for computing the wealth consumption ratio of the SSY
# model.
# 
# This code cannot be run when the model is high-dimensional.  It's only purpose
# is for cross-checking solutions produced by the multi-index code!
# 



# Indexing functions for mapping between multiple and single indices 

@njit
def split_index(i, M):
    div = i // M
    rem = i % M
    return (div, rem)

@njit
def single_to_multi(m, K, I, J):
    """
    A utility function for the multi-index.
    """
    l, rem = split_index(m, K * I * J)
    k, rem = split_index(rem, I * J)
    i, j = split_index(rem, J)
    return (l, k, i, j)

@njit
def multi_to_single(l, k, i , j, K, I, J):
    return l * (K * I * J) + k * (I * J) + i * J + j



# Discretize SSY model to single-index 


def discretize_ssy_single_index(ssy, L, K, I, J):
    """
    Build the single index state arrays.  The discretized version is
    converted into single index form to facilitate matrix operations.

    The rule for the index is

        n = n(l, k, i, j) = l * (K * I * J) + k * (I * J) + i * J + j

    where n is in range(N) with N = L * K * I * J.

    We store a Markov chain with states

        x_states[n] := (h_λ[l], σ_c[k], σ_z[i], z[i,j])

    A stochastic matrix P_x gives transition probabilitites, so

        P_x[n, np] = probability of transition x[n] -> x[np]


    """
    # Discretize on a multi-index
    shapes, params, arrays = discretize_ssy_multi_index(ssy, L, K, I, J)

    # Allocate single index arrays
    N = L * K * I * J
    P_x = np.zeros((N, N))
    x_states = np.zeros((4, N))

    # Populate single index arrays
    _build_single_index_arrays(shapes, params, arrays, x_states, P_x)

    # Return single index arrays
    return shapes, params, arrays, x_states, P_x

#@njit
def _build_single_index_arrays(shapes, params, arrays, x_states, P_x):
    # Unpack
    L, K, I, J = shapes
    (h_λ_states, h_λ_P,              
     h_c_states, h_c_P,
     h_z_states, h_z_P,
     z_states,   z_Q,
     σ_c_states, σ_z_states) = arrays

    N = L * K * I * J

    for m in range(N):
        l, k, i, j = single_to_multi(m, K, I, J)
        x_states[:, m] = (h_λ_states[l],
                          h_c_states[k], h_z_states[i], z_states[i, j])
        for mp in range(N):
            lp, kp, ip, jp = single_to_multi(mp, K, I, J)
            P_x[m, mp] = h_λ_P[l, lp] * h_c_P[k, kp] * h_z_P[i, ip] * z_Q[i, j, jp]


def compute_H(ssy, L=4, K=4, I=4, J=4):
    " An interface to the fast function _compute_H. "
    shapes, params, arrays, x_states, P_x = discretize_ssy_single_index(
            ssy, L, K, I, J)
    H = _compute_H(shapes, params, arrays, x_states, P_x)
    return H

#@njit
def _compute_H(shapes, params, arrays, x_states, P_x):
    """
    Compute the matrix H in the SSY model using the single-index
    framework.

    """
    # Unpack
    β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ = params
    L, K, I, J = shapes
    (h_λ_states, h_λ_P,              
     h_c_states, h_c_P,
     h_z_states, h_z_P,
     z_states,   z_Q,
     σ_c_states, σ_z_states) = arrays

    # Set up
    N = L * K * I * J
    θ = (1 - γ) / (1 - 1/ψ)
    H = np.empty((N, N))

    # Comptue
    for m in range(N):
        l, k, i, j = single_to_multi(m, K, I, J)
        σ_c, σ_z, z = σ_c_states[k], σ_z_states[i], z_states[i, j]
        for mp in range(N):
            lp, kp, ip, jp = single_to_multi(m, K, I, J)
            h_λp = h_λ_states[lp]
            a = np.exp(θ * h_λp + (1 - γ) * (μ_c + z) + 0.5 * (1 - γ)**2 * σ_c**2)
            H[m, mp] =  a * P_x[m, mp]

    return H


# Build the Koopmans operator == #

@jax.jit
def single_index_ssy_T(w, H, params):
    "T via JAX operations."
    β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ = params
    θ = (1 - γ) / (1 - 1/ψ)
    Tw = 1 + β * (H @ (w**θ))**(1/θ)
    return Tw



# ============= Tests ====================== #


def test_compute_wc_ratio_ssy_multi_index():
    """
    Solve a small version of the model.
    """
    L, K, I, J = 3, 3, 3, 3
    init_val = 800.0 
    ssy = SSY()

    # Build discrete rep of SSY
    shapes, params, arrays = discretize_ssy_multi_index(ssy, L, K, I, J)
    arrays = [jax.device_put(array) for array in arrays]

    # Marginalize T
    T = lambda w : multi_index_ssy_T(w, shapes, params, arrays)
    # Call the solver
    w_init = jnp.ones((L, K, I, J)) * init_val
    w_star = solver(T, w_init, algorithm="newton")
    return w_star


def test_compute_wc_ratio_ssy_single_index(single_index_output=False):
    """
    Solve a small version of the model.
    """
    L, K, I, J = 3, 3, 3, 3
    init_val = 800.0 
    ssy = SSY()

    H = compute_H(ssy, L, K, I, J)
    H = jax.device_put(H)  # Put H on the device (GPU)

    N = L * K * I * J
    w_init = jnp.ones(N) * init_val

    # Marginalize T given the parameters
    T = lambda w: single_index_ssy_T(w, H, ssy.params)

    # Call the solver
    w_star = solver(T, w_init, algorithm="newton")

    # Return output in desired shape
    if single_index_output:
        w_out = w_star
    else:
        w_out = jnp.reshape(w_star, (L, K, I, J))
    return w_out


def test_cross_check_single_multi():
    w_star_single = test_compute_wc_ratio_ssy_single_index()
    w_star_multi = test_compute_wc_ratio_ssy_multi_index()
    print(jnp.max(jnp.abs(w_star_single - w_star_multi)))
