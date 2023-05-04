"""

Single-index routines for computing the wealth consumption ratio of the SSY
model.

This code cannot be run when the model is high-dimensional.  It's only purpose
is for cross-checking solutions produced by the multi-index code!

"""


import jax
import jax.numpy as jnp
from jax.config import config

# Import local modules
import sys
sys.path.append('..')
from ssy_model import *
from utils import *


# Tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)


# == Indexing functions for mapping between multiple and single indices == #


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



# == Convert SSY model to single-index code == #


def discretize_single_index(ssy):
    """
    Build the single index state process.  The discretized version is
    converted into single index form to facilitate matrix operations.

    The rule for the index is

        n = n(l, k, i, j) = l * (K * I * J) + k * (I * J) + i * J + j

    where n is in range(N) with N = L * K * I * J.

    We store a Markov chain with states

        x_states[n] := (h_λ[l], σ_c[k], σ_z[i], z[i,j])

    A stochastic matrix P_x gives transition probabilitites, so

        P_x[n, np] = probability of transition x[n] -> x[np]


    """
    # Unpack
    L, K, I, J = ssy.L, ssy.K, ssy.I, ssy.J
    N = L * K * I * J

    # Allocate arrays
    P_x = jnp.zeros((N, N))
    x_states = jnp.zeros((4, N))

    # Populate arrays
    state_arrays = (ssy.h_λ_states, ssy.h_c_states,
                    ssy.h_z_states, ssy.z_states)
    prob_arrays = ssy.h_λ_P, ssy.h_c_P, ssy.h_z_P, ssy.z_Q
    _build_single_index_arrays(
            L, K, I, J, state_arrays, prob_arrays, x_states, P_x
    )
    return x_states, P_x

@njit
def _build_single_index_arrays(L, K, I, J,
                               state_arrays,
                               prob_arrays,
                               x_states,
                               P_x):
    """
    Read in SSY data and write to x_states and P_x.
    """

    h_λ_states, h_c_states, h_z_states, z_states = state_arrays
    h_λ_P, h_c_P, h_z_P, z_Q = prob_arrays

    N = L * K * I * J

    for m in range(N):
        l, k, i, j = single_to_multi(m, K, I, J)
        x_states[:, m] = (h_λ_states[l],
                          h_c_states[k], h_z_states[i], z_states[i, j])
        for mp in range(N):
            lp, kp, ip, jp = single_to_multi(mp, K, I, J)
            P_x[m, mp] = h_λ_P[l, lp] * h_c_P[k, kp] * h_z_P[i, ip] * z_Q[i, j, jp]


def compute_H(ssy, P_x):
    " An interface to the fast function _compute_H. "
    H = _compute_H(ssy.unpack(),
                      ssy.L, ssy.K, ssy.I, ssy.J,
                      ssy.h_λ_states,
                      ssy.σ_c_states,
                      ssy.σ_z_states,
                      ssy.z_states,
                      P_x)
    return H


@njit
def _compute_H(ssy_params,
               L, K, I, J,
               h_λ_states,
               σ_c_states,
               σ_z_states,
               z_states,
               P_x):
    """
    Compute the matrix H in the SSY model using the single-index
    framework.

    """
    # Unpack
    (β, γ, ψ,
        μ_c, ρ, ϕ_z, ϕ_c,
        ρ_z, ρ_c, ρ_λ,
        s_z, s_c, s_λ) = ssy_params
    N = L * K * I * J
    θ = (1 - γ) / (1 - 1/ψ)
    H = jnp.empty((N, N))

    for m in range(N):
        l, k, i, j = single_to_multi(m, K, I, J)
        σ_c, σ_z, z = σ_c_states[k], σ_z_states[i], z_states[i, j]
        for mp in range(N):
            lp, kp, ip, jp = single_to_multi(m, K, I, J)
            h_λp = h_λ_states[lp]
            a = jnp.exp(θ * h_λp + (1 - γ) * (μ_c + z) + 0.5 * (1 - γ)**2 * σ_c**2)
            H[m, mp] =  a * P_x[m, mp]

    return H



# == Operator == #

@jax.jit
def T(w, params):
    "T via JAX operations."
    H, β, θ = params
    Tw = 1 + β * (H @ (w**θ))**(1/θ)
    return Tw

# == Iteration == #

def wc_ratio_single_index(ssy, 
                          algorithm="newton",
                          init_val=800, 
                          single_index_output=False,   
                          verbose=True):
    """
    Iterate to convergence on the Koopmans operator associated with the 
    SSY model and return the wealth consumption ratio.

    """

    # Unpack 
    β, θ = ssy.β, ssy.θ

    N = ssy.L * ssy.K * ssy.I * ssy.J
    x_states, P_x = discretize_single_index(ssy)
    H = compute_H(ssy, P_x)

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
        w_out = jnp.reshape(w_star, (ssy.L, ssy.K, ssy.I, ssy.J))

    return w_out



# == Specialized iteration == #

@jax.jit
def jacobian(w, params):
    "Jacobian update via JAX operations."
    H, β, θ = params
    N, _ = H.shape
    Hwθ = H @ w**θ
    Fw = (Hwθ)**((1-θ)/θ)
    Gw = w**(θ-1)
    DF = jnp.diag(Fw.flatten())
    DG = jnp.diag(Gw.flatten())
    I = jnp.identity(N)
    J = β * DF @ H @ DG - I
    return J

@jax.jit
def Q(w, params):
    "Jacobian update via JAX operations."
    H, β, θ = params
    Tw = T(w, params)
    J = jacobian(w, params)
    #b = jax.scipy.sparse.linalg.bicgstab(J, Tw - w)[0], 
    b = jnp.linalg.solve(J, Tw - w)
    return w - b


def wc_ratio_single_index_specialized(ssy, 
                                      init_val=800, 
                                      single_index_output=False,   
                                      verbose=True):
    """
    Iterate to convergence on the Koopmans operator associated with the 
    SSY model and return the wealth consumption ratio.

    """

    # Unpack 
    β, θ = ssy.β, ssy.θ

    N = ssy.L * ssy.K * ssy.I * ssy.J
    x_states, P_x = discretize_single_index(ssy)
    H = compute_H(ssy, P_x)

    H = jax.device_put(H)  # Put H on the device (GPU)
    params = H, β, θ 

    w_init = jnp.ones(N) * init_val

    # Marginalize Q given the parameters
    Q_operator = lambda x: Q(x, params)
    # Call the solver
    w_star, num_iter = fwd_solver(Q_operator, w_init)

    # Return output in desired shape
    if single_index_output:
        w_out = w_star
    else:
        w_out = jnp.reshape(w_star, (ssy.L, ssy.K, ssy.I, ssy.J))

    return w_out

if __name__ == '__main__':
    ssy = SSY()
    β, θ = ssy.β, ssy.θ

    N = ssy.L * ssy.K * ssy.I * ssy.J
    x_states, P_x = discretize_single_index(ssy)
    H = compute_H(ssy, P_x)

    H = jax.device_put(H)  # Put H on the device (GPU)
    params = H, β, θ 

    w_init = jnp.ones(N) * 800
    out = Q(w_init, params)
