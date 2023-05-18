"""
This code is will either be moved to production or abandoned, and probably the
latter.

At this point it doesn't look like any will be useful.
"""




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


def discretize_single_index(ssy, shapes):
    """
    Build the single index state arrays.  The discretized version is
    converted into single index form to facilitate matrix operations.

    The rule for the index is

        shapes = L, K, I, J

        n = n(l, k, i, j) = l * (K * I * J) + k * (I * J) + i * J + j

    where n is in range(N) with N = L * K * I * J.

    We store a Markov chain with states

        x_states[n] := (h_λ[l], σ_c[k], σ_z[i], z[i,j])

    A stochastic matrix P_x gives transition probabilitites, so

        P_x[n, np] = probability of transition x[n] -> x[np]


    """
    # Discretize on a multi-index
    arrays = discretize_multi_index(ssy, shapes)
    params = ssy.params
    L, K, I, J = shapes

    # Allocate single index arrays
    N = L * K * I * J
    P_x = np.zeros((N, N))
    x_states = np.zeros((4, N))

    # Populate single index arrays
    _build_single_index_arrays(shapes, params, arrays, x_states, P_x)

    # Return single index arrays
    return params, arrays, x_states, P_x

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


def compute_H_single_index(ssy, shapes):
    " An interface to the fast function _compute_H_single_index. "
    params, arrays, x_states, P_x = discretize_single_index(ssy, shapes)
    H = _compute_H_single_index(shapes, params, arrays, x_states, P_x)
    return H

#@njit
def _compute_H_single_index(shapes, params, arrays, x_states, P_x):
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
        σ_c, z = σ_c_states[k], z_states[i, j]
        a2 = np.exp(0.5 * ((1 - γ) * σ_c)**2)
        a3 = np.exp((1 - γ) * (μ_c + z))
        for mp in range(N):
            lp, kp, ip, jp = single_to_multi(mp, K, I, J)
            h_λp = h_λ_states[lp]
            a1 = np.exp(θ * h_λp)
            H[m, mp] =  a1 * a2 * a3 * P_x[m, mp]

    return H


# Build the Koopmans operator == #

@jax.jit
def single_index_T(w, H, params):
    "T via JAX operations."
    β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ = params
    θ = (1 - γ) / (1 - 1/ψ)
    Tw = 1 + β * (H @ (w**θ))**(1/θ)
    return Tw


def test_compute_wc_ratio_single_index(L, K, I, J, 
                                           single_index_output=False):
    """
    Solve a small version of the model.
    """
    shapes = L, K, I, J
    init_val = 800.0 
    init_val = 800.0 
    ssy = SSY()

    H = compute_H_single_index(ssy, shapes)
    H = jax.device_put(H)  # Put H on the device (GPU)

    N = L * K * I * J
    w_init = jnp.ones(N) * init_val

    # Marginalize T given the parameters
    T = lambda w: single_index_T(w, H, ssy.params)

    # Call the solver
    w_star = solver(T, w_init, algorithm="newton")

    # Return output in desired shape
    if single_index_output:
        w_out = w_star
    else:
        w_out = jnp.reshape(w_star, (L, K, I, J))
    return w_out








# ===== Specialized iteration --- using explicit deriviates ============== #
#
#  Experiments show that this is *slower* than the code generated by autodiff.
#


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
