"""

Single-index routines for computing the wealth consumption ratio of the SSY
model.

CPU and CuPY based code.

"""

import numpy as np
from model import *


# == CPU version == #

def wc_ratio_single_index_cpu(ssy, 
                    tol=1e-7, 
                    init_val=np.exp(5), 
                    max_iter=1_000_000,
                    single_index_output=True,   # output as w[m] or w[l, k, i, j]
                    verbose=True):
    """
    Iterate to convergence on the Koopmans operator associated with the SSY
    model and then return the wealth consumption ratio.

    """

    # Unpack and set up parameters EpsteinZin parameters
    (β, γ, ψ, 
        μ_c, ρ, ϕ_z, ϕ_c, 
        ρ_z, ρ_c, ρ_λ, 
        s_z, s_c, s_λ) = ssy.unpack()
    θ = ssy.θ  

    K_matrix = compute_K(ssy)
    L, K, I, J = ssy.L, ssy.K, ssy.I, ssy.J
    M = ssy.M
    w = np.ones(M) * init_val
    iter = 0
    error = tol + 1

    if verbose:
        print("Beginning iteration\n\n")

    while error > tol and iter < max_iter:
        Tw = 1 + β * (K_matrix @ (w**θ))**(1/θ)
        error = np.max(np.abs(w - Tw))
        w = Tw
        iter += 1

    if verbose:
        print(f"Iteration converged after {iter} iterations") 

    if single_index_output:
        w_out = w

    else:
        w_out = np.empty((L, K, I, J))
        for m in range(M):
            l, k, i, j = single_to_multi(m, K, I, J)
            w_out[l, k, i, j] = w[m]

    return w_out


# == Utilities == #

def average_wealth_cons(ssy, L=5, K=5, I=5, J=5, verbose=False):
    """
    Computes the mean wealth consumption ratio under the stationary
    distribution pi.

    """

    w = wealth_cons_ratio(ssy, L=L, K=K, I=I, J=J, 
            single_index_output=True, verbose=verbose)

    x_mc = MarkovChain(ssy.P_x)
    x_pi = x_mc.stationary_distributions[0]

    mean_w = w @ x_pi

    return mean_w


