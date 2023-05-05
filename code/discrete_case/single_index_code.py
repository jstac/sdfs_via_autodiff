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
