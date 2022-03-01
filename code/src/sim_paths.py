
# ================================================= #
# == State updates and simulation of state paths == #
# ================================================= #

def consumption_state_update(ssy_params, x_in, x_out, η):
    """
    Update the state x = (z, h_z, h_c, h_λ) one period of time given 
    the standard normal innovation vector η.

    The innovation vector is supplied because it will be controlled 
    by other functions.

    The function writes to x_out and returns None.

        * ssy_params: array_like, contains SSY parameters

    """
    # Unpack consumption parameters
    (β, γ, ψ, 
        μ_c, ρ, ϕ_z, ϕ_c, 
        ρ_z, ρ_c, ρ_λ, 
        s_z, s_c, s_λ) = ssy_params

    h_λ, h_c, h_z, z = x_in
    η0, η1, η2, η3 = η

    # Map h to σ
    σ_z = φ_z * math.exp(h_z)

    # Update states
    h_λ = ρ_λ * h_λ + s_λ * η3
    h_c = ρ_c * h_c + s_c * η2
    h_z = ρ_z * h_z + s_z * η1
    z =   ρ   * z   + σ_z * η0

    x_out[:] = h_λ, h_c, h_z, z


@njit
def consumption_growth(ssy_params, x, ξ_c):
    """
    Evaluate current, one-period consumption growth given the current state
    and innovation to consumption.

        * ssy_params: array_like, contains SSY parameters
    """
    # Unpack consumption parameters
    (β, γ, ψ, 
        μ_c, ρ, ϕ_z, ϕ_c, 
        ρ_z, ρ_c, ρ_λ, 
        s_z, s_c, s_λ) = ssy_params

    h_λ, h_c, h_z, z = x

    # Update
    σ_c = ϕ_c * math.exp(h_c)
    return μ_c + z + σ_c * ξ_c



@njit
def simulate_consumption_state_path(ssy, T=1000, seed=1234):
    """
    Generate and return a time series of the state process 

        x[0], ..., x[T-1]

    This is useful for plotting and diagnostics.

    """
    np.random.seed(seed)
    num_states = 4

    # Allocate memory for states 
    h_λ_vec = np.zeros(T)
    h_c_vec = np.zeros(T)
    h_z_vec = np.zeros(T)
    z_vec = np.zeros(T)

    # Initialize at the stationary mean, which is zero 
    x = np.zeros(num_states)
    x_p = np.empty_like(x)

    # Simulate 
    for t in range(T):
        h_λ_vec[t], h_c_vec[t], h_z_vec[t], z_vec[t]  = x
        consumption_state_update_cpu(ssy, x, x_p, randn(num_states))
        x[:] = x_p

    return h_λ_vec, h_c_vec, h_z_vec, z_vec

