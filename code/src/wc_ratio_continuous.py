from ssy_model import SSY
import jax
import jax.numpy as jnp
from functools import partial
from utils import fwd_solver, fixed_point_interface
from jax.config import config

# Tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)


def build_grid(ssy,
               h_λ_grid_size,
               h_c_grid_size,
               h_z_grid_size,
               z_grid_size,
               num_std_devs=3.2):
    """
    Build a grid over the triple (z_vec, h_z_vec, h_c_vec) for linear
    interpolation.

    num_std_devs must be set or the code will fail.

    """
    # Unpack parameters
    (β, γ, ψ,
        μ_c, ρ, ϕ_z, ϕ_c,
        ρ_z, ρ_c, ρ_λ,
        s_z, s_c, s_λ) = ssy.unpack()

    # Build the h grids
    s_vals = s_λ, s_c, s_z
    rho_vals = ρ_λ, ρ_c, ρ_z
    grid_sizes = h_λ_grid_size, h_c_grid_size,  h_z_grid_size
    grids = []

    # The h processes are zero mean so we center the grids on zero.
    # The end points of the grid are multiples of the stationary std.
    for s, ρ, grid_size in zip(s_vals, rho_vals, grid_sizes):
        std = jnp.sqrt(s**2 / (1 - ρ**2))
        g_max = num_std_devs * std
        g_min = - g_max
        grids.append(jnp.linspace(g_min, g_max, grid_size))

    h_λ_grid, h_c_grid, h_z_grid = grids

    # grid for z, which has volatility σ_z = ϕ_z exp(h_z)
    h_z_max = num_std_devs * jnp.sqrt(s_z**2 / (1 - ρ_z**2))
    σ_z_max = ϕ_z * jnp.exp(h_z_max)
    z_max = num_std_devs * σ_z_max
    z_min = - z_max
    z_grid = jnp.linspace(z_min, z_max, z_grid_size)

    return h_λ_grid, h_c_grid, h_z_grid, z_grid

# ================================================= #
# == State updates and simulation of state paths == #
# ================================================= #


def next_state(ssy_params, x, η_array):
    """
    Generate an array of states in the next period given current state
    x = (z, h_z, h_c, h_λ) and an array of shocks.
    """
    (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = ssy_params
    σ_z = φ_z * jnp.exp(x[3])

    h_λ = ρ_λ * x[0] + s_λ * η_array[0]
    h_c = ρ_c * x[1] + s_c * η_array[1]
    h_z = ρ_z * x[2] + s_z * η_array[2]
    z = ρ * x[3] + σ_z * η_array[3]

    return jnp.array([h_λ, h_c, h_z, z])


@jax.jit
def jit_map_coordinates(vals, coords):
    return jax.scipy.ndimage.map_coordinates(vals, coords, order=1,
                                             mode='nearest')


def vals_to_coords(grids, x_vals):
    # jax.jit doesn't allow dynamic shapes
    dim = 4

    intervals = jnp.asarray([grid[1] - grid[0] for grid in grids])
    low_bounds = jnp.asarray([grid[0] for grid in grids])

    intervals = intervals.reshape(dim, 1)
    low_bounds = low_bounds.reshape(dim, 1)

    return (x_vals - low_bounds) / intervals


@partial(jax.vmap, in_axes=(0, None, None, None, None))
def Kg_vmap(x, ssy_params, g_vals, grids, mc_draws):
    (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = ssy_params
    θ = (1-γ) / (1-(1/ψ))
    h_λ, h_c, h_z, z = x
    # Compute the constant term, given x, which doesn't
    # require the new state.
    σ_c = ϕ_c * jnp.exp(h_c)
    const = jnp.exp((1 - γ) * (μ_c + z) +
                    (1/2) * (1 - γ)**2 * σ_c**2)

    # Ready to kick off the inner loop, which computes
    #
    #     E_x g(h_λ', h_c', h_z', z') exp(θ * h_λ')
    next_x = next_state(ssy_params, x, mc_draws)
    pf = jnp.exp(next_x[0] * θ)
    # interpolate g for next_x
    next_x_coords = vals_to_coords(grids, next_x)
    next_g = jit_map_coordinates(g_vals, next_x_coords)
    e_x = jnp.mean(next_g * pf)
    Kg = const * e_x
    return Kg


Kg_vmap = jax.jit(Kg_vmap)


def fun_factory(params, batch_size=10000):
    @jax.jit
    def wc_operator_continuous(ssy_params, w_in, grids, mc_draws):
        (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = ssy_params
        θ = (1-γ) / (1-(1/ψ))
        h_λ_grid, h_c_grid, h_z_grid, z_grid = grids

        # Grid sizes
        nh_λ = len(h_λ_grid)
        nh_c = len(h_c_grid)
        nh_z = len(h_z_grid)
        nz = len(z_grid)
        map_n = nh_λ * nh_c * nh_z * nz // batch_size

        g_vals = w_in**θ

        # flatten and reshape the states for computation
        mesh_grids = jnp.meshgrid(*grids, indexing='ij')
        x_3d = jnp.stack([grid.ravel() for grid in mesh_grids],
                         axis=1).reshape(map_n, batch_size, 4)

        def Kg_map_fun(x_array):
            return Kg_vmap(x_array, ssy_params, g_vals, grids, mc_draws)

        # Compute Kg and reshape back
        Kg_out = jax.lax.map(Kg_map_fun, x_3d).reshape(nh_λ, nh_c, nh_z, nz)
        w_out = 1 + β * Kg_out**(1/θ)

        return w_out

    @jax.jit
    def T(w, params):
        "T via JAX operations."
        ssy_params, grids, mc_draws = params
        w_out = wc_operator_continuous(ssy_params, w, grids, mc_draws)
        return w_out

    return T


def wc_ratio_continuous(ssy, h_λ_grid_size=10, h_c_grid_size=10,
                        h_z_grid_size=10, z_grid_size=20, num_std_devs=3.2,
                        mc_draw_size=2000, w_init=None, seed=1234,
                        ram_free=20, tol=1e-5, verbose=True, print_skip=10):
    """
    Iterate to convergence on the Koopmans operator associated with the SSY
    model and then return the wealth consumption ratio.
    """
    ssy_params = jnp.array(ssy.unpack())
    grids = build_grid(ssy, h_λ_grid_size, h_c_grid_size, h_z_grid_size,
                       z_grid_size, num_std_devs)

    # generate shocks to evaluate the inner expectation
    key = jax.random.PRNGKey(seed)
    mc_draws = jax.random.normal(key, shape=(4, mc_draw_size))

    if w_init is None:
        w_init = jnp.ones(shape=(h_λ_grid_size, h_c_grid_size, h_z_grid_size,
                                 z_grid_size))

    # determine batch_size
    state_size = h_λ_grid_size * h_c_grid_size * h_z_grid_size * z_grid_size
    batch_size = ram_free * 30000000 // mc_draw_size
    if state_size <= batch_size:
        batch_size = state_size
    else:
        while (state_size % batch_size > 0):
            batch_size -= 1

    print("batch_size =", batch_size)

    params = ssy_params, grids, mc_draws
    T = fun_factory(params, batch_size=batch_size)

    solver = fwd_solver

    w_star, iter = fixed_point_interface(solver, T, params, w_init, tol=tol,
                                         verbose=verbose,
                                         print_skip=print_skip)

    return w_star


# old versions

# def wc_operator_continuous(ssy_params, w_in, grids, mc_draws,
#                            batch_size=10000):
#     (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = ssy_params
#     θ = (1-γ) / (1-(1/ψ))
#     h_λ_grid, h_c_grid, h_z_grid, z_grid = grids

#     # Grid sizes
#     nh_λ = len(h_λ_grid)
#     nh_c = len(h_c_grid)
#     nh_z = len(h_z_grid)
#     nz = len(z_grid)

#     g_vals = w_in**θ

#     def Kg(x_array, batch_size=batch_size):
#         """Return Kg for an array of states with x_array.shape = (n, 4).
#         The computation is done in batches using vmap to reduce memory usage.
#         """
#         res = jnp.array([])
#         n = x_array.shape[0] // batch_size
#         rem = x_array.shape[0] - n*batch_size
#         for i in range(n):
#             start = i*batch_size
#             end = start + batch_size
#             res = jnp.append(res, Kg_vmap(x_array[start:end], ssy_params,
#                              g_vals, grids, mc_draws))
#         if rem > 0:
#             res = jnp.append(res, Kg_vmap(x_array[-rem:], ssy_params, g_vals,
#                                           grids, mc_draws))
#         return res

#     # flatten the states for computation
#     mesh_grids = jnp.meshgrid(*grids, indexing='ij')
#     x_array = jnp.stack([grid.ravel() for grid in mesh_grids], axis=1)

#     # Compute Kg and reshape back
#     if x_array.shape[0] <= batch_size:
#         Kg_out = Kg_vmap(x_array, ssy_params, g_vals, grids, mc_draws)
#         Kg_out = Kg_out.reshape(nh_λ, nh_c, nh_z, nz)
#     else:
#         Kg_out = Kg(x_array).reshape(nh_λ, nh_c, nh_z, nz)
#     w_out = 1 + β * Kg_out**(1/θ)

#     return w_out


# @jax.jit
# def next_state(ssy_params, x, η_array):
#     """
#     Generate an array of states in the next period given current state
#     x = (z, h_z, h_c, h_λ) and an array of shocks.
#     """
#     (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = ssy_params
#     ρ_array = jnp.array([ρ_λ, ρ_c, ρ_z, ρ])
#     σ_z = φ_z * jnp.exp(x[3])
#     s_array = jnp.array([s_λ, s_c, s_z, σ_z])

#     # reshape for broadcasting
#     # x_out.shape = η_array
#     x_out = (jnp.diag(ρ_array) @ x).reshape(4, 1) + jnp.diag(s_array) @ η_array
#     return x_out
