import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.config import config

from quantecon.quad import qnwnorm
import time

import sys
sys.path.append('..')
from gcy_model import *
sys.path.append('../..')
from solvers import solver
from utils import lin_interp

# Tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)

# order:
# h_λ_grid, h_c_grid, h_z_grid, h_zπ_grid, z_grid, zπ_grid

def build_grid(gcy,
               h_λ_grid_size,
               h_c_grid_size,
               h_z_grid_size,
               h_zπ_grid_size,
               z_grid_size,
               z_π_grid_size,
               num_std_devs=3.2):
    """Build a grid over the tuple
    (h_λ_grid, h_c_grid, h_z_grid, h_zπ_grid, z_grid, zπ_grid) for
    linear interpolation.

    """
    # Unpack parameters
    (β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z,
        ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = gcy.params

    # Build the h grids
    s_vals = s_λ, s_c, s_z, s_zπ
    rho_vals = ρ_λ, ρ_c, ρ_z, ρ_zπ
    grid_sizes = h_λ_grid_size, h_c_grid_size,  h_z_grid_size, h_zπ_grid_size
    grids = []

    # The h processes are zero mean so we center the grids on zero.
    # The end points of the grid are multiples of the stationary std.
    for s, ρ, grid_size in zip(s_vals, rho_vals, grid_sizes):
        std = jnp.sqrt(s**2 / (1 - ρ**2))
        g_max = num_std_devs * std
        g_min = - g_max
        grids.append(jnp.linspace(g_min, g_max, grid_size))

    h_λ_grid, h_c_grid, h_z_grid, h_zπ_grid = grids

    # Build grid for zπ, which has volatility σ_z = ϕ_zπ exp(h_zπ)
    h_zπ_max = num_std_devs * jnp.sqrt(s_zπ**2 / (1 - ρ_zπ**2))
    σ_zπ_max = φ_zπ * jnp.exp(h_zπ_max)
    zπ_max = num_std_devs * jnp.sqrt(σ_zπ_max**2 / (1 - ρ_ππ**2))
    zπ_min = - zπ_max
    zπ_grid = jnp.linspace(zπ_min, zπ_max, z_π_grid_size)

    # Build grid for z, which has volatility σ_z = ϕ_z exp(h_z)
    # z' = ρ * z + ρ_π * z_π + σ_z * η0
    h_z_max = num_std_devs * jnp.sqrt(s_z**2 / (1 - ρ_z**2))
    σ_z_max = φ_z * jnp.exp(h_z_max)
    z_max = (ρ_π * zπ_grid[-1] + num_std_devs * σ_z_max) / (1 - ρ)
    z_min = (ρ_π * zπ_grid[0] - num_std_devs * σ_z_max) / (1 - ρ)
    z_grid = jnp.linspace(z_min, z_max, z_grid_size)
    return h_λ_grid, h_c_grid, h_z_grid, h_zπ_grid, z_grid, zπ_grid


# ================================================= #
# == State updates and simulation of state paths == #
# ================================================= #

@jax.jit
def next_state(gcy_params, x, η_array):
    """Generate an array of states in the next period given current state
    x = (h_λ, h_c, h_z, h_zπ, z, z_π) and an array of shocks.

    η_array: a jnp array of shape (6, N).

    Return a jnp array of shape (6, N).

    z' = ρ * z + ρ_π * z_π + σ_z * η0

    z_π' = ρ_ππ * z_π + σ_zπ * η1

    h_z' = ρ_z * h_z + s_z * η2

    h_c' = ρ_c * h_c + s_c * η3

    h_zπ' = ρ_zπ * h_zπ + s_zπ * η4

    h_λ' = ρ_λ * h_λ + s_λ * η5

    """

    (β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z,
        ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = gcy_params
    # h_λ_grid, h_c_grid, h_z_grid, h_zπ_grid, z_grid, zπ_grid
    h_λ, h_c, h_z, h_zπ, z, z_π = x

    σ_z = φ_z * jnp.exp(h_z)
    σ_zπ = φ_zπ * jnp.exp(h_zπ)

    h_λ = ρ_λ * h_λ + s_λ * η_array[0]
    h_c = ρ_c * h_c + s_c * η_array[1]
    h_z = ρ_z * h_z + s_z * η_array[2]
    h_zπ = ρ_zπ * h_zπ + s_zπ * η_array[3]
    z = ρ * z + ρ_π * z_π + σ_z * η_array[4]
    z_π = ρ_ππ * z_π + σ_zπ * η_array[5]

    return jnp.array([h_λ, h_c, h_z, h_zπ, z, z_π])


# ============================================= #
# == Kernel for operator T using Monte Carlo == #
# ============================================= #

@partial(jax.vmap, in_axes=(0, None, None, None, None))
def Kg_vmap_mc(x, gcy_params, w_vals, grids, mc_draws):
    """Evaluate Hg(x) for one x using Monte Carlo, where w_vals are
    wealth-consumption ratios stored on grids.

    The function is vmap'd for parallel computation on the GPU.

    """
    (β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z,
        ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = gcy_params
    θ = (1-γ) / (1-(1/ψ))
    h_λ, h_c, h_z, h_zπ, z, z_π = x
    # Compute the constant term, given x, which doesn't require the new state.
    σ_c = ϕ_c * jnp.exp(h_c)
    const = jnp.exp((1 - γ) * (μ_c + z) + (1/2) * (1 - γ)**2 * σ_c**2)

    # Ready to kick off the inner loop, which computes
    # E_x g(h_λ', h_c', h_z', h_zπ', z', z_π') exp(θ * h_λ')
    next_x = next_state(gcy_params, x, mc_draws)
    pf = jnp.exp(next_x[0] * θ)

    # Interpolate g(next_x) given w_vals:
    next_g = lin_interp(next_x, w_vals, grids)**θ

    e_x = jnp.mean(next_g * pf)
    Kg = const * e_x
    return Kg


Kg_vmap_mc = jax.jit(Kg_vmap_mc)


# ==========================================================#
# == Kernel for operator T using Gauss-Hermite quadrature ==#
# ==========================================================#

@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def Kg_vmap_quad(x, gcy_params, w_vals, grids, nodes, weights):
    """Evaluate Hg(x) for one x using Gauss-Hermite quadrature, where w_vals
    are wealth-consumption ratios stored on grids.

    The function is vmap'd for parallel computation on the GPU.

    """
    (β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z,
        ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = gcy_params
    θ = (1-γ) / (1-(1/ψ))
    h_λ, h_c, h_z, h_zπ, z, z_π = x
    # Compute the constant term, given x, which doesn't require the new state.
    σ_c = ϕ_c * jnp.exp(h_c)
    const = jnp.exp((1 - γ) * (μ_c + z) + (1/2) * (1 - γ)**2 * σ_c**2)

    # Ready to kick off the inner loop, which computes
    # E_x g(h_λ', h_c', h_z', h_zπ', z', z_π') exp(θ * h_λ') using Gaussian quadrature:
    next_x = next_state(gcy_params, x, nodes)
    pf = jnp.exp(next_x[0] * θ)

    # Interpolate g(next_x) given w_vals:
    next_g = lin_interp(next_x, w_vals, grids)**θ

    e_x = jnp.dot(next_g*pf, weights)
    Kg = const * e_x
    return Kg


Kg_vmap_quad = jax.jit(Kg_vmap_quad)


def T_fun_factory(params, method="quadrature", batch_size=10000):
    """Function factory for operator T.

    batch_size is the length of an array to map over in Kg_vmap. When the
    state space is large, we need to divide it into batches. We use jax.vmap
    for each batch and use jax.lax.map to loop over batches.

    """
    gcy_params = params[0]
    grids = params[1]
    (β, ψ, γ, ρ_λ, s_λ, μ_c, φ_c, ρ, ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z,
        ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = gcy_params
    θ = (1-γ) / (1-(1/ψ))
    h_λ_grid, h_c_grid, h_z_grid, h_zπ_grid, z_grid, z_π_grid = grids

    # Get grid sizes
    shape = [len(grid) for grid in grids]
    total_size = np.prod(shape)
    # Determine how many batches to create
    n_batches = total_size // batch_size
    if total_size % batch_size != 0:
        raise ValueError("""Size of the state space cannot be evenly divided
        by batch_size.""")

    dim = len(grids)

    def get_x_3d(grids, n_batches, batch_size):
        """Flatten and reshape the state space for computation"""
        mesh_grids = jnp.meshgrid(*grids, indexing='ij')
        # Each x_3d[i] is one batch with shape (batch_size, dim)
        x_3d = jnp.stack([grid.ravel() for grid in mesh_grids],
                         axis=1).reshape(n_batches, batch_size, dim)
        return x_3d

    if method == "quadrature":
        gcy_params, grids, nodes, weights = params

        @jax.jit
        def T(w):
            def Kg_map_fun(x_array):
                return Kg_vmap_quad(x_array, gcy_params, w, grids, nodes,
                                    weights)

            # Run this inside T for faster compilation time (why?)
            x_3d = get_x_3d(grids, n_batches, batch_size)

            # We loop over axis-0 of x_3d using Kg_map_fun, which applies
            # Kg_vmap to each batch, and then reshape the results back.
            Kg_out = jax.lax.map(Kg_map_fun, x_3d).reshape(shape)
            w_out = 1 + β * Kg_out**(1/θ)
            return w_out

    elif method == "monte_carlo":
        gcy_params, grids, mc_draws = params

        @jax.jit
        def T(w):
            def Kg_map_fun(x_array):
                return Kg_vmap_mc(x_array, gcy_params, w, grids, mc_draws)

            # Run this inside T for faster compilation time (why?)
            x_3d = get_x_3d(grids, n_batches, batch_size)

            # We loop over axis-0 of x_3d using Kg_map_fun, which applies
            # Kg_vmap to each batch, and then reshape the results back.
            Kg_out = jax.lax.map(Kg_map_fun, x_3d).reshape(shape)
            w_out = 1 + β * Kg_out**(1/θ)
            return w_out
    else:
        raise KeyError("Method not found.")

    return T


def wc_ratio_continuous(gcy, h_λ_grid_size=10, h_c_grid_size=10,
                        h_z_grid_size=10, h_zπ_grid_size=10, z_grid_size=20,
                        z_π_grid_size=20, num_std_devs=3.2,
                        d=5, mc_draw_size=2000, seed=1234, w_init=None,
                        ram_free=20, tol=1e-5, method='quadrature',
                        algorithm="successive_approx", verbose=True,
                        write_to_file=True, filename='w_star_data.npy'):
    """
    Iterate to convergence on the Koopmans operator associated with the SSY
    model and then return the wealth consumption ratio.
    """
    gcy_params = jnp.array(gcy.params)
    grids = build_grid(gcy, h_λ_grid_size, h_c_grid_size, h_z_grid_size,
                       h_zπ_grid_size, z_grid_size, z_π_grid_size, num_std_devs)
    dim = len(grids)

    if w_init is None:
        w_init = jnp.ones(shape=(h_λ_grid_size, h_c_grid_size, h_z_grid_size,
                                 h_zπ_grid_size, z_grid_size, z_π_grid_size))


    if algorithm == 'newton':
        scale = 8
    else:
        scale = 1

    if method == 'quadrature':
        # Calculate nodes and weights for Gauss-Hermite quadrature
        nodes, weights = qnwnorm([d]*dim)
        nodes = jnp.asarray(nodes.T)
        weights = jnp.asarray(weights)
        params = gcy_params, grids, nodes, weights

        sim_size = weights.size
    elif method == 'monte_carlo':
        # Generate shocks to evaluate the inner expectation
        key = jax.random.PRNGKey(seed)
        mc_draws = jax.random.normal(key, shape=(dim, mc_draw_size))
        params = gcy_params, grids, mc_draws

        sim_size = mc_draw_size
    else:
        raise KeyError("Approximation method not found.")

    # Determine batch_size based on available GPU memory
    batch_size = (ram_free * 1024**3 // 14) // (dim * sim_size * scale)
    # Choose the largest batch_size that evenly divides state_size
    state_size = w_init.size
    if state_size <= batch_size:
        batch_size = state_size
    else:
        max_div = 1
        for i in range(1, int(np.sqrt(state_size)) + 1):
            if state_size % i == 0:
                if i <= batch_size:
                    max_div = max(max_div, i)
                z = state_size//i
                if z <= batch_size:
                    max_div = max(max_div, z)
        batch_size = max_div
    print("batch_size =", batch_size)

    T = T_fun_factory(params, method, batch_size)
    w_star = solver(T, w_init, algorithm=algorithm)

    if write_to_file:
        # Save results
        with open(filename, 'wb') as f:
            np.save(f, grids)
            np.save(f, w_star)

    return grids, w_star


# =============================== #
# == Build callables from data == #
# =============================== #

def construct_wstar_callable(w_star_vals=None, grids=None,
                             datafile='w_star_data.npy'):
    """
    Builds and returns a jitted callable that implements an approximation of
    the function w_star by linear interpolation over the grid.

    Data for the callable is read from disk.

    """

    if w_star_vals is None or grids is None:
        with open(datafile, 'rb') as f:
            grids = np.load(f)
            w_star_vals = np.load(f)

        grids = jnp.asarray(grids)
        w_star_vals = jnp.asarray(w_star_vals)

    @jax.jit
    def w_star_func(x):
        return lin_interp(x, w_star_vals, grids)

    return w_star_func



def compare_T_factories(T_fact_old, T_fact_new, shape=(2, 3, 4, 5, 6, 7),
                        seed=1234, n=50):
    """Compare the results and speed of two function factories for T"""
    gcy = GCY()
    n_h_λ, n_h_c, n_h_z, n_h_zπ, n_z, n_z_π = shape
    std_devs = 3.0

    ssy_params = jnp.array(gcy.params)
    grids = build_grid(gcy, n_h_λ, n_h_c, n_h_z, n_h_zπ, n_z, n_z_π, std_devs)

    d = 4
    nodes, weights = qnwnorm([d]*len(grids))
    nodes = jnp.asarray(nodes.T)
    weights = jnp.asarray(weights)

    state_size = np.prod(shape)
    batch_size = state_size

    params_quad = ssy_params, grids, nodes, weights

    T_old = T_fact_old(params_quad, 'quadrature', batch_size)
    T_new = T_fact_new(params_quad, 'quadrature', batch_size)

    print("----- Testing the Operator T -----")
    # Run them once to compile
    w0 = jnp.zeros((shape))
    t0 = time.time()
    T_old(w0)
    t1 = time.time()

    comp_time_old = (t1 - t0)*1000

    t0 = time.time()
    T_new(w0)
    t1 = time.time()

    comp_time_new = (t1 - t0)*1000

    print("Compilation time: {:.4f}ms vs {:.4f}ms".format(comp_time_old,
                                                          comp_time_new))

    key = jax.random.PRNGKey(seed)
    w0_array = jax.random.uniform(key, shape=(n, n_h_λ, n_h_c, n_h_z, n_h_zπ, n_z, n_z_π))

    t0 = time.time()
    w1_old_list = []
    for i in range(n):
        w1_old_list.append(T_old(w0_array[i]))
    t1 = time.time()
    t_old = 1000*(t1 - t0)

    t0 = time.time()
    w1_new_list = []
    for i in range(n):
        w1_new_list.append(T_new(w0_array[i]))
    t1 = time.time()
    t_new = 1000*(t1 - t0)

    comparison_result = jnp.asarray([jnp.allclose(*w1) for w1 in
                                     zip(w1_old_list, w1_new_list)])

    print("Speed comparison for {} runs: {:.4f}ms vs {:.4f}ms".format(n,
                                                                      t_old,
                                                                      t_new))
    print("Same results? {}".format(jnp.all(comparison_result)))

    print("\n----- Testing Newton's Method -----")

    # Generate the function for Newton's method
    def gen_newton_fun(f):
        def g(x): return f(x) - x

        @jax.jit
        def q(x):
            # First we define the map v -> J(x) v from x and g
            jac_x_prod = lambda v: jax.jvp(g, (x,), (v,))[1]
            # Next we compute J(x)^{-1} g(x).  Currently we use
            # sparse.linalg.bicgstab. Another option is sparse.linalg.bc
            # but this operation seems to be less stable.
            b = jax.scipy.sparse.linalg.bicgstab(
                    jac_x_prod, g(x),
                    atol=1e-4)[0]
            return x - b
        return q

    T_newton_old = gen_newton_fun(T_old)
    T_newton_new = gen_newton_fun(T_new)

    t0 = time.time()
    T_newton_old(w0)
    t1 = time.time()

    comp_newton_time_old = (t1 - t0)

    t0 = time.time()
    T_newton_new(w0)
    t1 = time.time()

    comp_newton_time_new = (t1 - t0)

    print("Compilation time: {:.4f}s vs {:.4f}s".format(comp_newton_time_old,
                                                        comp_newton_time_new))

    m = int(n/50)
    t0 = time.time()
    w1_old_list = []
    for i in range(m):
        w1_old_list.append(T_newton_old(w0_array[i]))
    t1 = time.time()
    t_old = t1 - t0

    t0 = time.time()
    w1_new_list = []
    for i in range(m):
        w1_new_list.append(T_newton_new(w0_array[i]))
    t1 = time.time()
    t_new = t1 - t0

    print("Speed comparison for {} runs: {:.4f}s vs {:.4f}s".format(m, t_old,
                                                                    t_new))
    comparison_result = jnp.asarray([jnp.allclose(*w1) for w1 in
                                     zip(w1_old_list, w1_new_list)])
    print("Same results? {}".format(jnp.all(comparison_result)))
