import jax
import jax.numpy as jnp

# == Interpolation related utilities == #


@jax.jit
def jit_map_coordinates(vals, coords):
    return jax.scipy.ndimage.map_coordinates(vals, coords, order=1,
                                             mode='nearest')


def vals_to_coords(grids, x_vals):
    """Transform values of the states to corresponding coordinates (array
    indices) on the grids.

    """
    # jax.jit doesn't allow dynamic shapes so we hard code its dimension
    dim = 4

    intervals = jnp.asarray([grid[1] - grid[0] for grid in grids])
    low_bounds = jnp.asarray([grid[0] for grid in grids])

    intervals = intervals.reshape(dim, 1)
    low_bounds = low_bounds.reshape(dim, 1)

    return (x_vals - low_bounds) / intervals


@jax.jit
def lin_interp(x, g_vals, grids):
    """x: jnp array of shape (N, 4)"""
    coords = vals_to_coords(grids, x)
    # Interpolate using coordinates
    next_g = jit_map_coordinates(g_vals, coords)
    return next_g


# def lininterp_funcvals(ssy, function_vals):
#     """
#     Builds and returns a jitted callable that implements an approximation of
#     the function determined by function_vals via linear interpolation over the
#     grid.

#         expected grid is (h_λ_states, h_c_states, h_z_states, z_states)


#     """

#     h_λ_states = ssy.h_λ_states
#     h_c_states = ssy.h_c_states
#     h_z_states = ssy.h_z_states
#     z_states = ssy.z_states

#     @njit
#     def interpolated_function(x):
#         h_λ, h_c, h_z, z = x
#         i = np.searchsorted(h_z_states, h_z)
#         # Don't go out of bounds
#         if i == len(h_z_states):
#             i = i - 1

#         return lininterp_4d(h_λ_states,
#                             h_c_states,
#                             h_z_states,
#                             z_states[i, :],
#                             function_vals,
#                             x)

#     return interpolated_function
