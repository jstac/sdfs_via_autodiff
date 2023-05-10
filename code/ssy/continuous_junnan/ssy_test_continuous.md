---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
# %env CUDA_VISIBLE_DEVICES=1
```

```{code-cell} ipython3
run ../ssy_model.py
```

```{code-cell} ipython3
run ../../solvers.py
```

```{code-cell} ipython3
run ssy_wc_ratio_continuous.py
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
```

```{code-cell} ipython3
from jax.config import config

# Tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)
```

```{code-cell} ipython3
ssy = SSY()
```

# Test wc ratio

```{code-cell} ipython3
zs   = 15
hzs  = 15
hcs  = 15
hλs  = 15
std_devs = 5.0
w_init = jnp.ones(shape=(hλs, hcs, hzs, zs))
```

```{code-cell} ipython3
wc_loglinear = wc_loglinear_factory(ssy)
```

```{code-cell} ipython3
mesh_grids = jnp.meshgrid(*build_grid(ssy, hλs, hcs, hzs, zs, std_devs), indexing='ij')
x_flat = np.asarray([grid.ravel() for grid in mesh_grids])
w_init = jnp.asarray([wc_loglinear(x_flat[:, i]) for i in range(x_flat.shape[1])]).reshape((hλs, hcs, hzs, zs))
```

## Successive evaluation

```{code-cell} ipython3
%%time

grids, out = wc_ratio_continuous(ssy, h_λ_grid_size=hλs, h_c_grid_size=hcs, 
                                 h_z_grid_size=hzs, z_grid_size=zs, 
                                 num_std_devs=std_devs, mc_draw_size=5000, 
                                 w_init=w_init, ram_free=20, tol=1e-5, write_to_file=True,
                                 filename='w_star_data.npy')
```

```{code-cell} ipython3
out
```

## Anderson Acceleration

```{code-cell} ipython3
%%time

grids, out = wc_ratio_continuous(ssy, h_λ_grid_size=hλs, h_c_grid_size=hcs, 
                                 h_z_grid_size=hzs, z_grid_size=zs, 
                                 num_std_devs=std_devs, mc_draw_size=5000, 
                                 w_init=w_init, ram_free=20, algorithm="anderson", 
                                 write_to_file=True, filename='w_star_data.npy')
```

```{code-cell} ipython3
out
```

```{code-cell} ipython3
wc_func = construct_wstar_callable('w_star_data.npy')
```

```{code-cell} ipython3
wc_loglinear = wc_loglinear_factory(ssy)
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(8, 8))


titles = 'h_λ', 'h_c', 'h_z', 'z'

for pos, grid, title in zip(range(4), grids, titles):
    ax = axes.flatten()[pos]
    y1 = np.empty_like(grid)
    y2 = np.empty_like(grid)

    for i, val in enumerate(grid):
        x = np.zeros(4)
        x[pos] = val
        y1[i] = np.log(wc_func(x)) 
        y2[i] = wc_loglinear(x)

    ax.plot(grid, y1, label='numerical')
    ax.plot(grid, y2, label='log-linear')
    ax.set_xlabel(title)
    ax.set_ylim(2, 8)
    ax.legend(loc='lower left')

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
h_λ_grid, h_c_grid, h_z_grid, z_grid = grids
```

```{code-cell} ipython3
xg = z_grid
yg = h_z_grid
x, y = np.meshgrid(xg, yg)

z = out[int(hλs/2), int(hcs/2), :, :]


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                np.log(z.T),
                rstride=2, cstride=2,
                cmap=cm.viridis,
                alpha=0.7,
                linewidth=0.25)

ax.set_xlabel('$z$', fontsize=14)
ax.set_ylabel('$h_z$', fontsize=14)
ax.set_xticks((-0.0015, 0, 0.0015))
ax.set_yticks((-1.5, -0.5, 0.5, 1.5))

ax.view_init(18, -147)

plt.show()
```

```{code-cell} ipython3
xg = h_c_grid
yg = h_λ_grid
x, y = np.meshgrid(xg, yg)

z = out[:, :, int(hzs/2), int(zs/2)]


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                np.log(z.T),
                rstride=2, cstride=2,
                cmap=cm.viridis,
                alpha=0.7,
                linewidth=0.25)
#ax.set_zlim(-0.5, 1.0)
ax.set_xlabel('$h_c$', fontsize=14)
ax.set_ylabel('$h_\lambda$', fontsize=14)
ax.view_init(18, -134)
plt.show()
```

```{code-cell} ipython3
xg = z_grid
yg = h_λ_grid
x, y = np.meshgrid(xg, yg)

z = out[:, int(hcs/2), int(hzs/2), :]


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                np.log(z.T),
                rstride=2, cstride=2,
                cmap=cm.viridis,
                alpha=0.7,
                linewidth=0.25)
#ax.set_zlim(-0.5, 1.0)
ax.view_init(18, -134)
plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

# Test Anderson Acceleration

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
def get_batch_size(state_size, mc_draw_size, ram_free):
    batch_size = ram_free * 30000000 // mc_draw_size
    if state_size <= batch_size:
        batch_size = state_size
    else:
        while (state_size % batch_size > 0):
            batch_size -= 1
    return batch_size
```

```{code-cell} ipython3
zs   = 15
hzs  = 15
hcs  = 15
hλs  = 15
std_devs = 5.0
w_init = jnp.ones(shape=(hλs, hcs, hzs, zs))
mc_draw_size = 5000
seed = 1234
```

```{code-cell} ipython3
batch_size = get_batch_size(zs*hzs*hcs*hλs, mc_draw_size=mc_draw_size, ram_free=10)
```

```{code-cell} ipython3
batch_size
```

```{code-cell} ipython3
w_init = jnp.zeros(shape=(hλs, hcs, hzs, zs)) 
```

```{code-cell} ipython3
mesh_grids = jnp.meshgrid(*build_grid(ssy, hλs, hcs, hzs, zs, std_devs), indexing='ij')
x_flat = np.asarray([grid.ravel() for grid in mesh_grids])
w_init = jnp.asarray([wc_loglinear(x_flat[:, i]) for i in range(x_flat.shape[1])]).reshape((hλs, hcs, hzs, zs))
```

## Test Jax's AA Algorithm

```{code-cell} ipython3
import jaxopt
```

```{code-cell} ipython3
ssy_params = jnp.array(ssy.params)
grids = build_grid(ssy, hλs, hcs, hzs, zs, std_devs)

# generate shocks to evaluate the inner expectation
key = jax.random.PRNGKey(1234)
mc_draws = jax.random.normal(key, shape=(4, mc_draw_size))

# determine batch_size
state_size = hλs* hcs * hzs * zs
batch_size = 20 * 30000000 // mc_draw_size
if state_size <= batch_size:
    batch_size = state_size
else:
    while (state_size % batch_size > 0):
        batch_size -= 1

print("batch_size =", batch_size)

params = ssy_params, grids, mc_draws
T_tmp = fun_factory(params, batch_size=batch_size)

def T(w):
    return T_tmp(w, params)
```

```{code-cell} ipython3
%%time
W_tmp = jnp.copy(w_init)
for i in range(20):
    W_tmp = T(W_tmp)
```

```{code-cell} ipython3
%%time
AA = jaxopt.AndersonAcceleration(T, verbose=False, mixing_frequency=5, tol=1e-7,
                                 maxiter=500, history_size=10, beta=5.0, implicit_diff=False, ridge=1e-6, 
                                 jit=True, unroll=True)

W_tmp = jnp.copy(w_init)
for i in range(10):
    W_tmp = T(W_tmp)

out = AA.run(W_tmp)
w_out = out[0]
current_iter = int(out[1][0])
```

```{code-cell} ipython3
print(np.max(np.abs(w_star-out[0])), "iter_num =", out[1].iter_num)
```

```{code-cell} ipython3
out[1][0]
```

```{code-cell} ipython3
out[0]
```

```{code-cell} ipython3
%%time
w_star, iter = anderson_solver(T, w_init, max_iter=500)
```

```{code-cell} ipython3
w_star
```

```{code-cell} ipython3
print(np.max(np.abs(w_accurate[0]-out[0])), "iter_num =", out[1].iter_num)
```

```{code-cell} ipython3
np.max(np.abs(w_est[0]-w_accurate[0].ravel()))
```

```{code-cell} ipython3

```

### Naive AA (deprecated)

```{code-cell} ipython3
# one dimension for jax
def get_c(AR_series, m, offset):
    U = jnp.empty((m+1, m+1))
    for j in range(m):
        for i in range(m+1):
            U = U.at[j, i].set(AR_series[offset + 1 + i + j] - AR_series[offset + i + j])
    U = U.at[m, :].set(1.0)
    rhs = jnp.zeros(m+1)
    rhs = rhs.at[-1].set(1.0)
    c = jnp.linalg.solve(U, rhs)
    return c

get_c = jax.jit(get_c, static_argnames=('m'))
get_c_vmap = jax.vmap(get_c, in_axes=(1, None, None), out_axes=(1))

dot_vmap = jax.vmap(jnp.dot, in_axes=(1))
```

```{code-cell} ipython3
def AA_solver(f, w_init, m = 2, tol=1e-7, max_iter=10000, print_skip=10, verbose=True):
    def build_list(w_in):
        # initialization
        # store results of 2m+1 iterations
        w_flat_list = []
        w_in = w_init
        for i in range(2 * m + 1):
            w_out = f(w_in)
            w_flat_list.append(w_out.ravel())
            w_in = w_out
        return w_flat_list, w_out

    w_flat_list, w_in = build_list(w_init)
    # get one extrapolated array
    w_flat_array = jnp.asarray(w_flat_list)
    w_est = dot_vmap(get_c_vmap(w_flat_array, m, 0), w_flat_array[0:m+1])
    
    # start the iteration
    error_aa = tol + 1.0
    i = 2*m
    while error_aa > tol and i < max_iter:
        w_out = f(w_in)
        w_flat_list.pop(0)
        w_flat_list.append(w_out.ravel())
        error = jnp.max(jnp.abs(w_in - w_out))
        w_in = w_out

        w_flat_array = jnp.asarray(w_flat_list)
        w_est_new = dot_vmap(get_c_vmap(w_flat_array, m, 0), w_flat_array[0:m+1])
        error_aa = jnp.max(jnp.abs(w_est - w_est_new))
        w_est = w_est_new
        if verbose and i % print_skip == 0:
            print("Error = {}; Error (AA) = {}".format(error, error_aa))
        i += 1

    if i == max_iter:
        print(f"Warning: Hit maximum iteration number {max_iter}")
    else:
        print(f"Iteration converged after {i} iterations") 
    return w_est, i
```

```{code-cell} ipython3

```

```{code-cell} ipython3
%%time
w_est = AA_solver(lambda x: T(x, params), w_init, m=3, tol=1e-5)
```

```{code-cell} ipython3
%%time
w_out = fwd_solver(lambda x: T(x, params), w_init, tol=1e-6)
```

```{code-cell} ipython3
%%time
w_accurate = fwd_solver(lambda x: T(x, params), w_init, tol=1e-11)
```

```{code-cell} ipython3
np.max(np.abs(w_out[0].ravel() - w_accurate[0].ravel()))
```

```{code-cell} ipython3
np.max(np.abs(w_est[0].ravel() - w_accurate[0].ravel()))
```

```{code-cell} ipython3
np.max(np.abs(w_out[0].ravel() - w_est[0].ravel()))
```
