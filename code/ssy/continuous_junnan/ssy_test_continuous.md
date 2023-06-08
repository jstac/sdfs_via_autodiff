---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
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
std_devs = 3.0
w_init = jnp.ones(shape=(hλs, hcs, hzs, zs))
```

```{code-cell} ipython3
wc_loglinear = wc_loglinear_factory(ssy)
```

```{code-cell} ipython3
# mesh_grids = jnp.meshgrid(*build_grid(ssy, hλs, hcs, hzs, zs, std_devs), indexing='ij')
# x_flat = np.asarray([grid.ravel() for grid in mesh_grids])
# w_init = jnp.asarray([wc_loglinear(x_flat[:, i]) for i in range(x_flat.shape[1])]).reshape((hλs, hcs, hzs, zs))
```

## Successive evaluation

+++

### Monte Carlo

```{code-cell} ipython3
%%time

grids, out = wc_ratio_continuous(ssy, h_λ_grid_size=hλs, h_c_grid_size=hcs, 
                                 h_z_grid_size=hzs, z_grid_size=zs, 
                                 num_std_devs=std_devs, mc_draw_size=2000, 
                                 w_init=w_init, ram_free=20, tol=1e-5, method='monte_carlo', 
                                 write_to_file=True, filename='w_star_data.npy')
```

### Quadrature

```{code-cell} ipython3
%%time

grids, out = wc_ratio_continuous(ssy, h_λ_grid_size=hλs, h_c_grid_size=hcs, 
                                h_z_grid_size=hzs, z_grid_size=zs, 
                                num_std_devs=std_devs, d=5, method='quadrature',
                                w_init=w_init, ram_free=20, tol=1e-5, write_to_file=True,
                                filename='w_star_data.npy')
```

## Anderson Acceleration

+++

### Monte Carlo

```{code-cell} ipython3
%%time

grids, out = wc_ratio_continuous(ssy, h_λ_grid_size=hλs, h_c_grid_size=hcs, 
                                 h_z_grid_size=hzs, z_grid_size=zs, 
                                 num_std_devs=std_devs, mc_draw_size=2000, method='monte_carlo',
                                 w_init=w_init, ram_free=20, tol=1e-5, algorithm="anderson", 
                                 write_to_file=True, filename='w_star_data.npy')
```

### Quadrature

```{code-cell} ipython3
%%time

grids, out = wc_ratio_continuous(ssy, h_λ_grid_size=hλs, h_c_grid_size=hcs, 
                                h_z_grid_size=hzs, z_grid_size=zs, 
                                num_std_devs=std_devs, d=5, method='quadrature',
                                w_init=w_init, ram_free=20, tol=1e-5, algorithm="anderson", 
                                write_to_file=True, filename='w_star_data.npy')
```

## Newton's Method

```{code-cell} ipython3
%%time

grids, out = wc_ratio_continuous(ssy, h_λ_grid_size=hλs, h_c_grid_size=hcs, 
                                h_z_grid_size=hzs, z_grid_size=zs, 
                                num_std_devs=std_devs, d=5, method='quadrature',
                                w_init=w_init, ram_free=20, tol=1e-5, algorithm="newton", 
                                write_to_file=True, filename='w_star_data.npy')
```

```{code-cell} ipython3

```

# Tune AA parameters

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
d = 5
nodes, weights = qnwnorm([d, d, d, d])
nodes = jnp.asarray(nodes.T)
weights = jnp.asarray(weights)
# determine batch_size
state_size = hλs* hcs * hzs * zs
batch_size = 20 * 30000000 // (weights.size * 2)
if state_size <= batch_size:
    batch_size = state_size
else:
    while (state_size % batch_size > 0):
        batch_size -= 1
print(batch_size)
params = ssy.params, grids, nodes, weights
T = T_fun_factory(params, method="quadrature", batch_size=batch_size)
```

```{code-cell} ipython3
%%time
AA = jaxopt.AndersonAcceleration(T, verbose=True, mixing_frequency=4, tol=1e-5,
                                 maxiter=2000, history_size=10, beta=8.0, implicit_diff=False, ridge=1e-6, 
                                 jit=True, unroll=True)

W_tmp = jnp.copy(w_init)
for i in range(10):
    W_tmp = T(W_tmp)

out = AA.run(W_tmp)
w_out = out[0]
current_iter = int(out[1][0])
print(current_iter, jnp.any(jnp.isnan(w_out)))
```

```{code-cell} ipython3
%%time
successive_approx(T, w_init)
```

```{code-cell} ipython3

```

# Plots

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
        y1[i] = np.log(wc_func(x.reshape(4, 1))) 
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
