---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%env CUDA_VISIBLE_DEVICES=1
```

```python
%run ../ssy_model.py
%run ../../solvers.py
%run ssy_wc_ratio_continuous.py
```

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from jax.config import config

# Tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)
```

```python
ssy = SSY()
wc_loglinear = wc_loglinear_factory(ssy)
```

```python
zs   = 15
hzs  = 15
hcs  = 15
hλs  = 15
std_devs = 3.0

mesh_grids = jnp.meshgrid(*build_grid(ssy, hλs, hcs, hzs, zs, std_devs), indexing='ij')
x_flat = np.asarray([grid.ravel() for grid in mesh_grids])
w_init = jnp.asarray([wc_loglinear(x_flat[:, i]) for i in range(x_flat.shape[1])]).reshape((hλs, hcs, hzs, zs))
w_init = jnp.exp(w_init)
```

### Generate Operator $T$

```python
ssy_params = jnp.array(ssy.params)
grids = build_grid(ssy, hλs, hcs, hzs, zs, std_devs)

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
params_quad = ssy_params, grids, nodes, weights
T_quad = T_fun_quad_factory(params_quad, batch_size=batch_size)
```

## Successive Evaluation

```python
%%time

xstar1, iter = successive_approx(T_quad, w_init, verbose=True, print_skip=1000)
```

## Anderson Acceleration

```python
%%time

xstar2, iter = anderson_solver(T_quad, w_init, tol=default_tolerance, max_iter=5000, verbose=False)
```

## Newton's Method

```python
%%time

xstar3, iter = newton_solver(T_quad, w_init, verbose=True, print_skip=1)
```

```python

```

```python
jnp.allclose(xstar1, xstar2, xstar3)
```

```python

```
