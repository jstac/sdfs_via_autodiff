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
%run ssy_wc_ratio_continuous_vectorized.py
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
from quantecon.quad import qnwnorm
```

```python
ssy = SSY()
zs   = 20
hzs  = 20
hcs  = 20
hλs  = 20
std_devs = 5.0
# w_init = jnp.ones(shape=(hλs, hcs, hzs, zs))
wc_loglinear = wc_loglinear_factory(ssy)
mesh_grids = jnp.meshgrid(*build_grid(ssy, hλs, hcs, hzs, zs, std_devs), indexing='ij')
x_flat = np.asarray([grid.ravel() for grid in mesh_grids])
w_init = jnp.asarray([wc_loglinear(x_flat[:, i]) for i in range(x_flat.shape[1])]).reshape((hλs, hcs, hzs, zs))
```

```python
ssy_params = jnp.array(ssy.params)
grids = build_grid(ssy, hλs, hcs, hzs, zs, std_devs)

d = 6
nodes, weights = qnwnorm([d, d, d, d])
nodes = jnp.asarray(nodes.T)
weights = jnp.asarray(weights)

params = ssy_params, grids, nodes, weights
T_quad = T_fun_quad_vectorized_factory(params)
```

```python

```

```python
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
```

```python
newton_fun = gen_newton_fun(T_quad)
```

```python
%%time
newton_fun(w_init)
```

```python
%%time 
xstar, iter = successive_approx(newton_fun, w_init, print_skip=1)
```

```python

```

```python
%%time 
xstar, iter = successive_approx(T_quad, w_init, print_skip=100)
```

```python
xstar
```

```python
%%time 
xstar, iter = anderson_solver(T_quad, w_init)
```
