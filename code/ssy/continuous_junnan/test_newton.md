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
mc_draws.shape
```

```python
1 != 0
```

```python
next_state(ssy_params, jnp.zeros(4), mc_draws)
```

```python
zs   = 15
hzs  = 15
hcs  = 15
hλs  = 15
std_devs = 3.0

ssy_params = jnp.array(ssy.params)
grids = build_grid(ssy, hλs, hcs, hzs, zs, std_devs)

mesh_grids = jnp.meshgrid(*grids, indexing='ij')
x_flat = np.asarray([grid.ravel() for grid in mesh_grids])
w_init = jnp.asarray([wc_loglinear(x_flat[:, i]) for i in range(x_flat.shape[1])]).reshape((hλs, hcs, hzs, zs))
w_init = jnp.exp(w_init)
```

#### Interpolate $\log(WC_t)$ instead of $WC_t^\theta$

```python
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def Kg_vmap_quad(x, ssy_params, ln_w, grids, nodes, weights):
    """Evaluate Hg(x) for one x using Gauss-Hermite quadrature, where g is
    calculated using log wealth-consumption ratio stored on grids.

    The function is vmap'd for parallel computation on the GPU.

    """
    (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = ssy_params
    θ = (1-γ) / (1-(1/ψ))
    h_λ, h_c, h_z, z = x
    # Compute the constant term, given x, which doesn't require the new state.
    σ_c = ϕ_c * jnp.exp(h_c)
    const = jnp.exp((1 - γ) * (μ_c + z) +
                    (1/2) * (1 - γ)**2 * σ_c**2)

    # Ready to kick off the inner loop, which computes
    # E_x g(h_λ', h_c', h_z', z') exp(θ * h_λ') using Gaussian quadrature:
    next_x = next_state(ssy_params, x, nodes)
    pf = jnp.exp(next_x[0] * θ)

    # Interpolate g(next_x) given ln_w:
    next_g = jnp.exp(lin_interp(next_x, ln_w, grids))**θ

    e_x = jnp.dot(next_g*pf, weights)
    Kg = const * e_x
    return Kg


Kg_vmap_quad = jax.jit(Kg_vmap_quad)


def T_fun_quad_factory(params, batch_size=10000):
    """Function factory for operator T.

    batch_size is the length of an array to map over in Kg_vmap. When the
    state space is large, we need to divide it into batches. We use jax.vmap
    for each batch and use jax.lax.map to loop over batches.

    """

    @jax.jit
    def wc_operator_continuous(ssy_params, w_in, grids, nodes, weights):
        (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = ssy_params
        θ = (1-γ) / (1-(1/ψ))
        h_λ_grid, h_c_grid, h_z_grid, z_grid = grids

        # Get grid sizes
        nh_λ = len(h_λ_grid)
        nh_c = len(h_c_grid)
        nh_z = len(h_z_grid)
        nz = len(z_grid)

        # Determine how many batches to create
        n_batches = nh_λ * nh_c * nh_z * nz // batch_size

        ln_w = jnp.log(w_in)

        # Flatten and reshape the state space for computation
        mesh_grids = jnp.meshgrid(*grids, indexing='ij')
        # Each x_3d[i] is one batch with shape (batch_size, 4)
        x_3d = jnp.stack([grid.ravel() for grid in mesh_grids],
                         axis=1).reshape(n_batches, batch_size, 4)

        def Kg_map_fun(x_array):
            return Kg_vmap_quad(x_array, ssy_params, ln_w, grids, nodes,
                                weights)

        # We loop over axis-0 of x_3d using Kg_map_fun, which applies Kg_vmap
        # to each batch, and then reshape the results back.
        Kg_out = jax.lax.map(Kg_map_fun, x_3d).reshape(nh_λ, nh_c, nh_z, nz)
        w_out = 1 + β * Kg_out**(1/θ)

        return w_out

    @jax.jit
    def T(w):
        "T via JAX operations."
        ssy_params, grids, nodes, weights = params
        w_out = wc_operator_continuous(ssy_params, w, grids, nodes, weights)
        return w_out

    return T
```

### Generate Operator $T$

```python
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

## Anderson Acceleration

```python
%%time

xstar2, iter = anderson_solver(T_quad, w_init, tol=default_tolerance, max_iter=50000, verbose=False)
```

## Newton's Method

```python
%%time

xstar3, iter = newton_solver(T_quad, jnp.ones_like(w_init), verbose=True, print_skip=1)
```

```python
jnp.allclose(xstar2, xstar3)
```

```python
seed = 1234
T = 1000000
key = jax.random.PRNGKey(seed)
mc_draws = jax.random.normal(key, shape=(4, T))
x_seq = next_state(ssy_params, jnp.zeros(4), mc_draws)
wc_seq = lin_interp(x_seq, xstar3, grids)
jnp.array([wc_seq.mean(), wc_seq.std()])
```

```python
seed = 1234
T = 1000000
key = jax.random.PRNGKey(seed)
mc_draws = jax.random.normal(key, shape=(4, T))
x_seq = next_state(ssy_params, jnp.zeros(4), mc_draws)
wc_seq = lin_interp(x_seq, xstar2, grids)
jnp.array([wc_seq.mean(), wc_seq.std()])
```

Interpolate $w^\theta$

20^4, std=2.5, d=8: [976.43571268,   8.62554633]

15^4, std=2.5, d=5: [983.28449407,   8.76520362]

15^4, std=2.8, d=5: [864.27515112,   8.0951512 ]

15^4, std=3.2, d=5: [670.75128139,   6.60051464]

15^4, std=2.5, d=8: [914.93331693,   7.87979566]


Interpolate $\ln(w)$

15^4, std=2.5, d=5: [1077.95676508,    9.61219993]

15^4, std=2.8, d=5: [981.03514072,   9.0978135 ]

15^4, std=3.2, d=5: [865.00929848,   8.35713019]

15^4, std=2.5, d=8: [1092.81133325,    9.62531072]

```python
x_seq.max(axis=1)
```

```python
[grid.max() for grid in grids]
```

```python
x_seq_np = np.asarray(x_seq)
wc_loglin_seq = np.asarray([np.exp(wc_loglinear(x)) + 1 for x in x_seq_np.T])
wc_loglin_seq.mean(), wc_loglin_seq.std()
```

```python

```
