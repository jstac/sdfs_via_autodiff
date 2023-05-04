---
jupytext:
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

# Test Newton vs Successive Approximation

Tests the timings of these two algorithms in the discretized case (with standard multi-index).

```{code-cell} ipython3
# Ignore warnings (QuantEcon user warning)
import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3

run ../src/wc_ratio.py
```

```{code-cell} ipython3
s = 3
ssy = SSY(L=s, K=s, I=s, J=s)
```

The size of the state space is 

```{code-cell} ipython3
s**4
```

```{code-cell} ipython3
%%time
w_star_multi_sa = wc_ratio(ssy, algorithm="successive_approx")
```

```{code-cell} ipython3
w_star_multi_sa
```

```{code-cell} ipython3
%%time
w_star_multi_newton = wc_ratio(ssy, algorithm="newton")
```

```{code-cell} ipython3
w_star_multi_newton
```

```{code-cell} ipython3
jnp.max(jnp.abs(w_star_multi_newton - w_star_multi_sa))
```

```{code-cell} ipython3

```
