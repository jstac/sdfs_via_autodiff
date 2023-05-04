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

# Discrete Solution Methods: Single vs Multi Index Methods

+++

We compare single and multi index methods for computing the wealth consumption ratio.

Note that the single index code is only for comparison purposes, to be used for **small** state spaces.  It's just a check that the multi index code works.

+++

(The size of the Markov matrix is $n \times n$ when $n$ is the size of the state space.  Similarly, the Jacobian for Newton's method is $n \times n$. This matrix cannot be instantiated or manipulated when $n$ is large.  Hence the single index code fails for large $n$.)

+++

## Single Index Code

```{code-cell} ipython3
# Ignore warnings (QuantEcon user warning)
import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3
run ../src/single_index_code.py
```

Let's start with $n=3^4$, just to verify that the code works.

```{code-cell} ipython3
s = 3
ssy = SSY(L=s, K=s, I=s, J=s)
```

```{code-cell} ipython3
print(solvers.keys())
```

```{code-cell} ipython3
%%time
w_star_single = wc_ratio_single_index(ssy, algorithm="newton")
```

```{code-cell} ipython3
w_star_single
```

When $n$ gets to around 12, the kernel dies or the program fails with an out of memory error.

+++

## Multi-Index Code

+++

With multi-index code, the state remains as a multidimensional grid and the Markov matrix of transitions does not need to be instantiated.  Transition probabilities are calculated at run time.

In addition, we use some JAX tricks to avoid instantiating the Jacobian used in Newton iteration.

To put this on the GPU via JAX, we use broadcasting to define the necessary calculations.

```{code-cell} ipython3
run ../src/wc_ratio.py
```

Here's a quick test at $n=3^4$:

```{code-cell} ipython3
s = 3
ssy = SSY(L=s, K=s, I=s, J=s)
```

```{code-cell} ipython3
%%time
w_star_multi_newton = wc_ratio(ssy, algorithm="newton")
```

Then numbers are very close to what we get with a single index calculation:

```{code-cell} ipython3
w_star_multi_newton
```

```{code-cell} ipython3

```
