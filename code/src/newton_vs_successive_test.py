# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %run ../src/wc_ratio_multi_index.py

from wc_ratio_multi_index import *

s = 18
ssy = SSY(L=s, K=s, I=s, J=s)

# %%time
w_star_multi_sa = wc_ratio_multi_index(ssy, algorithm="successive_approx")

# %%time
w_star_multi_newton = wc_ratio_multi_index(ssy, algorithm="newton")


