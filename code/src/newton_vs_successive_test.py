from wc_ratio_discrete import *
from single_index_code import *
import quantecon as qe

# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')

s = 8
ssy = SSY(L=s, K=s, I=s, J=s)

print("\n\nComputing multi-index successive approx.")
qe.tic()
w_star_multi_sa = wc_ratio_multi_index(ssy, algorithm="successive_approx")
qe.toc()

print("\n\nComputing multi-index Newton.")
qe.tic()
w_star_multi_newton = wc_ratio_multi_index(ssy, algorithm="newton")
qe.toc()

error = jnp.max(jnp.abs(w_star_multi_sa - w_star_multi_newton))
print(f"\n\nPairwise max error = {error}")

print("\n\nComputing single index successive approx.")
qe.tic()
w_star_single_sa = wc_ratio_single_index(ssy, algorithm="successive_approx")
qe.toc()

print("\n\nComputing single index Newton.")
qe.tic()
w_star_single_newton = wc_ratio_single_index(ssy, algorithm="newton")
qe.toc()

error = jnp.max(jnp.abs(w_star_single_sa - w_star_single_newton))
print(f"\n\nPairwise max error = {error}")

error = jnp.max(jnp.abs(w_star_single_newton - w_star_multi_newton))
print(f"\n\nPairwise max error NS vs NM = {error}")


print("\n\nComputing single index Newton specialized.")
qe.tic()
w_star_single_newton_s = wc_ratio_single_index_specialized(ssy)
qe.toc()


error = jnp.max(jnp.abs(w_star_single_newton - w_star_single_newton_s))
print(f"\n\nPairwise max error NS vs NSS = {error}")


