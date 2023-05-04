"""

Schorfheide--Song--Yaron code.

There are four states for the recursive utility / wealth-consumption ratio
problem, namely

    x = (h_λ, h_c, h_z, z)  # indexed by l, k, i, j

State dynamics are


    h_λ' = ρ_λ h_λ + s_λ η'
    h_c' = ρ_c h_c + s_c η'
    h_z' = ρ_z h_z + s_z η'
    z'   = ρ   z   + σ_z η'

The volatility coefficients are

    σ_z = ϕ_z exp(h_z)
    σ_c = ϕ_c exp(h_c)

Consumption growth and preference shock growth are given by

    g_c = μ_c + z + σ_c ξ_c'
    g_λ' = h_λ'

Changes from SSY notation (ours -> original):

    * ϕ_z  ->  ϕ_z σ_bar sqrt(1 - ρ^2)
    * ϕ_c  ->  ϕ_c σ_bar
    * β    ->  δ

All innovations are IID and N(0, 1).

Baseline parameter values stated below are from Table VII of the paper.
The symbol h_i used for volatility is σ_hi in SSY, while s_i is used for
volatility of this process.  For μ_c see the table footnote.
For ϕ_c and ϕ_d, see the sentence after eq (9).


"""

import jax
import jax.numpy as jnp
from jax.config import config

import numpy as np
from scipy.optimize import brentq
from quantecon import MarkovChain, rouwenhorst

from numba import njit, prange, float32, cuda
from numpy.random import rand, randn


class SSY:
    """
    Stores the SSY model parameters, along with data for a discretized
    (multi-index) version with indices (l, k, i, j).

    """

    def __init__(self,
                 β=0.999,                                # = δ in SSY
                 γ=8.89,
                 ψ=1.97,
                 ρ=0.987,
                 ρ_z=0.992,
                 ρ_c=0.991,
                 ρ_λ=0.959,
                 s_z=np.sqrt(0.0039),
                 s_c=np.sqrt(0.0096),
                 s_λ=0.0004,
                 μ_c=0.0016,
                 φ_z=0.215*0.0035*np.sqrt(1-0.987**2),   # *σ_bar*sqrt(1-ρ^2)
                 φ_c=1.00*0.0035,                        # *σ_bar
                 ):

        # Store parameters and derived parameters
        self.β, self.γ, self.ψ = β, γ, ψ
        self.μ_c,self.ϕ_z, self.ϕ_c = μ_c, ϕ_z, ϕ_c
        self.ρ, self.ρ_z, self.ρ_c, self.ρ_λ = ρ, ρ_z, ρ_c, ρ_λ
        self.s_z, self.s_c, self.s_λ = s_z, s_c, s_λ
        self.θ = (1 - γ) / (1 - 1/ψ)

        # Pack params into an array
        self.params = β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ

    def discretize_multi_index(self, L=4, K=4, I=4, J=4):
        """
        Discretizes the SSY model using a multi-index and returns a discrete
        representation of the model.

        The discretization uses iterations of the Rouwenhorst method.  The
        indices are

            h_λ[l] for l in range(L)
            h_c[k] for k in range(K)
            h_z[i] for i in range(I)
            z[i, j] is the j-th z state when σ_z = σ_z[i] for j in range(J)

            z_Q[i, j, jp] is trans prob from j to jp when σ_z index = i

        """

        β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ = self.params

        h_λ_mc = rouwenhorst(L, ρ_λ, s_λ, 0)
        h_c_mc = rouwenhorst(K, ρ_c, s_c, 0)
        h_z_mc = rouwenhorst(I, ρ_z, s_z, 0)

        # Build states
        h_λ_states = h_λ_mc.state_values
        h_c_states = h_c_mc.state_values
        h_z_states = h_z_mc.state_values
        σ_z_states = ϕ_z * np.exp(h_z_states)
        z_states = np.zeros((I, J))

        # Build transition probabilities
        z_Q = np.zeros((I, J, J))

        for i, σ_z in enumerate(σ_z_states):
            mc_z = rouwenhorst(J, 0, σ_z, ρ)
            for j in range(J):
                z_states[i, j] = mc_z.state_values[j]
                z_Q[i, j, :] = mc_z.P[j, :]
                 
        # For convenience, store the sigma states as well
        σ_c_states = ϕ_c * jnp.exp(h_c_states)
        σ_z_states = ϕ_z * jnp.exp(h_z_states)

        arrays = (h_λ_states,  h_λ_mc.P,
                  h_c_states,  h_c_mc.P,
                  h_z_states,  h_z_mc.P,
                  z_states,    z_Q,
                  σ_c_states, σ_z_states)

        shapes = L, K, I, J

        return shapes, self.params, arrays

    def discrete_T_operator_factory(self, L, K, I, J):

        shapes, params, arrays = self.discretize_multi_index(L, K, I, J)
         
        def _T(w):
            """
            Implement a discrete version of the operator T for the SSY model via
            JAX operations.  In the call signature, we pass array sizes to aid
            the JIT compiler.  In particular, changing sizes triggers a
            recompile. 
            """

            # Unpack
            L, K, I, J = shapes
            (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = params
            (h_λ_states, h_λ_P,              
             h_c_states, h_c_P,
             h_z_states, h_z_P,
             z_states,   z_Q,
             σ_c_states, σ_z_states) = arrays

            θ = (1 - γ) / (1 - 1/ψ)

            # Create intermediate arrays
            B1 = jnp.exp(θ * h_λ_states)
            B1 = jnp.reshape(B1, (1, 1, 1, 1, L, 1, 1, 1))
            B2 = jnp.exp(0.5*((1-γ)*σ_c_states)**2)
            B2 = jnp.reshape(B2, (1, K, 1, 1, 1, 1, 1, 1))
            B3 = jnp.exp((1-γ)*(μ_c+z_states))
            B3 = jnp.reshape(B3, (1, 1, I, J, 1, 1, 1, 1))

            # Reshape existing matrices prior to reduction
            Phλ = jnp.reshape(h_λ_P, (L, 1, 1, 1, L, 1, 1, 1))
            Phc = jnp.reshape(h_c_P, (1, K, 1, 1, 1, K, 1, 1))
            Phz = jnp.reshape(h_z_P, (1, 1, I, 1, 1, 1, I, 1))
            Pz = jnp.reshape(z_Q, (1, 1, I, J, 1, 1, 1, J))
            w = jnp.reshape(w, (1, 1, 1, 1, L, K, I, J))

            # Take product and sum along last four axes
            A = w**θ * B1 * B2 * B3 * Phλ * Phc * Phz * Pz
            Hwθ = jnp.sum(A, axis=(4, 5, 6, 7))

            # Define and return Tw
            Tw = 1 + β * Hwθ**(1/θ)
            return Tw

        # JIT compile the intermediate function _T
        # T = jax.jit(_T, static_argnums=(1, ))
        return jax.jit(_T)





## ==== Log linear approximation of the SSY W/C ratio === ##

def wc_loglinear_factory(ssy):
    """
    A factory function that computes the constant terms for the WC ratio log
    linear approximation and then creates a jitted function that evaluates the
    log linear approximation.

    """

    # Unpack parameters
    (β, γ, ψ,
        μ_c, ρ, ϕ_z, ϕ_c,
        ρ_z, ρ_c, ρ_λ,
        s_z, s_c, s_λ) = ssy.params
    θ = ssy.θ

    s_wc = 2*(φ_c)**2*s_c;
    s_wx = 2*(φ_z)**2*s_z;

    def fk1(x):
        return np.exp(x)/(1+np.exp(x))

    def fk0(x):
        return np.log(1+np.exp(x))-fk1(x)*x

    def fA1(x):
        return (1-1/ψ)/(1-fk1(x)*ρ)

    def fAλ(x):
        return ρ_λ/(1-fk1(x)*ρ_λ)

    def fAz(x):
        return (θ/2)*(fk1(x)*fA1(x))**2/(1-fk1(x)*ρ_z)

    def fAc(x):
        return (θ/2)*(1-1/ψ)**2/(1-fk1(x)*ρ_c)

    def fA0(x):
        return (np.log(β)+fk0(x)+μ_c * (1-1/ψ) \
                + fk1(x) * fAz(x) * (φ_z)**2 * (1-ρ_z)\
                + fk1(x) * fAc(x) * (φ_c)**2 * (1-ρ_c)\
                + (θ/2) * ((fk1(x) * fAλ(x)+1)**2 * s_λ**2 \
                + (fk1(x) * fAz(x) * s_wx)**2+(fk1(x) * fAc(x) * s_wc)**2)) \
                / (1-fk1(x))

    def fq_bar(x):
        return x - fA0(x) - fAc(x)*(φ_c)**2 - fAz(x)*(φ_z)**2

    qbar = brentq(fq_bar, -20, 20)
    Az = fA1(qbar)
    Ah_λ = fAλ(qbar)
    Ah_z = fAz(qbar)
    Ah_c = fAc(qbar)
    A0 = fA0(qbar)

    # Now build a jitted function 
    @njit
    def wc_loglinear(x):
        """
        Evaluates log-linear solution at state z,h_z,h_c,h_λ
        """

        h_λ, h_c, h_z, z = x
        s_z = h_z*2*(φ_z)**2 + (φ_z)**2;
        s_c = h_c*2*(φ_c)**2 + (φ_c)**2;

        return A0 + Ah_λ * h_λ + Ah_c * s_c + Ah_z * s_z + Az * z

    # Calling the factory returns the jitted function
    return wc_loglinear
