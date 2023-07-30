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

import numpy as np
from scipy.optimize import brentq

from numba import njit


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

    s_wc = 2*(φ_c)**2*s_c
    s_wx = 2*(φ_z)**2*s_z

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
        Evaluates log-linear solution at state h_λ, h_c, h_z, z
        """

        h_λ, h_c, h_z, z = x
        s_z = h_z*2*(φ_z)**2 + (φ_z)**2
        s_c = h_c*2*(φ_c)**2 + (φ_c)**2

        return A0 + Ah_λ * h_λ + Ah_c * s_c + Ah_z * s_z + Az * z

    # Calling the factory returns the jitted function
    return wc_loglinear
