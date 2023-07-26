"""

Gomez-Cram--Yaron model (2020).

There are six states for the recursive utility / wealth-consumption ratio
problem, given by

    x = (h_λ, h_c, h_z, h_zπ, z, z_π)

They evolve according to

    z' = ρ * z + ρ_π * z_π + σ_z * η0

    z_π' = ρ_ππ * z_π + σ_zπ * η1

    h_z' = ρ_z * h_z + s_z * η2

    h_c' = ρ_c * h_c + s_c * η3

    h_zπ' = ρ_zπ * h_zπ + s_zπ * η4

    h_λ' = ρ_λ * h_λ + s_λ * η5

with

    σ_z = φ_z * exp(h_z)

    σ_zπ = φ_zπ * exp(h_zπ)

Consumption growth is

    g_c' = μ_c + z' + φ_c * ξ



"""
import numpy as np
from scipy.optimize import brentq

from numba import njit


class GCY:

    def __init__(self,
                    β=0.9987,            # δ
                    ψ=1.5,
                    γ=13.01,
                    ρ_λ=0.981,
                    s_λ=0.12 * 0.0015,   # φ_λ * σ
                    μ_c=0.0016,
                    φ_c=0.0015,          # φ_c * σ
                    ρ=0.983,             # ρ_cc
                    ρ_π=-0.0075,         # ρ_cπ
                    φ_z=0.13 * 0.0015,   # φ_xc * σ
                    ρ_c=0.992,           # ρ_hc
                    s_c=0.104,           # σ_hc
                    ρ_z=0.980,           # ρ_hxc
                    s_z=0.09,            # σ_hxc
                    ρ_ππ=0.985,
                    φ_zπ=0.08 * 0.0015,  # φ_xπ * σ
                    ρ_zπ=0.970,          # ρ_hxπ
                    s_zπ=0.271):         # σ_hxπ

        self.β, self.ψ, self.γ = β, ψ, γ
        self.ρ_λ, self.s_λ, self.μ_c, self.φ_c, self.ρ = ρ_λ, s_λ, μ_c, φ_c, ρ
        self.ρ_π, self.φ_z, self.ρ_c = ρ_π, φ_z, ρ_c
        self.s_c, self.ρ_z, self.s_z = s_c, ρ_z, s_z
        self.ρ_ππ, self.φ_zπ, self.ρ_zπ, self.s_zπ = ρ_ππ, φ_zπ, ρ_zπ, s_zπ

        # Pack params into an array
        self.params = (β, ψ, γ,
                         ρ_λ, s_λ, μ_c, φ_c, ρ,
                         ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z,
                         ρ_ππ, φ_zπ, ρ_zπ, s_zπ)


## ==== Log linear approximation of the GCY W/C ratio === ##

def wc_loglinear_factory(gcy):
    """
    A factory function that computes the constant terms for the WC ratio log
    linear approximation and then creates a jitted function that evaluates the
    log linear approximation.

    """

    # Unpack parameters
    (β, ψ, γ,
        ρ_λ, s_λ, μ_c, φ_c, ρ,
        ρ_π, φ_z, ρ_c, s_c, ρ_z, s_z,
        ρ_ππ, φ_zπ, ρ_zπ, s_zπ) = gcy.params
    θ = (1 - γ) / (1 - 1/ψ)

    s_wc = 2*(φ_c)**2*s_c
    s_wx = 2*(φ_z)**2*s_z
    s_wxπ = 2*(φ_zπ)**2*s_zπ

    def fk1(x):
        return np.exp(x)/(1+np.exp(x))

    def fk0(x):
        return np.log(1+np.exp(x))-fk1(x)*x

    def fA1(x):
        return (1-1/ψ)/(1-fk1(x)*ρ)

    def fAλ(x):
        return ρ_λ/(1-fk1(x)*ρ_λ)

    def fAπ(x):
        return fk1(x)*(1-1/ψ)*ρ_π/((1-fk1(x)*ρ)*(1-fk1(x)*ρ_ππ))

    def fAz(x):
        return (θ/2)*(fk1(x)*fA1(x))**2/(1-fk1(x)*ρ_z)

    def fAzπ(x):
        return (θ/2)*(fk1(x)*fAπ(x))**2/(1-fk1(x)*ρ_zπ)

    def fAc(x):
        return (θ/2)*(1-1/ψ)**2/(1-fk1(x)*ρ_c)

    def fA0(x):
        return (np.log(β)+fk0(x)+μ_c * (1-1/ψ) \
                + fk1(x) * fAz(x) * (φ_z)**2 * (1-ρ_z)\
                + fk1(x) * fAc(x) * (φ_c)**2 * (1-ρ_c)\
                + fk1(x) * fAzπ(x) * (φ_zπ)**2 * (1-ρ_zπ)\
                + (θ/2) * ((fk1(x) * fAλ(x)+1)**2 * s_λ**2 \
                + (fk1(x) * fAz(x) * s_wx)**2+(fk1(x) * fAc(x) * s_wc)**2\
                + (fk1(x) * fAzπ(x) * s_wxπ)**2)) \
                / (1-fk1(x))

    def fq_bar(x):
        return x - fA0(x) - fAc(x)*(φ_c)**2 - fAz(x)*(φ_z)**2 - fAzπ(x)*(φ_zπ)**2

    qbar = brentq(fq_bar, -20, 20)
    Az = fA1(qbar)
    Az_π = fAπ(qbar)
    Ah_λ = fAλ(qbar)
    Ah_z = fAz(qbar)
    Ah_c = fAc(qbar)
    Ah_zπ = fAzπ(qbar)
    A0 = fA0(qbar)

    # Now build a jitted function
    @njit
    def wc_loglinear(x):
        """
        Evaluates log-linear solution at state h_λ, h_c, h_z, h_zπ, z, z_π
        """
        h_λ, h_c, h_z, h_zπ, z, z_π = x
        s_z_1 = h_z*2*(φ_z)**2 + (φ_z)**2
        s_c_1 = h_c*2*(φ_c)**2 + (φ_c)**2
        s_zπ_1 = h_zπ*2*(φ_zπ)**2 + (φ_zπ)**2
        return (A0 + Ah_λ * h_λ + Ah_c * s_c_1 +
                Ah_z * s_z_1 + Az * z + Ah_zπ * s_zπ_1 + Az_π * z_π)

    # Calling the factory returns the jitted function
    return wc_loglinear
