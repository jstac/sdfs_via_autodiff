"""

Gomez-Cram--Yaron model.  

There are six states for the recursive utility / wealth-consumption ratio
problem, namely

    z' = ρ * z + ρ_π * z_π + σ_z * η0

    z_π' = ρ_ππ * z_π + σ_zπ * η1

    h_z' = ρ_z * h_z + s_z * η2

    h_c' = ρ_c * h_c + s_c * η3

    h_zπ' = ρ_zπ * h_zπ + s_zπ * η4

    h_λ' = ρ_λ * h_λ + s_λ * η5

"""


class GCY:

    def __init__(self,
                    β=0.9987,  # δ
                    ψ=1.5,
                    γ=13.01,
                    ρ_λ=0.981,
                    s_λ=0.12 * 0.0015,  # φ_λ * σ
                    μ_c=0.0016,
                    φ_c=0.0015,  # φ_c * σ
                    ρ=0.983,  # ρ_cc
                    ρ_π=-0.0075,  # ρ_cπ
                    φ_z=0.13 * 0.0015,  # φ_xc * σ
                    ρ_c=0.992,  # ρ_hc
                    s_c=0.104,  # σ_hc
                    ρ_z=0.980,  # ρ_hxc
                    s_z=0.09,  # σ_hxc
                    ρ_ππ=0.985,
                    φ_zπ=0.08 * 0.0015,  # φ_xπ * σ
                    ρ_zπ=0.970,  # ρ_hxπ
                    s_zπ=0.271  # σ_hxπ

