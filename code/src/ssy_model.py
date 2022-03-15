"""

Schorfheide--Song--Yaron code.  

There are four states for the recursive utility / wealth-consumption ratio
problem, namely

    x = (h_λ, h_c, h_z, z)

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
from quantecon import MarkovChain, rouwenhorst
from numba import njit, prange, float32, cuda
from numpy.random import rand, randn


# == Some convenience functions == #

@njit
def draw_from_cdf(F, U):
    " Draws from F when U is uniform on (0, 1) "
    return np.searchsorted(F, U)


@njit
def split_index(i, M):
    """
    A utility function for the multi-index.
    """
    div = i // M
    rem = i % M
    return (div, rem)

@njit
def single_to_multi(m, K, I, J):
    l, rem = split_index(m, K * I * J)
    k, rem = split_index(rem, I * J)
    i, j = split_index(rem, J)
    return (l, k, i, j)


@njit
def multi_to_single(l, k, i , j, K, I, J):
    return l * (K * I * J) + k * (I * J) + i * J + j


def compute_spec_rad(Q):
    """
    Function to compute spectral radius of a matrix.

    """
    return np.max(np.abs(np.linalg.eigvals(Q)))


class SSY:
    """
    Stores the SSY model parameters, along with data for two discretized
    versions of the SSY model:

        - multi-index model with indices l, k, i, j
        - single index model with index n

    """

    def __init__(self,
                 β=0.999,   # = δ in SSY
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
                 φ_c=1.00*0.0035):                       # *σ_bar
                 L=4, K=4, I=4, J=4, 
                 build_single_index=True):          
        
        # Create and store an instance of SSY if one is not assigned
        self.β, self.γ, self.ψ = β, γ, ψ
        self.μ_c,self.ϕ_z, self.ϕ_c = μ_c, ϕ_z, ϕ_c
        self.ρ, self.ρ_z, self.ρ_c, self.ρ_λ = ρ, ρ_z, ρ_c, ρ_λ
        self.s_z, self.s_c, self.s_λ = s_z, s_c, s_λ
        self.θ = (1 - γ) / (1 - 1/ψ)
        self.ssy = SSY() if ssy is None else ssy 


        # Set up multi-index states and transitions
        (self.h_λ_states, self.h_λ_P,              
         self.h_c_states, self.h_c_P,
         self.h_z_states, self.h_z_P,
         self.z_states,   self.z_Q) = self.discretize_multi_index(L, K, I, J)                

        # For convenience, store the sigma states as well
        self.σ_c_states = self.ssy.ϕ_c * np.exp(self.h_c_states)
        self.σ_z_states = self.ssy.ϕ_z * np.exp(self.h_z_states)

        # Single index states and transitions
        if build_single_index:
            self.N = L * K * I * J
            self.x_states, self.P_x = self.discretize_single_index()
            self.H = self.compute_H()

    def unpack(self):
        return (self.β, self.γ, self.ψ,
                self.μ_c, self.ρ, self.ϕ_z, self.ϕ_c,
                self.ρ_z, self.ρ_c, self.ρ_λ,
                self.s_z, self.s_c, self.s_λ) 

    def discretize_multi_index(self, L, K, I, J):
        """
        Discretize the SSY model, using a multi-index, as discussed above.        

        The discretization uses iterations of the Rouwenhorst method.  The
        indices are

            h_λ[l] for l in range(L)
            h_c[k] for k in range(K)
            h_z[i] for i in range(I)
            z[i, j] is the j-th z state when σ_z = σ_z[i] for j in range(J)

            z_Q[i, j, jp] is trans prob from j to jp when σ_z index = i

        """
        (β, γ, ψ, μ_c, ρ, ϕ_z, ϕ_c, ρ_z, ρ_c, ρ_λ, s_z, s_c, s_λ) = self.unpack()

        h_λ_mc = rouwenhorst(L, 0, s_λ, ρ_λ)
        h_c_mc = rouwenhorst(K, 0, s_c, ρ_c)
        h_z_mc = rouwenhorst(I, 0, s_z, ρ_z)

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

        return (h_λ_states,  h_λ_mc.P,     
                h_c_states,  h_c_mc.P,  
                h_z_states,  h_z_mc.P,  
                z_states,    z_Q) 


    def discretize_single_index(self):
        """
        Build the single index state process.  The discretized version is
        converted into single index form to facilitate matrix operations.  

        The rule for the index is

            n = n(l, k, i, j) = l * (K * I * J) + k * (I * J) + i * J + j

        where n is in range(N) with N = L * K * I * J.

        We store a Markov chain with states 

            x_states[n] := (h_λ[l], σ_c[k], σ_z[i], z[i,j])
            
        A stochastic matrix P_x gives transition probabilitites, so

            P_x[n, np] = probability of transition x[n] -> x[np]


        """
        # Unpack 
        L, K, I, J = self.L, self.K, self.I, self.J
        N = L * K * I * J 

        # Allocate arrays
        P_x = np.zeros((N, N))
        x_states = np.zeros((4, N))

        # Populate arrays
        state_arrays = (self.h_λ_states, self.h_c_states, 
                        self.h_z_states, self.z_states)
        prob_arrays = self.h_λ_P, self.h_c_P, self.h_z_P, self.z_Q 
        _build_single_index_arrays(
                L, K, I, J, state_arrays, prob_arrays, x_states, P_x
        )
        return x_states, P_x


    def compute_H(ssy):
        """
        Compute the matrix H in the SSY model using the single-index
        framework.

        """

        H = _compute_H(ssy.unpack(),
                          ssy.L, ssy.K, ssy.I, ssy.J,
                          ssy.h_λ_states, 
                          ssy.σ_c_states, 
                          ssy.σ_z_states,
                          ssy.z_states,
                          ssy.P_x)

        return H



@njit
def _build_single_index_arrays(L, K, I, J, 
                               state_arrays, 
                               prob_arrays, 
                               x_states, 
                               P_x):

    h_λ_states, h_c_states, h_z_states, z_states = state_arrays
    h_λ_P, h_c_P, h_z_P, z_Q = prob_arrays

    N = L * K * I * J

    for m in range(N):
        l, k, i, j = single_to_multi(m, K, I, J)
        x_states[:, m] = (h_λ_states[l], 
                          h_c_states[k], h_z_states[i], z_states[i, j])
        for mp in range(N):
            lp, kp, ip, jp = single_to_multi(mp, K, I, J)
            P_x[m, mp] = h_λ_P[l, lp] * h_c_P[k, kp] * h_z_P[i, ip] * z_Q[i, j, jp]



@njit
def _compute_H(ssy_params,
               L, K, I, J, 
               h_λ_states,
               σ_c_states, 
               σ_z_states,
               z_states,
               P_x):
    # Unpack
    (β, γ, ψ, 
        μ_c, ρ, ϕ_z, ϕ_c, 
        ρ_z, ρ_c, ρ_λ, 
        s_z, s_c, s_λ) = ssy_params
    N = L * K * I * J
    θ = (1 - γ) / (1 - 1/ψ)
    H = np.empty((N, N))

    for m in range(N):
        l, k, i, j = single_to_multi(m, K, I, J)
        σ_c, σ_z, z = σ_c_states[k], σ_z_states[i], z_states[i, j]
        for mp in range(N):
            lp, kp, ip, jp = single_to_multi(m, K, I, J)
            h_λp = h_λ_states[lp] 
            a = np.exp(θ * h_λp + (1 - γ) * (μ_c + z) + 0.5 * (1 - γ)**2 * σ_c**2)
            H[m, mp] =  a * P_x[m, mp]
            
    return H



## == Other utilities == ##


def lininterp_funcvals(ssy, function_vals):
    """
    Builds and returns a jitted callable that implements an approximation of
    the function determined by function_vals via linear interpolation over the
    grid.

        expected grid is (h_λ_states, h_c_states, h_z_states, z_states)


    """

    h_λ_states = ssy.h_λ_states
    h_c_states = ssy.h_c_states
    h_z_states = ssy.h_z_states
    z_states = ssy.z_states

    @njit
    def interpolated_function(x):
        h_λ, h_c, h_z, z = x
        i = np.searchsorted(h_z_states, h_z)
        # Don't go out of bounds
        if i == len(h_z_states):
            i = i - 1

        return lininterp_4d(h_λ_states, 
                            h_c_states, 
                            h_z_states, 
                            z_states[i, :], 
                            function_vals, 
                            x)

    return interpolated_function






def wc_loglinear_factory(ssy):
    """
    A method factory .  It computes the constant terms for the WC ratio
    log linear approximation and then creates a jitted function that
    evaluates the log linear approximation.

    This factory is called by the __init__ method so that the jitted
    function is accessible through the class.

    """

    # Unpack parameters
    (β, γ, ψ, 
        μ_c, ρ, ϕ_z, ϕ_c, 
        ρ_z, ρ_c, ρ_λ, 
        s_z, s_c, s_λ) = ssy.unpack()
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

    # Record the constants within the instance
    ssy.Az, ssy.Ah_λ, ssy.Ah_z, ssy.Ah_c, ssy.A0 = \
        Az, Ah_λ, Ah_z, Ah_c, A0 

    # Now build the jitted function
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



