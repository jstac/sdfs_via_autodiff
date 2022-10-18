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
from quantecon import rouwenhorst
from numpy.random import rand, randn
import jax
import jax.numpy as jnp
from jax.config import config
import jaxopt
from textwrap import dedent

# Optionally tell JAX to use 64 bit floats
config.update("jax_enable_x64", True)



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
                 φ_c=1.00*0.0035,                        # *σ_bar
                 L=4, K=4, I=4, J=4):

        # Unpack
        self.β, self.γ, self.ψ = β, γ, ψ
        self.μ_c,self.ϕ_z, self.ϕ_c = μ_c, ϕ_z, ϕ_c
        self.ρ, self.ρ_z, self.ρ_c, self.ρ_λ = ρ, ρ_z, ρ_c, ρ_λ
        self.s_z, self.s_c, self.s_λ = s_z, s_c, s_λ
        self.θ = (1 - γ) / (1 - 1/ψ)

        # Set up states and transitions
        self.L, self.K, self.I, self.J = L, K, I, J
        (self.h_λ_states, self.h_λ_P,
         self.h_c_states, self.h_c_P,
         self.h_z_states, self.h_z_P,
         self.z_states,   self.z_Q) = self.discretize_multi_index(L, K, I, J)

        # For convenience, store the sigma states as well
        self.σ_c_states = self.ϕ_c * np.exp(self.h_c_states)
        self.σ_z_states = self.ϕ_z * np.exp(self.h_z_states)

    def unpack(self):
        return (self.β, self.γ, self.ψ,
                self.μ_c, self.ρ, self.ϕ_z, self.ϕ_c,
                self.ρ_z, self.ρ_c, self.ρ_λ,
                self.s_z, self.s_c, self.s_λ)

    def discretize_multi_index(self, L, K, I, J):
        """
        Discretize the SSY model using a multi-index.

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





# == Fixed point solvers == #

default_tolerance = 1e-7
default_max_iter = int(1e6)

def fwd_solver(f,
               x_init,
               tol=default_tolerance,
               max_iter=default_max_iter,
               verbose=True,
               print_skip=1000):
    "Uses successive approximation on f."

    if verbose:
        print("Beginning iteration\n\n")

    current_iter = 0
    x = x_init
    error = tol + 1
    while error > tol and current_iter < max_iter:
        x_new = f(x)
        error = jnp.max(jnp.abs(x_new - x))
        if verbose and current_iter % print_skip == 0:
            print("iter = {}, error = {}".format(current_iter, error))
        current_iter += 1
        x = x_new

    if current_iter == max_iter:
        print(f"Warning: Hit maximum iteration number {max_iter}")
    else:
        if verbose:
            print(f"Iteration converged after {current_iter} iterations")

    return x, current_iter


def newton_solver(f, 
                  x_init, 
                  tol=default_tolerance, 
                  max_iter=default_max_iter,
                  bicgstab_atol=1e-4,
                  verbose=True,
                  print_skip=1):
    """
    Apply Newton's algorithm to find a fixed point of f. The routine defines 
    g via g(x) = f(x) - x and then searches for a root of g via Newton's
    method, which iterates on 

        x_{n+1} = x_n - J(x_n)^{-1} g(x_n)

    until convergence, where J(x) is the Jacobian of g at x. The implementation 
    below defines 

        q(x) := x - J(x)^{-1} g(x)

    and passes this function to fwd_solver.

    To compute J(x)^{-1} g(x) we can in principle use
    `jnp.linalg.solve(jax.jacobian(g)(x), g(x))`. However, this operation is
    very memory intensive when x is high-dimensional. It also requires that g
    is a regular 2D array (matrix), which necessitates conversion to a single
    index. 

    To avoid instantiating the large matrix J(x), we use jax.jvp to define the
    linear map v -> J(x) v. This map is computed on demand for any given v,
    which avoids instantiating J(x).  We then pass this to a solver that can
    invert arbitrary linear maps.
    """
    g = lambda x: f(x) - x
    def q(x):
        # First we define the map v -> J(x) v from x and g
        jac_x_prod = lambda v: jax.jvp(g, (x,), (v,))[1]
        # Next we compute J(x)^{-1} g(x).  Currently we use 
        # sparse.linalg.bicgstab. Another option is sparse.linalg.bc
        # but this operation seems to be less stable.
        b = jax.scipy.sparse.linalg.bicgstab(
                jac_x_prod, g(x), 
                atol=bicgstab_atol)[0]
        return x - b
    return fwd_solver(q, x_init, tol, max_iter, verbose, print_skip)



def fixed_point_via_gradient_decent(f, x_init):

    def loss(x):
        v = f(x) - x
        return jnp.dot(v.flatten(), v.flatten())

    gd = jaxopt.GradientDescent(fun=loss, 
                                maxiter=1000, 
                                tol=0.0001,
                                stepsize=0.0)
    res = gd.run(init_params=x_init)
    solution, state = res

    return solution


# == List solvers for simple access == #

# A dictionary of available solvers.
solvers = dict((("newton", newton_solver),
                ("gd", fixed_point_via_gradient_decent),
                ("successive_approx", fwd_solver)))




"""
Multi-index routines for computing the wealth consumption ratio of the SSY
model.

"""

# == Operator == #

def _T(w, shapes, constants, arrays):
    """
    Implement the operator T via JAX operations.  In the call signature, we
    differentiate between static parameters and arrays to help the JIT
    compiler.
    """

    # Unpack
    L, K, I, J = shapes
    β, θ, γ, μ_c = constants
    (h_λ_states, h_λ_P,              
     h_c_states, h_c_P,
     h_z_states, h_z_P,
     z_states,   z_Q,
     σ_c_states, σ_z_states) = arrays

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
    Hwθ = np.sum(A, axis=(4, 5, 6, 7))

    # Define and return Tw
    Tw = 1 + β * Hwθ**(1/θ)
    return Tw

# JIT compile the intermediate function _T
_T = jax.jit(_T, static_argnums=(1, 2))

# Define a version of T with a simplified interface."
def T(w, params):
    (L, K, I, J,
    β, θ, γ, μ_c,
    h_λ_states, h_λ_P,              
      h_c_states, h_c_P,
      h_z_states, h_z_P,
      z_states,   z_Q,
      σ_c_states, σ_z_states) = params

    shapes = L, K, I, J
    constants = β, θ, γ, μ_c,
    arrays = (h_λ_states, h_λ_P,              
              h_c_states, h_c_P,
              h_z_states, h_z_P,
              z_states,   z_Q,
              σ_c_states, σ_z_states) 
    return _T(w, shapes, constants, arrays)


def wc_ratio(model, 
             algorithm="newton",
             init_val=800, 
             verbose=True):
    """
    Iterate solve for the fixed point of the Koopmans operator 
    associated with the model and return the wealth consumption ratio.

    - model is an instance of SSY or GCY

    """
    # Unpack 
    m = model
    params = (m.L, m.K, m.I, m.J,
              m.β, m.θ, m.γ, m.μ_c,
              m.h_λ_states, m.h_λ_P,              
              m.h_c_states, m.h_c_P,
              m.h_z_states, m.h_z_P,
              m.z_states,   m.z_Q,
              m.σ_c_states, m.σ_z_states) 

    # Marginalize T given the parameters
    T_operator = lambda x: T(x, params)
    # Set up the initial condition for the fixed point search
    w_init = jnp.ones((m.L, m.K, m.I, m.J)) * init_val

    try:
        solver = solvers[algorithm]
    except KeyError:
        msg = f"""\
                  Algorithm {algorithm} not found.  
                  Falling back to successive approximation.
               """
        print(dedent(msg))
        solver = fwd_solver

    # Call the solver
    w_star, num_iter = solver(T_operator, w_init)

    return w_star
