import numpy as np
import numba

@numba.njit()
def M2nu(M, gamma = 1.4): # Prandtl-Meyer Function
    # https://en.wikipedia.org/wiki/Prandtl-Meyer_function
    A = np.sqrt((gamma + 1)/(gamma - 1))
    B = np.sqrt(M**2 - 1)
    return A * np.arctan(B/A) - np.arctan(B)

@numba.njit()
def M2nu_prime(M, gamma = 1.4): # First Derivative of Prandtl-Meyer Function
    # https://en.wikipedia.org/wiki/Prandtl-Meyer_function
    A2 = (gamma + 1)/(gamma - 1)
    B = np.sqrt(M**2 - 1)
    return (A2 - 1) * B / (M * (M**2 + A2 - 1))

@numba.njit()
def nu2M(nu, M_guess = 1.1, gamma = 1.4): # Inverse Prandtl-Meyer Function, Newton's Method
    # https://en.wikipedia.org/wiki/Prandtl-Meyer_function
    # https://en.wikipedia.org/wiki/Newton%27s_method

    nu_guess = M2nu(M_guess, gamma)
    guess_count = 0
    while np.abs(nu_guess - nu) > 1e-16 and guess_count < 50:
        M_guess -= (nu_guess - nu)/M2nu_prime(M_guess, gamma)
        nu_guess = M2nu(M_guess, gamma)
        guess_count += 1
    
    return M_guess

@numba.njit()
def M2mu(M): # Mach Angle Function
    # https://en.wikipedia.org/wiki/Mach_wave
    return np.arcsin(1/M)

@numba.njit()
def p_ratio(M, gamma = 1.4): # ratio of p(0)/p(M)
    return (1 + (gamma-1)/2 * M**2)**(gamma/(gamma-1))

@numba.njit()
def T_ratio(M, gamma = 1.4): # ratio of T(0)/T(M)
    return 1 + (gamma-1)/2 * M**2

@numba.njit()
def A_ratio(M, gamma = 1.4): # ratio of A(M)/A(1)
    return 1/M * (gamma/2 - 1/2)**(-(gamma + 1)/(2*gamma - 2)) * (1 + (gamma-1)/2 * M**2)**(-gamma/(gamma-1))

@numba.njit()
def M_jet(p0, p_jet, gamma = 1.4):
    return np.sqrt(2/(gamma-1) * ((p0/p_jet)**((gamma-1)/gamma) - 1))