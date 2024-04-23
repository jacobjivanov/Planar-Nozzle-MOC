import numpy as np
import math
import numba
import time
import matplotlib.pyplot as plt

from scipy.optimize import root_scalar

# runtime parameters
y_inl = 0.25 # throat cross section radius
r_exp = 0.1 # expansion section radius
M_inl = 1.1 # throat Mach Number
M_exi = 2.5 # exit Mach Number
gamma = 1.4

@numba.njit()
def AERO_M2nu(M): # Prandtl-Meyer Function
    # https://en.wikipedia.org/wiki/Prandtl-Meyer_function
    A = np.sqrt((gamma + 1)/(gamma - 1))
    B = np.sqrt(M**2 - 1)
    return A * np.arctan(B/A) - np.arctan(B)

@numba.njit()
def AERO_M2nu_prime(M): # First-Derivative of Prandtl-Meyer Function
    # https://en.wikipedia.org/wiki/Prandtl-Meyer_function
    A2 = (gamma + 1)/(gamma - 1)
    B = np.sqrt(M**2 - 1)
    return (A2 - 1) * B / (M * (M**2 + A2 - 1))

@numba.njit()
def AERO_nu2M(nu): # Inverse Prandtl-Meyer Function
    # https://en.wikipedia.org/wiki/Prandtl-Meyer_function
    # https://en.wikipedia.org/wiki/Newton%27s_method

    M_below = M_inl
    M_above = M_exi
    
    nu_below = AERO_M2nu(M_below)
    nu_above = AERO_M2nu(M_above)

    max_iter = 1000
    iter_count = 0
    tolerance = 1e-16
    
    while iter_count < max_iter and np.abs(nu_above - nu_below) > tolerance:
        M_guess = (nu - nu_below)/(nu_above - nu_below) * (M_above - M_below) + M_below
        nu_guess = AERO_M2nu(M_guess)
        
        if np.abs(nu_guess - nu_below) < np.abs(nu_guess - nu_above):
            M_above = M_guess
            nu_above = nu_guess
        else:
            M_below = M_guess
            nu_below = nu_guess

        iter_count += 1
    
    return (M_below + M_above)/2

@numba.njit()
def AERO_M2mu(M): # Mach Angle Function
    # https://en.wikipedia.org/wiki/Mach_wave
    return np.arcsin(1/M)

@numba.njit()
def MESH_internal(p1data, p2data, p3data):
    # p1 is below, p2 is above
    x1, x2 = p1data[0], p2data[0]
    y1, y2 = p1data[1], p2data[1]
    M1, M2 = p1data[2], p2data[2]
    theta1, theta2 = p1data[3], p2data[3]

    mu1, mu2 = AERO_M2mu(M1), AERO_M2mu(M2)
    
    nu1, nu2 = AERO_M2nu(M1), AERO_M2nu(M2)
    nu3 = (nu1 + nu2)/2 - (theta1 - theta2)/2
    theta3 = (theta1 + theta2)/2 - (nu1 - nu2)/2
    
    M3 = AERO_nu2M(nu3)
    mu3 = AERO_M2mu(M3)

    phi1 = (theta1 + theta3 + mu1 + mu3)/2
    phi2 = (theta2 + theta3 - mu2 - mu3)/2

    x3 = x1 * math.tan(phi1) - x2 * math.tan(phi2) + y2 - y1
    x3 /= math.tan(phi1) - math.tan(phi2)

    y3 = y1 + math.tan(phi1) * (x3 - x1)
    
    p3data[0] = x3
    p3data[1] = y3
    p3data[2] = M3
    p3data[3] = theta3
    
    return p3data

@numba.njit()
def MESH_centerline(p1data, pcdata):
    # p1 is above centerline
    x1 = p1data[0]
    y1 = p1data[1]
    M1 = p1data[2]
    theta1 = p1data[3]

    mu1 = AERO_M2mu(M1)

    nu1 = AERO_M2nu(M1)
    nuc = theta1 + nu1
    thetac = 0

    Mc = AERO_nu2M(nuc)
    muc = AERO_M2mu(Mc)

    phi1 = theta1/2 - (mu1 + muc)/2
    
    xc = x1 - y1/math.tan(phi1)
    yc = 0

    pcdata[0] = xc
    pcdata[1] = yc
    pcdata[2] = Mc
    pcdata[3] = thetac
    
    return pcdata

@numba.njit()
def MESH_wall(p1data, p2data, pwdata):
    # p1 is an internal point, p2 is the last wall point
    x1, x2 = p1data[0], p2data[0]
    y1, y2 = p1data[1], p2data[1]
    M1, M2 = p1data[2], p2data[2]
    theta1, theta2 = p1data[3], p2data[3]

    mu1 = AERO_M2mu(M1)

    thetaw = theta1
    Mw = M1

    phi1 = theta1 + mu1
    phi2 = (theta1 + theta2)/2

    xw = x1 * math.tan(phi1) - x2 * math.tan(phi2) + y2 - y1
    xw /= math.tan(phi1) - math.tan(phi2)

    yw = y1 + math.tan(phi1) * (xw - x1)

    pwdata[0] = xw
    pwdata[1] = yw
    pwdata[2] = Mw
    pwdata[3] = thetaw
    
    return pwdata

def expansion_section(y_inl, r_exp, M_inl, M_exi):
    nu_thr = AERO_M2nu(M_inl)
    nu_exi = AERO_M2nu(M_exi)
    delta_max = nu_exi/2 - nu_thr/2 # Eq. 11.33 "Modern Compressible Flow"

    def wall_position(x): return y_inl + r_exp - np.sqrt(r_exp**2 - x**2)
    def wall_angle(x): return np.arctan(x / np.sqrt(r_exp**2 - x**2))
    
    wall_x = np.linspace(0, r_exp * np.sin(delta_max))
    wall_y = [wall_position(x) for x in wall_x]
    
    nu_thr = AERO_M2nu(M_inl)
    expansion_series = np.empty(shape = (N, 4))
    for n in range(0, N):
        expansion_series[n, 3] = n*delta_max/(N-1)
        nu = expansion_series[n, 3] + nu_thr
        expansion_series[n, 2] = AERO_nu2M(nu)
        expansion_series[n, 0] = r_exp * np.sin(expansion_series[n, 3])
        expansion_series[n, 1] = y_inl + r_exp - r_exp * np.cos(expansion_series[n, 3])

    return wall_x, wall_y, expansion_series

def internal_section(expansion_series):
    internal_series = np.empty(shape = ((N*(N+1)//2), 4))
    edge_series = np.empty(shape = (N, 4))

    last_series = expansion_series
    next_series = np.empty(shape = (N, 4))
    next_series[0] = MESH_centerline(last_series[0], next_series[0])
    for n in range(1, N):
        next_series[n] = MESH_internal(next_series[n-1], last_series[n], next_series[n])
    
    internal_series = np.vstack((next_series))
    edge_series[0] = internal_series[-1]

    def next_series_func(last_series):
        next_series = np.empty(shape = (last_series.shape[0] - 1, 4))
        next_series[0] = MESH_centerline(last_series[1], next_series[0])

        for n in range(1, last_series.shape[0] - 1):
            next_series[n] = MESH_internal(next_series[n-1], last_series[n+1], next_series[n])

        return next_series

    for i in range(1, N):
        next_series = next_series_func(next_series)
        internal_series = np.vstack((internal_series, next_series))
        edge_series[i] = internal_series[-1]
    
    return internal_series, edge_series

def straighten_section(expansion_series, edge_series):
    straighten_series = np.empty(shape = (N, 4))

    straighten_series[0] = MESH_wall(edge_series[0], expansion_series[-1], straighten_series[0])

    for n in range(1, N):
        straighten_series[n] = MESH_wall(edge_series[n], straighten_series[n-1], straighten_series[n])
    
    return straighten_series

if __name__ == '__main__':
    N = 1000
    
    wall_x, wall_y, expansion_series = expansion_section(y_inl, r_exp, M_inl, M_exi)
    
    internal_series, edge_series = internal_section(expansion_series)
    straighten_series = straighten_section(expansion_series, edge_series)
    full_series = np.vstack((
        np.array([0, 0, M_inl, 0]),
        expansion_series,
        internal_series,
        straighten_series,
        np.array([straighten_series[-1, 0], 0, straighten_series[-1, 2], 0])
    ))

    centerline_series = np.empty(shape = (N + 2, 4))
    c = 0
    for p in range(0, int(2*N + N*(N+1)/2 + 2)):
        if full_series[p, 1] == 0:
            centerline_series[c, :] = full_series[p, :]
            c += 1
    
    wall_series = np.vstack((
        expansion_series,
        straighten_series
    ))

    # plt.scatter(centerline_series[:, 0], centerline_series[:, 2])
    plt.plot(wall_series[:, 0], wall_series[:, 1]/y_inl, ':.', label = r"$\alpha_\mathrm{num}$", color = 'grey')

    def alpha(M_cen):
        alpha_ana = M_inl / M_cen * ((1 + (gamma-1)/2 * M_cen ** 2)/(1 + (gamma-1)/2 * M_inl ** 2))**((gamma+1)/(2*gamma - 2))

        return alpha_ana

    plt.title("Numerical Area Ratio vs Area Ratio of Centerline Mach Number")
    plt.plot(centerline_series[:, 0], alpha(centerline_series[:, 2]), ':.', label = r"$\alpha(M_\mathrm{cen})$", color = 'red')
    plt.plot(wall_series[:, 0], alpha(wall_series[:, 2]), ':.', label = r"$\alpha(M_\mathrm{wall})$", color = 'blue')
    plt.legend()
    plt.savefig("Centerline Mach Number Area Ratio.png", dpi = 200)
    plt.show()