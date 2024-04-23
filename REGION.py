import numpy as np
import MESH
import AERO

def expansion(y_inl, r_exp, M_inl, M_exi, N):
    nu_thr = AERO.M2nu(M_inl)
    nu_exi = AERO.M2nu(M_exi)
    delta_max = nu_exi/2 - nu_thr/2 # Eq. 11.33 "Modern Compressible Flow"

    def wall_position(x): return y_inl + r_exp - np.sqrt(r_exp**2 - x**2)
    def wall_angle(x): return np.arctan(x / np.sqrt(r_exp**2 - x**2))
    
    wall_x = np.linspace(0, r_exp * np.sin(delta_max))
    wall_y = [wall_position(x) for x in wall_x]

    expansion_series = np.empty(shape = (N, 4))
    for n in range(0, N):
        expansion_series[n, 3] = n*delta_max/(N-1)
        nu = expansion_series[n, 3] + nu_thr
        expansion_series[n, 2] = AERO.nu2M(nu)
        expansion_series[n, 0] = r_exp * np.sin(expansion_series[n, 3])
        expansion_series[n, 1] = y_inl + r_exp - r_exp * np.cos(expansion_series[n, 3])

    return wall_x, wall_y, expansion_series

def internal(expansion_series, N, F):
    internal_series = np.empty(shape = ((N*(N+1)//2), 4))
    edge_series = np.empty(shape = (N, 4))

    last_series = expansion_series
    next_series = np.empty(shape = (N, 4))
    expansion_fill = np.empty(shape = (N*F, 4))
    next_series[0] = MESH.centerline(last_series[0], next_series[0])

    f = 0
    expansion_fill[f:f+F, :] = MESH.fill(expansion_series[0], next_series[0], F)
    f += F
    for n in range(1, N):
        next_series[n] = MESH.internal(next_series[n-1], last_series[n], next_series[n])
        expansion_fill[f:f+F, :] = MESH.fill(last_series[n], next_series[n], F)
        f += F
    
    internal_series = np.vstack((next_series))
    edge_series[0] = internal_series[-1]

    def next_series_func(last_series):
        next_series = np.empty(shape = (last_series.shape[0] - 1, 4))
        next_series[0] = MESH.centerline(last_series[1], next_series[0])

        for n in range(1, last_series.shape[0] - 1):
            next_series[n] = MESH.internal(next_series[n-1], last_series[n+1], next_series[n])

        return next_series

    for i in range(1, N):
        next_series = next_series_func(next_series)
        internal_series = np.vstack((internal_series, next_series))
        edge_series[i] = internal_series[-1]
    
    return internal_series, edge_series, expansion_fill

def straighten(expansion_series, edge_series, N, F):
    straighten_series = np.empty(shape = (N, 4))
    straighten_fill = np.empty(shape = (N*F, 4))

    straighten_series[0] = MESH.wall(edge_series[0], expansion_series[-1], straighten_series[0])
    straighten_fill[:F, :] = MESH.fill(edge_series[0], straighten_series[0], F)
    f = F
    for n in range(1, N):
        straighten_series[n] = MESH.wall(edge_series[n], straighten_series[n-1], straighten_series[n])
        straighten_fill[f:f+F, :] = MESH.fill(edge_series[n], straighten_series[n], F)
        f += F

    return straighten_series, straighten_fill

def lip(straighten_series, M_jet, N):
    M_exi = straighten_series[-1, 2]
    nu_exi = AERO.M2nu(M_exi)
    nu_jet = AERO.M2nu(M_jet)
    delta_max = nu_jet/2 - nu_exi/2

    lip_series = np.empty(shape = (N, 4))
    for n in range(0, N):
        lip_series[n, 3] = n*delta_max/(N-1)
        nu = lip_series[n, 3] + nu_exi
        lip_series[n, 2] = AERO.nu2M(nu)
        lip_series[n, 0] = straighten_series[-1, 0]
        lip_series[n, 1] = straighten_series[-1, 1]

    return lip_series

def plume(lip_series, N, F):
    # leading_series = np.empty(shape = (N, 4))
    # leading_series[0] = MESH.centerline(lip_series[0], leading_series[0])
    # for n in range(1, N):
    #     leading_series[n] = MESH.internal(leading_series[n-1], lip_series[n], leading_series[n])

    plume_series, leading_series, lip_fill = internal(lip_series, N, F)

    jet_series = np.empty(shape = (N, 4))
    jet_fill = np.empty(shape = (N*F, 4))
    jet_series[0] = MESH.jet(leading_series[0], lip_series[N-1], jet_series[0])
    f = 0
    jet_fill[f:f+F] = MESH.fill(leading_series[0], jet_series[0], F)
    f += F
    for n in range(1, N):
        jet_series[n] = MESH.jet(leading_series[n], jet_series[n-1], jet_series[n])
        jet_fill[f:f+F, :] = MESH.fill(leading_series[n], jet_series[n], F)
        f += F

    # leading_series = np.empty(shape = (N, 4))
    # leading_series[1:N-1] = MESH.fill(lip_series[0], leading_point, N-2)
    # leading_series[0] = lip_series[0]
    # leading_series[N-1] = leading_point

    # plume_series = np.empty(shape = (N, 4))
    # plume_series[0] = MESH.internal(leading_series[1], lip_series[1], plume_series[0])

    return plume_series, jet_series, jet_fill, lip_fill