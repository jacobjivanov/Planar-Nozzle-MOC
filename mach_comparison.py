import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import REGION
import MESH
import AERO

def create_nozzle_figure():
    fig, ax = plt.subplots(figsize = (12, 6))
    triang_nozzle = tri.Triangulation(nozzle_series[:, 0], nozzle_series[:, 1])

    if jet_effects == True:
        triang_free = tri.Triangulation(free_series[:, 0], free_series[:, 1])
        M_plot1 = ax.tripcolor(triang_nozzle, nozzle_series[:, 2], cmap = 'plasma', vmin = 1, vmax = np.max(free_series[:, 2]))
        M_plot2 = ax.tripcolor(triang_free, free_series[:, 2], cmap = 'plasma', vmin = 1, vmax = np.max(free_series[:, 2]))
        # ax.scatter(nozzle_series[:, 0], nozzle_series[:, 1], s = 0.01, color = 'black')
        # ax.scatter(free_series[:, 0], free_series[:, 1], s = 0.01, color = 'black')

    else:
        M_plot1 = ax.tripcolor(triang_nozzle, nozzle_series[:, 2], cmap = 'plasma')
        # ax.scatter(nozzle_series[:, 0], nozzle_series[:, 1], s = 0.1, color = 'black')

    divider = make_axes_locatable(ax)
    wall_plot = ax.plot(wall_series[:, 0], wall_series[:, 1], color = 'black', label = 'Nozzle Wall')
    cl_plot = ax.plot([0, straighten_series[-1, 0]], [0, 0], linestyle = 'dashed', color = 'black', label = 'Nozzle Centerline')
    cax = divider.append_axes(position = "bottom", size = "5%", pad = 0.75)
    M_bar = fig.colorbar(M_plot1, cax = cax, orientation = 'horizontal')
    M_bar.set_label("$M$")
    ax.set_aspect('equal')
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    ax.set_title("Mach Number in Shock-Free Diverging Nozzle")
    ax.legend(loc = 'upper left')

    fig.tight_layout()
    plt.savefig("Mach Number in Shock-Free Diverging Nozzle.png", dpi = 200)
    plt.show()

if __name__ == '__main__':
    N = 10 # characteristic line count
    F = 0 # fill point count

    # runtime parameters
    y_inl = 0.25 # throat cross section radius
    r_exp = 0 # expansion section radius
    M_inl = 1.1 # throat Mach Number
    M_exi = 3 # exit Mach Number
    gamma = 1.2 # ratio of specific heats

    p_exi = 500
    p0 = p_exi * AERO.p_ratio(M_exi)

    jet_effects = False
    p_jet = 25
    M_jet = AERO.M_jet(p0, p_jet)

    # compute each region of points within nozzle
    wall_x, wall_y, expansion_series = REGION.expansion(y_inl, r_exp, M_inl, M_exi, N)
    internal_series, edge_series, expansion_fill = REGION.internal(expansion_series, N, F)
    straighten_series, straighten_fill = REGION.straighten(expansion_series, edge_series, N, F)
    # combine points
    
    nozzle_series = np.vstack((
        np.array([0, 0, M_inl, 0]),
        expansion_series,
        expansion_fill, 
        internal_series,
        straighten_series,
        straighten_fill,
        np.array([straighten_series[-1, 0], 0, straighten_series[-1, 2], 0]),
    ))

    wall_series = np.vstack((
        expansion_series,
        straighten_series
    ))

    # free jet effects
    if jet_effects == True:
        lip_series = REGION.lip(straighten_series, M_jet, N)
        plume_series, jet_series, jet_fill, lip_fill = REGION.plume(lip_series, N, F)
        
        free_series = np.vstack((
            np.array([straighten_series[-1, 0], 0, straighten_series[-1, 2], 0]),
            lip_series,
            plume_series,
            jet_series,
            jet_fill,
            lip_fill
        ))

    # create_nozzle_figure()
    plt.scatter(nozzle_series[:, 2], AERO.M2nu(nozzle_series[:, 2]) + nozzle_series[:, 3], color = 'blue')
    plt.show()