import numpy as np
import math

from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# runtime parameters
y_thr = 0.25 # throat cross section radius
r_exp = 0.15 # expansion section radius
M_init = 1.1 # init Mach Number
delta = np.deg2rad(25)
N = 20 # initial point count
gamma = 1.4
x_max = 3 * r_exp * math.sin(delta)

def wall_position(x):
    if x < 0: 
        return y_thr
    if x <= r_exp * math.sin(delta):
        return y_thr + r_exp - math.sqrt(r_exp**2 - x**2)
    else: 
        return y_thr + r_exp * (1 - math.cos(delta)) + math.tan(delta) * (x - r_exp * math.sin(delta))

def wall_angle(x): 
    if x < 0:
        return 0
    if x <= r_exp * math.sin(delta): 
        return math.atan(x / math.sqrt(r_exp**2 - x**2))
    else:
        return delta

class AeroFunc():
    def M2nu(M): # Prandtl-Meyer Function
        # https://en.wikipedia.org/wiki/Prandtl-Meyer_function
        A = math.sqrt((gamma + 1)/(gamma - 1))
        B = math.sqrt(M**2 - 1)
        return A * math.atan(B/A) - math.atan(B)

    def nu2M(nu): # Inverse Prandtl-Meyer Function
        # https://en.wikipedia.org/wiki/Prandtl-Meyer_function
        return root_scalar(lambda M: nu - AeroFunc.M2nu(M), x0 = M_init, x1 = 3 * M_init).root
    
    def M2mu(M): # Mach Angle Function
        # https://en.wikipedia.org/wiki/Mach_wave
        return math.asin(1/M)

class MeshPoint():
    def internal(p1data, p2data):
        # p1 is below, p2 is above
        x1, x2 = p1data[0], p2data[0]
        y1, y2 = p1data[1], p2data[1]
        M1, M2 = p1data[2], p2data[2]
        theta1, theta2 = p1data[3], p2data[3]

        mu1, mu2 = AeroFunc.M2mu(M1), AeroFunc.M2mu(M2)
        
        nu1, nu2 = AeroFunc.M2nu(M1), AeroFunc.M2nu(M2)
        nu3 = (nu1 + nu2)/2 - (theta1 - theta2)/2
        theta3 = (theta1 + theta2)/2 - (nu1 - nu2)/2
        
        M3 = AeroFunc.nu2M(nu3)
        mu3 = AeroFunc.M2mu(M3)

        phi1 = (theta1 + theta3 + mu1 + mu3)/2
        phi2 = (theta2 + theta3 - mu2 - mu3)/3

        x3 = x1 * math.tan(phi1) - x2 * math.tan(phi2) + y2 - y1
        x3 /= math.tan(phi1) - math.tan(phi2)

        y3 = y1 + math.tan(phi1) * (x3 - x1)
        
        return np.array([x3, y3, M3, theta3])
    
    def wall(p1data):
        x1 = p1data[0]
        y1 = p1data[1]
        M1 = p1data[2]
        theta1 = p1data[3]

        nu1 = AeroFunc.M2nu(M1)
        mu1 = AeroFunc.M2mu(M1)

        def yw(x):
            thetaw = wall_angle(x)
            nuw = nu1 + thetaw - theta1
            Mw = AeroFunc.nu2M(nuw)
            muw = AeroFunc.M2mu(Mw)

            phi1 = (theta1 + thetaw)/2 + (mu1 + muw)/2

            yw = y1 + np.tan(phi1) * (x - x1)
            return yw
        
        xw = root_scalar(lambda x: wall_position(x) - yw(x), x0 = x1, x1 = x_max).root
        
        yw = wall_position(xw)
        thetaw = wall_angle(xw)
        nuw = thetaw - theta1 + nu1
        Mw = AeroFunc.nu2M(nuw)

        return np.array([xw, yw, Mw, thetaw])

class MeshGen():
    def init_series():
        init_series = np.empty(shape = (N, 4))
        init_series[:, 0] = 0
        init_series[:, 1] = np.array([y_thr*n/(N-1) for n in range(0, N)])
        init_series[:, 2] = M_init
        init_series[:, 3] = 0

        return init_series

    def onWall_series(noWall_series):
        newN = noWall_series.shape[0]
        onWall_series = np.empty(shape = (newN, 4))

        for n in range(0, newN - 1):
            onWall_series[n] = MeshPoint.internal(noWall_series[n], noWall_series[n+1])
        onWall_series[newN - 1] = MeshPoint.wall(noWall_series[newN - 1])


        return onWall_series

    def noWall_series(onWall_series):
        newN = onWall_series.shape[0] - 1
        noWall_series = np.empty(shape = (newN, 4))

        for n in range(0, newN):
            noWall_series[n] = MeshPoint.internal(onWall_series[n], onWall_series[n+1])
        return noWall_series

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize = (6.5, 3.75))

    wall_x = np.linspace(0, x_max)
    wall_y = np.array([wall_position(x) for x in wall_x])
    
    init_series = MeshGen.init_series()
    noWall_series = MeshGen.noWall_series(init_series)
    onWall_series = MeshGen.onWall_series(noWall_series)

    all_series = np.vstack((init_series, noWall_series, onWall_series))

    while onWall_series.shape[0] > 2:
        noWall_series = MeshGen.noWall_series(onWall_series)
        onWall_series = MeshGen.onWall_series(noWall_series)

        all_series = np.vstack((all_series, noWall_series, onWall_series))


    ax.plot(wall_x, wall_y, color = 'black')
    all_series_plot = ax.scatter(all_series[:, 0], all_series[:, 1], c = all_series[:, 2])
    M_bar = fig.colorbar(all_series_plot)
    M_bar.set_label("$M$")

    ax.set_xlim(0, x_max)

    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color = 'black', label = "Wall Boundary"),
        Line2D([0], [0], color = 'grey', label = "$C_+$ Characteristics"),
        Line2D([0], [0], color = 'red', label = "$C_-$ Characteristics"),
    ]

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.set_title(r"Mach Number at Intersections, $N = 20$, $\delta = 25º$, $M_1 = 1.1$")
    # plt.show()
    plt.savefig("PMC Plot 1", dpi = 400)