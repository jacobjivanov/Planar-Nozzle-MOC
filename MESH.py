import numpy as np
import AERO
import numba

@numba.njit()
def internal(p1data, p2data, p3data):
    # p1 is below, p2 is above
    x1, x2 = p1data[0], p2data[0]
    y1, y2 = p1data[1], p2data[1]
    M1, M2 = p1data[2], p2data[2]
    theta1, theta2 = p1data[3], p2data[3]

    mu1, mu2 = AERO.M2mu(M1), AERO.M2mu(M2)
    
    nu1, nu2 = AERO.M2nu(M1), AERO.M2nu(M2)
    nu3 = (nu1 + nu2)/2 - (theta1 - theta2)/2
    theta3 = (theta1 + theta2)/2 - (nu1 - nu2)/2
    
    M3 = AERO.nu2M(nu3)
    mu3 = AERO.M2mu(M3)

    phi1 = (theta1 + theta3 + mu1 + mu3)/2
    phi2 = (theta2 + theta3 - mu2 - mu3)/2

    x3 = x1 * np.tan(phi1) - x2 * np.tan(phi2) + y2 - y1
    x3 /= np.tan(phi1) - np.tan(phi2)

    y3 = y1 + np.tan(phi1) * (x3 - x1)
    
    p3data[0] = x3
    p3data[1] = y3
    p3data[2] = M3
    p3data[3] = theta3
    
    return p3data

@numba.njit()
def centerline(p1data, pcdata):
    # p1 is above centerline
    x1 = p1data[0]
    y1 = p1data[1]
    M1 = p1data[2]
    theta1 = p1data[3]

    mu1 = AERO.M2mu(M1)

    nu1 = AERO.M2nu(M1)
    nuc = theta1 + nu1
    thetac = 0

    Mc = AERO.nu2M(nuc)
    muc = AERO.M2mu(Mc)

    phi1 = theta1/2 - (mu1 + muc)/2
    
    xc = x1 - y1/np.tan(phi1)
    yc = 0

    pcdata[0] = xc
    pcdata[1] = yc
    pcdata[2] = Mc
    pcdata[3] = thetac
    
    return pcdata

@numba.njit()
def wall(p1data, p2data, pwdata):
    # p1 is an internal point, p2 is the last wall point
    x1, x2 = p1data[0], p2data[0]
    y1, y2 = p1data[1], p2data[1]
    M1, M2 = p1data[2], p2data[2]
    theta1, theta2 = p1data[3], p2data[3]

    mu1 = AERO.M2mu(M1)

    thetaw = theta1
    Mw = M1

    phi1 = theta1 + mu1
    phi2 = (theta1 + theta2)/2

    xw = x1 * np.tan(phi1) - x2 * np.tan(phi2) + y2 - y1
    xw /= np.tan(phi1) - np.tan(phi2)

    yw = y1 + np.tan(phi1) * (xw - x1)

    pwdata[0] = xw
    pwdata[1] = yw
    pwdata[2] = Mw
    pwdata[3] = thetaw
    
    return pwdata

def jet(p1data, p2data, pjdata):
    # p1 is an internal point, p2 is the last jet point
    x1, x2 = p1data[0], p2data[0]
    y1, y2 = p1data[1], p2data[1]
    M1, M2 = p1data[2], p2data[2]
    theta1, theta2 = p1data[3], p2data[3]

    mu1 = AERO.M2mu(M1)
    phi1 = theta1 + mu1
    nu1 = AERO.M2nu(M1)

    xj = y1 - y2 + x2 * np.tan(theta2) - x1 * np.tan(phi1)
    xj /= np.tan(theta2) - np.tan(phi1)

    yj = y1 + np.tan(phi1) * (xj - x1)
    Mj = M2
    nuj = AERO.M2nu(Mj)

    thetaj = theta1 + nuj - nu1

    pjdata[0] = xj
    pjdata[1] = yj
    pjdata[2] = Mj
    pjdata[3] = thetaj
    
    return pjdata

@numba.njit()
def fill(p1data, p2data, F):
    # will include non-intersection fill points for visualization
    fill = np.empty(shape = (F, 4))
    for n in range(0, F):
        fill[n, :] = p1data + (n+1)/(F+1) * (p2data - p1data)

    return fill