# -*- coding: utf-8 -*-
"""
Created on Wed 14 Oct 15:20 2020

mcsquared.fz@gmail.com
miguel.cruces@rai.usc.es

@author:
Miguel Cruces

"""
from scipy import stats
from const import *  # Numpy as np imported in const.py
from kalmanTRAG_lineal import diag_matrix, gene_tracks, trag_digitization, set_transport_func, set_jacobi, set_mKgain
from kalmanFilter import plot_vec, plot_rays

np.set_printoptions(formatter={'float': '{:.3f}'.format})


# np.random.seed(16)  # If commented ==> random tracks


def tragaldabas_output(m_dat):
    """
    Creates a matrix TRAGALDABAS output data like, in realistic format.

    :param m_dat: Matrix of generated and digitized tracks.
    :return: Equivalent matrix to m_data, in realistic TRAGALDABAS format.
    """
    if np.all(m_dat == 0):  # Check if mdat is all zero
        raise Exception('No tracks available! Matrix mdat is all zero ==> Run the program Again '
                        f'because actual random seed is not valid for {NTRACK} number of tracks')
    ntrk, _ = m_dat.shape  # Number of tracks, number of plans
    ncol = len(["Plane ID", "kX", "kY", "X", "Y", "Z", "T"])  # One more column to store number of tracks
    m_tragal = np.zeros([0, ncol])
    idat = 0
    for ip in range(NPLAN):  # Number of planes
        for it in range(ntrk):  # Number of tracks
            idatf = idat + NDAC
            kx, ky, t = m_dat[it, idat:idatf]  # kx, ky (index cells), t (time in ps)
            z = VZI[::-1][ip]  # Height in mm
            x = kx * WCX - WCX / 2  # X [mm]
            y = ky * WCY - WCY / 2  # Y [mm]
            row = np.hstack((ip, kx, ky, x, y, z, t))
            m_tragal = np.vstack((m_tragal, row))
        idat += NDAC
    np.random.shuffle(m_tragal)
    return m_tragal


def input_saeta_2planes(xi, yi, ti, zi, xj, yj, tj, zj):
    """
    Saeta2Planes calculates a saeta between 2 planes. A simple non-linear model used.
    Used to fill an non-zero input saeta.

    :param xi: To
    :param xj: From
    :return: The output saeta has the form (X0,X',Y0,Y',T0,S)
    """
    S2 = np.zeros([6])
    dz = zi - zj
    S2[0] = (xj * zi - xi * zj) / dz
    S2[1] = (xi - xj) / dz
    S2[2] = (yj * zi - yi * zj) / dz
    S2[3] = (yi - yj) / dz
    S2[4] = (tj * zi - ti * zj) / dz
    S2[5] = (ti - tj) / dz
    return S2


def fit_kalman(ri, Ci, xi, yi, ti, zi, xf, yf, tf, zf):

    # Step 2. - PREDICTION
    F = set_transport_func(1, z3 - z4)  # ks = 1 due to initial hypothesis
    r4p = np.dot(F, ri)
    C4p = np.dot(F, np.dot(C4, F.T))

    # # Step 3. - PROCESS NOISE [At most 10%]
    # r4p *= 1 + np.random.uniform(low=-0.1, high=0.1, size=r4p.shape)
    # C4p *= 1 + np.random.uniform(low=-0.1, high=0.1, size=C4p.shape)

    # Step 4. - FILTRATION
    m = np.array([x3, y3, t3])  # Measurement

    H = set_jacobi()

    # Matrix K gain
    K, weights = set_mKgain(H, C4p, V)

    # New r vector
    mr = np.dot(H, r4p)
    delta_m = m - mr
    delta_r = np.dot(K, delta_m)
    r3 = r4p + delta_r

    # New C matrix
    C3 = C4p - np.dot(K, np.dot(H, C4p))
    pass


# =============== GENERATE TRACKS ============== #
mtrk, nt = gene_tracks()
# mtrk --> Initial Saetas
# nt ----> Number of tracks in the detector

# ================ DIGITIZATION ================ #
mdpt, mdat = trag_digitization(nt, mtgen=mtrk)  # Digitization for TRAGALDABAS detector
# mdat --> (kx1, ky2, time1,   kx2, ky2, time2, ...)
# mdpt --> (X1, Y1, T1,   X2, Y2, T2,   ...)  Real points of impact / mm

# =============== RECONSTRUCTION =============== #
mdet = tragaldabas_output(mdat)  # Matrix with columns: ["Plane ID", "kX", "kY", "X", "Y", "Z", "T"]

# ================ MAIN FUNCTION =============== #

C0 = diag_matrix(NPAR, [1 / WX, VSLP, 1 / WY, VSLP, 1 / WT, VSLN])  # Error matrix
V = diag_matrix(NDAC, [SIGX ** 2, SIGY ** 2, SIGT ** 2])
dcut = 0.995

nhits = mdet.shape[0]
for i4 in range(nhits):  # ============================================================================ PLANE 4
    hit4 = mdet[i4]
    if np.all(hit4 == 0): continue
    plane4, kx4, ky4, x4, y4, z4, t4 = hit4
    if plane4 != 3: continue
    plane_hits = 1

    for i3 in range(nhits):  # ======================================================================== PLANE 3
        if i3 == i4: continue
        hit3 = mdet[i3]
        if np.all(hit3 == 0): continue
        plane3, kx3, ky3, x3, y3, z3, t3 = hit3
        if plane3 != 2: continue

        # Step 1. - INITIALIZATION
        r4 = [x4, 0, y4, 0, t4, SC]  # Initial Guess for the saeta in fourth plane
        C4 = diag_matrix(NPAR, [1 / WX, VSLP, 1 / WY, VSLP, 1 / WT, VSLN])  # Error matrix (Initial Guess)

        # Step 2. - PREDICTION
        F = set_transport_func(1, z3 - z4)  # ks = 1 due to initial hypothesis
        r4p = np.dot(F, r4)
        C4p = np.dot(F, np.dot(C4, F.T))

        # # Step 3. - PROCESS NOISE [At most 10%]
        # r4p *= 1 + np.random.uniform(low=-0.1, high=0.1, size=r4p.shape)
        # C4p *= 1 + np.random.uniform(low=-0.1, high=0.1, size=C4p.shape)

        # Step 4. - FILTRATION
        m = np.array([x3, y3, t3])  # Measurement

        H = set_jacobi()

        # Matrix K gain
        K, weights = set_mKgain(H, C4p, V)

        # New r vector
        mr = np.dot(H, r4p)
        delta_m = m - mr
        delta_r = np.dot(K, delta_m)
        r3 = r4p + delta_r

        # New C matrix
        C3 = C4p - np.dot(K, np.dot(H, C4p))

        for i2 in range(nhits):  # ==================================================================== PLANE 2
            if i2 == i3 or i2 == i4: continue
            hit2 = mdet[i2]
            if np.all(hit2 == 0): continue
            plane2, kx2, ky2, x2, y2, z2, t2 = hit2
            if plane2 != 1: continue
            for i1 in range(nhits):  # ================================================================ PLANE 1
                if i1 == i2 or i1 == i3 or i1 == i4: continue
                hit1 = mdet[i1]
                if np.all(hit1 == 0): continue
                plane1, kx1, ky1, x1, y1, z1, t1 = hit1
                if plane1 != 0: continue
                dist = np.sqrt((x1 - x4) * (x1 - x4) +
                               (y1 - y4) * (y1 - y4) +
                               (z1 - z4) * (z1 - z4))  # Total track distance
                if plane4 == 3 and plane3 == 2 and plane2 == 1 and plane1 == 0:
                    s_init = input_saeta_2planes(x3, y3, t3, z3, x4, y4, t4, z4)
                    print(f"({i1}, {i2}, {i3}, {i4}) s_init = {s_init}")

                    hits = [i1, i2, i3, i4]

                    # for hit in hits:
