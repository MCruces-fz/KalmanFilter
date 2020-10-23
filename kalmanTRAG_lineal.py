# -*- coding: utf-8 -*-
"""
Created on Fri 9 Oct 18:47 2020

mcsquared.fz@gmail.com
miguel.cruces@rai.usc.es

@author:
Miguel Cruces

"""
from scipy import stats
from const import *  # Numpy as np imported in const.py
from kalmanFilter import plot_rays, plot_saetas

import matplotlib.pyplot as plt

plt.close("all")

np.set_printoptions(formatter={'float': '{:.3f}'.format})

# np.random.seed(21)  # peq -> 3/3 Gran -> 3/3


# np.random.seed(20)  # peq -> 3/3 Gran -> 3/3
# np.random.seed(19)  # peq -> 2/3 Gran -> 2/3
# np.random.seed(18)  # peq -> 2/3 Gran -> 2/3
# np.random.seed(17)  # peq -> 3/3 Gran -> 2/3
# np.random.seed(16)  # If commented ==> random tracks
# np.random.seed(15)  # peq -> 1/3 Gran -> 2/3                    ||  dcut=0.812 -> 3/2    dcut=0.813  -> 1/2
# np.random.seed(14)  # peq -> 1/3 bien 2/3 mal Gran -> 2/3 bien    ||  dcut=0.8 ---> 2/2    dcut=0.3  ---> 1/2


# np.random.seed(13)  # peq -> 3/3 bien +1 mal Gran -> 3/3
# np.random.seed(12)  # peq -> 2/3 bien +1 mal -> 2/3 bien +1 mal
# np.random.seed(11)  # peq -> 2/3 Gran -> 2/3
# np.random.seed(6)  # Fallo al digitalizar (THMAX = 30)


def diag_matrix(dim: int, diag: list):
    """
    Create squared matrix of dimXdim dimension with diag in the diagonal.

    :param dim: Quantity of rows/columns.
    :param diag: String of length dim with the diagonal values.
    :return: Squared matrix of dimXdim dimension with diag in the diagonal.
    """
    arr = np.zeros([dim, dim])
    row, col = np.diag_indices(arr.shape[0])
    arr[row, col] = np.array(diag)
    return arr


def gene_tracks(all_tracks_in: bool = True):
    """
    It generates random parameters to construct the tracks as Saetas. If the
    track doesn't enter in the detector, it is deleted from the list.

    :param all_tracks_in: True if force nt == NTRACKS or False if nt <= NTRACKS
        randomly deleting outisders.

    :return mtgen: Matrix of generated tracks (initial saetas).
    :return nt: Total number of tracks in the detector
    """
    ctmx = np.cos(np.deg2rad(THMAX))  # Theta max angle cosine
    # lenz = abs(VZI[0] - VZI[-1])  # Distance from bottom to top planes
    it = 0  # Number of tracks actually
    mtgen = np.zeros([NTRACK, NPAR])  # generated tracks matrix
    i = 1
    while i <= NTRACK:
        # Uniform distribution in cos(theta) and phi
        rcth = 1 - np.random.random() * (1 - ctmx)
        tth = np.arccos(rcth)  # theta random angle
        tph = np.random.random() * 2 * np.pi  # phi random angle

        X0 = np.random.random() * LENX
        Y0 = np.random.random() * LENY
        T0 = TINI
        S0 = SINI

        # Director Cosines
        cx = np.sin(tth) * np.cos(tph)
        cy = np.sin(tth) * np.sin(tph)
        cz = np.cos(tth)
        XP = cx / cz  # projected slope in the X-Z plane
        YP = cy / cz  # projected slope in the Y-Z plane

        # Coordinate where would the particle come out
        xzend = X0 - XP * LENZ
        yzend = Y0 - YP * LENZ

        # We refer the coordinate to the detector center (xmid, ymid)
        xmid = xzend - (LENX / 2)
        ymid = yzend - (LENY / 2)

        if not all_tracks_in:
            i += 1
        # We check if the particle has entered the detector
        if np.abs(xmid) < (LENX / 2) and np.abs(ymid) < (LENY / 2):
            mtgen[it, :] = [X0, XP, Y0, YP, T0, S0]
            it += 1
            if all_tracks_in:
                i += 1
    nt = it  # number of tracks in the detector
    mtgen = mtgen[~(mtgen == 0).all(1)]
    return mtgen, nt


def trag_digitization(nt: int, mtgen):
    """
    # ======== DIGITIZATION FOR TRAGALDABAS DETECTOR ======== #

    It converts the parameters inside mtgen to discrete
    numerical values, which are the cell indices (m_dat) and
    cell central positions (m_dpt).

    :param nt: Number of tracks generated across the detector.
    :param mtgen: Matrix of generated tracks
    :return: m_dat (cell indices matrix) and m_dpt (cell central
        positions matrix).
    """
    v_dat = np.zeros(NPLAN * NDAC)  # Digitalizing tracks vector
    v_dpt = np.zeros(NPLAN * NDAC)  # Vector with impact point
    m_dat = np.zeros(NPLAN * NDAC)  # Detector data matrix
    m_dpt = np.zeros(NPLAN * NDAC)  # Impact point
    nx = 0
    for it in range(nt):
        x0, xp, y0, yp = mtgen[it, 0:4]  # dz = np.cos(th)

        it = 0
        for ip in range(NPLAN):
            zi = VZI[ip]  # current Z
            zt = VZI[0]  # Z top
            dz = zi - zt  # dz <= 0

            xi = x0 + xp * dz
            yi = y0 + yp * dz
            ks = np.sqrt(1 + xp ** 2 + yp ** 2)
            ti = TINI + ks * SC * (- dz)  # Time Flies (dz < 0)

            # Position indices of the impacted cells (cell index)
            kx = np.int((xi + (WCX / 2)) / WCX)
            ky = np.int((yi + (WCY / 2)) / WCY)
            kt = np.int((ti + (DT / 2)) / DT) * DT
            # Cell position (distance)
            # xic = kx * WCX + (WCX / 2)
            # yic = ky * WCX + (WCX / 2)
            vpnt = np.asarray([xi, yi, ti])  # (X,Y,T) impact point
            vxyt = np.asarray([kx, ky, kt])  # impact index
            v_dpt[it:it + NDAC] = vpnt[0:NDAC]
            v_dat[it:it + NDAC] = vxyt[0:NDAC]
            it += 3
        m_dpt = np.vstack((m_dpt, v_dpt))
        m_dat = np.vstack((m_dat, v_dat))
        nx += 1
    m_dpt = np.delete(m_dpt, 0, axis=0)
    m_dat = np.delete(m_dat, 0, axis=0)
    return m_dpt, m_dat


def gcut(vstat, vr, k):
    """
    Function that returns quality factor by the second method
    """
    bm = 0.2  # beta min
    cmn = bm * VC
    smx = 1 / cmn
    nhits = vstat[0]
    ndat = int(nhits * NDAC)  # Number of measurement coordinates (x, y, t)
    vhits = np.array(np.split(vstat[1:ndat + 1], nhits))
    ndf = ndat - NPAR  # Degrees of Freedom

    # sigx2, _, sigy2, _, sigt2, _ = np.diag(C[k])
    sigx2, sigy2, sigt2 = SIGX ** 2, SIGY ** 2, SIGT ** 2

    x0, xp, y0, yp, t0, s0 = vr
    ks = np.sqrt(1 + xp ** 2 + yp ** 2)
    print("==============================")
    print(f"vr = {vr}")

    if s0 < 0 or s0 > smx:
        cut_f = 0
    else:
        if ndf > 0:
            s_2 = 0
            for ip, (xd, yd, td) in enumerate(vhits):
                dtp = (td - t0)  # Almost always is t0 < td
                dt = dtp * np.sqrt(1 - xp ** 2 - yp ** 2)
                dz = (VZI[ip] - VZI[k])
                print(f"dz = {VZI[ip]} - {VZI[k]} = {dz}; dtp = {dtp}; dt = {dt}; dtz = {dt/s0}")
                # Saeta intersection at instant td [coordinates (x, y)]
                x = x0 - xp * dt
                y = y0 - yp * dt
                t = t0 + ks * dt
                # Measured coordinates in mm
                xd *= WCX
                yd *= WCY
                print(f"(x, y, t) = ({x}, {y}, {t})")
                print(f"(xd, yd, td) = ({xd:.3f}, {yd:.3f}, {td:.3f})")
                print("-------")
                s_2 += (xd - x) ** 2 / sigx2 + (yd - y) ** 2 / sigy2 + (td - t) ** 2 / sigt2
            cut_f = stats.chi2.sf(x=s_2, df=ndf)  # Survival function
            print(f"s2 = {s_2}, DoF = {ndf}, cutf = {cut_f}, dcut = {dcut}")
        elif not ndf:  # DoF == 0
            cut_f = 1
        else:
            print(f'WARNING! ndf = {ndf}')
            cut_f = np.nan
    # print(f'Plane T{k + 1}: cutf = {cut_f:.4f}; DoF = {int(vstat[0])}x{NDAC} - {NPAR} = {ndf}')
    return cut_f


def matrix_det(m_dat):
    """
    Creates a matrix similar to TRAGALDABAS output data

    :param m_dat: Matrix of generated and digitized tracks.
    :return: Equivalent matrix to m_data, in TRAGALDABAS format.
    """
    if np.all(m_dat == 0):  # Check if mdat is all zero
        raise Exception('No tracks available! Matrix mdat is all zero ==> Run the program Again '
                        f'because actual random seed is not valid for {NTRACK} '
                        'number of tracks')
    ntrk, _ = m_dat.shape  # Number of tracks, number of plans
    ncol = 1 + NDAC * ntrk  # One more column to store number of tracks
    mdet = np.zeros([NPLAN, ncol])
    idat = 0
    for ip in range(NPLAN):
        idet = 0
        for it in range(ntrk):
            ideti = idet + 1
            idetf = ideti + NDAC
            idatf = idat + NDAC
            mdet[ip, ideti:idetf] = m_dat[it, idat:idatf]
            if not np.all((mdet[ip, ideti:idetf] == 0)):  # checks if all are zero
                mdet[ip, 0] += 1
            idet += NDAC
        idat += NDAC
    return mdet


def set_transport_func(ks: float, dz: int):
    """
    It sets the transport matrix between both planes separated by dz

    :param ks: sqrt( 1 + XP**2 + YP**2)
    :param dz: distance between planes
    :return: Transport function (matrix | Numpy array)
    """
    F = diag_matrix(NPAR, [1] * NPAR)  # Identity 6x6
    F[0, 1] = dz
    F[2, 3] = dz
    F[4, 5] = - ks * dz
    return F


def set_jacobi():
    #  zf, rn):
    # Vector with noise: rn = (X0n, XPn, Y0n, YPn, T0n, S0n)
    """
    Jacobian || I(NDACxNPAR): Parameters (NPAR dim) --> Measurements (NDAC dim)

    :return: Jacobi matrix H
    """
    H = np.zeros([NDAC, NPAR])
    rows = range(NDAC)
    cols = range(0, NPAR, 2)
    H[rows, cols] = 1
    # X0n, XPn, Y0n, YPn, T0n, S0n = rn[k, i3]
    # ksn = np.sqrt(1 + XPn ** 2 + YPn ** 2)
    # H[0, 1] = zf
    # H[1, 3] = zf
    # H[2, 1] = - S0n * XPn * zf / ksn
    # H[2, 3] = - S0n * YPn * zf / ksn
    # H[2, 5] = ksn * zf
    return H


def set_mdet_xy(m_det):
    """
    It Calculates the mdet equivalent in mm, in spite of in indices.
    mdet with x & y in mm

    Columns:
    Hits per plane | X [mm] | Y [mm] | Time [ps]

    :param m_det: Matrix with TRAGALDABAS output data
    :return: Matrix equivalent to m_det with positions in mm.
    """
    mdet_xy = np.copy(m_det)
    no_tracks = (len(m_det[0]) - 1) // 3  # Number of tracks
    for idn in range(no_tracks):
        iX = 1 + idn * NDAC
        iY = iX + 1
        mdet_xy[:, iX] = mdet_xy[:, iX] * WCX  # - (WCX / 2)  # X [mm]
        mdet_xy[:, iY] = mdet_xy[:, iY] * WCY  # - (WCY / 2)  # Y [mm]
    return mdet_xy


def set_params(iplanN: int, iN: int):
    """
    It sets parameters of indexes of cells and positions respectively,
    taking data from mdet
    :param iN: Integer which defines the data index.
    :param iplanN: Integer which defines the plane index.
    :return: Parameters kxN, kyN, ktN, x0, y0, t0 where N is the plane (TN).
    """
    icel = 1 + iN * NDAC
    kxN, kyN, ktN = mdet[iplanN, icel:icel + NDAC]
    x0 = kxN * WCX  # - (WCX / 2)
    y0 = kyN * WCY  # - (WCY / 2)
    t0 = ktN
    return kxN, kyN, ktN, x0, y0, t0


def m_coord(k, idn):
    """
    It sets the coordinates in mm of the hit to thew measurement vector
    """
    ini = 1 + idn * NDAC
    fin = ini + NDAC
    return mdet_xy[k, ini:fin]


def set_vstat(k: int, i1: int, i2: int, i3: int, i4: int, plane_hits: int):
    """
    It sets the array:
    [ Nbr. hits, kx1, ky1, kt1, ..., 0, 0, 0, X0, XP, Y0, YP, T0, S0 ]
    """
    idx = [[0, i1], [1, i2], [2, i3], [3, i4]]
    k_list = []
    for plane, hit in reversed(idx[k:]):
        kx, ky, kt, _, _, _ = set_params(plane, hit)
        k_list.extend([kx, ky, kt])
    k_vec = np.zeros(NDAC * NPLAN)
    k_vec[:len(k_list)] = k_list
    vstat = np.hstack([plane_hits, k_vec, r[k]])
    return vstat


def fcut(vstat, vm, vr, s2_prev):
    """
    Function that returns quality factor by the first method
    """
    bm = 0.2  # beta min
    cmn = bm * VC
    smx = 1 / cmn
    ndat = int(vstat[0] * NDAC)  # Number of measurement coordinates (x, y, t)
    ndf = ndat - NPAR  # Degrees of Freedom

    xd, yd, td = vm

    # sigx2, _, sigy2, _, sigt2, _ = np.diag(C[k])
    sigx2, sigy2, sigt2 = SIGX ** 2, SIGY ** 2, SIGT ** 2

    x0, _, y0, _, t0, s0 = vr

    s2 = s2_prev + (xd - x0) ** 2 / sigx2 + (yd - y0) ** 2 / sigy2 + (td - t0) ** 2 / sigt2

    if s0 < 0 or s0 > smx:
        cut_f = 0
    else:
        if ndf > 0:
            cut_f = stats.chi2.sf(x=s2, df=ndf)  # Survival function
        elif not ndf:  # DoF == 0
            cut_f = 1
        else:
            print(f'WARNING! ndf = {ndf}')
            cut_f = np.nan
    # print(f'Plane T{k + 1}: cutf = {cut_f:.4f}; DoF = {int(vstat[0])}x{NDAC} - {NPAR} = {ndf}')
    return cut_f, s2


def set_mKgain(H, Cn, V):
    """
    It sets the K matrix of gain and weights.

    :param H: Jacobi matrix
    :param Cn: Noised uncertainty matrix.
    :param V: Error matrix.
    :return: K gain matrix and weights.
    """
    H_Cn_Ht = np.dot(H, np.dot(Cn, H.T))
    wghts = np.linalg.inv(V + H_Cn_Ht)
    K = np.dot(Cn, np.dot(H.T, wghts))
    return K, wghts


# =============== GENERATE TRACKS ============== #
mtrk, nt = gene_tracks()
# mtrk --> Initial Saetas
# nt ----> Number of tracks in the detector

# ================ DIGITIZATION ================ #
mdpt, mdat = trag_digitization(nt, mtgen=mtrk)  # Digitization for TRAGALDABAS detector
# mdat --> (kx1, ky2, time1,   kx2, ky2, time2, ...)
# mdpt --> (X1, Y1, T1,   X2, Y2, T2,   ...)  Real points of impact / mm

# =============== RECONSTRUCTION =============== #
mdet = matrix_det(mdat)  # Matrix with columns: (nhits, kx, ky, time)

mdet_xy = set_mdet_xy(mdet)  # Matrix (mdet) with columns: (nhits, x [mm], y [mm], time)

r = np.zeros([NPLAN, NPAR])  # Vector (parameters); dimension -> Number of Planes x maximum hits x parameters
C = np.zeros([NPLAN, NPAR, NPAR])  # Error Matrices

rp = np.zeros(r.shape)  # Projected vector and matrices
Cp = np.zeros(C.shape)

rn = np.zeros(r.shape)  # UNUSED projected vectors with noises
Cn = np.zeros(C.shape)

# ================ MAIN FUNCTION =============== #

C0 = diag_matrix(NPAR, [5 / WX, 50 * VSLP, 5 / WY, 50 * VSLP, 5 / WT, 10 * VSLN])  # Error matrix
# C0 = diag_matrix(NPAR, [1 / WX, VSLP, 1 / WY, VSLP, 1 / WT, VSLN])  # Error matrix
V = diag_matrix(NDAC, [SIGX ** 2, SIGY ** 2, SIGT ** 2])
dcut = 0.75
Chi2 = 0
m_stat = np.zeros([0, 20])

iplan1, iplan2, iplan3, iplan4 = 0, 1, 2, 3  # Index for planes T1, T2, T3, T4 respectively
ncel1, ncel2, ncel3, ncel4 = mdet[:, 0].astype(np.int)  # Nr. of hits in each plane

# iN is the index of the hit in the plane N
for i4 in range(ncel4):
    for i3 in range(ncel3):
        for i2 in range(ncel2):
            for i1 in range(ncel1):
                s2 = 0
                hits = [i1, i2, i3, i4]  # Combination of hit indices
                # hits = [0] * 4
                # Step 1. - INITIALIZATION
                kx4, ky4, kt4, x0, y0, t0 = set_params(iplan4, i4)
                r0 = [x0, 0, y0, 0, t0, SC]  # Hypothesis
                r[iplan4] = r0
                C[iplan4] = C0
                plane_hits = 1

                # k: index of plane and zk: height of plane k in mm
                for k, zk in reversed(list(enumerate(VZI))[:-1]):  # [[2, 600], [1, 900], [0, 1800]]
                    s2_prev = s2 + 0
                    # Step 2. - PREDICTION
                    zi = VZI[k + 1]  # Lower plane, higher index
                    zf = VZI[k]  # This plane
                    dz = zf - zi
                    hiti = hits[k + 1]  # hit index in lower plane
                    hitf = hits[k]  # hit index in upper plane

                    # print(f"Plane {k + 1 + 1} hit {hiti + 1} -> Plane {k + 1} hit {hitf + 1}")

                    _, xp, _, yp, _, _ = r[k + 1]
                    ks = np.sqrt(1 + xp ** 2 + yp ** 2)

                    F = set_transport_func(ks, dz)
                    rp[k] = np.dot(F, r[k + 1])
                    Cp[k] = np.dot(F, np.dot(C[k + 1], F.T))

                    # Step 3. - PROCESS NOISE  [UNUSED YET]
                    rn[k] = rp[k]
                    Cn[k] = Cp[k]  # + Q (random matrix)

                    # Step 4. - FILTRATION
                    m = m_coord(k, hitf)  # Measurement

                    H = set_jacobi()

                    # Matrix K gain
                    K, weights = set_mKgain(H, Cn[k], V)
                    # weights = diag_matrix(NDAC, [SIGX, SIGY, SIGT])

                    # New rk vector
                    mr = np.dot(H, rn[k])
                    delta_m = m - mr
                    delta_r = np.dot(K, delta_m)
                    r[k] = rn[k] + delta_r

                    # New Ck matrix
                    C[k] = Cn[k] - np.dot(K, np.dot(H, Cn[k]))

                    # Chi2
                    Chi2 += np.dot(delta_m.T, np.dot(weights, delta_m))
                    # print(f"chi2 = {np.dot(delta_m.T, np.dot(weights, delta_m))}")
                    plane_hits += 1

                    vstat = set_vstat(k, i1, i2, i3, i4, plane_hits)
                    # _ = gcut(vstat, r[k], k)
                    cutf, s2 = fcut(vstat, m, r[k], s2_prev)
                    vstat_cutf = np.hstack([vstat, cutf])
                    # print(f"vstat = {vstat_cutf}, dcut ({dcut})")
                    if cutf > dcut and k != 0:
                        # vstat_cutf_ok = vstat_cutf.copy()
                        continue  # Continues going up in planes
                    else:
                        if vstat_cutf[-1] > dcut:
                            m_stat = np.vstack((m_stat, vstat_cutf))
                        break  # It takes another hit configuration and saves vstat in m_stat
    #             break
    #         break
    #     break
    # break

saeta = m_stat[:, 13:-1]
plot_rays(mdat, name=f"Generated || dcut = {dcut}", cells=True, mtrack=mtrk, mrec=saeta)
plot_rays(m_stat, name="mstat")

# TODO: Estudio de eficiencia:
#  NTRK = 1 -> 1000 lanzamientos -> distintos dcut -> Número de reconstruídas
#  NTRK = 2 -> 1000 lanzamientos -> distintos dcut -> Número de reconstruídas
#  ...
#  ...
#  Representar número de trazas reconstruídas sobre generadas, frente a dcut

# TODO: Cambiar en plot_rays:
#  - Saeta generada: linea con rayitas
#  - Saeta reconstruida: línea contínua
#  - Trazas mdat: línea punteada
#  Corregir aceptancia (ajustar el tamaño de la grafica al tamaño de los planos)

# TODO: Lluvias a distintas alturas (Preguntar a Hans)