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
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
import time

# ========================================================================== #
# ======= I N I T I A L   V A L U E S --- C O N F I G U R A T I O N ======== #
# ========================================================================== #

rd_seed: int or None = None
""" Choose an integer seed for numpy random generator, or keep it random with 
'None' """

single_run = True
kfcut: float = 0.6
ttcut: float = 1e-5
do_efficiency = False

if_repr: bool = True
""" Set if shows the 3D representation of rays on the detector """

if_final_prints: bool = False
""" Set if print final data """

if_save_diff: bool = False
"""
Set if save differences between parameters of the generated and reconstructed 
SAETAs,
    Sgen = [X0g, XPg, Y0g, YPg, T0g, S0g]
    Srec = [X0r, XPr, Y0r, YPr, T0r, S0r]
on 'saetas_file.csv'
(X0r - X0g), (XPr - XPg), (Y0r - Y0g), (YPr - YPg), (T0r - T0g), (S0r - S0g)
[..., ..., ..., ..., ..., ..., ..., ..., ...]
on append mode.
"""

np.set_printoptions(formatter={'float': '{:.3f}'.format})

# ========================================================================== #
# ============= K A L M A N   F I L T E R   F U N C T I O N S ============== #
# ========================================================================== #

# Randomize if rd_seed is an integer seed
if rd_seed is not None:
    np.random.seed(rd_seed)

if if_repr:
    plt.close("all")


def diag_matrix(dim: int, diag: list):
    """
    Create squared k_mat of dimXdim dimension with diag in the diagonal.

    :param dim: Quantity of rows/columns.
    :param diag: String of length dim with the diagonal values.
    :return: Squared k_mat of dimXdim dimension with diag in the diagonal.
    """
    arr = np.zeros([dim, dim])
    row, col = np.diag_indices(arr.shape[0])
    arr[row, col] = np.asarray(diag)
    return arr


def gene_tracks(all_tracks_in: bool = True, ntrack=NTRACK):
    """
    It generates random parameters to construct the tracks as Saetas. If the
    track doesn't enter in the detector, it is deleted from the list.

    :param all_tracks_in: True if force nt == NTRACKS or False if nt <= NTRACKS
        randomly deleting outisders.
    :param ntrack: Number of generated tracks.

    :return mtgen: Matrix of generated tracks (initial saetas).
    :return nt: Total number of tracks in the detector
    """
    ctmx = np.cos(np.deg2rad(THMAX))  # Theta max angle cosine
    # lenz = abs(VZI[0] - VZI[-1])  # Distance from bottom to top planes
    it = 0  # Number of tracks actually
    mtgen = np.zeros([ntrack, NPAR])  # generated tracks k_mat
    i = 1
    while i <= ntrack:
        # Uniform distribution in cos(theta) and phi
        rcth = 1 - np.random.random() * (1 - ctmx)
        tth = np.arccos(rcth)  # theta random angle
        tph = np.random.random() * 2 * np.pi  # phi random angle

        X0 = np.random.random() * LENX
        Y0 = np.random.random() * LENY
        T0 = (0.5 + np.random.random()) * TINI
        # T0 = TINI
        S0 = SINI

        # Director Cosines
        cx = np.sin(tth) * np.cos(tph)
        cy = np.sin(tth) * np.sin(tph)
        cz = np.cos(tth)
        XP = cx / cz  # projected slope in the X-Z plane
        YP = cy / cz  # projected slope in the Y-Z plane

        # Coordinate where would the particle come out
        xzend = X0 + XP * LENZ
        yzend = Y0 + YP * LENZ

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
    :return: m_dat (cell indices k_mat) and m_dpt (cell central
        positions k_mat).
    """
    v_dat = np.zeros(NPLAN * NDAC)  # Digitalizing tracks vector
    v_dpt = np.zeros(NPLAN * NDAC)  # Vector with impact point
    m_dat = np.zeros(NPLAN * NDAC)  # Detector data k_mat
    m_dpt = np.zeros(NPLAN * NDAC)  # Impact point
    nx = 0
    zt = VZ1[0]  # Z top
    for it in range(nt):
        x0, xp, y0, yp, t0, s0 = mtgen[it, :]  # dz = np.cos(th)

        it = 0
        for ip in range(NPLAN):
            zi = VZ1[ip]  # current Z
            dz = zi - zt  # dz >= 0

            xi = x0 + xp * dz
            yi = y0 + yp * dz
            ks = np.sqrt(1 + xp ** 2 + yp ** 2)
            ti = t0 + ks * s0 * dz  # Time Flies (dz > 0)

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


def matrix_det(m_dat):
    """
    Creates a k_mat similar to TRAGALDABAS output data

    :param m_dat: Matrix of generated and digitized tracks.
    :return: Equivalent k_mat to m_data, in TRAGALDABAS format.
    """
    if np.all(m_dat == 0):  # Check if mdat is all zero
        raise Exception('No tracks available! Matrix mdat is all zero ==> Run '
                        'the program Again because actual random seed is not '
                        f'valid for {NTRACK} number of tracks')
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
    It sets the transport k_mat between both planes separated by dz

    :param ks: sqrt( 1 + XP**2 + YP**2)
    :param dz: distance between planes
    :return: Transport function (k_mat | Numpy array)
    """
    F = diag_matrix(NPAR, [1] * NPAR)  # Identity 6x6
    F[0, 1] = dz
    F[2, 3] = dz
    F[4, 5] = ks * dz  # - ks * dz
    return F


def set_jacobi():
    #  zf, rn):
    # Vector with noise: rn = (X0n, XPn, Y0n, YPn, T0n, S0n)
    """
    Jacobian || I(NDACxNPAR): Parameters (NPAR dim) --> Measurements (NDAC dim)

    :return: Jacobi k_mat H
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


def set_vstat(k: int, i1: int, i2: int, i3: int, i4: int, plane_hits: int, r):
    """
    It sets the array:
    [ Nbr. hits, 0, 0, 0, ..., kx1, ky1, kt1, X0, XP, Y0, YP, T0, S0 ]
    """
    idx = [[0, i1], [1, i2], [2, i3], [3, i4]]
    k_list = []
    for plane, hit in reversed(idx[k:]):
        kx, ky, kt, _, _, _ = set_params(plane, hit)
        # k_list.extend([kx, ky, kt])
        k_list = [kx, ky, kt] + k_list
    k_vec = np.zeros(NDAC * NPLAN)
    # k_vec[:len(k_list)] = k_list
    k_vec[-len(k_list):] = k_list
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

    s2 = s2_prev + \
         (xd - x0) ** 2 / sigx2 + \
         (yd - y0) ** 2 / sigy2 + \
         (td - t0) ** 2 / sigt2

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
    return cut_f, s2


def set_mKgain(H, Cn, V):
    """
    It sets the K k_mat of gain and weights.

    :param H: Jacobi k_mat
    :param Cn: Noised uncertainty k_mat.
    :param V: Error k_mat.
    :return: K gain k_mat and weights.
    """
    H_Cn_Ht = np.dot(H, np.dot(Cn, H.T))
    wghts = np.linalg.inv(V + H_Cn_Ht)
    K = np.dot(Cn, np.dot(H.T, wghts))
    return K, wghts


def plot_saetas(vector, fig_id: int or str or None = None,
                plt_title=None, lbl: str = 'Vector', grids: bool = False,
                frmt_color: str = "green", frmt_marker: str = "--", prob_s=None):
    """
    Config Function for plot any SAETA with 6 parameters

    :param vector: The SAETA vector [X0, XP, Y0, YP, T0, S0]
    :param fig_id: Identification for the plot window
    :param plt_title:  Title for the plot
    :param lbl: Label for the SAETA
    :param grids: Set cell grids (higher CPU requirements, not recommendable)
    :param frmt_color: Format color for the SAETA representation
    :param frmt_marker: Format marker for the SAETA representation
    :param prob_s: value with alpha to fade SAETA.
    """
    # Plot configuration
    if fig_id is None:
        fig_id = 'State Vectors'
    fig = plt.figure(fig_id)
    ax = fig.gca(projection='3d')
    if plt_title is not None:
        ax.set_title(plt_title)
    ax.set_xlabel('X axis / mm')
    ax.set_ylabel('Y axis / mm')
    ax.set_zlabel('Z axis / mm')
    ax.set_xlim([0, LENX])
    ax.set_ylim([0, LENY])
    ax.set_zlim([VZ0[-1], VZ0[0]])

    # Unpack values
    x0, xp, y0, yp, t0, s0 = vector

    # Definition of variables
    z0 = VZ0[0]  # Detector Top Height
    z1 = VZ0[-1]  # Detector Bottom Height
    dz = z0 - z1  # Detector Height
    x1 = xp * dz
    y1 = yp * dz

    # Plot Vector
    x = np.array([x0, x0 + x1])
    y = np.array([y0, y0 + y1])
    z = np.array([z0, z1])
    if prob_s is not None:
        if 1 >= prob_s >= 0.9:
            frmt_color = "#FF0000"
        elif 0.9 > prob_s >= 0.6:
            frmt_color = "#FF5000"
        elif 0.6 > prob_s >= 0.3:
            frmt_color = "#FFA000"
        elif 0.3 > prob_s >= 0:
            frmt_color = "#FFF000"
        else:
            raise Exception(f"Ojo al dato: Prob = {prob_s}")
    ax.plot(x, y, z, linestyle=frmt_marker, color=frmt_color, label=lbl)
    ax.legend(loc='best')

    # Plot cell grid
    if grids:
        for zi in [-7000]:
            for yi in np.arange(-0.5, 10.5 + 1):
                for xi in np.arange(-0.5, 12.5 + 1):
                    plt.plot([-0.5, 12.5], [yi, yi], [zi, zi], 'k', alpha=0.1)
                    plt.plot([xi, xi], [-0.5, 10.5], [zi, zi], 'k', alpha=0.1)
    ax.legend(loc='best')
    plt.show()


def plot_hit_ids(k_vec, fig_id: str = None, plt_title: str or None = None,
                 digi_trk: bool = True, cells: bool = True,
                 lbl: str = 'Digitized', frmt_color: str = "green", frmt_marker: str = ":"):
    """
    Config Function for plot any set of hits

    :param k_vec: Set of hits
    :param fig_id: Identification for the plot window
    :param plt_title: Title for the plot
    :param digi_trk: Set if reconstructed digitized track is shown
    :param cells: Set if hit cell squares are shown
    :param lbl: Label for the SAETA
    :param frmt_color: Format of color for the SAETA representation
    :param frmt_marker: Format of marker for the SAETA representation
    """
    # Set Plot - Initial Config
    if fig_id is None:
        fig_id = plt_title
    fig = plt.figure(fig_id)
    ax = fig.gca(projection='3d')
    if plt_title is not None:
        ax.set_title(plt_title)
    ax.set_xlabel('X axis / mm')
    ax.set_ylabel('Y axis / mm')
    ax.set_zlabel('Z axis / mm')
    ax.set_xlim([0, LENX])
    ax.set_ylim([0, LENY])
    ax.set_zlim([VZ0[-1], VZ0[0]])

    x = k_vec[np.arange(0, 12, 3)] * WCX
    y = k_vec[np.arange(1, 12, 3)] * WCY

    if cells:
        for ip in range(NPLAN):
            p = Rectangle(xy=(x[ip] - 0.5 * WCX, y[ip] - 0.5 * WCY),
                          width=WCX, height=WCY, alpha=0.5,
                          facecolor='#AF7AC5', edgecolor='#9B59B6', fill=True)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=VZ0[ip], zdir="z")

    if digi_trk:
        ax.plot(x, y, VZ0, linestyle=frmt_marker, color=frmt_color, label=lbl)
    ax.plot(x, y, VZ0, 'k.', alpha=0.9)

    ax.legend(loc='best')
    plt.show()
    # fig.show()


def plot_detector(k_mat=None, fig_id=None, plt_title='Matrix Rays',
                  cells: bool = False, mtrack=None, mrec=None, prob_ary=None):
    """
    Config function for plot sets of hits and SAETAs

    :param k_mat: Matrix with all hits indices and times
    :param fig_id: Identification for the plot window
    :param plt_title: Title for the plot
    :param cells: Set if hit cell squares are shown
    :param mtrack: Array with all SAETAs generated
    :param mrec: Array with all SAETAs reconstructed
    :param prob_ary: Array with probabilities sorted by tracks order.
    """
    # Set Plot - Initial Config
    if fig_id is None:
        fig_id = plt_title
    fig = plt.figure(fig_id)
    ax = fig.gca(projection='3d')
    ax.set_title(plt_title)
    ax.set_xlabel('X axis / mm')
    ax.set_ylabel('Y axis / mm')
    ax.set_zlabel('Z axis / mm')
    ax.set_xlim([0, LENX])
    ax.set_ylim([0, LENY])
    ax.set_zlim([VZ0[-1], VZ0[0]])

    # Plot Generated Tracks (SAETAs)
    if mtrack is not None:
        for trk in range(mtrack.shape[0]):
            plot_saetas(mtrack[trk], fig_id=fig_id,
                        lbl=f'Gene. {trk + 1}', frmt_color='#3498DB', frmt_marker='--')

    # Plot Digitized Tracks (Hits By Indices)
    if k_mat is not None:
        for trk in range(k_mat.shape[0]):
            plot_hit_ids(k_mat[trk], fig_id=fig_id,
                         lbl=f'Digi. {trk + 1}', frmt_color='#196F3D', frmt_marker=':', cells=cells)

    # Plot Reconstructed Tracks (SAETAs)
    if mrec is not None:
        for rec in range(mrec.shape[0]):
            plot_saetas(mrec[rec], fig_id=fig_id,
                        lbl=f'Reco. {rec + 1}', frmt_color='b', frmt_marker='-',
                        prob_s=prob_ary[rec])

    plt.show()


# ========================================================================== #
# ================= T I M   T R A C K   F U N C T I O N S ================== #
# ========================================================================== #

def v_g0_tt(vs, z):
    """
    Sets the g0 value

    :param vs: State vector (SAETA)
    :param z: Height of the current plane of the detector
    """
    vg0 = np.zeros(3)
    _, XP, _, YP, _, S0 = vs
    ks = np.sqrt(1 + XP ** 2 + YP ** 2)
    vg0[2] = - S0 * (XP ** 2 + YP ** 2) * z / ks
    return vg0


'''
def m_K_a_tt(vs, z, vw, vdat):
    """
    It calculates the K k_mat and the new state vector.
    NPAR =  6 parameters version: X0, XP, Y0, YP, T0, S0

    :param vs: State vector input
    :param z: Height in mm of the plane
    :param vw: Vector of cell widths
    :param vdat: Vector of measured data in mm
    :return: K k_mat and vector for a pad plane
    """
    mk = np.zeros([NPAR, NPAR])
    va = np.zeros(NPAR)
    xp = vs[1]
    yp = vs[3]
    # t0   = vs[4]
    s0 = vs[5]
    ks2 = 1 + xp * xp + yp * yp  # slope factor
    wx = vw[0]
    wy = vw[1]
    wt = vw[2]
    dx = vdat[0]
    dy = vdat[1]
    dt = vdat[2]

    mk[0, 0] = wx
    mk[0, 1] = z * wx
    mk[1, 1] = z ** 2 * (wx + wt * (1 / ks2) * xp * xp * s0 * s0)
    mk[1, 3] = z ** 2 * wt * (1 / ks2) * xp * yp * s0 ** 2
    mk[1, 4] = z * wt * (1 / np.sqrt(ks2)) * xp * s0
    mk[1, 5] = z ** 2 * wt * xp * s0
    mk[2, 2] = wy
    mk[2, 3] = z * wy
    mk[3, 3] = z ** 2 * (wy + wt * (1 / ks2) * yp * yp * s0 * s0)
    mk[3, 4] = z * wt * (1 / ks2) * yp * s0
    mk[3, 5] = z ** 2 * wt * yp * s0
    mk[4, 4] = wt
    mk[4, 5] = z * wt * np.sqrt(ks2)
    mk[5, 5] = z ** 2 * wt * ks2

    # Vai =
    va[0] = wx * dx
    va[1] = z * (wx * dx + wt * xp * s0 * (dt * (1 / np.sqrt(ks2)) + z * (1 / ks2) * (xp ** 2 + yp ** 2) * s0))
    va[2] = wy * dy
    va[3] = z * (wy * dy + wt * yp * s0 * (dt * (1 / np.sqrt(ks2)) + z * (1 / ks2) * (xp ** 2 + yp ** 2) * s0))
    va[4] = wt * (dt + z * (1 / np.sqrt(ks2)) * s0 * (xp ** 2 + yp ** 2))
    va[5] = z * wt * ks * (dt + z * (1 / np.sqrt(ks2)) * (xp ** 2 + yp ** 2) * s0)

    # Por ser simetrica, mK=mK' (traspuesta)
    mk = mk + mk.T - np.diag(mk.diagonal())

    return mk, va
'''


def set_mG_tt(saeta, zi):
    """
    Jacobian k_mat

    :param saeta: State vector.
    :param zi: Height of the plane.
    :return: Jacobian Matrix
    """
    mG = np.zeros([NDAC, NPAR])

    X0, XP, Y0, YP, T0, S0 = saeta
    ks = np.sqrt(1 + XP ** 2 + YP ** 2)
    ksi = 1 / ks

    mG[0, 0] = 1
    mG[0, 1] = zi
    mG[1, 2] = 1
    mG[1, 3] = zi
    mG[2, 1] = S0 * XP * zi * ksi
    mG[2, 3] = S0 * YP * zi * ksi
    mG[2, 4] = 1
    mG[2, 5] = ks * zi
    return mG


def set_K_tt(saeta, zi, mW):
    """
    Calculates the K k_mat Gain

    :param saeta: State vector
    :param zi: Height of the plane
    :param mW: Weights Matrix diagonal (WX, WY, WT)
    :return: K k_mat = mG.T * mW * mG.
    """
    mG = set_mG_tt(saeta, zi)  # mG: k_mat = partial m(s) / partial s
    mK = np.dot(mG.T, np.dot(mW, mG))
    return mK


def set_vstat_tt(mG, mW, vdat, vg0):
    d_g0 = vdat - vg0
    va_out = np.dot(mG.T, np.dot(mW, d_g0))
    return va_out


def tim_track_fit(v_stat):
    vw = np.array([WX, WY, WT])  # Vector of Weights
    mvw = diag_matrix(3, [WX, WY, WT])  # Wieght Matrix (inverse of the V diagonal k_mat)

    saeta = v_stat[13:-1]
    k_vector = v_stat[:13]

    vs = saeta  # m_stat[it, 13:-1]  # State vector
    mK = np.zeros([NPAR, NPAR])  # K k_mat initialization
    va = np.zeros(NPAR)  # Final state vector initialization
    so = 0  # So (Store soi values on loop)

    for ip in range(NPLAN):  # Loop over hits in each track from TOP
        zi = VZ1[ip]  # [0, 522, 902, 1739] mm
        ii = ip * 3 + 1  # Initial index to search values
        dxi = k_vector[ii] * WCX  # - (WCX / 2)
        dyi = k_vector[ii + 1] * WCY  # - (WCY / 2)
        dti = k_vector[ii + 2]
        vdat = np.array([dxi, dyi, dti])  # Measured data

        mKi = set_K_tt(vs, zi, mvw)
        mG = set_mG_tt(vs, zi)
        vg0 = v_g0_tt(vs, zi)
        vai = set_vstat_tt(mG, mvw, vdat, vg0)

        mK += mKi
        va += vai
        # print(f"vdat = {vdat}\n vg0 = {vg0}")
        so += np.dot((vdat - vg0).T, np.dot(mvw, (vdat - vg0)))  # soi values

    mK = np.asmatrix(mK)  # K k_mat (symmetric)
    mErr = mK.I  # Error k_mat (symmetric)

    va = np.asmatrix(va).T  # Vertical Measurement Vector
    vsol = np.dot(mErr, va)  # SEA equation

    sks = float(np.dot(vsol.T, np.dot(mK, vsol)))  # s'·K·s
    sa = float(np.dot(vsol.T, va))  # s'·a
    S = sks - 2 * sa + so  # S = s'·K·s - 2s'·a + So
    # print(f"S = sks - 2*sa + so = {sks:.3f} - 2*{sa:.3f} + {so:.3f} = {S:.3f}")

    DoF = NPLAN * NDAC - NPAR  # Degrees of Freedom
    prob = stats.chi2.sf(S, DoF)
    # print(f"prob = {prob:.3e} EXP")
    # print(f"prob = {prob:.4f}  FLOAT")

    # mtrec = np.vstack((mtrec, vsol))
    vsol = np.asarray(vsol.T)[0]  # Make it a normal array again
    return vsol, prob


def kalman_filter_find(mdet, dcut=kfcut, tcut=ttcut):
    """
    Main Finding Function using Kalman Filter Algorithm

    :param mdet: Matrix with columns: (nhits, kx, ky, time)
    :param dcut:
    :param tcut:
    :return:
        - mstat: Hello
        - mtrec: World!
    """
    r = np.zeros([NPLAN, NPAR])  # Vector (parameters); dimension -> Number of
    # Planes x maximum hits x parameters
    C = np.zeros([NPLAN, NPAR, NPAR])  # Error Matrices

    rp = np.zeros(r.shape)  # Projected vector and matrices
    Cp = np.zeros(C.shape)

    rn = np.zeros(r.shape)  # UNUSED projected vectors with noises
    Cn = np.zeros(C.shape)

    C0 = diag_matrix(NPAR, [5 / WX, 50 * VSLP, 5 / WY, 50 * VSLP, 5 / WT, 10 * VSLN])  # Error k_mat
    # C0 = diag_matrix(NPAR, [1 / WX, VSLP, 1 / WY, VSLP, 1 / WT, VSLN])  # Error k_mat
    V = diag_matrix(NDAC, [SIGX ** 2, SIGY ** 2, SIGT ** 2])
    # Chi2 = 0
    m_stat = np.zeros([0, 20])
    mtrec = np.zeros([0, 20])

    iplan1, iplan2, iplan3, iplan4 = 0, 1, 2, 3  # Index for planes T1, T2, T3, T4 respectively
    ncel1, ncel2, ncel3, ncel4 = mdet[:, 0].astype(np.int)  # Nr. of hits in each plane

    # ================== MAIN LOOP ================= #

    # iN is the index of the hit in the plane N
    for i4 in range(ncel4):
        for i3 in range(ncel3):
            for i2 in range(ncel2):
                for i1 in range(ncel1):
                    s2 = 0
                    hits = [i1, i2, i3, i4]  # Combination of chosen hit indices

                    # Step 1. - INITIALIZATION
                    kx4, ky4, kt4, x0, y0, t0 = set_params(iplan4, i4)
                    r0 = [x0, 0, y0, 0, t0, SC]  # Hypothesis
                    r[iplan4] = r0
                    C[iplan4] = C0
                    plane_hits = 1

                    # k: index of plane and zk: height of plane k in mm
                    for k, zk in list(enumerate(VZ1))[::-1][1:]:  # Iterates over [[2, 924], [1, 1304], [0, 1826]]
                        s2_prev = s2 + 0
                        # Step 2. - PREDICTION
                        zi = VZ1[k + 1]  # Lower plane, higher index
                        zf = VZ1[k]  # This plane
                        dz = zf - zi  # dz < 0; It goes up, against Z axis
                        hiti = hits[k + 1]  # hit index in lower plane
                        hitf = hits[k]  # hit index in upper plane

                        _, xp, _, yp, _, _ = r[k + 1]
                        ks = np.sqrt(1 + xp ** 2 + yp ** 2)

                        F = set_transport_func(ks, dz)
                        rp[k] = np.dot(F, r[k + 1])
                        Cp[k] = np.dot(F, np.dot(C[k + 1], F.T))

                        # Step 3. - PROCESS NOISE  [UNUSED YET]
                        rn[k] = rp[k]
                        Cn[k] = Cp[k]  # + Q (random k_mat)

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

                        # New Ck k_mat
                        C[k] = Cn[k] - np.dot(K, np.dot(H, Cn[k]))

                        plane_hits += 1
                        vstat = set_vstat(k, i1, i2, i3, i4, plane_hits, r)
                        cutf, s2 = fcut(vstat, m, r[k], s2_prev)
                        vstat_cutf = np.hstack([vstat, cutf])
                        # print(f"vstat = {vstat_cutf}, dcut ({dcut})")
                        if cutf > dcut and k != 0:
                            continue  # Continues going up in planes
                        else:
                            if vstat_cutf[-1] > dcut:
                                m_stat = np.vstack((m_stat, vstat_cutf))

                                # Tim Track Analysis (Track Fitting)
                                vs, prob = tim_track_fit(vstat_cutf)
                                if prob > tcut:
                                    k_vector = vstat_cutf[0:13]
                                    v_stat_tt = np.hstack((k_vector, vs, prob))
                                    mtrec = np.vstack((mtrec, v_stat_tt))
                            break  # It takes another hit configuration and saves vstat in m_stat
    to_delete = []
    for i in range(len(mtrec)):
        for j in range(i + 1, len(mtrec)):
            if np.all(mtrec[i, 1:4] == mtrec[j, 1:4]):
                if mtrec[i, -1] > mtrec[j, -1]:
                    # print(f"Deleted index {j} because {mtrec[j, -1]:.4f} < {mtrec[i, -1]:.4f}")
                    to_delete.append(j)
                else:
                    # print(f"Deleted index {i} because {mtrec[i, -1]:.4f} < {mtrec[j, -1]:.4f}")
                    to_delete.append(i)
    m_trec = np.delete(mtrec, to_delete, axis=0)
    return m_stat, m_trec


# ========================================================================== #
# ====================== G E N E - D I G I T - A N A ======================= #
# ========================================================================== #

if single_run:
    # ============== TRACKS GENERATION ============= #
    mtrk, nt = gene_tracks()
    """
    - mtrk --> Initial Saetas
    - nt ----> Number of tracks in the detector
    """

    # ================ DIGITIZATION ================ #
    mdpt, mdat = trag_digitization(nt, mtgen=mtrk)  #
    """
    Digitization for TRAGALDABAS detector
    - mdat --> (kx1, ky2, time1, kx2, ky2, time2, ...)  Indices of impact
    - mdpt --> ( X1,  Y1,    T1,  X2,  Y2,    T2, ...)  Real points of impact / mm
    """

    mdet = matrix_det(mdat)
    """
    Matrix with columns: (nhits, kx, ky, time)
    """
    mdet_xy = set_mdet_xy(mdet)
    """
    Matrix (mdet) with columns: (nhits, x [mm], y [mm], time)
    """

    # ================== ANALYSIS ================== #

    m_stat, mtrec = kalman_filter_find(mdet, dcut=kfcut, tcut=ttcut)

    saeta_kf = m_stat[:, 13:-1]
    saeta_tt = mtrec[:, 13:-1]

# ========================================================================== #
# ===================== R E P R E S E N T A T I O N S ====================== #
# ========================================================================== #

    if if_repr:
        prob_tt = mtrec[:, -1]
        prob_kf = m_stat[:, -1]
        k_mat_gene = mdat
        # plot_detector(fig_id=f"cut = {kfcut}", plt_title=f"Track Finding (KF)", cells=True,
        #               k_mat=k_mat_gene, mtrack=mtrk, mrec=saeta_kf, prob_ary=prob_kf)
        plot_detector(fig_id=f"cut = {ttcut}", plt_title=f"Track Fitting (TT)", cells=True,
                      k_mat=k_mat_gene, mtrack=mtrk, mrec=saeta_tt, prob_ary=prob_tt)
        k_mat_rec = mtrec[:, 1:13]
        # plot_detector(fig_id='Id-Rec', plt_title="Reconstructed by Indices", cells=True,
        #               k_mat=k_mat_rec)

    if if_final_prints:
        print("# ================ P R I N T S ================ #")
        print(f"Generated SAETAs:\n{mtrk}\n")
        print(f"Track Finding SAETAs:\n{saeta_kf}\n")
        print(f"Track Fitting SAETAs:\n{saeta_tt}")
        try:
            print(f"\nTrack Finding DIFFERENCES:\n{saeta_kf - mtrk} || Prob {mtrec[:, -1]}\n")
            print(f"Track Fitting DIFFERENCES:\n{saeta_tt - mtrk} || Prob {mtrec[:, -1]}")
        except ValueError:
            print(f"\nProb KF {m_stat[:, -1]}")
            print(f"Prob TT {mtrec[:, -1]}")
            pass
        print("# ============================================= #")

    if if_save_diff:
        with open("saetas_file.csv", "a+") as f:
            try:
                relative_saeta = saeta_tt[0] - mtrk[0]
                X0, XP, Y0, YP, T0, S0 = relative_saeta
                prb = mtrec[:, -1][0]
                row = f"{X0},{XP},{Y0},{YP},{T0},{S0},{prb}\n"
                f.write(row)
            except IndexError:
                print('IndexError: Wait, man...')
                pass

# ========================================================================== #
# ========================== E F F I C I E N C Y =========================== #
# ========================================================================== #

elif do_efficiency:

    nb_tracks = 3
    bins_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    all_bins = np.zeros([0, len(bins_list) - 1], dtype=np.uint16)

    print("Completed percentage of efficiency:")
    points = 100
    for cut in range(1, points):
        cut /= points
        kf_cut = cut
        tt_cut = 0
        n_rec = np.array([], dtype=np.uint8)
        print(f"{int(cut * 100)}%")
        for run in range(1000):
            np.random.seed(int(time.time() * 1e6) % 2 ** 32)
            mtrk, nt = gene_tracks(ntrack=nb_tracks)
            mdpt, mdat = trag_digitization(nt, mtgen=mtrk)
            mdet = matrix_det(mdat)
            mdet_xy = set_mdet_xy(mdet)
            m_stat, mtrec = kalman_filter_find(mdet, dcut=kf_cut, tcut=tt_cut)

            saeta_kf = m_stat[:, 13:-1]
            saeta_tt = mtrec[:, 13:-1]

            n_rec = np.append(n_rec, saeta_kf.shape[0])
        # plot histogram
        plt.figure(f"Cut {cut}")
        n, bins, _ = plt.hist(n_rec, bins=bins_list)
        mids = (bins[1:] + bins[:-1]) / 2
        mean = np.average(mids, weights=n)
        var = np.average((mids - mean) ** 2, weights=n)
        std = np.sqrt(var)
        plt.title(f"kf_cut: {kf_cut} | Mean: {mean:.3f}, Var: {var:.3f}, Std: {std:.3f}")
        # plt.show()
        plt.close(f"Cut {cut}")
        all_bins = np.vstack((all_bins, n))
    all_bins = all_bins.astype(np.uint16).T
    plt.matshow(all_bins)
    np.savetxt(f"all_bins_{nb_tracks}_tracks.txt", all_bins)

else:
    print("Ojo cuidao, atento a los Settings (if_single_run = do_efficiency = False)")

# TODO: Lluvias a distintas alturas (Preguntar a Hans)

# TODO: Create different branches:
#  - (kf_lineal) Kalman Filter Lineal
#  - (master) Kalman Filter Classes

