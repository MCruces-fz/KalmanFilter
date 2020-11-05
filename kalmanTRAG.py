# -*- coding: utf-8 -*-
"""
Created on Fri 31 July 10:47 2020

mcsquared.fz@gmail.com
miguel.cruces@rai.usc.es

@author:
Miguel Cruces

"""
# from kalmanFilter import GenerateTracks, diag_matrix
# from kalmanFilter import plot_detector, plot_vec
from scipy import stats
from const import *

# Numpy as np imported in const

np.set_printoptions(formatter={'float': '{:.3f}'.format})
# np.random.seed(16)
'''
Seed with first vertical line (NTRACK = 1) -> 34
Seed with 5 tracks (NTRACK = 5) -> 17
'''


def diag_matrix(dim: int, diag: list):
    """
    Create squared k_mat of dimXdim dimension with diag in the diagonal.

    :param dim: Quantity of rows/columns.
    :param diag: String of length dim with the diagonal values.
    :return: Squared k_mat of dimXdim dimension with diag in the diagonal.
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
    ctmx = np.cos(np.deg2rad(THMAX))  # theta_max cosine
    lenz = abs(VZI[0] - VZI[-1])  # Distance from bottom to top planes
    it = 0  # Number of tracks actually
    mtgen = np.zeros([NTRACK, NPAR])  # generated tracks k_mat
    i = 1
    while i <= NTRACK:
        # Uniform distribution in cos(theta) and phi
        rcth = 1 - np.random.random() * (1 - ctmx)
        tth = np.arccos(rcth)  # theta
        tph = np.random.random() * 2 * np.pi  # phi

        X0 = np.random.random() * LENX
        Y0 = np.random.random() * LENY
        T0 = TINI
        S0 = SINI

        cx = np.sin(tth) * np.cos(tph)  # Director Cosines
        cy = np.sin(tth) * np.sin(tph)
        cz = np.cos(tth)
        XP = cx / cz  # projected slope in the X-Z plane
        YP = cy / cz  # projected slope in the Y-Z plane

        # Coordinate where would the particle come out
        xzend = X0 - XP * lenz
        yzend = Y0 - YP * lenz

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

    :param nt: Number of tracks generated accross the detector.
    :param mtgen: Matrix of generated tracks
    :return: m_dat (cell indices k_mat) and m_dpt (cell central
        positions k_mat).
    """
    v_dat = np.zeros(NPLAN * NDAC)  # Digitalizing tracks vector
    v_dpt = np.zeros(NPLAN * NDAC)  # Vector with impact point
    m_dat = np.zeros(NPLAN * NDAC)  # Detector data k_mat
    m_dpt = np.zeros(NPLAN * NDAC)  # Impact point
    nx = 0
    for it in range(nt):
        x0, xp, y0, yp = mtgen[it, 0:4]  # dz = np.cos(th)

        it = 0
        for ip in range(NPLAN):
            zi = VZI[ip]  # current Z
            zt = VZI[0]  # Z top
            dz = zi - zt  # dz <= 0
            # print(f'dz = {zi} - {zt} = {dz}')
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


def matrix_det(m_dat):
    """
    Creates a k_mat similar to TRAGALDABAS output data

    :param m_dat: Matrix of generated and digitized tracks.
    :return: Equivalent k_mat to m_data, in TRAGALDABAS format.
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
    It sets the transport k_mat between both planes separated by dz

    :param ks: sqrt( 1 + XP**2 + YP**2)
    :param dz: distance between planes
    :return: Transport function (k_mat | Numpy array)
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
        mdet_xy[:, iX] = mdet_xy[:, iX] * WCX - (WCX / 2)  # X [mm]
        mdet_xy[:, iY] = mdet_xy[:, iY] * WCY - (WCY / 2)  # Y [mm]
    return mdet_xy


class KalmanTRAG:

    def __init__(self, m_det):
        self.mdet = m_det

        self.mdet_xy = set_mdet_xy(m_det)

        max_hits = int(max(self.mdet[:, 0]))
        self.r = np.zeros([NPLAN, max_hits, NPAR])
        self.C = np.zeros([NPLAN, max_hits, NPAR, NPAR])

        self.rp = np.zeros(self.r.shape)
        self.Cp = np.zeros(self.C.shape)

        self.rn = np.zeros(self.r.shape)
        self.Cn = np.zeros(self.C.shape)

        self.mstat = self.main()

    def set_params(self, iplanN: int, iN: int):
        """
        It sets parameters of indexes of cells and possitions respectively,
        taking data from self.mdet
        :param iN: Integer which defines the data index.
        :param iplanN: Integer which defines the plane index.
        :return: Parameters kxN, kyN, ktN, x0, y0, t0 where N is the plane (TN).
        """
        icel = 1 + iN * NDAC
        kxN, kyN, ktN = self.mdet[iplanN, icel:icel + NDAC]
        x0 = kxN * WCX - (WCX / 2)
        y0 = kyN * WCY - (WCY / 2)
        t0 = ktN
        return kxN, kyN, ktN, x0, y0, t0

    def m_coord(self, k, idn):
        ini = 1 + idn * NDAC
        fin = ini + NDAC
        return self.mdet_xy[k, ini:fin]

    def set_vstat(self, k: int, i1: int, i2: int, i3: int, i4: int, plane_hits: int):
        idx = [[0, i1], [1, i2], [2, i3], [3, i4]]
        k_list = []
        for plane, hit in reversed(idx[k:]):
            kx, ky, kt, _, _, _ = self.set_params(plane, hit)
            k_list.extend([kx, ky, kt])
        k_vec = np.zeros(NDAC * NPLAN)
        k_vec[:len(k_list)] = k_list
        vstat = np.hstack([plane_hits, k_vec, self.r[k, i3]])
        return vstat

    def main(self):

        C0 = diag_matrix(NPAR, [1 / WX, VSLP, 1 / WY, VSLP, 1 / WT, VSLN])  # Error k_mat
        V = diag_matrix(NDAC, [SIGX ** 2, SIGY ** 2, SIGT ** 2])
        dcut = 0.995
        Chi2 = 0

        m_stat = np.zeros([0, 15])

        iplan1, iplan2, iplan3, iplan4 = 0, 1, 2, 3  # Index for planes T1, T2, T3, T4 respectively
        ncel1, ncel2, ncel3, ncel4 = self.mdet[:, 0].astype(np.int)  # Nr. of hits in each plane

        for i4 in range(ncel4):
            for i3 in range(ncel3):
                for i2 in range(ncel2):
                    for i1 in range(ncel1):
                        # Step 1. - INITIALIZATION
                        kx4, ky4, kt4, x0, y0, t0 = self.set_params(iplan4, i4)
                        r0 = [x0, 0, y0, 0, t0, SC]  # Hypothesis
                        self.r[iplan4, i4] = r0
                        self.C[iplan4, i4] = C0
                        plane_hits = 1

                        for k, zk in reversed(list(enumerate(VZI))[:-1]):
                            # Step 2. - PREDICTION
                            zi = VZI[k + 1]  # Lower plane, higher index
                            zf = VZI[k]  # This plane
                            dz = zf - zi

                            _, xp, _, yp, _, _ = self.r[k + 1, i4]
                            ks = np.sqrt(1 + xp ** 2 + yp ** 2)

                            F = set_transport_func(ks, dz)
                            self.rp[k, i3] = np.dot(F, self.r[k + 1, i4])
                            self.Cp[k, i3] = np.dot(F, np.dot(self.C[k + 1, i4], F.T))

                            # Step 3. - PROCESS NOISE
                            self.rn[k, i3] = self.rp[k, i3]
                            self.Cn[k, i3] = self.Cp[k, i3]  # + Q (random k_mat)

                            # Step 4. - FILTRATION
                            m = self.m_coord(k, i3)  # Measurement

                            H = set_jacobi()

                            # Matrix K gain
                            H_Cn_Ht = np.dot(H, np.dot(self.Cn[k, i3], H.T))
                            weights = np.linalg.inv(V + H_Cn_Ht)
                            K = np.dot(self.Cn[k, i3], np.dot(H.T, weights))

                            # New rk vector
                            mr = np.dot(H, self.rn[k, i3])
                            delta_m = m - mr
                            delta_r = np.dot(K, delta_m)
                            self.r[k, i3] = self.rn[k, i3] + delta_r

                            # New Ck k_mat
                            self.C[k, i3] = self.Cn[k, i3] - np.dot(K, np.dot(H, self.Cn[k, i3]))

                            # Chi2
                            Chi2 += np.dot(delta_m.T, np.dot(weights, delta_m))
                            plane_hits += 1

                            vstat = self.set_vstat(k, i1, i2, i3, i4, plane_hits)
                            cutf = self.fcut(vstat, m, k, i3)
                            vstat_cutf = np.hstack([vstat, cutf])
                            if cutf > dcut:
                                continue
                            # else:
                            #     break

                        kx4, ky4, kt4, _, _, _ = self.set_params(3, i4)
                        kx3, ky3, kt3, _, _, _ = self.set_params(2, i3)
                        kx2, ky2, kt2, _, _, _ = self.set_params(1, i2)
                        kx1, ky1, kt1, _, _, _ = self.set_params(0, i1)

                        vr4 = self.r[0, 0]
                        m_stat = np.hstack([plane_hits,
                                            kx4, ky4, kt4, kx3, ky3, kt3, kx2, ky2, kt2, kx1, ky1, kt1,
                                            vr4, Chi2])
        return m_stat

    def fcut(self, vstat, vm, k, i3):  # Function that returns quality factor
        bm = 0.2  # beta  min
        cmn = bm * C
        smx = 1 / cmn
        ndat = vstat[0] * NDAC
        ndf = ndat - NPAR  # Degrees of Freedom

        xd, yd, td = vm

        x0, _, y0, _, t0, s0 = self.r[k, i3]

        sigx = np.sqrt(self.C[k, i3][0, 0])
        sigy = np.sqrt(self.C[k, i3][2, 2])
        sigt = np.sqrt(self.C[k, i3][4, 4])

        # if s0 < 0 or s0 > smx:
        # print(f'Inside fcut: {s0} > 0 or {s0} < - {smx}')
        if s0 > 0 or s0 < - smx:
            # print('CUTF = 0')
            cutf = 0
        else:
            # print('CUTF > 0')
            if ndf > 0:
                s2 = ((xd - x0) / sigx) ** 2 + ((yd - y0) / sigy) ** 2 + ((td - t0) / sigt) ** 2
                cutf = stats.chi2.sf(s2, ndf)  # Survival function
                # print(f'cutf (chi2) = {cutf}; s2 = {s2}; ndf = {ndf}')
            elif not ndf:
                cutf = 1
                # print(f'cutf (= 1) = {cutf}')
            else:
                # print(f'WARNING! ndf = {ndf}')
                cutf = np.nan

        return cutf


if __name__ == "__main__":
    # =============== GENERATE TRACKS ============== #
    mtrk, nt = gene_tracks()
    # mtrk --> Initial Saetas
    # nt ----> Number of tracks in the detector

    # ================ DIGITIZATION ================ #
    mdpt, mdat = trag_digitization(nt, mtgen=mtrk)  # Digitization for TRAGALDABAS detector
    # mdat --> (kx1, ky2, time1,   kx2, ky2, time2, ...)
    # mdpt --> (X1, Y1, T1,   X2, Y2, T2,   ...)  Real points of impact / mm

    mdet = matrix_det(mdat)  # (nhits, kx, ky, time)

    # =============== RECONSTRUCTION =============== #
    kTRAG = KalmanTRAG(mdet)
    mdet_xy = kTRAG.mdet_xy
    r = kTRAG.r
    C = kTRAG.C
    rp = kTRAG.rp
    Cp = kTRAG.Cp
    mstat = kTRAG.mstat

    print(f"mtrk  = {mtrk}  <--- Saeta generada")
    print(f"r     = {r[0]}  <--- Saeta reconstruída")
    print(f"Dr(%) = {abs(r[0] - mtrk) * 100/ mtrk}  <--- Diferencia en porcentaje de mtrk")

# TODO: Generar 1000 veces una traza para poder guardar los datos y ver las
#  dispersiones que hay entre los datos generados y los reconstruidos
