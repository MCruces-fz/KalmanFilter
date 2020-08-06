# -*- coding: utf-8 -*-
"""
Created on Fri 31 July 10:47 2020

mcsquared.fz@gmail.com
miguel.cruces@rai.usc.es

@author:
Miguel Cruces

"""
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# from numpy.linalg import inv
# import matplotlib.pyplot as plt
from scipy import stats

from const import *
from kalmanFilter import GenerateInputData, plot_rays, diag_matrix, plot_vec

np.set_printoptions(formatter={'float': '{:.3f}'.format})
np.random.seed(16)
'''
Seed with first vertical line (NTRACK = 1) -> 34
Seed with 5 tracks (NTRACK = 5) -> 17
'''


class KalmanTRAG:

    def __init__(self, m_det):
        self.mdet = m_det

        self.mdet_xy = np.vstack([m_det[:, 0],  # Hits per plane
                                  m_det[:, 1] * WCX - (WCX / 2),  # Y [mm]
                                  m_det[:, 2] * WCY - (WCY / 2),  # X [mm]
                                  m_det[:, 3]  # Time [ps]
                                  ]).T  # mdet1 with x & y in mm

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
        return self.mdet_xy[k, ini:fin + 1]

    def main(self):

        C0 = diag_matrix(NPAR, [1 / WX, VSLP, 1 / WY, VSLP, 1 / WT, VSLN])
        V = diag_matrix(NDAC, [SIGX ** 2, SIGY ** 2, SIGT ** 2])
        dcut = 0.995
        Chi2 = 0

        iplan1, iplan2, iplan3, iplan4 = 0, 1, 2, 3  # Index for planes T1, T2, T3, T4 respectively
        ncel1, ncel2, ncel3, ncel4 = self.mdet[:, 0].astype(np.int)  # Nr. of hits in each plane

        for i4 in range(ncel4):
            for i3 in range(ncel3):
                for i2 in range(ncel2):
                    for i1 in range(ncel1):
                        print('Plane 4 (index: 3), z = 0')
                        print(f'i4: {i4}, i3: {i3}, i2: {i2}, i1: {i1}')

                        # Step 1. - INITIALIZATION
                        kx4, ky4, kt4, x0, y0, t0 = self.set_params(iplan4, i4)
                        r0 = [x0, 0, y0, 0, t0, - SC]  # Hypothesis
                        self.r[iplan4, i4] = r0
                        self.C[iplan4, i4] = C0
                        print(f'r0 = {self.r[iplan4, i4]}')
                        # print(f'C0 = {self.C[iplan4, i4]}')

                        plane_hits = 1

                        for k, zk in reversed(list(enumerate(VZI))[:-1]):  # iplanN; z / mm
                            print(f'Plane {k + 1} (index: {k}), z = {zk}')

                            # Step 2. - PREDICTION
                            zi = VZI[k + 1]  # Lower plane, higher index
                            zf = VZI[k]  # This plane
                            dz = zf - zi
                            print(f'dz = {zf} - {zi} = {dz}')

                            _, xp, _, yp, _, _ = self.r[k + 1, i4]
                            ks = np.sqrt(1 + xp ** 2 + yp ** 2)

                            F = diag_matrix(NPAR, [1] * NPAR)  # Identity 6x6
                            F[0, 1] = dz
                            F[2, 3] = dz
                            F[4, 5] = ks * dz
                            print(f'r{k + 2} = {self.r[k + 1, i4]}')
                            self.rp[k, i3] = np.dot(F, self.r[k + 1, i4])
                            self.Cp[k, i3] = np.dot(F, np.dot(self.C[k + 1, i4], F.T))
                            print(f'F = {F}')
                            print(f'rp{k + 1} = {self.rp[k, i3]}')
                            plot_vec(self.r[k + 1, i4], name=f'r{k + 2}')
                            plot_vec(self.rp[k, i3], name=f'rp{k + 1}')

                            # Step 3. - PROCESS NOISE
                            self.rn[k, i3] = self.rp[k, i3]
                            self.Cn[k, i3] = self.Cp[k, i3]  # + Q (random matrix)

                            # Step 4. - FILTRATION
                            # x0n, xpn, y0n, ypn, t0n, s0n = self.rn[k, i3]
                            m = self.m_coord(k, i3)  # Measurement
                            print(f'Measurement m{k + 1} = ', m)
                            # ksn = np.sqrt(1 + xpn ** 2 + ypn ** 2)

                            # Jacobian || I(NDACxNPAR): Parameters (NPAR dim) --> Measurements (NDAC dim)
                            H = np.zeros([NDAC, NPAR])
                            rows = range(NDAC)
                            cols = range(0, NPAR, 2)
                            H[rows, cols] = 1
                            # H[0, 1] = zf
                            # H[1, 3] = zf
                            # H[2, 1] = - s0n * xpn * zf / ksn
                            # H[2, 3] = - s0n * ypn * zf / ksn
                            # H[2, 5] = ksn * zf
                            # print(f'ksn * zi = {ksn:.3f} * {zi} = {ksn*zi:.3f}')

                            # Matrix K gain
                            H_Cn_Ht = np.dot(H, np.dot(self.Cn[k, i3], H.T))
                            weights = np.linalg.inv(V + H_Cn_Ht)
                            K = np.dot(self.Cn[k, i3], np.dot(H.T, weights))

                            # New rk vector
                            print(f'rn{k + 1} before H: ', self.rn[k, i3])
                            mr = np.dot(H, self.rn[k, i3])
                            print(f'rn{k + 1} after H: mr = ', np.dot(H, self.rn[k, i3]))
                            delta_m = m - mr
                            print(f'delta_m{k + 1} (before K):', delta_m)
                            print(f'K{k + 1} =', K)
                            delta_r = np.dot(K, delta_m)
                            print(f'delta_r{k + 1} (after K):', delta_r)
                            self.r[k, i3] = self.rn[k, i3] + delta_r
                            print(f'r{k + 1} final: ', self.r[k, i3])
                            plot_vec(self.r[k, i3], name=f'r{k + 1}')

                            # New Ck matrix
                            self.C[k, i3] = self.Cn[k, i3] - np.dot(K, np.dot(H, self.Cn[k, i3]))

                            # Chi2
                            Chi2 += np.dot(delta_m.T, np.dot(weights, delta_m))
                            print(f'Chi2 = {Chi2}')

                            plane_hits += 1
                            k_vec = np.zeros(NDAC * NPLAN)
                            idx = [[3, i4], [2, i3], [1, i2], [0, i1]]  # FIXME: Automatico al quitar los loops
                            k_list = []
                            for plane, hit in idx[:-(k+1)]:
                                kx, ky, kt, _, _, _ = self.set_params(plane, hit)
                                k_list.extend([kx, ky, kt])
                            # print(f'(k_vec dim: {k_vec.shape}) k_list = {k_list}')
                            k_vec[:len(k_list)] = k_list
                            print(f'(len {k_vec.shape}) k_vec = {k_vec}')
                            vstat = np.hstack([plane_hits, k_vec, self.r[k, i3]])
                            cutf = self.fcut(vstat, self.r[k, i3], self.C[k, i3], m)
                            vstat = np.hstack([vstat, cutf])
                            print(f'CUTF: {cutf} ?> DCUT: {dcut}')
                            if cutf > dcut:
                                continue
                            else:
                                break

                        # kx4, ky4, kt4, _, _, _ = self.set_params(3, i4)
                        # kx3, ky3, kt3, _, _, _ = self.set_params(2, i3)
                        # kx2, ky2, kt2, _, _, _ = self.set_params(1, i2)
                        # kx1, ky1, kt1, _, _, _ = self.set_params(0, i1)

        #                 vr4 = self.r[0, 0]
        #                 mstat = np.hstack([plane_hits, kx4, ky4, kt4, kx3, ky3, kt3, kx2, ky2, kt2, kx1, ky1, kt1, vr4, Chi2])
        # return mstat

    def fcut(self, vstat, vr, mC, vm):  # Function that returns quality factor
        bm = 0.2  # beta  min
        cmn = bm * C
        smx = 1 / cmn
        ndat = vstat[0] * NDAC
        ndf = ndat - NPAR  # Degrees of Freedom

        xd, yd, td = vm

        x0, _, y0, _, t0, s0 = vr

        sigx = np.sqrt(mC[0, 0])
        sigy = np.sqrt(mC[2, 2])
        sigt = np.sqrt(mC[4, 4])

        # if s0 < 0 or s0 > smx:
        print(f'Inside fcut: {s0} > 0 or {s0} < - {smx}')
        if s0 > 0 or s0 < - smx:
            print('CUTF = 0')
            cutf = 0
        else:
            print('CUTF > 0')
            if ndf > 0:
                s2 = ((xd - x0) / sigx) ** 2 + ((yd - y0) / sigy) ** 2 + ((td - t0) / sigt) ** 2
                cutf = stats.chi2.sf(s2, ndf)  # Survival function
                print(f'cutf (chi2) = {cutf}; s2 = {s2}; ndf = {ndf}')
            elif not ndf:
                cutf = 1
                print(f'cutf (= 1) = {cutf}')
            else:
                print(f'WARNING! ndf = {ndf}')
                cutf = np.nan

        return cutf


if __name__ == "__main__":
    #                   x  y  Time
    m_dat1 = np.array([[8, 2, 1000,
                        8, 1, 4000,
                        8, 1, 5000,
                        8, 0, 7000]])  # NTRACK = 5; Ray 1; seed 16

    #                   x  y  Time
    m_dat2 = np.array([[3, 4, 1000,
                        4, 4, 4000,
                        4, 4, 5000,
                        5, 4, 7100]])  # NTRACK = 5; Ray 4; seed 16

    #                   x  y  Time
    m_dat3 = np.array([[3, 3, 1000,
                        2, 2, 4000,
                        2, 2, 5000,
                        2, 1, 7100]])  # NTRACK = 5; Ray 3; seed 16

    GI = GenerateInputData()  # m_dat=m_dat3)

    mdat1 = GI.mdat
    mdet1 = GI.mdet

    plot_rays(mdat1, name='mdat')

    KT = KalmanTRAG(mdet1)
    mdet_xy = KT.mdet_xy
    r = KT.r
    C = KT.C
    rp = KT.rp
    Cp = KT.Cp
    mstat = KT.mstat
