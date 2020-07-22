# -*- coding: utf-8 -*-
"""
Created on Sun 5 July 19:17:15 2020

mcsquared.fz@gmail.com
miguel.cruces@rai.usc.es

@author:
Miguel Cruces

"""

import numpy as np
from numpy.linalg import inv
from scipy import stats
import sys
from const import *

np.set_printoptions(formatter={'float': '{:.3f}'.format})
np.random.seed(17)


class GenerateInputData:
    """
    It generates the input data randomly.
    """

    def __init__(self):
        self.nt = 0

        # Data vectors
        vdx = np.zeros(NPLAN)
        vdy = np.zeros(NPLAN)
        vdt = np.zeros(NPLAN)

        mtrec = np.zeros([NTRACK, NPAR])  # reconstructed tracks matrix
        self.mtgen = np.zeros([NTRACK, NPAR])  # generated tracks matrix
        self.vdat = np.zeros(NPLAN * NDAC)  # digitalizing tracks vector
        self.vdpt = np.zeros(NPLAN * NDAC)  # vector with impact point
        self.mdat = np.zeros(NPLAN * NDAC)  # detector data matrix
        self.mdpt = np.zeros(NPLAN * NDAC)  # impact point

        self.tracks_generation()
        self.digitization()
        # self.mdat[0, 3:6] = [0, 0, 0]
        self.mdet = self.matrix_det()

    def tracks_generation(self):
        """
        It generates the parameters to construct the tracks as lines.
        """
        ctmx = np.cos(np.deg2rad(THMAX))  # theta_max cosine
        lenz = VZI[NPLAN - 1] - VZI[0]  # Distance from bottom to top planes
        it = 0  # Number of tracks actually

        for i in range(NTRACK):
            # Uniform distribution in cos(theta) and phi
            rcth = 1 - np.random.random() * (1 - ctmx)
            tth = np.arccos(rcth)  # theta
            tph = np.random.random() * 2 * np.pi  # phi

            x0 = np.random.random() * LENX
            y0 = np.random.random() * LENY
            t0 = TINI
            s0 = SINI

            cx = np.sin(tth) * np.cos(tph)  # cosenos directores
            cy = np.sin(tth) * np.sin(tph)
            cz = np.cos(tth)
            xp = cx / cz  # projected slope in the X-Z plane
            yp = cy / cz  # projected slope in the Y-Z plane

            # Coordinate where would the particle come out
            xzend = x0 + xp * lenz
            yzend = y0 + yp * lenz

            # We refer the coordinate to the detector center (xmid, ymid)
            xmid = xzend - (LENX / 2)
            ymid = yzend - (LENY / 2)

            # We check if the particle has entered the detector
            if np.abs(xmid) < (LENX / 2) and np.abs(ymid) < (LENY / 2):
                self.mtgen[it, :] = [x0, xp, y0, yp, t0, s0]
                it += 1
            else:
                continue

        self.nt = it  # number of tracks in the detector
        self.mtgen = self.mtgen[~(self.mtgen == 0).all(1)]

    def digitization(self):
        """
        It converts the parameters inside mtgen to discrete numerical
        values, which are the cell indices (mdat) and cell central
        positions (mdpt).
        """
        nx = 0
        for it in range(self.nt):
            x0, xp, y0, yp = self.mtgen[it, 0:4]  # dz = np.cos(th)

            it = 0
            for ip in range(NPLAN):
                zi = VZI[ip]
                xi = x0 + xp * zi
                yi = y0 + yp * zi
                ks = np.sqrt(1 + xp * xp + yp * yp)
                ti = TINI + ks * SC * zi
                # Position indices of the impacted cells (cell index)
                kx = np.int((xi + (WCX / 2)) / WCX)
                ky = np.int((yi + (WCY / 2)) / WCY)
                kt = np.int((ti + (DT / 2)) / DT) * DT
                # Cell position (distance)
                # xic = kx * WCX + (WCX / 2)
                # yic = ky * WCX + (WCX / 2)
                vpnt = np.asarray([xi, yi, ti])  # (X,Y,T) impact point
                vxyt = np.asarray([kx, ky, kt])  # impact index
                self.vdpt[it:it + NDAC] = vpnt[0:NDAC]
                self.vdat[it:it + NDAC] = vxyt[0:NDAC]
                it += 3
            self.mdpt = np.vstack((self.mdpt, self.vdpt))
            self.mdat = np.vstack((self.mdat, self.vdat))
            nx += 1
        self.mdpt = np.delete(self.mdpt, 0, axis=0)
        self.mdat = np.delete(self.mdat, 0, axis=0)

    def matrix_det(self):  # mdat -> mdet
        if np.all(self.mdat == 0):  # Check if mdat is all zero
            raise Exception('No tracks available! Matrix mdat is all zero.')
        ndac = 3
        ntrk, nplan = self.mdat.shape  # Number of tracks, number of plans
        nplan = int(nplan / ndac)
        ncol = 1 + ndac * ntrk  # One more column to store number of tracks
        mdet = np.zeros([nplan, ncol])  # TODO: Preguntar a Sara por qué estas vueltas.
        idat = 0
        for ip in range(nplan):
            idet = 0
            for it in range(ntrk):
                ideti = idet + 1
                idetf = ideti + ndac
                idatf = idat + ndac
                mdet[ip, ideti:idetf] = self.mdat[it, idat:idatf]
                if not np.all((mdet[ip, ideti:idetf] == 0)):  # checks if all are zero
                    mdet[ip, 0] += 1
                idet += ndac
            idat += ndac
        return mdet


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


class KalmanFilter:
    def __init__(self, m_det):

        self.mdet = m_det
        mdet_list = self.mdet.tolist()

        # Error matrix for vr
        self.mErr = diag_matrix(NPAR, [1 / WX, VSLP, 1 / WY, VSLP, 1 / WT, VSLN])

        self.mVd = diag_matrix(NDAC, [SIGX ** 2, SIGY ** 2, SIGT ** 2])  # Matrix V_d -> measurement uncertainties

        # self.mvr = np.zeros([4, 6])  # Initiation of state vectors matrix

        self.mstat = self.main()

    def main(self):
        ntrmx = 1  # Max number of tracks
        for ip in range(NPLAN):
            nceli = int(self.mdet[ip, 0])
            ntrmx *= nceli  # Total number of collisions in all planes

        nvar = NPAR + 2  # npar + 0 column + 1 for quality
        ncol = NPLAN * NDAC + nvar

        mstat = np.zeros([0, ncol])

        ncomb = ntrmx  # Number of possible combinations == Maximum number of possibles tracks

        dcut = 0.995  # Defined threshold to consider positives

        iplan1, iplan2, iplan3, iplan4 = 0, 1, 2, 3  # Index for planes T1, T2, T3, T4 respectively
        ncel1, ncel2, ncel3, ncel4 = self.mdet[:, 0].astype(np.int)  # Nr. of hits in each plane

        for i4 in range(ncel4):
            kx4, ky4, kt4, x0, y0, t0 = self.set_params(i4, iplan4)

            vr = np.asarray([x0, 0, y0, 0, t0, SC])  # We assume a normal state vector

            for i3 in range(ncel3):
                kx3, ky3, kt3, x0, y0, t0 = self.set_params(i3, iplan3)

                vdat = np.asarray([x0, y0, t0])

                vrp = vr  # save previous values
                mErrp = self.mErr

                vr2, mErr2 = self.fitkalman(vr, self.mErr, vdat, iplan3)

                phits = 2
                vstat = np.hstack([phits, kx4, ky4, kt4, kx3, ky3, kt3, 0, 0, 0, 0, 0, 0, vr2])
                cutf = self.fcut(vstat, vr2, mErr2, vdat)
                vstat = np.hstack([vstat, cutf])

                if cutf > dcut:
                    for i2 in range(ncel2):
                        kx2, ky2, kt2, x0, y0, t0 = self.set_params(i2, iplan2)

                        vdat = np.asarray([x0, y0, t0])

                        vrp2 = vr2
                        mErrp2 = mErr2

                        vr3, mErr3 = self.fitkalman(vr2, mErr2, vdat, iplan2)

                        phits = 3
                        vstat = np.hstack([phits, kx4, ky4, kt4, kx3, ky3, kt3, kx2, ky2, kt2, 0, 0, 0, vr3])
                        cutf = self.fcut(vstat, vr3, mErr3, vdat)
                        vstat = np.hstack([vstat, cutf])

                        if cutf > dcut:
                            for i1 in range(ncel1):
                                kx1, ky1, kt1, x0, y0, t0 = self.set_params(i1, iplan1)

                                vdat = np.asarray([x0, y0, t0])

                                vr4, mErr4 = self.fitkalman(vr3, mErr3, vdat, iplan1)

                                vrp3 = vr3
                                mErrp3 = mErr3

                                phits = 4
                                vstat = np.hstack(
                                    [phits, kx4, ky4, kt4, kx3, ky3, kt3, kx2, ky2, kt2, kx1, ky1, kt1, vr4])
                                cutf = self.fcut(vstat, vr4, mErr4, vdat)
                                vstat = np.hstack([vstat, cutf])

                                if cutf > dcut:
                                    # nr of planes hit, cells, saeta, fit quality
                                    mstat = np.vstack([mstat, vstat])
        return mstat

    def set_params(self, iN: int, iplanN: int):
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
        y0 = kyN * WCX - (WCY / 2)
        t0 = ktN
        return kxN, kyN, ktN, x0, y0, t0

    def fprop(self, vr, mErr, zi, zf):  # Transport function
        dz = zf - zi
        XP = vr[1]
        YP = vr[3]
        ks = np.sqrt(1 + XP * XP + YP * YP)

        mF = np.zeros([NPAR, NPAR])
        np.fill_diagonal(mF, 1)
        mF[0, 1] = dz
        mF[2, 3] = dz
        mF[4, 5] = ks * dz

        vr = np.dot(mF, vr)
        vr = np.asarray(vr)
        # Propagation of the error matrix
        mErr = np.dot(mF, np.dot(mErr, mF.T))

        return vr, mErr

    def jacobi(self, vr, zi):  # Jacobian matrix
        mH = np.zeros([NDAC, NPAR])

        XP = vr[1]
        YP = vr[3]
        S0 = vr[5]
        ks = np.sqrt(1 + XP * XP + YP * YP)
        ksi = 1 / ks

        mH[0, 0] = 1
        mH[0, 1] = zi
        mH[1, 2] = 1
        mH[1, 3] = zi
        mH[2, 1] = ksi * S0 * XP * zi
        mH[2, 3] = ksi * S0 * YP * zi
        mH[2, 4] = 1
        mH[2, 5] = ks * zi

        return mH

    def fpar2dat(self, vr, mErr, mH, zi, zf):  # Projection in the measurement space
        # Fitting model
        X0 = vr[0]
        Y0 = vr[2]
        T0 = vr[4]

        vdr = [X0, Y0, T0]

        ndac = 3
        npar = 6
        mid_s = np.zeros([ndac, npar])
        mid_s[0, 0] = 1
        mid_s[1, 2] = 1
        mid_s[2, 4] = 1

        vdr = np.dot(mid_s, np.dot(mH.T, vdr))

        mVr = np.dot(mH, np.dot(mErr, mH.T))

        return vdr, mVr

    def fdat2par(self, mVr, mVd, mVc, mErr, mH, zi, zf):  # Projection in the parameter space

        mWc = inv(mVc)  # weight matrix
        mKgain = np.dot(mErr, np.dot(mH.T, mWc))
        mWr = np.dot(mH.T, np.dot(mWc, mH))

        return mKgain, mWr

    def update(self, mKgain, vdd, mWr, vr, mErr):  # Update the state vector and error matrix

        dvr = np.dot(mKgain, vdd)
        mdE = np.dot(mErr, np.dot(mWr, mErr))
        vr = vr + dvr
        mErr = mErr - mdE

        return vr, mErr

    def fitkalman(self, vr, mErr, vdat, iplan):
        for ip in range(iplan + 1, iplan, -1):  # loop on planes
            zi = VZI[ip]
            zf = VZI[ip - 1]

            # Propagation step
            vrp = vr
            vr, mErr = self.fprop(vr, mErr, zi, zf)

            mH = self.jacobi(vr, zi)  # Jacobian matrix

            vdr, mVr = self.fpar2dat(vr, mErr, mH, zi, zf)  # Parameter  -> Measurument

            # new measurement
            ndac = 3
            vdi = np.zeros(ndac)
            vdi[0] = vdat[0]
            vdi[1] = vdat[1]
            vdi[2] = vdat[2]
            vdd = vdi - vdr  # Difference between measurement and expected data
            mVc = mVr + self.mVd  # Joint uncertainties matrix

            mKgain, mWr = self.fdat2par(mVr, self.mVd, mVc, mErr, mH, zi, zf)  # Meas. -> Proj.

            vr, mErr = self.update(mKgain, vdd, mWr, vr, mErr)  # New state vector

        return vr, mErr

    def fcut(self, vstat, vr, mErr, vdat):  # Function that returns quality factor
        bm = 0.2  # beta  min
        cmn = bm * C
        smx = 1 / cmn
        ndat = vstat[0] * NDAC
        ndf = ndat - NPAR

        xd, yd, td = vdat

        x0, _, y0, _, t0, s0 = vr

        sigx = np.sqrt(mErr[0, 0])
        sigy = np.sqrt(mErr[2, 2])
        sigt = np.sqrt(mErr[4, 4])

        if s0 < 0 or s0 > smx:
            cutf = 0
        else:
            if ndf != 0:
                s2 = ((xd - x0) / sigx) ** 2 + ((yd - y0) / sigy) ** 2 + ((td - t0) / sigt) ** 2
                cutf = stats.chi2.sf(s2, ndf)
            else:
                cutf = 1

        return cutf


if __name__ == "__main__":
    GI = GenerateInputData()

    mdpt1 = GI.mdpt
    mdat1 = GI.mdat
    mdet1 = GI.mdet

    KF = KalmanFilter(mdet1)

    mstat1 = KF.mstat
    mVd1 = KF.mVd
    mErr1 = KF.mErr
