# -*- coding: utf-8 -*-
"""
Created on Sun 5 July 19:17:15 2020

mcsquared.fz@gmail.com
miguel.cruces@rai.usc.es

@author:
Miguel Cruces

"""

#  REVIEW

#  GI filter for a pads detector
#  TRAGALDABAS Version
# *****************************
#   JA Garzon. labCAF / USC
#   - Abril 2020
#   2020 Abril. Sara Costa
# ******************************************************************** GENE
# It generates ntrack tracks from a charged particle and propagates them
# in the Z axis direction through nplan planes
##################################################################### DIGIT
# It simulates the digital answer in nplan planes of detectors, in which
# - the coordinates (nx,ny) of the crossed pad are determined
# - the flight time is determined integrating tint
# *******************************************************************GI
# - It reconstructs the track through the GI Filter
# *************************************************************** Comments
# Some coding criteria:
# - The variable names, follow, in general, some mnemonic rule
# - Names of vectors start with v
# - Names of matrixes start with m
# ********************************************************************
# Typical units:
# Mass, momentum and energy: MeV
# Distances in mm
# Time of ps
# ********************************************************************

# import numpy as np
from numpy.linalg import inv
from scipy import stats
import sys
from const import *

np.set_printoptions(formatter={'float': '{:.3f}'.format})
np.random.seed(42)


class GenerateInputData:
    """
    It generates the input data
    """

    def __init__(self):

        # # Constants
        # c = 0.3  # [mm/ps]
        # self.sc = 1 / c  # slowness associated with light speed
        # mele = 0.511
        # mmu = 105.6
        # mpro = 938.3
        #
        # # Data
        # mass = mmu
        # kene = 1000  # MeV, kinetic energy
        # ene = mass + kene
        # gamma = ene / mass
        # beta = np.sqrt(1 - 1 / (gamma * gamma))
        # betgam = beta * gamma
        # vini = beta * c  # initial velocity
        # sini = 1 / vini
        # pmom = betgam * mass
        #
        # self.ntrack = 5  # num. of tracks to be generated
        # self.nt = 0  # num. of tracks in the detector
        # self.thmax = 10  # max theta in degrees
        # self.npar = 6  # num. of parameters to be fitted
        # self.ndac = 3  # num data per cell: x, y, t
        #
        # # Initial Values
        # self.sini = self.sc
        # self.tini = 1000
        #
        # # DETECTOR DESIGN
        # # Rectangular detector with ncx*ncy rectangular electrodes
        # # It is assumed that the origin is in one edge of the detector
        # self.nplan = 4  # num. of planes
        # ncx = 12  # num. of cells in x
        # ncy = 10  # num. of cells in y
        # self.vzi = [0, 600, 900, 1800]  # position of the planes in mm
        # self.lenx = 1500  # length in x
        # self.leny = 1200  # length in y
        # self.wcx = self.lenx / ncx  # cell width in x
        # self.wcy = self.leny / ncy  # cell width in y
        # wdt = 100
        #
        # # Uncertainties
        # self.sigx = (1 / np.sqrt(12)) * self.wcx
        # self.sigy = (1 / np.sqrt(12)) * self.wcy
        # self.sigt = 300  # [ps]
        # wx = 1 / self.sigx ** 2
        # wy = 1 / self.sigy ** 2
        # wt = 1 / self.sigt ** 2
        # self.dt = 100  # digitizer precission

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

    def tracks_generation(self):
        ctmx = np.cos(np.deg2rad(THMAX))  # theta_max cosine
        lenz = VZI[NPLAN - 1] - VZI[0]  # Toma la posicion del plano nplan con respecto al inicial
        it = 0  # numero de tracks inicialmente

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

            # Patch to simulate a specific track
            #    x0 = 1000
            #    xp = 0.1
            #    y0 = 600
            #    yp = -0.1

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

        nx = 0

        for it in range(self.nt):
            x0 = self.mtgen[it, 0]
            xp = self.mtgen[it, 1]
            y0 = self.mtgen[it, 2]
            yp = self.mtgen[it, 3]
            # dz = np.cos(th)

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
                xic = kx * WCX + (WCX / 2)
                yic = ky * WCX + (WCX / 2)
                vpnt = np.asarray([xi, yi, ti])  # (X,Y,T) impact point
                vxyt = np.asarray([kx, ky, kt])
                self.vdpt[it:it + 3] = vpnt[0:3]
                self.vdat[it:it + 3] = vxyt[0:3]
                it += 3
            self.mdpt = np.vstack((self.mdpt, self.vdpt))
            self.mdat = np.vstack((self.mdat, self.vdat))
            nx += 1
        self.mdpt = np.delete(self.mdpt, 0, axis=0)
        self.mdat = np.delete(self.mdat, 0, axis=0)


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
    def __init__(self, m_dat):

        self.mdat = m_dat

        self.mdet = self.matrix_det()
        mdet_list = self.mdet.tolist()

        # Error matrix for vr
        self.mErr = diag_matrix(NPAR, [1 / WX, VSLP, 1 / WY, VSLP, 1 / WT, VSLN])

        self.mVd = diag_matrix(NDAC, [SIGX ** 2, SIGY ** 2, SIGT ** 2])  # Matrix V_d -> measurement uncertainties

        self.mstat = self.main()

    def main(self):
        ntrmx = 1  # Max number of tracks
        for ip in range(NPLAN):
            ''' Combinatoria de los positivos en cada plano'''
            nceli = int(self.mdet[ip, 0])
            ntrmx *= nceli

        nvar = NPAR + 2  # npar + 0 column + 1 for quality
        ncol = NPLAN * NDAC + nvar

        mstat = np.zeros([1, ncol])

        ncomb = ntrmx  # Number of possible combinations == Maximum number of possibles tracks

        dcut = 0.995  # Defined threshold to consider positives

        iplan3 = 3  # plane n. 4
        ncel3 = int(self.mdet[iplan3, 0])  # nr. of hits plane 4
        for i3 in range(ncel3):

            icel = 1 + i3 * NDAC
            kx3 = self.mdet[iplan3, icel]
            ky3 = self.mdet[iplan3, icel + 1]
            kt3 = self.mdet[iplan3, icel + 2]
            x0 = kx3 * WCX - (WCX / 2)
            y0 = ky3 * WCX - (WCY / 2)
            t0 = kt3
            # state vector
            vr = np.asarray([x0, 0, y0, 0, t0, SC])  # we assume a normal svector

            iplan2 = 2  # iplan - 1       # plane n. 3
            ncel2 = int(self.mdet[iplan2, 0])  # nr. of hits plane 3
            #
            for i2 in range(ncel2):
                icel = 1 + i2 * NDAC
                kx2 = self.mdet[iplan2, icel]
                ky2 = self.mdet[iplan2, icel + 1]
                kt2 = self.mdet[iplan2, icel + 2]
                x0 = kx2 * WCX - (WCX / 2)
                y0 = ky2 * WCY - (WCY / 2)
                t0 = kt2
                vdat = np.asarray([x0, y0, t0])
                #
                vrp = vr  # save previous values
                mErrp = self.mErr
                #
                vr2, mErr2 = self.fitkalman(VZI, vr, self.mErr, vdat, iplan2)  # <â€” ajuste

                phits = 2
                vstat = np.hstack([phits, kx3, ky3, kt3, kx2, ky2, kt2, 0, 0, 0, 0, 0, 0, vr2])
                cutf = self.fcut(vstat, vr2, mErr2, vdat)
                vstat = np.hstack([vstat, cutf])

                if cutf > dcut:
                    iplan1 = 1  # iplan - 1
                    ncel1 = int(self.mdet[iplan1, 0])

                    for i1 in range(ncel1):
                        icel = 1 + 3 * i1

                        kx1 = self.mdet[iplan1, icel]
                        ky1 = self.mdet[iplan1, icel + 1]
                        kt1 = self.mdet[iplan1, icel + 2]

                        x0 = kx1 * WCX - (WCX / 2)
                        y0 = ky1 * WCY - (WCY / 2)
                        t0 = kt1

                        vdat = np.asarray([x0, y0, t0])
                        vrp2 = vr2
                        mErrp2 = mErr2

                        vr3, mErr3 = self.fitkalman(VZI, vr2, mErr2, vdat, iplan1)

                        phits = 3
                        vstat = np.hstack([phits, kx3, ky3, kt3, kx2, ky2, kt2, kx1, ky1, kt1, 0, 0, 0, vr3])
                        cutf = self.fcut(vstat, vr3, mErr3, vdat)
                        vstat = np.hstack([vstat, cutf])

                        if cutf > dcut:
                            iplan0 = 0  # iplan - 1
                            ncel0 = int(self.mdet[iplan0, 0])

                            for i0 in range(ncel0):
                                icel = 1 + 3 * i0

                                kx0 = self.mdet[iplan0, icel]
                                ky0 = self.mdet[iplan0, icel + 1]
                                kt0 = self.mdet[iplan0, icel + 2]

                                x0 = kx0 * WCX - (WCX / 2)
                                y0 = ky0 * WCY - (WCY / 2)
                                t0 = kt0

                                vdat = np.asarray([x0, y0, t0])
                                vrp3 = vr3
                                mErrp3 = mErr3

                                vr4, mErr4 = self.fitkalman(VZI, vr3, mErr3, vdat, iplan0)

                                phits = 4
                                vstat = np.hstack([phits, kx3, ky3, kt3, kx2, ky2, kt2, kx1, ky1, kt1, kx0, ky0, kt0, vr4])
                                cutf = self.fcut(vstat, vr4, mErr4, vdat)
                                vstat = np.hstack([vstat, cutf])

                                if cutf > dcut:
                                    # nr of planes hitted,cells, saeta, fit quality
                                    mstat = np.vstack([mstat, vstat])
                                else:
                                    continue
                        else:
                            continue
                else:
                    continue

        return mstat[~np.all(mstat == 0, axis=1)]

    def matrix_det(self):  # mdat -> mdet
        # Check if mdat is all zero
        all_zeros = np.all(self.mdat == 0)
        if all_zeros:
            print('No tracks available')
            sys.exit()
        else:
            ndac = 3
            ntrk, nplan = self.mdat.shape  # number of tracks, number of plans
            nplan = int(nplan / ndac)
            ncol = 1 + ndac * ntrk
            mdet = np.zeros([nplan, ncol])
            idat = 0
            for ip in range(nplan):
                idet = 0
                count = []
                for it in range(ntrk):
                    ideti = idet + 1
                    idetf = ideti + ndac
                    idatf = idat + ndac
                    mdet[ip, ideti:idetf] = self.mdat[it, idat:idatf]
                    a = np.all((mdet[ip, ideti:idetf] == 0))  # checks if all are zero
                    if a:
                        count.append(0)
                    else:
                        count.append(1)
                    mdet[ip, 0] = sum(count)
                    idet = idet + ndac
                idat = idat + ndac
        return mdet

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

    def fitkalman(self, vzi, vr, mErr, vdat, iplan):
        for ip in range(iplan + 1, iplan, -1):  # loop on planes
            zi = vzi[ip]
            zf = vzi[ip - 1]

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
        c = 0.3  # [mm/ps]
        ndac = 3
        cmn = bm * c
        smx = 1 / cmn
        ndat = vstat[0] * ndac
        npar = 6
        ndf = ndat - npar
        #
        xd = vdat[0]
        yd = vdat[1]
        td = vdat[2]
        #
        x0 = vr[0]
        y0 = vr[2]
        t0 = vr[4]
        s0 = vr[5]
        #
        sigx = np.sqrt(mErr[0, 0])
        sigy = np.sqrt(mErr[2, 2])
        sigt = np.sqrt(mErr[4, 4])
        if (s0 < 0) or (s0 > smx):
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

    KF = KalmanFilter(mdat1)

    mdet1 = KF.mdet
    mstat1 = KF.mstat
    mVd1 = KF.mVd
    mErr1 = KF.mErr


    # GI.mdat[0][0] = 5
