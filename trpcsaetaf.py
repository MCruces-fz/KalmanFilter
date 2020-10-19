# -*- coding: utf-8 -*-
"""
Created on Wed 8 August 16:36 2020

mcsquared.fz@gmail.com
miguel.cruces@rai.usc.es

@author:
Miguel Cruces

"""

from const import *
from kalmanFilter import GenerateTracks

np.set_printoptions(formatter={'float': '{:.3f}'.format})
np.random.seed(18)
'''
Seed with first vertical line (NTRACK = 1) -> 34
Seed with 5 tracks (NTRACK = 5) -> 17
'''


class TRpcHit(GenerateTracks):
    def setHits(self):
        """
        Returns array with hits
        """
        if np.all(self.mdat == 0):  # Check if mdat is all zero
            raise Exception('No tracks available! Matrix mdat is all zero.')
        columns = ["trbnum", "cell", "col", "row", "x", "y", "z", "time", "charge"]
        output = np.zeros([0, len(columns)])
        ntrk, _ = self.mdat.shape  # Number of tracks, number of plans
        ncol = 1 + NDAC * ntrk  # One more column to store number of tracks
        mdet = np.zeros([NPLAN, ncol])
        idat = 0
        for ip in range(NPLAN):
            idet = 0
            for it in range(ntrk):
                ideti = idet + 1
                idetf = ideti + NDAC
                idatf = idat + NDAC
                x = self.mdat[it, idat:idatf][0]
                y = self.mdat[it, idat:idatf][1]
                time = self.mdat[it, idat:idatf][2]
                row = np.hstack(())
                if not np.all((mdet[ip, ideti:idetf] == 0)):  # checks if all are zero
                    mdet[ip, 0] += 1
                idet += NDAC
            idat += NDAC
        return mdet

    def GetEntriesFast(self):
        """
        Returns the total number of hits
        """
        pass

    def fRpcHitHits(self):
        pass


class TRpcSaetaF:
    def __init__(self):
        pass

    def execute(self):
        pass


if __name__ == "__main__":
    TH = TRpcHit()
    TH.setHits()
