# -*- coding: utf-8 -*-
"""    z   = zini;
/genedigitana_tragas_2.py
Created on Tue Aug 25 18:34:36 2015

@author: jag
Sara Costa
"""

#  REVIEW

#  Kalman filter for a pads detector
#  TRAGALDABAS Version
#*****************************
#   JA Garzon. labCAF / USC
#   - Abril 2020
#   2020 Abril. Sara Costa
#******************************************************************** GENE
# It generates ntrack tracks from a charged particle and propagates them
# in the Z axis direction through nplan planes
##################################################################### DIGIT
# It simulates the digital answer in nplan planes of detectors, in which
# - the coordinates (nx,ny) of the crossed pad are determined
# - the flight time is determined integrating tint 
# *******************************************************************KALMAN
# - It reconstructs the track through the Kalman Filter
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
import numpy as np
from numpy.linalg import inv
from scipy import stats
import sys

#
np.set_printoptions(formatter={'float': '{:.3f}'.format})
np.random.seed(17)
#
###################################################
####################Constants######################
###################################################
c     = 0.3 # [mm/ps]
sc    = 1/c # slowness associated with light speed
mele  = 0.511 
mmu   = 105.6
mpro  = 938.3

###################################################
#################Data to be modified###############
###################################################
mass   = mmu;
kene   = 1000 # MeV, kinetic energy
ene    = mass + kene
gamma  = ene/mass
beta   = np.sqrt(1 - 1/(gamma*gamma))
betgam = beta * gamma
vini   = beta * c  # initial velocity
sini   = 1/vini
pmom   = betgam * mass;

ntrack = 5  # num. of tracks to be generated
thmax  = 10   # max theta in degrees
npar   = 6    # num. of parameters to be fitted
ndac   = 3    # num data per cell: x, y, t

####################################################
###########Initial values of S0 and T0##############
####################################################
sini   = sc
tini   = 1000
###############################################################
###############DETECTOR DESIGN#################################
###############################################################


# Rectangular detector with ncx*ncy rectangular electrodes
# It is assumed that the origin is in one edge of the detector


nplan  = 4     # num. of planes
ncx    = 12    # num. of cells in x
ncy    = 10    # num. of cells in y
vzi    = [0, 600, 900, 1800]    # position of the planes
lenx   = 1500    # length in x 
leny   = 1200    # length in y 
wcx    = lenx / ncx   # cell width in x
wcy    = leny / ncy   # cell width in y
wdt    = 100

# Uncertainties
sigx   = (1/np.sqrt(12)) * wcx
sigy   = (1/np.sqrt(12)) * wcy
sigt   = 300  # [ps]
wx     = 1/sigx**2
wy     = 1/sigy**2
wt     = 1/sigt**2
dt     = 100 # digitizer precission

# Data vectors
vdx    = np.zeros(nplan)
vdy    = np.zeros(nplan)
vdt    = np.zeros(nplan)

mtrec  = np.zeros([ntrack,npar]) # reconstructed tracks matrix
mtgen  = np.zeros([ntrack,npar]) # generated tracks matrix
vdat   = np.zeros(nplan * ndac)   # digitazing tracks vector 
vdpt   = np.zeros(nplan * ndac)   # vector with impact point
mdat   = np.zeros(nplan * ndac)   # detector data matrix
mdpt   = np.zeros(nplan * ndac)   # impact point


#####################################################################
################TRACKS GENERATION####################################
#####################################################################

ctmx = np.cos(np.deg2rad(thmax)) # theta_max cosine
lenz   = vzi[nplan-1]-vzi[0]
it   = 0


for i in range(ntrack):
    # Uniform distribution in cos(theta) and phi
    rcth  = 1 - np.random.random() * (1 - ctmx)
    tth   = np.arccos(rcth)               # theta
    tph   = np.random.random() * 2*np.pi     # phi 
    
    x0 = np.random.random() * lenx 
    y0 = np.random.random() * leny
    t0 = tini
    s0 = sini
    
    cx = np.sin(tth) * np.cos(tph)  # cosenos directores
    cy = np.sin(tth) * np.sin(tph) 
    cz = np.cos(tth)
    xp = cx/cz  # projected slope in the X-Z plane
    yp = cy/cz  # projected slope in the Y-Z plane       

    # Patch to simulate a specific track
#    x0 = 1000
#    xp = 0.1
#    y0 = 600
#    yp = -0.1
   
    # Coordenate where would the particle come out
    xzend =  x0 + xp * lenz
    yzend =  y0 + yp * lenz
    
    # We refere the coordinate to the detector center (xmid, ymid)
    xmid  = xzend - (lenx/2)
    ymid  = yzend - (leny/2)
    
    # We check if the particle has entered the detector
    if((np.abs(xmid) < (lenx/2)) and (np.abs(ymid) < (leny/2))): 
        mtgen[it,:]=[x0,xp,y0,yp,t0,s0]
        it   = it + 1
    else:
        continue
nt = it       # number of tracks in the detector
mtgen=mtgen[~(mtgen==0).all(1)]   

     

#########################################################################
############################DIGITAZION###################################
#########################################################################

nx = 0

for it in range(nt):
    x0 = mtgen[it,0]
    xp = mtgen[it,1]
    y0 = mtgen[it,2]
    yp = mtgen[it,3]
    #dz = np.cos(th)
    
    it=0
    for ip in range(nplan):
        zi = vzi[ip]
        xi = x0 + xp * zi
        yi = y0 + yp * zi
        ks = np.sqrt(1 + xp*xp + yp*yp)
        ti = tini + ks * sc * zi
        # Position indices of the impacted cells
        kx   = np.int((xi + (wcx/2))/wcx) 
        ky   = np.int((yi + (wcy/2))/wcy) 
        kt   = np.int((ti + (dt/2))/dt) * dt
        xic  = kx * wcx + (wcx/2)
        yic  = ky * wcy + (wcy/2)
        vpnt = np.asarray([xi, yi, ti])     # (X,Y,T) impact point
        vxyt = np.asarray([kx, ky, kt])
        vdpt[it:it+3] = vpnt[0:3]
        vdat[it:it+3] = vxyt[0:3]
        it = it + 3
    mdpt = np.vstack((mdpt, vdpt))    
    mdat = np.vstack((mdat, vdat))
    nx = nx + 1
mdpt = np.delete(mdpt, (0), axis=0)
mdat = np.delete(mdat, (0), axis=0)


########################################################################
########################## MDET MATRIX##################################
########################################################################

def matrix_det(mdat):                      # mdat -> mdet 
    #Check if mdat is all zero
    all_zeros = np.all(mdat==0)
    if all_zeros == True:
        print('No tracks available')
        sys.exit()
    else:
        ndac = 3
        ntrk, nplan = mdat.shape
        nplan =  int(nplan/ndac)
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
                mdet[ip, ideti:idetf] = mdat[it, idat:idatf]
                a = np.all((mdet[ip,ideti:idetf] == 0)) #checks if all are zero
                if a == True:
                    count.append(0)
                else:
                    count.append(1)   
                mdet[ip,0] = sum(count)
                idet = idet + ndac            
            idat = idat + ndac
    return mdet

mdet=matrix_det(mdat) 
mdet_list=mdet.tolist()



#############################################################################
######################KALMAN FILTER##########################################
#############################################################################


def fprop(vr, mErr, zi, zf):     # Transport function    
    dz = zf - zi
    XP = vr[1]
    YP = vr[3]
    ks = np.sqrt(1 + XP*XP + YP*YP)
    
    mF  = np.zeros([npar, npar])
    np.fill_diagonal(mF,[1, 1, 1, 1, 1, 1])
    mF[0,1] = dz
    mF[2,3] = dz
    mF[4,5] = ks * dz 
    
    
    vr  = np.dot(mF, vr)
    vr  = np.asarray(vr)
    # Propagation of the error matrix
    mErr = np.dot(mF, np.dot(mErr, mF.T)) 
           
    return vr, mErr


def jacobi(vr, zi):           # Jacobian matrix
    ndac = 3
    npar = 6
    mH   = np.zeros([ndac,npar])
    
    XP   = vr[1]
    YP   = vr[3]
    S0   = vr[5]
    ks   = np.sqrt(1 + XP*XP + YP*YP)
    ksi  = 1/ks
    
    mH[0,0] = 1
    mH[0,1] = zi
    mH[1,2] = 1
    mH[1,3] = zi
    mH[2,1] = ksi * S0 * XP * zi
    mH[2,3] = ksi * S0 * YP * zi
    mH[2,4] = 1
    mH[2,5] = ks * zi
    
    return mH
    
def fpar2dat(vr, mErr, mH, zi, zf):    # Projection in the measurement space
    # Fitting model
    X0 = vr[0]
    Y0 = vr[2]
    T0 = vr[4]
    
    vdr = [X0, Y0, T0]

    ndac  = 3
    npar  = 6    
    mid_s = np.zeros([ndac,npar])
    mid_s[0,0] = 1
    mid_s[1,2] = 1
    mid_s[2,4] = 1    
    
    vdr = np.dot(mid_s,np.dot(mH.T,vdr))
    
    
    mVr = np.dot(mH, np.dot(mErr, mH.T))
    
    return vdr,mVr


def fdat2par(mVr, mVd, mVc, mErr, mH, zi, zf):  # Projection in the parameter space
    
    mWc     = inv(mVc)   # weight matrix
    mKgain  = np.dot(mErr, np.dot(mH.T, mWc))
    mWr     = np.dot(mH.T, np.dot(mWc, mH))
    
    return mKgain, mWr


def update(mKgain,vdd,mWr, vr, mErr):    # Update the state vector and error matrix
    
    dvr  = np.dot(mKgain, vdd)
    mdE  = np.dot(mErr, np.dot(mWr, mErr))
    vr   = vr + dvr
    mErr = mErr - mdE
    
    return vr, mErr

# Matrix V_d -> measurement uncertainties
mVd = np.zeros([ndac,ndac])
np.fill_diagonal(mVd, [sigx*sigx, sigy*sigy, sigt*sigt])



#########################################################################
#################KALMAN FILTER###########################################
#########################################################################

def fitkalman(vzi,vr,mErr,vdat,iplan):

    for ip in range(iplan+1, iplan, -1):    # loop on planes
        zi   = vzi[ip]
        zf   = vzi[ip-1]       
                
        # Propagation step
        vrp = vr
        vr, mErr = fprop(vr, mErr, zi, zf)
                    
        mH = jacobi(vr, zi)  # Jacobian matrix
                    
        vdr, mVr = fpar2dat(vr, mErr, mH, zi, zf) #  Parameter  -> Measurument
                       
        # new measurement
        ndac   = 3
        vdi    = np.zeros(ndac)  
        vdi[0] = vdat[0]   
        vdi[1] = vdat[1] 
        vdi[2] = vdat[2]  
        vdd    = vdi - vdr     # Difference between measurement and expected data
        mVc    = mVr + mVd     # Joint uncertainties matrix       
                    
        mKgain, mWr = fdat2par(mVr,mVd,mVc,mErr,mH, zi, zf) # Meas. -> Proj.
                  
        vr, mErr = update(mKgain,vdd,mWr, vr, mErr)  # New state vector
               

    return vr, mErr

def fcut(vstat, vr, mErr, vdat): #Function that returns quality factor
    bm   = 0.2     # beta  min
    #c    = 0.3
    ndac = 3
    cmn  = bm * c
    smx  =  1/cmn
    ndat = vstat[0] * ndac
    npar = 6
    ndf  = ndat - npar
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
    sigx = np.sqrt(mErr[0,0])
    sigy = np.sqrt(mErr[2,2])
    sigt = np.sqrt(mErr[4,4])
    if (s0 < 0) or (s0 > smx):
        cutf = 0
    else:
        if ndf !=0:
            s2 = ((xd-x0)/sigx)**2 +  ((yd-y0)/sigy)**2 + ((td-t0)/sigt)**2      
            cutf = stats.chi2.sf(s2, ndf)
        else:
            cutf = 1

    return cutf
  
#############################################################################
############################ MAIN ###########################################
#############################################################################

nplan = len(mdet)
ntrmx = 1
for ip in range(nplan):
    nceli = int(mdet[ip, 0])
    ntrmx = ntrmx * nceli

                  
nvar  = npar + 2               # npar + 0 column + 1 for quality
ncol  = nplan * ndac + nvar


mstat = np.zeros([1, ncol]) 

ncomb = ntrmx 
    
dcut  = 0.995



# Default variances for slopes and slowness
vslp = 0.1 * 0.1
vsln = 0.01 * 0.01


         
iplan3 = 3                 # plane n. 4
ncel3 = int(mdet[iplan3,0]) # nr. of hits plane 4
for i3 in range(ncel3): 

    icel = 1 + i3 * ndac 
    kx3   = mdet[iplan3, icel]
    ky3   = mdet[iplan3, icel+1]
    kt3   = mdet[iplan3, icel+2]   
    x0   = kx3 * wcx - (wcx/2)
    y0   = ky3 * wcy - (wcy/2)
    t0   = kt3
    # state vector
    vr   = np.asarray([x0, 0, y0, 0, t0, sc])  # we assume a normal svector
    # Error matrix for vr
    mErr  = np.zeros([npar,npar])
    np.fill_diagonal(mErr,[1/wx, vslp, 1/wy, vslp, 1/wt, vsln])
    


    iplan2 = 2 #iplan - 1       # plane n. 3
    ncel2 = int(mdet[iplan2,0])  # nr. of hits plane 3
	#
    for i2 in range (ncel2):
        icel = 1 + i2 * ndac        
        kx2   = mdet[iplan2, icel]
        ky2   = mdet[iplan2, icel+1]
        kt2   = mdet[iplan2, icel+2]        
        x0   = kx2 * wcx - (wcx/2)
        y0   = ky2 * wcy - (wcy/2)
        t0   = kt2   
        vdat   = np.asarray([x0, y0, t0]) 
        #
        vrp    = vr     # save previous values
        mErrp  = mErr
        #
        vr2, mErr2 = fitkalman(vzi, vr, mErr, vdat, iplan2)    #  <â€” ajuste

        phits = 2
        vstat = np.hstack([phits, kx3,ky3,kt3,kx2,ky2,kt2,0,0,0,0,0,0, vr2])
        cutf = fcut(vstat, vr2, mErr2, vdat)
        vstat = np.hstack([vstat,cutf])
        
        if cutf > dcut:
            iplan1 = 1#iplan - 1
            ncel1 = int(mdet[iplan1,0])
            
            for i1 in range(ncel1):
                icel   = 1 + 3*i1
                
                kx1     = mdet[iplan1, icel]
                ky1     = mdet[iplan1, icel+1]
                kt1     = mdet[iplan1, icel+2]
                
                x0     = kx1 * wcx - (wcx/2)
                y0     = ky1 * wcy - (wcy/2)
                t0     = kt1
                
                vdat   = np.asarray([x0, y0, t0])
                vrp2   = vr2
                mErrp2 = mErr2
                
                
                vr3, mErr3  = fitkalman(vzi,vr2,mErr2,vdat,iplan1)

                phits = 3
                vstat = np.hstack([phits, kx3,ky3,kt3,kx2,ky2,kt2,kx1,ky1,kt1,0,0,0, vr3 ])
                cutf = fcut(vstat, vr3, mErr3, vdat)
                vstat = np.hstack([vstat,cutf])
                
                if cutf > dcut:               
                    iplan0 = 0#iplan - 1
                    ncel0 = int(mdet[iplan0,0])
					   
                    for i0 in range(ncel0):
                        icel   = 1 + 3*i0
                        
                        kx0     = mdet[iplan0,icel]
                        ky0     = mdet[iplan0, icel+1]
                        kt0     = mdet[iplan0, icel+2]
                        
                        x0     = kx0 * wcx - (wcx/2)
                        y0     = ky0 * wcy - (wcy/2)
                        t0     = kt0
                        
                        vdat   = np.asarray([x0,y0,t0])
                        vrp3   = vr3
                        mErrp3 = mErr3
                        
                        vr4, mErr4  = fitkalman(vzi,vr3,mErr3,vdat, iplan0)
                        
                        phits = 4
                        vstat = np.hstack([phits, kx3,ky3,kt3,kx2,ky2,kt2,kx1,ky1,kt1,kx0,ky0,kt0, vr4])    
                        cutf  = fcut(vstat, vr4, mErr4, vdat)
                        vstat = np.hstack([vstat,cutf])

                        
                        if cutf > dcut:
                             #nr of planes hitted,cells, saeta, fit quality
                            mstat = np.vstack([mstat,vstat])
                            
                        else: 
                            continue
                            
                else:
                    continue
                
        else:
            continue

mstat = mstat[~np.all(mstat == 0, axis=1)]





