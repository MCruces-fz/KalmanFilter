import numpy as np

# =================== #
# ==== Constants ==== #
# =================== #
C = 0.3  # [MM/PS]
SC = 1 / C  # SLOWNESS ASSOCIATED WITH LIGHT SPEED
MELE = 0.511
MMU = 105.6
MPRO = 938.3

# ============================= #
# ==== Data to be Modified ==== #
# ============================= #
MASS = MMU
KENE = 1000  # MEV, KINETIC ENERGY
ENE = MASS + KENE
GAMMA = ENE / MASS
BETA = np.sqrt(1 - 1 / (GAMMA * GAMMA))
BETGAM = BETA * GAMMA
VINI = BETA * C  # INITIAL VELOCITY
SINI = 1 / VINI
PMOM = BETGAM * MASS

NTRACK = 5  # NUM. OF TRACKS TO BE GENERATED
THMAX = 10  # MAX THETA IN DEGREES
NPAR = 6  # NUM. OF PARAMETERS TO BE FITTED
NDAC = 3  # NUM DATA PER CELL: X, Y, T


# ===================================== #
# ==== Initial values of 20 and t0 ==== #
# ===================================== #
SINI = SC
TINI = 1000

# ========================= #
# ==== Detector design ==== #
# ========================= #

# RECTANGULAR DETECTOR WITH NCX*NCY RECTANGULAR ELECTRODES
# IT IS ASSUMED THAT THE ORIGIN IS IN ONE EDGE OF THE DETECTOR


NPLAN = 4  # NUM. OF PLANES
NCX = 12  # NUM. OF CELLS IN X
NCY = 10  # NUM. OF CELLS IN Y
VZI = [0, 600, 900, 1800]  # POSITION OF THE PLANES
LENX = 1500  # LENGTH IN X
LENY = 1200  # LENGTH IN Y
WCX = LENX / NCX  # CELL WIDTH IN X
WCY = LENY / NCY  # CELL WIDTH IN Y
WDT = 100

# ======================= #
# ==== Uncertainties ==== #
# ======================= #
SIGX = (1 / np.sqrt(12)) * WCX
SIGY = (1 / np.sqrt(12)) * WCY
SIGT = 300  # [PS]
WX = 1 / SIGX ** 2
WY = 1 / SIGY ** 2
WT = 1 / SIGT ** 2
DT = 100  # DIGITIZER PRECISSION

# DEFAULT VARIANCES:
VSLP = 0.1**2  # VARIANCE FOR SLOPE
VSLN = 0.01**2  # VARIANCE FOR SLOWNESS