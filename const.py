import numpy as np

# =================== #
# ==== Constants ==== #
# =================== #

VC = 0.3  # [MM/PS]
SC = 1 / VC  # SLOWNESS ASSOCIATED WITH LIGHT SPEED
MELE = 0.511  # (MeV/c**2) ELECTRON
MMU = 105.6  # (MeV/c**2) MUON
MPRO = 938.3  # (MeV/c**2) PROTON

# ============================= #
# ==== Data to be Modified ==== #
# ============================= #

MASS = MMU
KENE = 1000  # (MeV) KINETIC ENERGY
ENE = MASS + KENE  # (MeV) TOTAL ENERGY
GAMMA = ENE / MASS  # GAMMA FACTOR
BETA = np.sqrt(1 - 1 / GAMMA ** 2)  # BETA FACTOR
BETGAM = BETA * GAMMA
VINI = BETA * VC  # INITIAL VELOCITY
SINI = 1 / VINI  # INITIAL SLOWNESS
PMOM = BETGAM * MASS  # MOMENTUM

NTRACK = 2  # NUM. OF TRACKS TO BE GENERATED
THMAX = 90  # MAX THETA IN DEGREES
NPAR = 6  # NUM. OF PARAMETERS TO BE FITTED
NDAC = 3  # NUM DATA PER CELL: X, Y, T

# ===================================== #
# ==== Initial values of 20 and t0 ==== #
# ===================================== #

SINI = SC  # INITIAL SLOWNESS
TINI = 1000  # INITIAL TIME

# ========================= #
# ==== Detector design ==== #
# ========================= #

# RECTANGULAR DETECTOR WITH NCX*NCY RECTANGULAR ELECTRODES
# IT IS ASSUMED THAT THE ORIGIN IS IN ONE EDGE OF THE DETECTOR

# Planes distribution:               [mm]   [mm]
# T1 # -------------------------- # 1826      0  TOP
#
# T2 # -------------------------- # 1304    522
# T3 # -------------------------- #  924    902
#
#
# T4 # -------------------------- #   87   1739  BOTTOM
#                                      0         GROUND

# VZI = [1800, 1200, 900, 0]  # mm. POSITION OF THE PLANES IN Z AXIS, MEASURED FROM GROUND TO TOP
# VZ = [0, 600, 900, 1800]  # mm. POSITION OF PLANES MEASURED FROM TOP TO BOTTOM
VZ0 = np.array([1826, 1304, 924, 87])  # mm. REAL HEIGHTS
VZ1 = VZ0[0] - VZ0  # mm. HEIGHTS MEASURED FROM TOP: [0, 522, 902, 1739]
NPLAN = len(VZ0)  # NUM. OF PLANES
NCX = 12  # NUM. OF CELLS IN X
NCY = 10  # NUM. OF CELLS IN Y
LENX = 1500  # mm. LENGTH IN X
LENY = 1200  # mm. LENGTH IN Y
LENZ = VZ0[0] - VZ0[-1]  # mm. LENGTH IN Z (HEIGHT OF THE DETECTOR)
WCX = LENX / NCX  # mm. CELL WIDTH IN X
WCY = LENY / NCY  # mm. CELL WIDTH IN Y
DT = 100  # DIGITIZER PRECISION

# ======================= #
# ==== Uncertainties ==== #
# ======================= #

SIGX = (1 / np.sqrt(12)) * WCX
SIGY = (1 / np.sqrt(12)) * WCY
SIGT = 300  # [PS]
WX = 1 / SIGX**2
WY = 1 / SIGY**2
WT = 1 / SIGT**2

# DEFAULT VARIANCES:
VSLP = 0.1**2  # VARIANCE FOR SLOPE
VSLN = 0.01**2  # VARIANCE FOR SLOWNESS
