#INPUT PARAMETERS *****************************************************
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from SSFM import SSFM_FD
from numpy.fft import fftshift, ifftshift
c = 299792.458                    #%speed of ligth nm/ps

#% Input Field Paramenters
tfwhm = 28.4e-3                   #% ps
ni = 1/tfwhm                     # % ps^-1
lamda_central = 835
fo=c/lamda_central               # central pulse frequency (Thz)

# Fiber Parameters
gamma = 110                      # W^-1 * km^-1
alpha = 0                         # atenuation coef. (km^-1)
L = 0.00001                       # fiber length (km)
betaw =np.array ([0,0, -11.830, 8.1038e-2, -9.5205e-5, 2.0737e-7, -5.3943e-10, 1.3486e-12, -2.5495e-15, 3.0524e-18, -1.714e-21]) # beta coefficients (ps^n/ nm)


# Numerical Parameters
nt = 2**15                              # number of spectral points`
time = 32                              # ps
dt = time/nt                           # ps
t=np.arange(-(time / 2), (time / 2 ), dt) # ps
dz = 1e-8                              #initial longitudinal step (km)
v=np.append(np.arange(0, nt/2 ), np.arange(-nt/2, 0))/(dt*nt) # frequencies frequencies (THz)

# INPUT FIELD ***********************************************************
PeakPower = 10000 # W, change here the Input Power!
#1 / np.sinc(np.pi * self.T_ps/(T0_ps*np.pi)
 #initial field shape in W^0.5


u0= np.sqrt(PeakPower) / np.cosh(t / tfwhm)

# PR0PAGATE finding numerical solution **********************************
#************************************************************************
print('Interaction Picture Method started')
tol = 1e-1 # photon error number, or local error, depending on the used method.# #
u = SSFM_FD(u0,dt,L,dz,alpha,betaw,gamma,fo,tol)