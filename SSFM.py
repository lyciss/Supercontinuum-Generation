import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.fft import fftshift, ifftshift


def SSFM_FD(u0, dt, L, dz, alpha, betap, gamma, fo, tol):
    nt = len(u0)  # number of sample points
    #nt = 32768
    w = np.append(np.arange(0, nt / 2 ), np.arange(-nt / 2, 0))
    w = 2 *  math.pi * np.transpose(w) / (dt * nt)
    t = np.arange(-(nt / 2)*dt, (nt / 2 )*dt, dt)  # vector temporal (en ps)

    t1 = 12.2e-3  # raman parameter t1 [ps]
    t2 = 32e-3 # raman parameter t2 [ps]
    tb = 96e-3  # ps
    fc = 0.04
    fb = 0.21
    fa = 1 - fc - fb
    fr = 0.245  # fraccion de respuesta retardada Raman
    tres = t - t[0]  # time starting in 0


    ha=((t1**2 + t2**2) / (t1 * (t2**2)))*np.exp(-tres / t2)*np.sin(tres / t1)

    #ha= ((t1**2 + t2**2) / (t1 * (t**2))) * np.exp(-tres / t2) * np.sin(tres / t1)
    hb = ((2 * tb - tres) / (tb ** 2)) * np.exp(-tres / tb)
    hr = (fa + fc) * ha + fb * hb  # Raman responce function (ps^-1)
    hrw = np.fft.fft(hr)

    linearoperator = np.zeros(nt)-alpha / 2
    for i in np.arange(0,len(betap) - 1):
        linearoperator = linearoperator -1.0j * betap[i] * np.power(w,i) / np.math.factorial(i)
    np.transpose(linearoperator)
    #linearoperator=np.conjugate(linearoperator)

    print( '\nSimulation running...      ')
    ufft = np.fft.fft(u0)
    propagedlength = 0
    u1 = u0
    nf = 1
    w1= w

    while (propagedlength < L):
        if (dz + propagedlength) > L:
            dz = L - propagedlength

        halfstep = np.exp(linearoperator*dz/2)
        uhalf = halfstep*ufft
        # NON LINEAR OPERATOR COARSE
        uip = uhalf

        k1 = -dz*1.0j*gamma*(1 + w1/(2* math.pi*fo))*np.fft.fft( u1*((1-fr)*np.power(abs(u1),2)) + fr*dt*u1* np.fft.ifft(hrw*np.fft.fft( np.power(abs(u1),2))))
        k1 = halfstep*k1

        uhalf2 = np.fft.ifft(uip + k1/2)
        k2 = -dz*1.0j*gamma*(1 + w1/(2* math.pi*fo))*np.fft.fft( uhalf2*((1-fr)*np.power(abs(uhalf2),2)) + fr*dt*uhalf2* np.fft.ifft(hrw*np.fft.fft( np.power(abs(uhalf2),2))))

        uhalf3 = np.fft.ifft(uip + k2/2)
        k3 = -dz*1.0j*gamma*(1 + w1/(2* math.pi*fo))*np.fft.fft( uhalf3*((1-fr)*np.power(abs(uhalf3),2)) + fr*dt*uhalf3* np.fft.ifft(hrw*np.fft.fft( np.power(abs(uhalf3),2) )))

        uhalf4 = np.fft.ifft(halfstep*(uip + k1))
        k4 = -dz*1.0j*gamma*(1 + w1/(2* math.pi*fo))*np.fft.fft( uhalf4*((1-fr)*np.power(abs(uhalf4),2)) + fr*dt*uhalf4* np.fft.ifft(hrw*np.fft.fft( np.power(abs(uhalf4),2))))

        uc = halfstep*(uip + k4 )

        nf = nf + 15
        # END COARSE
        uc = np.fft.ifft(uc)
        nf = nf +2
        u1 = uc
        ufft = np.fft.fft(u1)
        propagedlength = propagedlength + dz
        print( '%.1f' % (propagedlength * 100.0 /L))

    u1 = np.fft.ifft(ufft)
    return u1