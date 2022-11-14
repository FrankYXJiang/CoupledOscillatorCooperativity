from scipy.integrate import solve_ivp
import numpy as np
from numba import njit, prange
import time
import matplotlib.pyplot as plt

a = 1
b = .1
c = 0.01
d = 1
e = 1
f = 1
g = 0.2
Nsim = 200

DS = np.zeros((Nsim,3))
STO = np.zeros((Nsim,3))
MEA = np.zeros((Nsim,3))
for iw in range(Nsim):
    print(iw)
    for iA in range(3):
        A1 = 0.1
        A2 = 0.2
        if (iA == 0):
            A2 = 0
        elif (iA == 1):
            A1 = 0
        pha = 3/2*np.pi

        omega = 0.005*50 + 0.005*(iw+1);

        Tmax = 500
        ts = np.linspace(0, Tmax, Tmax*100)
        bru1 =  lambda T,Y: [a*(1+A1*np.sin(omega*T)) - b*Y[2]*Y[0]/(Y[0]+c),
                             d*Y[0]**2 - e*Y[1],
                             f*(1-A2*np.sin(omega*T-pha))*Y[1] - g*Y[2]]
        sol = solve_ivp (bru1, [0, Tmax], [2, 0,0],t_eval=ts,rtol=1e-8)
        T = sol.t
        Y = sol.y
        y0 = Y[:][0]; y1 = Y[:][1]; y2 = Y[:][2]
        idx  = [];
        for i2 in range(1,len(y2)-1):
            if (y2[i2]>y2[i2-1] and y2[i2]>y2[i2+1]):
                idx.append(i2)            
        idx =np.array(idx); idxF = idx[-20:]; DS[iw,iA] = omega/(2*np.pi/np.mean(T[idxF[1:]]-T[idxF[:-1]]))
        idxF = idx[2:]; MEA[iw,iA] = np.mean(y2[idxF])-np.mean(y2); STO[iw,iA] = np.std(y2[idxF])
plt.plot(DS[:,0],lw=2, color = '#F97306');
plt.plot(DS[:,1],lw=2, color = 'g');
plt.plot(DS[:,2],lw=2, color = 'b'); plt.show()
