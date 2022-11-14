from __future__ import division
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange
import time
import matplotlib.pyplot as plt

@njit
def Single_run_numba(o1b,o1u,o1p,o2b,o2u,o2p,o12ph,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12):
    SIRfile = []

    #### %%%%%%%% Original Parameters %%%%%%%%%%                                                                                                                                          

    ka = P0; kd = P1;
    kiN = P2; keN = P3;
    Vm = P4; Km = P5;
    ktrlt = P6;
    Vh = P7; kh = P8;
    kv = P9; h  = P10;
    ket = P11;
    xT = P12;

    Km0 = Km;
    Vh0 = Vh;

    #### %%%%%%%%%%%% Volume effect %%%%%%%%%%%%%                                                                                                                                         
    Vol = 1.0*10**(-18); NA = 6.02*10**(23); Cal = NA*Vol*10**(-6);
    ka = ka/Cal;
    xT = xT*Cal;
    Vm = Vm*Cal;
    Vh = Vh*Cal;
    Vh0 = Vh0*Cal;
    Km = Km*Cal;
    kh = kh*Cal;
    Km0 = Km0*Cal;

    pi = 3.141592;
    
    x = np.zeros(4); x[1] = xT/4.0; dx = np.zeros(4)
    x[2] = 1000;
    RT = 0; Tmax = 240*30*60; ts = 180; click = 0;
    dt = 0.5;
    TNF = 7.0
    ETH = 0

    m1 = 0; P1 = 0;
    m2 = 0; P2 = 0;

    if (o1b < 0.5):
        ETHmin = -0.0
        ETHmax = 0.0
        TNFmin = -0.06
        TNFmax = 0.06
    elif (o1b < 1.5):
        ETHmin = -0.02
        ETHmax = 0.02
        TNFmin = -0.0
        TNFmax = 0.0
    elif (o1b < 2.5):
        ETHmin = -0.02
        ETHmax = 0.02
        TNFmin = -0.06
        TNFmax = 0.06
    
    ome1 = 1./o1p
    ph = o12ph;
    
    TNF0 = 6
    while RT < Tmax:
        if (np.sin(RT*2*3.141592/(ome1*60.0)) > 0):
            TNF = TNFmax;
        else:
            TNF = TNFmin;

        Km = Km0/(6.0 + 1.0*TNF)

        if (np.sin(RT*2*3.141592/(ome1*60.0)-ph) > 0):
            ETH = ETHmax;
        else:
            ETH = ETHmin;

        Vh = Vh0/(1.0 + 0.1*ETH)

        xN = (xT-x[0]-x[1]);
        dx[0] = kd*x[1] - ka*x[0]*x[2] + keN*kv*xN - kiN*x[0] + Vm*x[1]/(x[1]+Km);
        dx[1] = ka*x[0]*x[2] - kd*x[1] - Vm*x[1]/(x[1]+Km);
        dx[2] = kd*x[1] - ka*x[0]*x[2] - Vm*x[2]/(x[2]+Km) + ktrlt*x[3];
        dx[3] = kv*(Vh*((kv*xN)**h/(kh**h+(kv*xN)**h)) - ket*x[3]);
    
        x += dt*dx;        
        RT += dt

    
        if (RT > click*ts):
            SIRfile.append(xN)
            click += 1

    return SIRfile

@njit(parallel = True)
def multiple_loops(N_loops,o1b,o1u,o1p,o2b,o2u,o2p,o12ph,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12):
    SIRfiles = []
    for i in prange(N_loops):
        SIRfile = Single_run_numba(o1b,o1u,o1p,o2b,o2u,o2p,o12ph,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12)
        SIRfiles.append(SIRfile)
    return SIRfiles

N_loops = 1


P = np.zeros(13); filename = '../Hope_New_FinalParameters.txt'; cline = 0
with open( filename, 'r' ) as f :
    for line in f:
        line = line.strip()
        columns = line.split('\t')
        P[cline] = float(columns[1])
        cline+=1

NN = 400
AT3 = np.zeros((NN))
for test2 in range (3):
    Pers = []
    for test3 in range(15):
        for test in range(NN):
            start = time.time()
        
            o1b = test2
            o1u = 2.0
            o1p = 0.1+0.1*test
            o2b = 0
            o2u = 0.0
            o2p = 30.0
            o12ph = (test3/15.0)*2*3.141592

            P0 = P[0]
            P1 = P[1]
            P2 = P[2]
            P3 = P[3]
            P4 = P[4]
            P5 = P[5]
            P6 = P[6]
            P7 = P[7]
            P8 = P[8]
            P9 = P[9]
            P10 = P[10]
            P11 = P[11]
            P12 = P[12]

            SIRfiles = multiple_loops(N_loops,o1b,o1u,o1p,o2b,o2u,o2p,o12ph,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12)

            end = time.time()
            print(f"\nElapsed (with compilation) = {end - start:.2f}",test,test2)

            for i0 in range(len(SIRfiles)):
                XN = SIRfiles[i0]
                x = XN;
                mx = np.mean(XN)

                t = np.linspace(0,len(XN)*3,len(XN))
                Pea = []
                TP = []
                for j1 in range(len(XN)-2):
                    if (x[j1+1]>x[j1] and x[j1+1]>x[j1+2] and x[j1+1]>mx):
                        Pea.append(t[j1+1])
                        if (len(Pea)>5):
                            TP.append(Pea[len(TP)+1]-Pea[len(TP)])
            

                L1 = int(len(TP) - 10); L2 = len(TP)
                AT3[test] = np.mean(TP[L1:L2])*o1p


        np.savetxt("DS_%s_c%s.txt"%(test2,test3),AT3)
