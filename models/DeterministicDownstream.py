from __future__ import division
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange
import time


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

    Km0 = Km/8.0;
    Vh0 = Vh;

    #### %%%%%%%%%%%% Volume effect %%%%%%%%%%%%%                                                                                                                                         
    Vol = 1.0*10**(-18); NA = 6.02*10**(23); Cal = NA*Vol*10**(-6);
    ka = ka/Cal;
    xT = xT*Cal;
    Vm = Vm*Cal;
    Vh = Vh*Cal;
    Km = Km*Cal;
    kh = kh*Cal;
    Km0 = Km0*Cal;
    Vh0 = Vh0*Cal;
    mp1 = 0.01;
    mdel1 = 0.001;
    pp1 = 0.01
    pdel1 = 0.0001

    
    ka0 = ka;
    kd0 = kd;    
    kiN0 = kiN;
    keN0 = keN
    Vm0 = Vm
    ktrlt0 = ktrlt
    kh0 = kh
    kv0 = kv
    ket0 = ket
    mp10 = mp1
    mdel10 = mdel1
    pp10 = pp1
    pdel10 = pdel1

    dx = np.zeros(4); pi = 3.141592;
    x = np.zeros(4);
    x[0] = 111.19988481
    x[1] = 110.22829659;
    x[2] = 19.64971775
    x[3] =  4.44401498
    m1 = 0; P1 = 0;
    m2 = 0; P2 = 0;
    
    RT = 0; Tmax = 4*240*60; ts = 10*60; click = 0;
    dt = 0.1;

    
    ome1 = 30
    ph = o12ph;
    mp10 = 0.001;
    A1 = 0.2;
    A2 = 0.2
    Km01 = Km/2.0
    Km02 = Km/8.0
    while RT < Tmax:
        if (RT < (3*240+30)*60):
            Km = Km0*2;
            Vh = Vh0/1.
        else:            
            Km = Km0/(1.0 + A1*np.sin(2*3.141592/(ome1*60)*RT))
            Vh = Vh0/(1.0 + A2*(1 + np.sin(2*3.141592/(ome1*60)*RT - ph)))

        xN = (xT-x[0]-x[1]);
        dx[0] = kd*x[1] - ka*x[0]*x[2] + keN*kv*xN - kiN*x[0] + Vm*x[1]/(x[1]+Km);
        dx[1] = ka*x[0]*x[2] - kd*x[1] - Vm*x[1]/(x[1]+Km);
        dx[2] = kd*x[1] - ka*x[0]*x[2] - Vm*x[2]/(x[2]+Km) + ktrlt*x[3];
        dx[3] = kv*(Vh*((kv*xN)**h/(kh**h+(kv*xN)**h)) - ket*x[3]);

        if (RT > 0*60):            
            m1 = m1 + dt*(0.01*(xN**4.)/((xN**4.) + (220.**4)) - 0.001*m1)
            P1 = P1 + dt*(0.02*m1 - 0.0001*P1)
            m2 = m2 + dt*(0.01*xN**2/(xN**2 + 15.**2) - 0.001*m2)
            P2 = P2 + dt*(0.001*m2 - 0.001*P2)
    
        x += dt*dx;        
        RT += dt

    
        if (RT > click*ts):
            SirTmp = np.zeros(4)
            SirTmp[0] = RT/60-3*240;
            SirTmp[1] = xN;
            SirTmp[2] = P1;
            SirTmp[3] = P2;
            SIRfile.append(SirTmp)
            click += 1
            
    print(x)
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

for test2 in range (1):
    Pers = []
    for test in range(2):
        start = time.time()
        
        o1b = 0
        o1u = 2.0
        o1p = 30.0
        o2b = 0
        o2u = 0.0
        if (test2 == 0):
            o2p = 0.0
        elif (test2 == 1):
            o2p = 20.0/10
        else:
            o2p = 40.0/10
        
        o12ph = (0.5+test)*3.141592

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
        print(f"\nElapsed (with compilation) = {end - start:.2f}",test)


        for i0 in range(len(SIRfiles)):
            XN = SIRfiles[i0]
            TT = np.zeros((len(XN),4))
            for j in range(len(XN)):
                TT[j][0] = XN[j][0]
                TT[j][1] = XN[j][1]
                TT[j][2] = XN[j][2]
                TT[j][3] = XN[j][3]
                    
            np.savetxt("data/Testdata_%s_%s.txt"%(test,test2),TT)


