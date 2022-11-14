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


    x = np.zeros(4); x[1] = xT/4.0; dx = np.zeros(4); pi = 3.141592;
    x[2] = 1000;
    m1 = 0; P1 = 0;
    m2 = 0; P2 = 0;
    
    RT = 0; Tmax = 120*30*20; ts = 10; click = 0;
    dt = 0.1;

    
    ome1 = 30
    ph = o12ph;
    mp10 = 0.001;
    
    AM = []
    for i in range(14):
        AM.append(0.0)

    AM[o2u] = o1b;
    if (o12ph==10):
        AM[o2u] = 0.0
    
    while RT < Tmax:

        Km = Km0*(1+o1b*np.sin(RT*2*3.141592/(ome1*60.0)))

        SI = np.sin(RT*2*3.141592/(ome1*60.0)-ph);
        Vh = Vh0*(1+AM[0]*(SI))
        ka = ka0*(1+AM[1]*(SI));
        kd = kd0*(1+AM[2]*(SI));
        kiN = kiN0*(1+AM[3]*(SI));
        keN = keN0*(1+AM[4]*(SI))
        Vm = Vm0*(1+AM[5]*(SI))
        ktrlt = ktrlt0*(1+AM[6]*(SI))
        kh = kh0*(1+AM[7]*(SI))
        kv = kv0*(1+AM[8]*(SI))
        ket = ket0*(1+AM[9]*(SI))
        mp1 = mp10*(1+AM[10]*(SI))
        mdel1 = mdel10*(1+AM[11]*(SI))
        pp1 = pp10*(1+AM[12]*(SI))
        pdel1 = pdel10*(1+AM[13]*(SI))

        

        xN = (xT-x[0]-x[1]);
        dx[0] = kd*x[1] - ka*x[0]*x[2] + keN*kv*xN - kiN*x[0] + Vm*x[1]/(x[1]+Km);
        dx[1] = ka*x[0]*x[2] - kd*x[1] - Vm*x[1]/(x[1]+Km);
        dx[2] = kd*x[1] - ka*x[0]*x[2] - Vm*x[2]/(x[2]+Km) + ktrlt*x[3];
        dx[3] = kv*(Vh*((kv*xN)**h/(kh**h+(kv*xN)**h)) - ket*x[3]);

        if (RT > Tmax/4):
            m1 = m1 + dt*(mp1*(xN**4.)/((xN**4.) + (250.**4)) - mdel1*m1)
            P1 = P1 + dt*(pp1*m1 - 2*pdel1*P1)

            m2 = m2 + dt*(0.01*xN**2/(xN**2 + 15.**2) - 0.001*m2)
            P2 = P2 + dt*(0.001*m2 - 0.001*P2)

    
        x += dt*dx;        
        RT += dt

    
        if (RT > click*ts):
            SirTmp = np.zeros(4)
            SirTmp[0] = RT;
            SirTmp[1] = xN;
            SirTmp[2] = P1;
            SirTmp[3] = P2;
            SIRfile.append(SirTmp)
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
        cline += 1


N1 = 14; N2 = 21;

for testamp in range(25):
    Means1 = np.zeros((N1,N2))
    Final1 = np.zeros((N1,N2))

    for test2 in range (N2):
        Pers = []
        for test in range(N1):
            start = time.time()
        
            o1b = 0.01*testamp
            o1u = 2.0
            o1p = 30.0
            o2b = 0
            o2u = test
            o2p = 1
            o12ph = 3.141592*test2/((N2-1)/2.0)
            if (test2==N2-1):
                o12ph = 10

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

            LL = len(TT)
            LL0 = int(LL*0.75)
            Means1[test,test2] = np.mean(TT[LL0:LL,2])
            Final1[test,test2] = np.max(TT[:,2])

            
    np.savetxt("data/Figure3NewOscimean_A%s_Neut_P1.txt"%(testamp),Means1)
    np.savetxt("data/Figure3Newosci_A%s_Neut_P1.txt"%(testamp),Final1)


