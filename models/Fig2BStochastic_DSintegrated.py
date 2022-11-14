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
    kh0 = kh;
    
    #### %%%%%%%%%%%% Volume effect %%%%%%%%%%%%%
    Vol = 5.0*10**(-18); NA = 6.02*10**(23); Cal = NA*Vol*10**(-6);
    ka = ka/Cal;
    xT = xT*Cal;
    Vm = Vm*Cal;
    Vh = Vh*Cal;    
    Vh0 = Vh0*Cal;
    Km = Km*Cal;
    kh = kh*Cal;
    Km0 = Km0*Cal;
    kh0 = kh0*Cal;
        
    pi = 3.141592;

# %%%%%%%%%% Oscillation Parameters %%%%%%%%%%%%
    ph = o12ph;
    ome1 = o1p

        ### %%%%%%%%%%%%%%%%%% Stochastic Parameters %%%%%%%%%%%%%
    x = np.zeros(4); x[1] = xT;
    RT = 0; Tmax = 100*20*30; ts = 180; click = 0;
    deltaT = 0.01;
    TNF0 = 6.0
    if (o1b < 0.5):
        ETHmin = -0.0
        ETHmax = 0.0
        TNFmin = -0.15*(o1u+1)
        TNFmax = 0.15*(o1u+1)
    elif (o1b < 1.5):
        ETHmin = -0.06*(o1u+1)
        ETHmax = 0.06*(o1u+1)
        TNFmin = -0.0
        TNFmax = 0.0
    elif (o1b < 2.5):
        ETHmin = -0.15*(o1u+1)
        ETHmax = 0.15*(o1u+1)
        TNFmin = -0.06*(o1u+1)
        TNFmax = 0.06*(o1u+1)
            
    while RT < Tmax:

        if (np.sin(RT*2*3.141592/(ome1*60.0)) > 0):
            TNF = TNFmax;
        else:
            TNF = TNFmin;

        Km = Km0/(TNF0 + 1.0*TNF)

        if (np.sin(RT*2*3.141592/(ome1*60.0)-ph) > 0):
            ETH = ETHmax;
        else:
            ETH = ETHmin;

        Vh = Vh0/(1.0 + 0.1*ETH)

        xN = (xT-x[0]-x[1]);
        T1 = kd*x[1];
        T2 = ka*x[0]*x[2];
        T3 = Vm*x[1]/(x[1]+Km);
        T4 = keN*kv*xN;
        T5 = kiN*x[0];
        T6 = Vm*x[2]/(x[2]+Km);
        T7 = ktrlt*x[3];
        T8 = kv*Vh*((kv*xN)**h/(kh**h+(kv*xN)**h));
        T9 = kv*ket*x[3];
        TT = T1+T2+T3+T4+T5+T6+T7+T8+T9;
        
        DelT = -np.log(np.random.random())/TT;
        RT = RT + DelT;
        AA = np.random.random();
        if (AA < T1/TT):
            x[0] += 1
            x[1] -= 1
            x[2] += 1
        elif (AA < (T1+T2)/TT):
            x[0] -= 1
            x[1] += 1
            x[2] -= 1
        elif (AA < (T1+T2+T3)/TT):
            x[0] += 1
            x[1] -= 1
        elif (AA < (T1+T2+T3+T4)/TT):
            x[0] += 1
        elif (AA < (T1+T2+T3+T4+T5)/TT):
            x[0] -= 1
        elif (AA < (T1+T2+T3+T4+T5+T6)/TT):
            x[2] -= 1;
        elif (AA < (T1+T2+T3+T4+T5+T6+T7)/TT):
            x[2] += 1
        elif (AA < (T1+T2+T3+T4+T5+T6+T7+T8)/TT):
            x[3] += 1
        else:
            x[3] -= 1

        if (RT > ts*click):            
            SirTmp = np.zeros(3)
            SirTmp[0] = RT;
            SirTmp[1] = xN;
            SirTmp[2] = x[2];

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

N_loops = 4


P = np.zeros(13); filename = '../Hope_New_FinalParameters.txt'; cline = 0
with open( filename, 'r' ) as f :
    for line in f:
        line = line.strip()
        columns = line.split('\t')
        P[cline] = float(columns[1])
        cline+=1




AT3 = np.zeros((100))
for test2 in range (12,15):
    Pers = []
    for test in range(100):
        print(test,test2)
        start = time.time()
        o1b = 2
        o1u = test2
        o1p = 10+0.4*test
        o2b = 0
        o2u = 0.0
        o2p = 30.0
        o12ph = 3*3.141592/2
        r = -0.05
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

        Pers = []
        for i0 in range(len(SIRfiles)):
            XN = SIRfiles[i0]
            TT = np.zeros((len(XN),3))
            for j in range(len(XN)):
                TT[j][0] = XN[j][0]
                TT[j][1] = XN[j][1]
                TT[j][2] = XN[j][2]

            x = TT[:,1]
            Pea = []
            for j in range(1,len(TT)-1):
                if (x[j]>=x[j-1] and x[j]>=x[j+1]):
                    Pea.append(j)

            Per = (TT[Pea[1:len(Pea)],0]- TT[Pea[0:len(Pea)-1],0])/60.0
            Pers.append(np.mean(Per)/o1p)
            
        AT3[test] = np.mean(Pers)

    np.savetxt("Data/Fig2BGillespie_S3_%s_A3.txt"%(test2),AT3)



