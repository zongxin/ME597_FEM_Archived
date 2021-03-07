import os
import sys
import numpy as np
from pdb import set_trace
from matplotlib import rc as matplotlibrc
import matplotlib.pyplot as plt
import pickle
import copy
import parameters as para
import module as mod
import basefunc as base
import transformation as transf
import assemble as assemb
import mesh as Mesh

# initialize gauss points and weight
transf.init_gauss()

Mesh.generate_map()

Mesh.generate_mesh()



# initialize jacobian 

for inb in range(para.Num_Ele):
    transf.cal_jacobian(inb)

# Initial guess
assemb.initial_guess()

# Simulation start
res = 1
iter = 0
tol = 1e-5
itermax = 20
U = mod.U
while res>tol and iter<itermax:

    RR,KK = assemb.assembleRRKK()

    ##bc############################
    #left
    ie1 = 0
    for ie2 in range(para.Num_Ele2):
        ele = mod.ie2ele[ie2,ie1]
    
        ib1 = 0
        for ib2 in range(para.Num_Basis2):
            bas = mod.ib2bas[ib2,ib1]
        
            Rindex = mod.Map[ele,bas,0]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
            Rindex = mod.Map[ele,bas,1]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
            Rindex = mod.Map[ele,bas,2]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
    #right
    ie1 = para.Num_Ele1-1
    for ie2 in range(para.Num_Ele2):
        ele = mod.ie2ele[ie2,ie1]
    
        ib1 = para.Num_Basis1-1
        for ib2 in range(para.Num_Basis2):
            bas = mod.ib2bas[ib2,ib1]
        
            Rindex = mod.Map[ele,bas,0]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
            Rindex = mod.Map[ele,bas,1]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
            Rindex = mod.Map[ele,bas,2]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
    #Top
    ie2 = para.Num_Ele2-1
    for ie1 in range(para.Num_Ele1):
        ele = mod.ie2ele[ie2,ie1]
    
        ib2 = para.Num_Basis2-1
        for ib1 in range(para.Num_Basis1):
            bas = mod.ib2bas[ib2,ib1]
        
            Rindex = mod.Map[ele,bas,0]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
            Rindex = mod.Map[ele,bas,1]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
            Rindex = mod.Map[ele,bas,2]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
    #bottom
    ie2 = 0
    for ie1 in range(para.Num_Ele1):
        ele = mod.ie2ele[ie2,ie1]
    
        ib2 = 0
        for ib1 in range(para.Num_Basis1):
            bas = mod.ib2bas[ib2,ib1]
        
            Rindex = mod.Map[ele,bas,0]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
            Rindex = mod.Map[ele,bas,1]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
        
            Rindex = mod.Map[ele,bas,2]
            RR[Rindex] = 0.
            KK[Rindex] = np.zeros(para.Num_Unknown)
            KK[Rindex,Rindex] = 1.
    ################################
    
    res = np.max(np.abs(RR))
    incr_u = -np.linalg.solve(KK,RR)

    U += incr_u
    mod.U = U
    
    iter +=1
    print('iter %i'%iter)
    print(res)
