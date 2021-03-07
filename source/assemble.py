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

Num_Basis1  = para.Num_Basis1
Num_Basis2  = para.Num_Basis2
Num_UNode   = para.Num_UNode
Num_Ele     = para.Num_Ele
Num_Basis   = para.Num_Basis
Num_Unknown = para.Num_Unknown

h   = para.h
f   = para.f
c   = para.c
E   = para.E
mu  = para.mu
mu2 = mu*mu
# To be Done
#--> Pick known in this element from large matrix
# --> Define interpolation points IP_xi and weight IP_wi

def assembleRRKK():
    R = np.zeros(Num_Unknown)
    K = np.zeros((Num_Unknown,Num_Unknown))

    Map     = mod.Map
    U       = mod.U
    for inb in range(Num_Ele):

        ux_e     = U[Map[inb,:,0]]#.reshape(Num_Basis2, Num_Basis1)
        uy_e     = U[Map[inb,:,1]]#.reshape(Num_Basis2, Num_Basis1)
        w_e      = U[Map[inb,:,2]]#.reshape(Num_Basis2, Num_Basis1)
        W_xx_e   = U[Map[inb,:,3]]#.reshape(Num_Basis2, Num_Basis1)
        W_yy_e   = U[Map[inb,:,4]]#.reshape(Num_Basis2, Num_Basis1)
        W_xy_e   = U[Map[inb,:,5]]#.reshape(Num_Basis2, Num_Basis1)
        Xia_xx_e = U[Map[inb,:,6]]#.reshape(Num_Basis2, Num_Basis1)
        Xia_yy_e = U[Map[inb,:,7]]#.reshape(Num_Basis2, Num_Basis1)
        Xia_xy_e = U[Map[inb,:,8]]#.reshape(Num_Basis2, Num_Basis1)
        Xiw_xx_e = U[Map[inb,:,9]]#.reshape(Num_Basis2, Num_Basis1)
        Xiw_yy_e = U[Map[inb,:,10]]#.reshape(Num_Basis2, Num_Basis1)
        Xiw_xy_e = U[Map[inb,:,11]]#.reshape(Num_Basis2, Num_Basis1)
        va_xx_e  = U[Map[inb,:,12]]#.reshape(Num_Basis2, Num_Basis1)
        va_yy_e  = U[Map[inb,:,13]]#.reshape(Num_Basis2, Num_Basis1)
        va_xy_e  = U[Map[inb,:,14]]#.reshape(Num_Basis2, Num_Basis1)
        
        dxi_dx  = 1./mod.Joc[inb,:]*mod.yY[inb,:]
        deta_dx = 1./mod.Joc[inb,:]*(-mod.yX[inb,:])
        dxi_dy  = 1./mod.Joc[inb,:]*(-mod.xY[inb,:])
        deta_dy = 1./mod.Joc[inb,:]*mod.xX[inb,:]
        J   = mod.Joc[inb,:]

        Re = np.zeros((Num_UNode*Num_Basis))
        Ke = np.zeros((Num_UNode*Num_Basis,Num_UNode*Num_Basis))
        for ip in range(25):
            xi  = mod.IP_xi[ip,0]
            eta = mod.IP_xi[ip,1]
            wi  = mod.IP_wi[ip]
            # eval shape functions 
            Ns      = base.Nvec(xi,eta)
            dNsdxi  = base.dNvecdxi(xi,eta)
            dNsdeta = base.dNvecdeta(xi,eta)

            for ni in range(Num_Basis):
                deltav_e    = np.zeros((Num_Basis))
                deltav_e[ni]= 1.
                deltav_e    = deltav_e

                deltav      = np.dot(deltav_e,Ns)
                ddeltav_dx,ddeltav_dy = transf.derv_xy(deltav_e,dNsdxi,dNsdeta,inb,ip)



                #===============================================================
                # Eq. 1.2
                va_xx  = np.dot(va_xx_e,Ns)
                va_yy  = np.dot(va_yy_e,Ns)      
                va_xy  = np.dot(va_xy_e,Ns)
                Xia_xx = np.dot(Xia_xx_e,Ns)
                Xia_yy = np.dot(Xia_yy_e,Ns)        
                Xia_xy = np.dot(Xia_xy_e,Ns)

                r1 = (va_xx - c*Xia_xx)*ddeltav_dx + (va_xy - c*Xia_xy)*ddeltav_dy  
                r2 = (va_yy - c*Xia_yy)*ddeltav_dy + (va_xy - c*Xia_xy)*ddeltav_dx

                Re[Num_UNode*ni]   += wi*J[ip]*r1
                Re[Num_UNode*ni+1] += wi*J[ip]*r2

                #===============================================================
                # Eq. 3
                vw_xx_e = E/(1-mu2)*((1-mu)*W_xx_e+mu*(W_xx_e+W_yy_e))
                vw_yy_e = E/(1-mu2)*((1-mu)*W_yy_e+mu*(W_xx_e+W_yy_e))
                vw_xy_e = E/(1-mu2)*((1-mu)*W_xy_e)

                dvw_xx_dx,dvw_xx_dy   = transf.derv_xy(vw_xx_e,dNsdxi,dNsdeta,inb,ip)
                dvw_xy_dx,dvw_xy_dy   = transf.derv_xy(vw_xy_e,dNsdxi,dNsdeta,inb,ip)
                dvw_yy_dx,dvw_yy_dy   = transf.derv_xy(vw_yy_e,dNsdxi,dNsdeta,inb,ip)

                dXiw_xx_dx,dXiw_xx_dy = transf.derv_xy(Xiw_xx_e,dNsdxi,dNsdeta,inb,ip)
                dXiw_xy_dx,dXiw_xy_dy = transf.derv_xy(Xiw_xy_e,dNsdxi,dNsdeta,inb,ip)
                dXiw_yy_dx,dXiw_yy_dy = transf.derv_xy(Xiw_yy_e,dNsdxi,dNsdeta,inb,ip)

                r3 = (dvw_xx_dx-c*dXiw_xx_dx)*ddeltav_dx+\
                     (dvw_xy_dy-c*dXiw_xy_dy)*ddeltav_dx+\
                     (dvw_xy_dx-c*dXiw_xy_dx)*ddeltav_dy+\
                     (dvw_yy_dy-c*dXiw_yy_dy)*ddeltav_dy+\
                      -12*f/h**3 * deltav   

                Re[Num_UNode*ni+2] += wi*J[ip]*r3

                #===============================================================
                # Eq. 4.5.6
                # Xia_xx calculated in Eq.1


                dva_xx_dx,dva_xx_dy = transf.derv_xy(va_xx_e,dNsdxi,dNsdeta,inb,ip)
                dva_xy_dx,dva_xy_dy = transf.derv_xy(va_xy_e,dNsdxi,dNsdeta,inb,ip)
                dva_yy_dx,dva_yy_dy = transf.derv_xy(va_yy_e,dNsdxi,dNsdeta,inb,ip)

                r4 = (dva_xx_dx*ddeltav_dx+dva_xx_dy*ddeltav_dy) + Xia_xx*deltav
                r5 = (dva_yy_dx*ddeltav_dx+dva_yy_dy*ddeltav_dy) + Xia_yy*deltav        
                r6 = (dva_xy_dx*ddeltav_dx+dva_xy_dy*ddeltav_dy) + Xia_xy*deltav

                Re[Num_UNode*ni+3] += wi*J[ip]*r4
                Re[Num_UNode*ni+4] += wi*J[ip]*r5   
                Re[Num_UNode*ni+5] += wi*J[ip]*r6   

                #===============================================================
                # Eq. 7.8.9
                # dvw_xx_dx,dvw_xx_dy calculated in Eq.2    

                Xiw_xx = np.dot(Xiw_xx_e,Ns)
                Xiw_yy = np.dot(Xiw_yy_e,Ns)        
                Xiw_xy = np.dot(Xiw_xy_e,Ns)

                r7 = (dvw_xx_dx*ddeltav_dx+dvw_xx_dy*ddeltav_dy) + Xiw_xx*deltav
                r8 = (dvw_yy_dx*ddeltav_dx+dvw_yy_dy*ddeltav_dy) + Xiw_yy*deltav        
                r9 = (dvw_xy_dx*ddeltav_dx+dvw_xy_dy*ddeltav_dy) + Xiw_xy*deltav

                Re[Num_UNode*ni+6] += wi*J[ip]*r7
                Re[Num_UNode*ni+7] += wi*J[ip]*r8   
                Re[Num_UNode*ni+8] += wi*J[ip]*r9

                #===============================================================
                # Eq. 10.11.12

                W_xx = np.dot(W_xx_e,Ns)
                W_yy = np.dot(W_yy_e,Ns)        
                W_xy = np.dot(W_xy_e,Ns)
                dw_dx,dw_dy = transf.derv_xy(w_e,dNsdxi,dNsdeta,inb,ip)

                r10 = W_xx*deltav + dw_dx * ddeltav_dx
                r11 = W_yy*deltav + dw_dy * ddeltav_dy  
                r12 = W_xy*deltav + dw_dy * ddeltav_dx  

                Re[Num_UNode*ni+9] += wi*J[ip]*r10
                Re[Num_UNode*ni+10] += wi*J[ip]*r11 
                Re[Num_UNode*ni+11] += wi*J[ip]*r12


                #===============================================================
                # Eq. 13.14.15
                # va_xx calculated 
                # dw_dx calculated 

                dux_dx,dux_dy = transf.derv_xy(ux_e,dNsdxi,dNsdeta,inb,ip)
                duy_dx,duy_dy = transf.derv_xy(uy_e,dNsdxi,dNsdeta,inb,ip)

                epsa_xx = 0.5*(dux_dx+dux_dx+dw_dx*dw_dx)
                epsa_yy = 0.5*(duy_dy+duy_dy+dw_dy*dw_dy)       
                epsa_xy = 0.5*(dux_dy+duy_dx+dw_dx*dw_dy)

                r13 = (va_xx-E/(1-mu2)*((1-mu)*epsa_xx+\
                                   mu*(epsa_xx+epsa_yy)))*deltav
                r14 = (va_yy-E/(1-mu2)*((1-mu)*epsa_yy+\
                               mu*(epsa_xx+epsa_yy)))*deltav
                r15 = (va_xy-E/(1-mu2)*(1-mu)*epsa_xy)*deltav   

                Re[Num_UNode*ni+12] += wi*J[ip]*r13
                Re[Num_UNode*ni+13] += wi*J[ip]*r14
                Re[Num_UNode*ni+14] += wi*J[ip]*r15

                for nj in range(Num_Basis):
                    deltau_e = np.zeros((Num_Basis))
                    deltau_e[nj] = 1.
                    

                    deltau = np.dot(deltau_e,Ns)
                    ddeltau_dx,ddeltau_dy = transf.derv_xy(deltau_e,dNsdxi,dNsdeta,inb,ip)
                    #===============================================================
                    Ke[Num_UNode*ni+0,Num_UNode*nj+12] += wi*J[ip] * ddeltav_dx * deltau 
                    Ke[Num_UNode*ni+0,Num_UNode*nj+14] += wi*J[ip] * ddeltav_dy * deltau 
                    Ke[Num_UNode*ni+0,Num_UNode*nj+6]  += wi*J[ip] * ddeltav_dx * deltau *(-c)
                    Ke[Num_UNode*ni+0,Num_UNode*nj+8]  += wi*J[ip] * ddeltav_dy * deltau *(-c)

                    #===============================================================
                    Ke[Num_UNode*ni+1,Num_UNode*nj+14] += wi*J[ip] * ddeltav_dx * deltau 
                    Ke[Num_UNode*ni+1,Num_UNode*nj+13] += wi*J[ip] * ddeltav_dy * deltau 
                    Ke[Num_UNode*ni+1,Num_UNode*nj+8]  += wi*J[ip] * ddeltav_dx * deltau *(-c)
                    Ke[Num_UNode*ni+1,Num_UNode*nj+7]  += wi*J[ip] * ddeltav_dy * deltau *(-c)

                    #===============================================================
                    ke3_4 = E/(1-mu2)*((1-mu)*ddeltav_dx*ddeltau_dx\
                                +mu*(ddeltav_dx*ddeltau_dx+ddeltav_dy*ddeltau_dy))
                    ke3_5 = E/(1-mu2)*((1-mu)*ddeltav_dy*ddeltau_dy\
                                +mu*(ddeltav_dx*ddeltau_dx+ddeltav_dy*ddeltau_dy))      
                    ke3_6 = E/(1-mu2)*((1-mu)*ddeltav_dx*ddeltau_dy)  

                    Ke[Num_UNode*ni+2,Num_UNode*nj+3]  += wi*J[ip] * ke3_4    
                    Ke[Num_UNode*ni+2,Num_UNode*nj+4]  += wi*J[ip] * ke3_5
                    Ke[Num_UNode*ni+2,Num_UNode*nj+5]  += wi*J[ip] * ke3_6
                    Ke[Num_UNode*ni+2,Num_UNode*nj+9]  += wi*J[ip] * ddeltav_dx * ddeltau_dx *(-c)
                    Ke[Num_UNode*ni+2,Num_UNode*nj+11] += wi*J[ip] * ddeltav_dx * ddeltau_dy *(-c)
                    Ke[Num_UNode*ni+2,Num_UNode*nj+10] += wi*J[ip] * ddeltav_dy * ddeltau_dy *(-c)       

                    #===============================================================
                    Ke[Num_UNode*ni+3,Num_UNode*nj+6]   += wi*J[ip] * deltav * deltau                    
                    Ke[Num_UNode*ni+3,Num_UNode*nj+12]  += wi*J[ip] * (ddeltav_dx*ddeltau_dx + ddeltav_dy*ddeltau_dy)


                        #===============================================================
                    Ke[Num_UNode*ni+4,Num_UNode*nj+7]   += wi*J[ip] * deltav * deltau
                    Ke[Num_UNode*ni+4,Num_UNode*nj+13]  += wi*J[ip] * (ddeltav_dx*ddeltau_dx + ddeltav_dy*ddeltau_dy)

                    #===============================================================
                    Ke[Num_UNode*ni+5,Num_UNode*nj+8]   += wi*J[ip] * deltav * deltau  
                    Ke[Num_UNode*ni+5,Num_UNode*nj+14]  += wi*J[ip] * (ddeltav_dx*ddeltau_dx + ddeltav_dy*ddeltau_dy)  

                    #===============================================================
                    Ke[Num_UNode*ni+6,Num_UNode*nj+9]  += wi*J[ip] * deltav * deltau
                    Ke[Num_UNode*ni+6,Num_UNode*nj+3]  += wi*J[ip] * E/(1-mu2) * (ddeltav_dx*ddeltau_dx + ddeltav_dy*ddeltau_dy)  
                    Ke[Num_UNode*ni+6,Num_UNode*nj+4]  += wi*J[ip] * E*mu/(1-mu2) * (ddeltav_dx*ddeltau_dx + ddeltav_dy*ddeltau_dy)

                    #===============================================================
                    Ke[Num_UNode*ni+7,Num_UNode*nj+10]  += wi*J[ip] * deltav * deltau
                    Ke[Num_UNode*ni+7,Num_UNode*nj+3]  += wi*J[ip] * E*mu/(1-mu2) * (ddeltav_dx*ddeltau_dx + ddeltav_dy*ddeltau_dy)
                    Ke[Num_UNode*ni+7,Num_UNode*nj+4]  += wi*J[ip] * E/(1-mu2) * (ddeltav_dx*ddeltau_dx + ddeltav_dy*ddeltau_dy)  

                    #===============================================================
                    Ke[Num_UNode*ni+8,Num_UNode*nj+11]  += wi*J[ip] * deltav * deltau  
                    Ke[Num_UNode*ni+8,Num_UNode*nj+5]  += wi*J[ip] * E*(1-mu)/(1-mu2) * (ddeltav_dx*ddeltau_dx + ddeltav_dy*ddeltau_dy)   

                    #===============================================================
                    Ke[Num_UNode*ni+9,Num_UNode*nj+3]  += wi*J[ip] * deltav * deltau
                    Ke[Num_UNode*ni+9,Num_UNode*nj+2]  += wi*J[ip] * ddeltav_dx * ddeltau_dx 
                    #===============================================================
                    Ke[Num_UNode*ni+10,Num_UNode*nj+4] += wi*J[ip] * deltav  * deltau
                    Ke[Num_UNode*ni+10,Num_UNode*nj+2] += wi*J[ip] * ddeltav_dy * ddeltau_dy

                    #===============================================================
                    Ke[Num_UNode*ni+11,Num_UNode*nj+5] += wi*J[ip] * deltav * deltau
                    Ke[Num_UNode*ni+11,Num_UNode*nj+2] += wi*J[ip] * ddeltav_dx * ddeltau_dy

                    #===============================================================
                    Ke[Num_UNode*ni+12,Num_UNode*nj+12]+= wi*J[ip] * deltav * deltau
                    Ke[Num_UNode*ni+12,Num_UNode*nj+0] += wi*J[ip] * (-E)/(1-mu2) * ddeltau_dx * deltav
                    Ke[Num_UNode*ni+12,Num_UNode*nj+1] += wi*J[ip] * (-E)*mu/(1-mu2) * ddeltau_dy * deltav

                    Ke[Num_UNode*ni+12,Num_UNode*nj+2] += wi*J[ip] * (-E)/(1-mu2) * ((1-mu)*ddeltau_dx*dw_dx+\
                                                         mu*(ddeltau_dx*dw_dx + ddeltau_dy*dw_dy)  )*deltav

                    #===============================================================
                    Ke[Num_UNode*ni+13,Num_UNode*nj+13]+= wi*J[ip] * deltav * deltau
                    Ke[Num_UNode*ni+13,Num_UNode*nj+0] += wi*J[ip] * (-E)*mu/(1-mu2) * ddeltau_dx * deltav
                    Ke[Num_UNode*ni+13,Num_UNode*nj+1] += wi*J[ip] * (-E)/(1-mu2) * ddeltau_dy * deltav
                
                    Ke[Num_UNode*ni+13,Num_UNode*nj+2] += wi*J[ip] * (-E)/(1-mu2) * ((1-mu)*ddeltau_dy*dw_dy+\
                                                         mu*(ddeltau_dx*dw_dx + ddeltau_dy*dw_dy)  )*deltav     
                    #===============================================================
                    Ke[Num_UNode*ni+14,Num_UNode*nj+14]+= wi*J[ip] * deltav * deltau 
                    Ke[Num_UNode*ni+14,Num_UNode*nj+0] += wi*J[ip] * (-E)*(1-mu)/(2*(1-mu2)) * ddeltau_dy * deltav
                    Ke[Num_UNode*ni+14,Num_UNode*nj+1] += wi*J[ip] * (-E)*(1-mu)/(2*(1-mu2)) * ddeltau_dx * deltav

                    Ke[Num_UNode*ni+14,Num_UNode*nj+2] += wi*J[ip] * (-E)*(1-mu)/(2*(1-mu2)) \
                                       *(ddeltau_dx*dw_dy + ddeltau_dy*dw_dx)* deltav
                    
        ########################
        #global assembling
        ###########################
        for ni in range(Num_Basis):
            for ui in range(Num_UNode):
                Row=Map[inb,ni,ui]
                R[Row] += Re[Num_UNode*ni+ui]
            
                for nj in range(Num_Basis):
                    for uj in range(Num_UNode):
                        Col=Map[inb,nj,uj]
                        K[Row,Col] += Ke[Num_UNode*ni+ui,Num_UNode*nj+uj]              
    return R,K



def initial_guess():
    Num_Node = para.Num_Node

    ux      = np.zeros(Num_Node)
    uy      = np.zeros(Num_Node)
    w       = np.zeros(Num_Node)
    W_xx    = np.zeros(Num_Node)
    W_yy    = np.zeros(Num_Node)
    W_xy    = np.zeros(Num_Node)
    Xia_xx  = np.zeros(Num_Node)
    Xia_yy  = np.zeros(Num_Node)
    Xia_xy  = np.zeros(Num_Node)
    Xiw_xx  = np.zeros(Num_Node)
    Xiw_yy  = np.zeros(Num_Node)
    Xiw_xy  = np.zeros(Num_Node)
    va_xx   = np.zeros(Num_Node)
    va_yy   = np.zeros(Num_Node)
    va_xy   = np.zeros(Num_Node)

    Num_Unknown = para.Num_Unknown
    Num_UNode   = para.Num_UNode
    U           = np.zeros(Num_Unknown)
    for nod in range(Num_Node):
        U[Num_UNode*nod+0]  = ux[nod]
        U[Num_UNode*nod+1]  = uy[nod]
        U[Num_UNode*nod+2]  = w[nod]
        U[Num_UNode*nod+3]  = W_xx[nod]
        U[Num_UNode*nod+4]  = W_yy[nod]
        U[Num_UNode*nod+5]  = W_xy[nod]
        U[Num_UNode*nod+6]  = Xia_xx[nod]
        U[Num_UNode*nod+7]  = Xia_yy[nod]
        U[Num_UNode*nod+8]  = Xia_xy[nod]
        U[Num_UNode*nod+9]  = Xiw_xx[nod]
        U[Num_UNode*nod+10] = Xiw_yy[nod]
        U[Num_UNode*nod+11] = Xiw_xy[nod]
        U[Num_UNode*nod+12] = va_xx[nod]
        U[Num_UNode*nod+13] = va_yy[nod]
        U[Num_UNode*nod+14] = va_xy[nod]
        
    mod.U = U
