import os
import sys
import numpy as np
from pdb import set_trace
import parameters as para
import module as mod

# nl 		= para.Num_Basis1
# xl 		= mod.xl

def init_gauss():

	x=np.zeros(5)
	x[0]=-1./3*(5+2*(10./7)**0.5)**0.5
	x[1]=-1./3*(5-2*(10./7)**0.5)**0.5
	x[2]=0
	x[3]= 1./3*(5-2*(10./7)**0.5)**0.5
	x[4]= 1./3*(5+2*(10./7)**0.5)**0.5

	XX,YY=np.meshgrid(x,x)
	XXYY=np.zeros((25,2))
	
	XXYY[:,0]=XX.flatten()
	XXYY[:,1]=YY.flatten()	

	w=np.zeros(5)
	w[0]=(322.-13*(70)**0.5)/900
	w[1]=(322.+13*(70)**0.5)/900
	w[2]=128./225
	w[3]=(322.+13*(70)**0.5)/900
	w[4]=(322.-13*(70)**0.5)/900
	W=np.outer(w,w)
	W=W.flatten()

	mod.xg 		= x
	mod.IP_xi	= XXYY
	mod.IP_wi	= W



def get_LINE_eq(X,x_end ,x_0):
	P=x_0+(x_end-x_0)*(X+1)/2
	return P


def cal_jacobian(nb):

	xg 		= mod.xg 

	x_no 	= mod.Node_X[nb,:,0]
	y_no 	= mod.Node_X[nb,:,1] 

	P1x=get_LINE_eq(xg,x_no[1],x_no[0]);
	P1y=get_LINE_eq(xg,y_no[1],y_no[0]);
	P2x=get_LINE_eq(xg,x_no[2],x_no[1]);
	P2y=get_LINE_eq(xg,y_no[2],y_no[1]);
	P3x=get_LINE_eq(xg,x_no[2],x_no[3]);
	P3y=get_LINE_eq(xg,y_no[2],y_no[3]);
	P4x=get_LINE_eq(xg,x_no[3],x_no[0]);
	P4y=get_LINE_eq(xg,y_no[3],y_no[0]);

	der_P1x=(x_no[1]-x_no[0])/2.
	der_P1y=(y_no[1]-y_no[0])/2.
	der_P2x=(x_no[2]-x_no[1])/2.
	der_P2y=(y_no[2]-y_no[1])/2.
	der_P3x=(x_no[2]-x_no[3])/2.
	der_P3y=(y_no[2]-y_no[3])/2.
	der_P4x=(x_no[3]-x_no[0])/2.
	der_P4y=(y_no[3]-y_no[0])/2.

	xX = np.zeros((5,5))
	xY = np.zeros((5,5))
	yX = np.zeros((5,5))
	yY = np.zeros((5,5))	

	for i in range(5): 		# y	coord	
		for j in range(5):		# x	coord	
			xX[i,j]=(1-xg[i])/2.*der_P1x\
					+(1+xg[i])/2.*der_P3x\
					-P4x[i]/2.\
					+P2x[i]/2.\
					+x_no[0]*(1-xg[i])/4.\
					-x_no[1]*(1-xg[i])/4.\
					-x_no[2]*(1+xg[i])/4.\
					+x_no[3]*(1+xg[i])/4.
			# set_trace()
	for i in range(5): 		# y	coord	
		for j in range(5):		# x	coord	
			yX[i,j]=(1-xg[i])/2.*der_P1y\
					+(1+xg[i])/2.*der_P3y\
					-P4y[i]/2.\
					+P2y[i]/2.\
					+y_no[0]*(1-xg[i])/4.\
					-y_no[1]*(1-xg[i])/4.\
					-y_no[2]*(1+xg[i])/4.\
					+y_no[3]*(1+xg[i])/4.



	for i in range(5): 		# y	coord	
		for j in range(5):		# x	coord	
			xY[i,j]= -P1x[j]/2.\
					+P3x[j]/2.\
					+(1-xg[j])/2.*der_P4x\
					+(1+xg[j])/2.*der_P2x\
					+x_no[0]*(1-xg[j])/4.\
					+x_no[1]*(1+xg[j])/4.\
					-x_no[2]*(1+xg[j])/4.\
					-x_no[3]*(1-xg[j])/4


	for i in range(5): 		# y	coord	
		for j in range(5):		# x	coord	
			yY[i,j]= -P1y[j]/2.\
					+P3y[j]/2.\
					+(1-xg[j])/2.*der_P4y\
					+(1+xg[j])/2.*der_P2y\
					+y_no[0]*(1-xg[j])/4.\
					+y_no[1]*(1+xg[j])/4.\
					-y_no[2]*(1+xg[j])/4.\
					-y_no[3]*(1-xg[j])/4


	mod.xX[nb,:] = xX.flatten()	
	mod.xY[nb,:] = xY.flatten()
	mod.yX[nb,:] = yX.flatten()
	mod.yY[nb,:] = yY.flatten()
	mod.Joc[nb,:]= (mod.xX[nb,:]*mod.yY[nb,:]-mod.xY[nb,:]*mod.yX[nb,:])


def derv_xy(u,dNsdxi,dNsdeta,inb,ip):

	dxi_dx  = 	1./mod.Joc[inb,ip]*mod.yY[inb,ip]
	deta_dx = 	1./mod.Joc[inb,ip]*mod.yX[inb,ip]*(-1.)
	dxi_dy 	= 	1./mod.Joc[inb,ip]*mod.xY[inb,ip]*(-1.)
	deta_dy = 	1./mod.Joc[inb,ip]*mod.xX[inb,ip]

	du_dxi	= np.dot(u,dNsdxi)
	du_deta	= np.dot(u,dNsdeta)

	dudx = du_dxi*dxi_dx+du_deta*deta_dx
	dudy = du_dxi*dxi_dy+du_deta*deta_dy	
	return dudx,dudy



