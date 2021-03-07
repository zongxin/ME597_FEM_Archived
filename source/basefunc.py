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


#Reading from input para
nl 		= para.Num_Basis1
N       = nl-1
base    = np.arange(nl)*1.0

# Define LG-mesh
mod.xl  = - np.cos(np.pi*base/N)


# base function for any single value
def base(xg,x):
	h=np.ones((len(xg)))
	for j in range(len(xg)):
		for i in range(len(xg)):
			if (i!=j):
				m=(x-xg[i])/(xg[j]-xg[i])
				h[j]=h[j]*m
	return h


# first order base func 
def der_base(xl,x):
	k={}
	h=np.ones((len(xl)))

	temp=np.zeros(len(xl))

	for j in range(len(xl)):
		b=np.ones((len(xl)-1))

		xx=np.ones((len(xl)-1))*x
		for i in range(len(xl)):
			k=list(copy.deepcopy(xl))
			bb=1
			k.remove(xl[j])
			b=xl[j]-k
			t=xx-k
			tt=np.ones((len(xl)-1))     
			for z,value in enumerate(t):
				bb=bb*b[z]#over
				for zz in range(len(t)):
					if (zz!=z):
						tt[z]=tt[z]*t[zz]
		temp[j]=np.sum(tt)
		h[j]=temp[j]/bb
		#set_trace()
	return h  

def Nvec(xi,eta):
	h_X	=	base(mod.xl,xi)
	h_Y	=	base(mod.xl,eta)
	h 	=	np.outer(h_Y,h_X)
	# set_trace()
	return h.flatten()

def dNvecdxi(xi,eta):
	h_X	=	der_base(mod.xl,xi)
	h_Y	=	base(mod.xl,eta)
	h 	=	np.outer(h_Y,h_X)
	return h.flatten()

def dNvecdeta(xi,eta):
	h_X =	base(mod.xl,xi)
	h_Y =	der_base(mod.xl,eta)
	h 	=	np.outer(h_Y,h_X)
	return h.flatten()





