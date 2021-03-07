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

import basefunc as basef

def cal_u(x,y):
	uu=x**2*y
	return uu

def cal_ux(x,y):
	uu=2*x*y
	return uu

def cal_uy(x,y):
	uu=x**2
	return uu		

x=mod.xl
y=mod.xl
XX,YY=np.meshgrid(x,y)
u=XX**2*YY


xi  = 0.12
eta = 0.8

# test interpolation
N=basef.Nvec(xi,eta)
uint=np.sum(N*u)
print('real:',cal_u(xi,eta))
print('intp:',uint)

# test derivative

Nx=basef.dNvecdxi(xi,eta)
uxint=np.sum(Nx*u)
print('real ux:',cal_ux(xi,eta))
print('intp ux:',uxint)

Ny=basef.dNvecdeta(xi,eta)
uyint=np.sum(Ny*u)
print('real uy:',cal_uy(xi,eta))
print('intp uy:',uyint)


x=np.zeros(5)
x[0]=-1./3*(5+2*(10./7)**0.5)**0.5
x[1]=-1./3*(5-2*(10./7)**0.5)**0.5
x[2]=0
x[3]= 1./3*(5-2*(10./7)**0.5)**0.5
x[4]= 1./3*(5+2*(10./7)**0.5)**0.5
XX,YY=np.meshgrid(x,x)
w=np.zeros(5)
w[0]=(322.-13*(70)**0.5)/900
w[1]=(322.+13*(70)**0.5)/900
w[2]=128./225
w[3]=(322.+13*(70)**0.5)/900
w[4]=(322.-13*(70)**0.5)/900
set_trace()