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

# module.py stores all the variables depending on parameters
# Including derivative operators, Jacobian, nodes mapping,
# CAUTIONS: all variables can be initialized and modified outside this file

# The nodes of element, the order MATTERS
#	4-----3
#	| inb |
#	1-----2



ie2ele = np.zeros( (para.Num_Ele2,para.Num_Ele1) )

ib2bas = np.zeros( (para.Num_Basis2,para.Num_Basis1) )

Map    = np.zeros( (para.Num_Ele,para.Num_Basis,para.Num_UNode) )

Node_X = np.zeros((para.Num_Ele,4,2),dtype=np.int32)


nl 	= para.Num_Basis1
nb 	= para.Num_Ele

#Define LG-mesh
xl 	  = np.zeros(nl)
xg 	  = np.zeros(5)
# Vector of Unknown
U 	  = np.zeros((para.Num_Unknown))

# The jacobian of each block/element
xX 	= np.zeros((nb,25))
yX 	= np.zeros((nb,25))
xY 	= np.zeros((nb,25))
yY 	= np.zeros((nb,25))
Joc = np.zeros((nb,25))

# Gauss point
IP_xi=np.zeros((25,2))
IP_wi=np.zeros((25))
