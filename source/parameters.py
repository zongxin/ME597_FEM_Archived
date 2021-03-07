import os
import sys
import numpy as np
from pdb import set_trace
from matplotlib import rc as matplotlibrc
import matplotlib.pyplot as plt
import pickle
import copy
# import module as mod
# parameters.py stores all the input parameters including
# physical parameter, mesh parameters

Num_Ele1 	= 2
Num_Ele2 	= 2
Num_Ele 	= Num_Ele1*Num_Ele2

Num_Basis1 	= 3
Num_Basis2 	= 3
Num_Basis  	= Num_Basis1*Num_Basis2

Num_UNode  	= 15


Num_Node1 	= (Num_Basis1-1)*Num_Ele1+1
Num_Node2 	= (Num_Basis2-1)*Num_Ele2+1
Num_Node  	= Num_Node1*Num_Node2


Num_Unknown = Num_Node*Num_UNode



X1_Start = 0.
X1_Final = 1.
X2_Start = 0.
X2_Final = 1.

c = 0.1
E = 6.9138
mu = 2.5967
f = 1000000
h = 1.
