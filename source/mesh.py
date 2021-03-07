import numpy as np
import math
from numpy import sqrt, cos, sin
import module as mod
import parameters as para
from pdb import set_trace
#################################################################

########Mapping##################################
def generate_map():

  Num_Ele1    = para.Num_Ele1
  Num_Ele2    = para.Num_Ele2
  Num_Ele     = para.Num_Ele
  Num_Basis1  = para.Num_Basis1
  Num_Basis2  = para.Num_Basis2
  Num_Basis   = para.Num_Basis
  Num_UNode   = para.Num_UNode

  ie2ele = np.zeros( (Num_Ele2,Num_Ele1) )
  ele2ie = np.zeros( (Num_Ele,2) )
  ele = 0
  for ie2 in range(Num_Ele2):
    for ie1 in range(Num_Ele1):
      # Given ie1,ie2, obtain ele
      ie2ele[ie2,ie1] = ele
      # Given ele, obtain ie1,ie2
      ele2ie[ele,0] = ie1
      ele2ie[ele,1] = ie2
      #update ele
      ele += 1
  ie2ele = ie2ele.astype (int)
  ele2ie = ele2ie.astype(int)
  mod.ie2ele = ie2ele*1
  #mod.ele2ie = ele2ie*1

  ib2bas = np.zeros( (Num_Basis2,Num_Basis1) )
  bas2ib = np.zeros( (Num_Basis,2) )
  bas = 0
  for ib2 in range(Num_Basis2):
    for ib1 in range(Num_Basis1):
      #Given ib1,ib2, obtain bas
      ib2bas[ib2,ib1] = bas
      #Given bas, obtain ib1,ib2
      bas2ib[bas,0] = ib1
      bas2ib[bas,1] = ib2
      #update bas
      bas += 1
  ib2bas = ib2bas.astype(int)
  bas2ib = bas2ib.astype(int)
  mod.ib2bas = ib2bas*1
  #mod.bas2ib = bas2ib*1
  
  Map = np.zeros( (Num_Ele,Num_Basis,Num_UNode) )
  nod = 0
  for ele in range(Num_Ele):
    for bas in range(Num_Basis):
      #obtain ie1,ie2
      ie1 = ele2ie[ele,0]
      ie2 = ele2ie[ele,1]
      #obtain ib1,ib2
      ib1 = bas2ib[bas,0]
      ib2 = bas2ib[bas,1]

      #left-most basis
      if ie1 != 0 and ib1 == 0:
        ele_l = ie2ele[ie2,ie1-1]
        bas_r = ib2bas[ib2,-1]
        for unode in range(Num_UNode):
          Map[ele,bas,unode] = Map[ele_l,bas_r,unode]
      #bottom-most basis
      elif ie2 != 0 and ib2 == 0:
        ele_b = ie2ele[ie2-1,ie1]
        bas_t = ib2bas[-1,ib1]
        for unode in range(Num_UNode):
          Map[ele,bas,unode] = Map[ele_b,bas_t,unode]
      else:
        for unode in range(Num_UNode):
          Map[ele,bas,unode] = nod
          nod += 1   
  Map = Map.astype(int)
  mod.Map = Map*1
  # set_trace()
###################
##################################

########Mesh##################################
# fill in mod.Node_X
def generate_mesh():

  X1_Start    = para.X1_Start
  X1_Final    = para.X1_Final
  X2_Start    = para.X2_Start
  X2_Final    = para.X2_Final
  Num_Ele1    = para.Num_Ele1
  Num_Ele2    = para.Num_Ele2
  Num_Ele     = para.Num_Ele

  X1 = np.linspace(X1_Start,X1_Final,Num_Ele1+1)
  X2 = np.linspace(X2_Start,X2_Final,Num_Ele2+1)
  
  Node_X = np.zeros((Num_Ele,4,2))
  for ie1 in range(Num_Ele1):
    for ie2 in range(Num_Ele2):
      x1e = np.array([X1[ie1],X2[ie2]])
      x2e = np.array([X1[ie1+1],X2[ie2]])
      x3e = np.array([X1[ie1+1],X2[ie2+1]])
      x4e = np.array([X1[ie1],X2[ie2+1]])
      
      ele = mod.ie2ele[ie2,ie1]
      Node_X[ele,0] = x1e
      Node_X[ele,1] = x2e
      Node_X[ele,2] = x3e
      Node_X[ele,3] = x4e
  
  mod.Node_X = Node_X*1.
#################################################
