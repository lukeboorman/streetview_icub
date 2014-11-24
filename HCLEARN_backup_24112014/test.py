from numpy import *
import numpy as np
import pdb
from makeMaze import *
from paths import *
from cffun import *
from rbm import *

def err(ps, hids):
    return sum( (ps-hids)**2 ) / hids.shape[0]


def fuse2(p1,p2):
    return 1.0 / (1.0 +   ((1-p1)*(1-p2)/(p1 * p2)  ))

#data was noisy GPS firing
WR = np.load('tWR.npy')
WO = np.load('tWO.npy')
WS = np.load('tWS.npy')
WB = np.load('tWB.npy'); WB=WB.reshape((86,1)) ## WHY 86 LB!

WR = np.random.random(WR.shape)

hids=np.load('hids.npy')
odom=np.load('odom.npy')
senses=np.load('senses.npy')


hidslag = lag(hids,1)

#predictions from lam only

T = hids.shape[0]

p_null = stripBias(boltzmannProbs(WB, np.ones((T,1)).transpose() ).transpose())
e_null = sum( (p_null-hids)**2 ) / T

p_odom = stripBias(boltzmannProbs(WO, addBias(odom).transpose()).transpose())
e_odom = sum( (p_odom-hids)**2 ) / T

p_senses = stripBias(boltzmannProbs(WS, addBias(senses).transpose()).transpose())

e_senses = sum( (p_senses-hids)**2 ) / T

#predictions from pi only
p_trans = stripBias(boltzmannProbs(WR, addBias(hidslag).transpose()).transpose())
e_trans = sum( (p_trans-hids)**2 ) / T

p_senses_null = fuse(p_senses, p_null)
p_odom_null   = fuse(p_odom, p_null)
p_trans_null  = fuse(p_trans, p_null)


p_all = fuse(p_null, p_odom)
p_all = fuse(p_all, p_senses)
p_all = fuse(p_all, p_trans) 
e_all = sum( (p_all-hids)**2 ) / T


##TODO convert input and output odom back to EC form.  Plot decoded place errors.


pdb.set_trace()

dictGrids = DictGrids()
ca3_hat_b = addBias(p_all)
ca3_gnd_b = addBias(hids)
xy_hat = np.zeros((T,2))
xy_gnd = np.zeros((T,2))
for t in range(0,T):
    v_ca3_hat = ca3_hat_b[t,:]
    v_ca3_gnd = ca3_gnd_b[t,:]

    #TODO make CA3 smart structure -- for hat only (its already optimal in gnd)
    ca3 = CA3StateFromVector(v_ca3_hat,N_places) # Altered by luke, 13)
    ca3.smartCollapse()
    v_ca3_hat = hstack((ca3.toVector(),1))

    p_odom   = boltzmannProbs(WO.transpose(), v_ca3_hat) #probs for CA1 cells
    p_senses = boltzmannProbs(WS.transpose(), v_ca3_hat) #probs for CA1 cells
    ca1 = CA1State(p_odom, p_senses,_,dictGrids) # Luke added dictGrids (removed n_places)
    loc = Location(dictGrids) # Luke modified
    loc.setGrids(ca1.grids) # Luke modified
    xy_hat[t,:] = loc.getXY()
    

    #get ground truth xy, from decoding ground truth hids?
    p_odom   = boltzmannProbs(WO.transpose(), v_ca3_gnd) #probs for CA1 cells
    p_senses = boltzmannProbs(WS.transpose(), v_ca3_gnd) #probs for CA1 cells
    ca1 = CA1State(p_odom, p_senses,_,dictGrids) # Luke added dictGrids (removed n_places)
    loc = Location(dictGrids)# Luke modified
    loc.setGrids(ca1.grids)# Luke modified
    xy_gnd[t,:] = loc.getXY()
    

e =  sum (sum(xy_hat!=xy_gnd, 1)!=0)

pdb.set_trace()
