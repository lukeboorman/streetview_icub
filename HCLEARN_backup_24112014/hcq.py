from rbm import ECState, boltzmannProbs, fuse, DGState, CA3StateFromVector, CA1State
import numpy as np

SEED=2942875  #95731  #73765
#np.random.seed(SEED)    #careful, these are different RNGs!
np.random.seed(SEED)


class History:
    def __init__(self,  ec_log, dg_log, ca3_log, ca1_log, sub_ints, sub_errs, sub_fires, surf_gnd_log, str_title):
        self.ecs=ec_log
        self.dgs=dg_log
        self.ca3s=ca3_log
        self.ca1s=ca1_log
        self.sub_ints=sub_ints
        self.sub_errs=sub_errs
        self.sub_fires=sub_fires
        self.surf_gnd_log=surf_gnd_log
        self.str_title = str_title

def makeMAPPredictions(path,dictGrids, dictSenses, WB, WR, WS, WO, dghelper, b_obsOnly, b_usePrevGroundTruthCA3, b_useGroundTruthGrids, b_useSub, str_title, b_learn):  
#    (ecs_gnd, dgs_gnd, ca3s_gnd) = path.getGroundTruthFirings(dictSenses, dictGrids, path.N_mazeSize, dghelper)     
    (ecs_gnd, dgs_gnd, ca3s_gnd) = path.getGroundTruthFirings(dictSenses, dictGrids, dghelper)   # Luke: MOD 

    ca3 = ca3s_gnd[0]
    ca1 = ECState(ecs_gnd[0])    #used to update grids!    
    T = len(ecs_gnd)
    sub_int = 0
    sub_errs = np.zeros(T)                         #current instant error
    sub_ints = np.zeros(T)                         #filtered/integrated error.  Weaken prior in proportion to this.
    sub_fires = np.zeros(T)                        #log of when sub is active
    ec_log  = [ ecs_gnd[0] ]  #log ground truth at 0th time (hack)
    dg_log  = [ dgs_gnd[0] ]
    ca3_log = [ ca3s_gnd[0] ]
    ca1_log = [ ca1 ]
    
    if dghelper is not None:
        surf_gnd_log = [ ecs_gnd[0].surfs ]
    else:
        surf_gnd_log = []

    for t in range(1,T): 
        if b_useGroundTruthGrids:
            ec = ecs_gnd[t].makeNoisyCopy(dictGrids, b_GPSNoise=True) # LB added 
#            ec = ecs_gnd[t].makeNoisyCopy(b_GPSNoise=True) 
        else:
#            ec = ecs_gnd[t].makeNoisyCopy(b_GPSNoise=False) #add observation noise (inc. noisy GPS, which may be overriden by odometry)        
            ec = ecs_gnd[t].makeNoisyCopy(dictGrids,b_GPSNoise=False) #add observation noise (inc. noisy GPS, which may be overriden by odometry)    # LB: added    
            ec.hd = []        #kill old values to prevent any bugs creeping in!
            ec.placeCells=[]
            b_odom = sum(path.posLog[t,0:2] != path.posLog[t-1  ,0:2])>0    #have I moved?
            d_th    = (path.posLog[t,2] - path.posLog[t-1,2])%4             #have I rotated, which dir?
            
            #TODO add realistic odom noise here!!!!

            p_noise = 0.05
            if np.random.random() < p_noise:
                b_odom = not b_odom
            if np.random.random() < p_noise:
                d_th = (d_th + (-1)**(np.floor(2*np.random.random())) )%4
            ec.updateGrids(ca1.grids, ca1.hd, b_odom, dictGrids)  # Luke MOD  #overwrite grids with odom (NB uses PREVIOUS hd)
#            ec.updateGrids(ca1.grids, ca1.hd, b_odom, path.N_mazeSize, dictGrids)    #overwrite grids with odom (NB uses PREVIOUS hd)
            ec.updateHeading(ca1.hd, d_th) 

        (dg, ca3, ca1,  sub_err, sub_int, sub_fire) = makeMAPPredictionsStep(dictGrids, ec, ca3, ca3s_gnd[t-1], sub_int, WB, WR, WS, WO,  b_obsOnly, b_usePrevGroundTruthCA3, b_useSub, dghelper)

        sub_errs[t] = sub_err
        sub_ints[t] = sub_int  
        sub_fires[t] = sub_fire
        ec_log.append(ec)
        dg_log.append(dg)
        ca3_log.append(ca3)
        ca1_log.append(ca1)
        if dghelper is not None:
            surf_gnd_log.append(ecs_gnd[t].surfs) # ALAN

        #learning steps (weight changes are made in-place)
        if b_learn:
            b_fakeSub = np.floor(2*np.random.random())                       #train with/without rec+odom, to encourage WB to learn true biases
            wakeStep(ec, ca3, WB, WR, WS, WO, b_fakeSub, dictGrids) # Luke added dictGrids
            sleepStep(ec, ca3, WB, WR, WS, WO, b_fakeSub, dictGrids)  # Luke added dictGrids

    return History(ec_log, dg_log, ca3_log, ca1_log, sub_ints, sub_errs, sub_fires, surf_gnd_log, str_title)           #remove bias column from CA3


def getWeightChange(hids, input):
    alpha=0.01
    C = np.outer(hids, input);
    return alpha*C;

#CONCEPTUALLY:
#Copy ECDG -> CA1  (how??)
#sample CA3 (at T=1?)
#hebb learning: EC->CA3, DG->CA3, CA3->CA3, CA3->CA1
#
#PROGRAMATICALLY:
#Sample CA3.
#Just use ECDG, CA3 to Hebb update weights ECDG->CA3 and CA3->CA1 (pretend CA1 was forced to match ECDG)
#
def wakeStep(ec, ca3, WB, WR, WS, WO, b_fakeSub, dictGrids): # Luke added dictGrids
    v_senses      = np.hstack((ec.toVectorSensesOnlyD(dictGrids), 1))
    v_odom        = np.hstack((ec.toVectorOdomOnlyD(dictGrids), 1))
    v_ca3_prev    = np.hstack((ca3.toVector(), 1))       

    p   = boltzmannProbs(WB, np.array((1.0)))           #global prior
    lamSenses = boltzmannProbs(WS,v_senses)        
    p = fuse(p,lamSenses)
    if not b_fakeSub:
        pi  = boltzmannProbs(WR,v_ca3_prev)                 #probs for x next
        p = fuse(p,pi)
        lamOdom = boltzmannProbs(WO, v_odom)          #may be GPS ground truth if passed in
        p = fuse(p, lamOdom)

    #sample from CA3, T=1
    v_ca3 = ( np.random.random(p.shape) < p )

    #learn WB -- is hard?? no, just regusar WS in this setting, was hard when based on offline correls?
    WS += getWeightChange(v_ca3, v_senses)
    WB += getWeightChange(v_ca3, np.array((1.0))).reshape(WB.shape[0])   ##TODO do we need to do something special to make this the real prior?? eg. train it in the absence of the odom inputs as well as their presence, as in the real world??
    if not b_fakeSub:
        WO += getWeightChange(v_ca3, v_odom)
        WR += getWeightChange(v_ca3, v_ca3_prev)


#CONCEPTUALLY:
# EC and DG = f(CA1)   (how??) -- use a fake CA3 state to do ideal recoding?? (in reality, would just be 1-1 connections)
# sample CA3 (T=1?)
# sample CA1 (T=1?)
# EC and DG = f(CA1)   (again)
#antihebb learning: EC->CA3, DG->CA3, CA3->CA3, CA3->CA1
#
#PROGRAMATRICALLY:
# Get a T=1 sample in CA1 (assuming CA3 was just T=1 sampled in the wake step)
# Use the denoise ECd out as an obs sample:
# Instantiate a copy in ECs, DG
# Sample CA3 (T=1)
# Unlearn
#
def sleepStep(ec, ca3, WB, WR, WS, WO, b_fakeSub, dictGrids):  # Luke added dictGrids

    #for encapsulation -- sample again from CA3 at T=1  (could speed up by reusing wake step?)
    v_senses      = np.hstack((ec.toVectorSensesOnlyD(dictGrids), 1))
    v_odom        = np.hstack((ec.toVectorOdomOnlyD(dictGrids), 1))
    v_ca3_prev    = np.hstack((ca3.toVector(), 1))       

    p   = boltzmannProbs(WB, np.array((1.0)))           #global prior
    lamSenses = boltzmannProbs(WS,v_senses)        
    p = fuse(p,lamSenses)
    if not b_fakeSub:
        pi  = boltzmannProbs(WR,v_ca3_prev)                 #probs for x next
        p = fuse(p,pi)
        lamOdom = boltzmannProbs(WO, v_odom)          #may be GPS ground truth if passed in
        p = fuse(p, lamOdom)

    #sample from CA3, T=1
    v_ca3 = ( np.random.random(p.shape) < p )

    #sample CA1 (assume ECDG are also updated to these same values)
    p_odom   = boltzmannProbs(WO.transpose(), v_ca3)
    p_senses = boltzmannProbs(WS.transpose(), v_ca3)

    v_odom   = ( np.random.random(p_odom.shape) < p_odom )
    v_senses = ( np.random.random(p_senses.shape) < p_senses )

    #ANTIlearn WB -- is hard?? no, just regusar WS in this setting, was hard when based on offline correls?
    WB -= getWeightChange(v_ca3, np.array((1.0))).reshape(WB.shape[0])   ##TODO do we need to do something special to make this the real prior?? eg. train it in the absence of the odom inputs as well as their presence, as in the real world??
    WS -= getWeightChange(v_ca3, v_senses)
    if not b_fakeSub:
        WO -= getWeightChange(v_ca3, v_odom)
        WR -= getWeightChange(v_ca3, v_ca3_prev)


###** TODO: half the time, disconnect odom and/or rec from the net, to model what really happens
####  hopefully this will encourage WB to learn proper priors, rather than prior components being stuck in the other Ws ?

### TODO can we do w/s at zero temperature instead of T=1 ?




#***I HAVE CHANGED THIS VERSION TO TAKE CA3_GND_{T-1} TO COMAPRE WITH TEST, THIS DIFFERS FROM IJCNN ONE!!!***
def makeMAPPredictionsStep(dictGrids, ec, ca3, ca3_PREV_gnd, sub_int, WB, WR, WS, WO,  b_obsOnly, b_usePrevGroundTruthCA3, b_useSub, dghelper=None): # Luke added Nmax

    sub_thresh=0.26;  sub_fire=0
    if (b_useSub and sub_int>sub_thresh):
        sub_fire=1

    dg = DGState(ec, dictGrids, dghelper)
    v_senses = np.hstack((ec.toVectorSensesOnlyD(dictGrids, dghelper), 1))
    v_odom   = np.hstack((ec.toVectorOdomOnlyD(dictGrids), 1))


    if b_usePrevGroundTruthCA3:
        v_ca3 = np.hstack((ca3_PREV_gnd.toVector(), 1))      #overwrite if cheating with perfect PREVIOUS state
    else:
        v_ca3    = np.hstack((ca3.toVector(), 1))                   

    N_places = dg.place.shape[0] # Does this work to get correct no places!!!!!!
    #N_grids = ec.grids.shape[1]*2  # Luke -> Unused
    #N_ec = ec.toVector().shape[0]  # Luke -> Unused
    #N_ca3 = len(v_ca3) # Luke -> Unused

    pb   = boltzmannProbs(WB, np.array((1.0)))           #global prior

   # pdb.set_trace()
    if sub_fire:
        p=pb  
    else:
        #print("odoms: %s" % v_odom)
        lamOdom = boltzmannProbs(WO, v_odom)          #may be GPS ground truth if passed in        
        p = fuse(pb, lamOdom)
        if not b_obsOnly:
            #print("CA3 recursive: %s" % v_ca3)
            pi  = boltzmannProbs(WR,v_ca3)                 #probs for x next
            p = fuse(p,pi)


    #print("senses before: %s\n%d" % (v_senses, len(v_senses)))
    lamSenses = boltzmannProbs(WS,v_senses)        
    #print("senses after: %s\n%d" % (lamSenses, len(v_senses)))
    #print("senses after: %s" % lamSenses)
    #print("p before fusing with senses: %s" % p)
    p = fuse(p,lamSenses)

#    pdb.set_trace()
#    ca3_new = CA3StateFromVector(p, N_places=13)
#    ca3_new.smartCollapse()
#    v_ca3 = hstack((ca3_new.toVector(),1))
    #print("p before thresholding: %s" % p)
    
    v_ca3 = (p>0.5) ##WAS just p!!
    #print("v_ca3: %s, len: %d" % (v_ca3, len(v_ca3)))

    p_odom   = boltzmannProbs(WO.transpose(), v_ca3)
    p_senses = boltzmannProbs(WS.transpose(), v_ca3)

    #print("p_odom: %s" % p_odom)
    #print("p_senses: %s" % p_senses)
    #ALAN - The last sense is "whiskers left and right on" this is almost always on, thus in the CA1State can be decoded to mean "whiskers left and right on"
    # Need to get from grid size!!!!!

    ca1 = CA1State(p_odom, p_senses, dghelper, dictGrids)   #dictGrids added by luke           #lots of smart decoding done in here
    
    #HOOK-ALAN set weights for each error in the subiculum so they add up to one (so surfs don't count much more towards the final error than the others)
    whiskersWeighted = np.sum(ca1.whiskers!=ec.whiskers)/float(len(ca1.whiskers))
    rgbWeighted = np.sum(ca1.rgb!=ec.rgb)/float(len(ca1.rgb))
    lightAheadWeighted = np.sum(ca1.lightAhead!=ec.lightAhead)/float(len(ca1.lightAhead))
    if dghelper is not None:
        surfsWeighted = np.sum(ca1.surfs!=ec.surfs)/float(len(ca1.surfs))
        sub_err = whiskersWeighted + rgbWeighted + lightAheadWeighted + surfsWeighted
        #Value of float should vary with how many parts contribute towards it
        sub_err = sub_err/float(4)
    else:
        sub_err = whiskersWeighted + rgbWeighted + lightAheadWeighted
        sub_err = sub_err/float(3)

    #sub_err =  sum(ca1.whiskers!=ec.whiskers) + sum(ca1.rgb!=ec.rgb) + sum(ca1.lightAhead!=ec.lightAhead) + sum(ca1.surfs!=ec.surfs)

    mix=0.02
    sub_int = (1-mix)*sub_int + mix*sub_err 

    ca3_out = CA3StateFromVector(v_ca3, N_places)

    return (dg, ca3_out, ca1,  sub_err, sub_int, sub_fire)


