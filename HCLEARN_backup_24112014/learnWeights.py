import numpy as np
from makeMazeResizeable import * # LB: MODified
# from makeMaze import *
from paths import * 
from cffun import *
from rbm import *
import DGStateAlan

def err(ps, hids):
    return sum( (ps-hids)**2 ) / hids.shape[0]


def fuse2(p1,p2):
    return 1.0 / (1.0 +   ((1-p1)*(1-p2)/(p1 * p2)  ))

#def learn(path, dictSenses, dictGrids, N_mazeSize, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=True, b_learnTrained=False, b_learnDGWeights=True, learningRate=0.01,tr_epochs=10):
def learn(path, dictSenses, dictGrids, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=True, b_learnTrained=False, b_learnDGWeights=True, learningRate=0.01,tr_epochs=10):
    dghelper=None
    #Learn DG weights when you have visual input
    if b_learnDGWeights:
        #Extract data from dictionary of senses to train on 
        allSURFS = [ sense.surfs for sense in dictSenses.values() ]
        #print allSURFS
        #print len(allSURFS)

        #Select a selection?
        percent = 100
        numOfImages = int(len(allSURFS)*(percent/float(100)))
        allIndices = range(0,len(allSURFS))
        np.random.shuffle(allIndices)
        randomIndices = allIndices[0:numOfImages]
        randomSURFs = [ allSURFS[ind] for ind in randomIndices ] 


        ## Luke -> THESE MAY NEED TO BE ADJUSTED FOR LARGER MAZ = LARGE NO images....
        #Train weights
        X=7 #  Number of inputs to each Lamellae
        N=45 # Number of Lamellae 
        presentationOfData = 30
        learningrate = 0.05 # Why is this changed here 0.01 elsewhere?
        dghelper = DGStateAlan.train_weights(randomSURFs, X, N, presentationOfData, learningrate)

    #Learn ideal weights (with perfect look ahead training)
    if b_learnIdeal: #latest :  lots of odom noise    
        print "TRAINING IDEAL WEIGHTS..."
#        (ecs_nsy, dgs_nsy, ca3s_nsy) = path.getNoiseyGPSFirings(dictSenses, dictGrids, N_mazeSize, dghelper)  #ideal percepts for path, for trainign and for inference
        (ecs_nsy, dgs_nsy, ca3s_nsy) = path.getNoiseyGPSFirings(dictSenses, dictGrids, dghelper)  #ideal percepts for path, for trainign and for inference

        senses = ecs2vd_so(ecs_nsy,dictGrids, dghelper) 

        #Use test2.py for learning, use this for generating ground truth path to learn with test2.py
        odom   = ecs2vd_oo(ecs_nsy,dictGrids) 
        hids   = ca3s2v(ca3s_gnd) ##NB this is the ground truth, no noise

        WB = trainPriorBias(hids)

        WR = trainW(lag(hids,1), hids, WB, N_epochs=1, alpha=0.01)
        WS = trainW(senses,      hids, WB, N_epochs=1, alpha=0.01)
        WO = trainW(odom,        hids, WB, N_epochs=1, alpha=0.01)

        np.save('WR',WR)
        np.save('WS',WS)
        np.save('WB',WB)
        np.save('WO',WO)

        np.save('hids',hids)
        np.save('odom',odom)

        np.save('senses',senses)
     
        print "DONE TRAINING"

    if b_learnTrained:
        print "TRAINING TRAINED WEIGHTS..."

        foo = np.random.random()

        WR = np.load('WR.npy')
        WO = np.load('WO.npy')
        WS = np.load('WS.npy')
        WB = np.load('WB.npy')
        WB = WB.reshape((WB.shape[0],1))#((86,1)) # LB: Why the %%%% is this 86!!!!
        #No longer need the above lines as they are a hack and the wrong size

        ##these are all to be learned, so overwrite them with rands
        WR = 1-2*np.random.random(WR.shape) #Weights recurrent
        WB = 1-2*np.random.random(WB.shape) #Weights bias
        WO = 1-2*np.random.random(WO.shape) #Odom sensors
        WS = 1-2*np.random.random(WS.shape) #Non-odom sensors - INCLUDING SURF - Need to include DG size within it
        #ALAN Seems like by changing the ecs2vs_so to include the surf features this size has been now modified correctly?
        #Above need resizing to include SURF weights

        #hids_gnd=addBias(hids)    #NB each pop has a local bias, in addition to the global prior bias
        hids_gnd=addBias(np.load('hids.npy'))    #NB each pop has a local bias, in addition to the global prior bias
        #ALAN - Keep hids_gnd the same 
        #odom=addBias(odom)
        odom=addBias(np.load('odom.npy'))
        #senses=addBias(senses)
        senses=addBias(np.load('senses.npy'))
        #ALAN - odom and senses are matrices with rows as time columns as what is observed at that time (light, whiskers, SURF, DGencodedSURF etc.)

        hidslag_gnd = lag(hids_gnd,1)

        T = odom.shape[0]

        b = np.array([1.0])  #bias
        alpha = learningRate #0.01 ##0.0001 is stable, starting at perfects; 0.001 diverges.

        ### train with proper wake-sleep (no peeking at hid_t, though hid_{t-1} is OK )

        print 'Optimising weights for %d epochs!'%(tr_epochs)
        #FIXME: TURN BACK TO 1000 or so!
        for epoch in range(0,tr_epochs):
            
            err_epoch = 0  #accumulator

            for t in range(0,T):

                b_fakeSub = np.floor(2*np.random.random())  #learn on full data or on hist-indep subset?

                hids_prev = hidslag_gnd[t,:]
                s = senses[t,:]
                o = odom[t,:]

                #WAKE

                p_b  = boltzmannProbs(WB, np.array([1.0]))
                p_s  = boltzmannProbs(WS,s)          
                p = p_b.copy()
                p=fuse(p, p_s)

                if not b_fakeSub:
                    p_o  = boltzmannProbs(WO,o)          
                    p_r  = boltzmannProbs(WR,hids_prev)
                    p=fuse(p, p_o)
                    p=fuse(p, p_r)

                hids = (p > np.random.random(p.shape)).astype('d')    #sample, T=1

                CS = np.outer(hids,s)
                CB = np.outer(hids,b)
                WS += alpha*CS
                WB += alpha*CB

                if not b_fakeSub:
                    CO = np.outer(hids,o)
                    CR = np.outer(hids,hids_prev)
                    WR += alpha*CR
                    WO += alpha*CO

                #SLEEP

                #retain the hid sample from the wake step -- draw samples from obs, then hid again, CD style.

                if not b_fakeSub:
                    po = boltzmannProbs(WO.transpose(), hids) 
                    o = (po > np.random.random(po.shape)).astype('d')    #sleep sample (at temp=1)

                ps = boltzmannProbs(WS.transpose(), hids) 
                s = (ps > np.random.random(ps.shape)).astype('d')    #sleep sample (at temp=1)

                p_b  = boltzmannProbs(WB, np.array([1.0]))
                p_s  = boltzmannProbs(WS,s)          
                p = p_b.copy()
                p=fuse(p, p_s)

                if not b_fakeSub:
                    p_o  = boltzmannProbs(WO,o)          
                    p_r  = boltzmannProbs(WR,hids_prev)
                    p=fuse(p, p_o)
                    p=fuse(p, p_r)

                #resample hids (needed to antii learn recs!)
                hids = (p > np.random.random(p.shape)).astype('d')    #sample, T=1

                CS = np.outer(hids,s)
                CB = np.outer(hids,b)
                WS -= alpha*CS
                WB -= alpha*CB

                if not b_fakeSub:
                    CO = np.outer(hids,o)
                    CR = np.outer(hids,hids_prev)
                    WR -= alpha*CR
                    WO -= alpha*CO


                #TODO report error rate?
                e = err( hids, hids_gnd[t,:] )
                err_epoch += e

            print('epoch:'+str(epoch)+' err:'+str(err_epoch))


        np.save('tWR',WR)
        np.save('tWS',WS)
        np.save('tWB',WB)
        np.save('tWO',WO)
    # 1 see if these give ok results
    # 2 try disbaling odom info again, and learning in its absence half the time 

    if b_learnDGWeights:
        return dghelper
