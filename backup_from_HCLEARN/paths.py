from cffun import *
#from makeMaze import Senses
from makeMazeResizeable import Senses
from location import Location
from DGStateAlan import DGState, smartCollapse
#Senses, makeMaze

#A path is a log of random walk locations and sensations.
#CHANGED FROM PATH TO PATHS AS IPYTHON DOESN'T LIKE FILES/DIRS CALLED PATH
class Paths:
    
    def __init__(self, dictNext, N_mazeSize, T_max, start_location=[3,3,0]):
        self.N_mazeSize = N_mazeSize
        self.posLog        = np.zeros((T_max, 3),dtype='int16')  #for training, ground truth states. Includes lightState
        self.lightAheadLog = np.zeros((T_max, 1),dtype='uint8')  #true if there is a light ahead of the agent
        self.lightStateLog = np.zeros((T_max, 1),dtype='uint8')  #true if there is a light ahead of the agent
        # Luke - fixed dtype to save time as not fp data -> better for bigger paths

        #s=[3,3,0]  # Luke send in argument this tile may not exist #state (of agent only).  start at center, facing towards east.
        lightState=0        #there are 4 lights which move when agent reaches NESW respectively
        
        ### This looks to be the main path generator -> uses start_location and T (steps) 
        for t in range(0,T_max):

            if start_location[0]==2*N_mazeSize and lightState==0: #E arm
                lightState=1 
                print "light to N"
            if start_location[1]==2*N_mazeSize and lightState==1: #N arm
                lightState=2
                print "light to W"
            if start_location[0]==0 and lightState==2: #W
                lightState=3
                print "light to S"
            if start_location[1]==0 and lightState==3: #S
                lightState=0 
                print "light to E"
            self.lightStateLog[t] = lightState


            if start_location[2]==lightState:            #agent facing in same direction as the lit arm
                self.lightAheadLog[t] = 1   #there is a visible light ahead

            self.posLog[t,0:3]=start_location

            s_nexts = dictNext[tuple(start_location)]          #possible next locations
            i = random.randrange(0,len(s_nexts))  #choose a random next location
            start_location = s_nexts[i]


    def getGroundTruthFiring(self,dictSenses,dictGrids,N_mazeSize,t,dghelper=None):

         loc        = self.posLog[t,:]
         lightState = self.lightStateLog[t,0]     #which physical light (eg 
         lightAhead = self.lightAheadLog[t,0]
         senses = dictSenses[tuple(loc)]    
         #HOOK include SURF features in dictSenses structure
         ecState = ECState((senses, lightAhead))
         dgState = DGState(ecState, dictGrids, dghelper)
         ca3State = CA3StateFromInputs(ecState, dgState, lightState) #ideal state, No need to know what a surf feature is...

         if t==0:
             odom=np.zeros((1,2))
         else:
             odom = self.posLog[t,:]-self.posLog[t-1,:]
         return (ecState,dgState,ca3State,odom)


    #makes a "data" matrix, with cols of IDEAL EC and DG outputs. (No noise, and perfect GPS).
    #also returns the ideal CA3 output (used for training) and the raw positions.
    def getGroundTruthFirings(self, dictSenses, dictGrids, N_mazeSize, dghelper=None):
        print "get ground truth firings"
        T_max = self.posLog.shape[0]
        ecStates = []  #fill with ECState objects
        dgStates = [] #fill with DGState objects
        ca3States = []
        for t in range(0,T_max):
            (ecState,dgState,ca3State,odom) = self.getGroundTruthFiring(dictSenses,dictGrids,N_mazeSize,t,dghelper) 
            ecStates.append(ecState)           
            dgStates.append(dgState)         
            ca3States.append(ca3State)
        print "done"
        return (ecStates, dgStates, ca3States)

    
    #for training only. (Real inference uses noiseless, and adds its own noise AND odometry)
    #this assumes the noise is due to noisy GPS -- not to lost odometry
    def getNoiseyGPSFirings(self, dictSenses, dictGrids, N_mazeSize, dghelper=None):
        T_max = self.posLog.shape[0]
        ecStates = []  #fill with ECState objects
        dgStates = [] #fill with DGState objects
        ca3States = []
        for t in range(0,T_max):
            (ecState,dgState,ca3State,odom) = self.getGroundTruthFiring(dictSenses,dictGrids,N_mazeSize,t,dghelper)
            
            lightState = self.lightStateLog[t,0]

            ecState = ecState.makeNoisyCopy()
            dgState = DGState(ecState, dictGrids, dghelper)
            ca3State = CA3StateFromInputs(ecState, dgState, lightState)

            ecStates.append(ecState)           
            dgStates.append(dgState)         
            ca3States.append(ca3State)
        return (ecStates, dgStates, ca3States)


class CA1State:
    def __init__(self, p_odom, p_senses, dghelper=None, n_places=13):
        i=0
        n_grids=6
        n_hd=4
        # AGAIN n_places
        # Luke removed! n_places=13

        #pdb.set_trace()

        p_grids  = p_odom[i:i+n_grids];   i+=n_grids
        p_hd     = p_odom[i:i+n_hd];      i+=n_hd
        p_places = p_odom[i:i+n_places];  i+=n_places

        i=0
        n_whiskers=3
        n_rgb=3
        n_lightAhead=1
        n_whiskerCombis=3
        
        p_whiskers = p_senses[i:i+n_whiskers]; i+=n_whiskers
        p_rgb = p_senses[i:i+n_rgb]; i+=n_rgb
        p_lightAhead = p_senses[i:i+n_lightAhead]; i+=n_lightAhead
        p_whiskerCombis = p_senses[i:i+n_whiskerCombis]; i+=n_whiskerCombis

        #HOOK: put your decoding of output (whatever representation that is...., here)
        #decode remaining sensors which are the features previously encoded
        if dghelper is not None:
            #Get the number of surf features
            n_surfFeatures = dghelper.numOfSurfFeatures
            #Get the number of encoded features
            n_encoded = dghelper.numOfEncodedFeatures
            #print("Num of surf features: %d\nNum of encodedFeatures: %d\nNum of all feautres: %d" % (n_surfFeatures, n_encoded, (n_surfFeatures+n_encoded)))
            p_surfFeatures = p_senses[i:i+n_surfFeatures]; i+=n_surfFeatures
            p_encoded = p_senses[i:i+n_encoded]; i+=n_encoded

            #We now have two sources of surf, one from the probabilities that came from EC into CA3, and one from the DG encoded going into CA3
            #Dumb decode the former:
            surfFromEC = (p_surfFeatures>0.5)

            #Very smart decode... use the weights learnt to decode back to EC space
            surfFromDG = dghelper.decode(p_encoded)

            #Experiment with using both see what advantage DG gives over EC
            self.surfs = surfFromDG

        #print("Total length of senses:%d, used:%d" % (len(p_senses), i))

        #smart decoding, use smart feature collapse, then create ECd pops here too
        self.places = smartCollapse(p_places)
        self.hd = smartCollapse(p_hd)
        #print("p_whiskerCombis: %s" % p_whiskerCombis)
        self.whiskerCombis = smartCollapse(p_whiskerCombis)
        
        loc=Location()
        loc.setPlaceId(argmax(self.places))
        self.grids=loc.getGrids()

        #dumb decodes
        self.lightAhead = (p_lightAhead>0.5)
        self.rgb = (p_rgb>0.5)

        #print("whisker combis: %s" % self.whiskerCombis)
        #whiskers
        if self.whiskerCombis[0]:
            self.whiskers=np.array([1,1,1])  #all
        elif self.whiskerCombis[1]:
            #print(self.places)
            #print("no whiskers touching")
            self.whiskers=np.array([0,0,0])  #none
        elif self.whiskerCombis[2]:
            #print("left right whiskers touching")
            self.whiskers=np.array([1,0,1])  #L+R


    def toString(self):
        r="CA1:\n  grids:"+str(self.grids)+"\n  hd:"+str(self.hd)+"\n  whiskers:"+str(self.whiskers)+"\n  rgb:"+str(self.rgb)+"\n  lightAhead:"+str(self.lightAhead)+"\n  place:"+str(self.places)+"\n  wcombis:"+str(self.whiskerCombis)
        return r


class ECState:           #just flattens grids, and adds lightAhead to the Senses object!
    def __init__(self, arg):
        self.N_places=13 ## Set to default as 13 but update if included in arguments -> senses
        if isinstance(arg,ECState):  #copy constructor
            self.grids=arg.grids.copy()
            self.hd=arg.hd.copy()
            self.whiskers=arg.whiskers.copy()
            self.rgb=arg.rgb.copy()
            self.lightAhead=arg.lightAhead.copy()
            self.surfs=arg.surfs.copy() #ALAN
        elif isinstance(arg[0], Senses):    #contruct from a (s:Senses, lightAhead:bool) tuple
            senses=arg[0]
            lightAhead=arg[1]
            self.grids=senses.grids.copy() 
            self.hd=senses.hd.copy()
            self.whiskers=senses.whiskers.copy()
            self.rgb=senses.rgb.copy()
            self.lightAhead=lightAhead.copy()
            self.surfs=senses.surfs.copy() #ALAN # LB if error here changes makeMaze import to correct version!!
            if len(arg)>=3:
                self.N_places=arg[3]
        elif isinstance(arg[0], np.ndarray):   #TODO test. COnstruct from a (v_ec:vector, nPlaces:int) tuple
            N_grids=arg[1]
            if arg[1]>6:
                pdb.set_trace()
                print "ERROR TOO MANY GRIDS!"
            self.grids=arg[0][0:N_grids].reshape((2,N_grids/2))
            self.hd=arg[0][N_grids:N_grids+4]
            self.whiskers=arg[0][N_grids+4:N_grids+4+3]
            self.rgb=arg[0][N_grids+4+3:N_grids+4+3+3]
            self.lightAhead=arg[0][N_grids+4+3+3:N_grids+4+3+3+1]
            print("HOOK THIS NEED TO BE IMPLEMENTED FOR SURF")

    def collapseToMax(self):    #use this if I was created from a prob vec
        #i=argmax(self.placeCells)
        #self.placeCells*=0
        #self.placeCells[i]=1

        i=argmax(self.hd)
        self.hd*=0
        self.hd[i]=1

        self.whiskers = (self.whiskers>0.5)
        self.rgb = (self.rgb>0.5)
        self.lightAhead = (self.lightAhead>0.5)
        self.surfs = (self.surfs>0.5) #ALAN


    #NB my grids arne't Vechure-style attractors; rahter they passively sum CA1 with odom
    #doign so assumes that the HC output is always right, unless speciafically lost. (Noisy GPS)
    def updateGrids(self, ca1grids, ca1hd, b_odom, N_mazeSize, dictGrids):  

        loc=Location()
        loc.setGrids(ca1grids, dictGrids)

        (x_hat_prev, y_hat_prev) = loc.getXY()
        ## Hard coded again here North negative.....
        dxys = [[1,0],[0,1],[-1,0],[0,-1]]  #by hd cell
        ihd = argmax(ca1hd)
        odom_dir = dxys[ihd]

        odom = [0,0]
        if b_odom:
            odom=odom_dir
        
        x_hat_now = x_hat_prev + odom[0]
        y_hat_now = y_hat_prev + odom[1]       
        
        ##SMART UPDATE -- if odom took us outside the maze, then ignore it
        #pdb.set_trace()

        ##if this takes me to somewhere not having a '3'(=N_mazeSize) in the coordinate, then the move was illegal?
        if sum( (x_hat_now==N_mazeSize) + (y_hat_now==N_mazeSize))==0:
            print "OFFMAZE FIX: OLD:" ,x_hat_now, y_hat_now
            x_hat_now = x_hat_prev
            y_hat_now = y_hat_prev
            print "NEW:",x_hat_now, y_hat_now
        x_hat_now = crop(x_hat_now, 0, 2*N_mazeSize)
        y_hat_now = crop(y_hat_now, 0, 2*N_mazeSize)  #restrict to locations in the maze
            
        loc=Location()
        loc.setXY(x_hat_now, y_hat_now)
        #self.placeCells=zeros(ca1placeCells.shape)
        #self.placeCells[loc.placeId] = 1
        self.grids = loc.getGrids().copy()

    #dth in rads; HDs are four bool cells
    def updateHeading(self, ca1hd, d_th):
        self.hd=np.zeros((4))
        i_old = argmax(ca1hd)
        i_new = (i_old+d_th)%4
        self.hd[i_new]=1

    def toVector(self):
        return np.hstack((self.grids.flatten(), self.hd, self.whiskers, self.rgb, self.lightAhead, self.surfs) )
   
    def toVectorSensesOnly(self):
        senses=  np.hstack((self.whiskers, self.rgb, self.lightAhead, self.surfs) )
        return senses

    def toVectorOdomOnly(self):
        return np.hstack((self.grids.flatten(), self.hd) )

    def toVectorD(self,dictGrids, dghelper=None):  #with dentate and bias
        return np.hstack(( self.toVector(), DGState(self, dictGrids, dghelper).toVector() ))

    def toVectorSensesOnlyD(self,dictGrids, dghelper=None):
        senses = np.hstack((self.toVectorSensesOnly(), DGState(self, dictGrids, dghelper).toVectorSensesOnly()))
        return senses

    def toVectorOdomOnlyD(self,dictGrids):
        return np.hstack((self.toVectorOdomOnly(), DGState(self,dictGrids).toVectorOdomOnly()))

    def toString(self):
        r="EC:\n  grids:"+str(self.grids)+"\n  hd:"+str(self.hd)+"\n  whiskers:"+str(self.whiskers)+"\n  rgb:"+str(self.rgb)+"\n  lightAhead:"+str(self.lightAhead)+"\n surfs:"+str(self.surfs)
        return r

    #GPSnoise:use ONLY to simulate occasional lostness for TRAINING, not during inference
    #(might want to make noisy odom elsewhere for inference)
    def makeNoisyCopy(self, b_GPSNoise=True):   #makes and returns a noisy copy
        ec = ECState(self)

        p_flip = 0.2
        p_flip_odom = 0.2   #testing, make the grids,hds very unreliable (TODO iterative training??)

        if b_GPSNoise:
            if random.random()<p_flip_odom:    #simulate grid errors- fmove to a random place (as when lost)
                #N_places = 13 # Luke arguments sent in at top!
                i = random.randrange(0,self.N_places) # Luke converted to self 
                loc = Location()
                loc.setPlaceId(i)
                ec.grids = loc.getGrids().copy()

            if random.random()<p_flip_odom:    #simulate HD errors
                i = random.randrange(0,4) 
                ec.hd[:] = 0
                ec.hd[i] = 1
            ##if random.random()< 0.05:  ####simulate lost/reset events WRITEUP: EM like estimation of own error rate needed here (cf. Mitch's chanel equalisation decision feedback/decision directed)
            ##    ec.placeCells = 0.0 * ec.placeCells
            ##    ec.hd = 0.0 * ec.hd  ##no this isnt what we want to do -- we dont want to leatn flatness as an OUTPUT!
    

        if random.random()<p_flip:    #flip whiskers
            ec.whiskers[0] = 1-ec.whiskers[0]
        if random.random()<p_flip:    #flip whiskers
            ec.whiskers[0] = 1-ec.whiskers[0]
        if random.random()<p_flip:    #flip whiskers
            ec.whiskers[1] = 1-ec.whiskers[1]
        if random.random()<p_flip:    #flip whiskers
            ec.whiskers[2] = 1-ec.whiskers[2]
        if random.random()<p_flip:    #flip lightAhead
            ec.lightAhead = 1-ec.lightAhead
        if random.random()<p_flip:    #flip colors
            ec.rgb[0] = 1-ec.rgb[0]
        if random.random()<p_flip:    #flip colors
            ec.rgb[1] = 1-ec.rgb[1]
        for featureInd, feature in enumerate(ec.surfs): #ALAN implemented flipping
            if random.random()<p_flip:
                ec.surfs[featureInd] = 1-feature
        return ec

        


#God's eye, ideal CA3 response to ideal EC and DG states
class CA3State:  
    def __init__(self, place, place_hd, light, light_hd):
        self.place=place
        self.place_hd=place_hd
        self.light=light
        self.light_hd=light_hd  #this is the light STATE not lightAhead
        #self.surfs=surfs #ALAN not needed because we are looking at ideal?
        
    def toVector(self):
        return np.hstack(( self.place, self.place_hd.flatten(), self.light, self.light_hd.flatten())) #without Bias

    def toString(self):
        r = "CA3state:\n  place="+str(self.place)+"\n  phace_hd:"+str(self.place_hd)+"\n  light:"+str(self.light)+"\n  light_hd:"+str(self.light_hd)
        return r

    def smartCollapse(self):
        self.place = smartCollapse(self.place)
        self.place_hd = smartCollapse(self.place_hd)
        self.light = smartCollapse(self.light)
        self.light_hd = smartCollapse(self.light_hd)
        #self.surfs = smartCollapse(self.encodedValues) #ALAN is this necessary?

def CA3StateFromInputs(ec, dg, lightState):
    place    = dg.place.copy()
    hd       = ec.hd.copy()

    place_hd=np.zeros((place.shape[0],hd.shape[0]))
    for i_place in range(0,place.shape[0]):
        for i_hd in range(0, hd.shape[0]):
            place_hd[i_place,i_hd] = place[i_place]*hd[i_hd]

    light = np.zeros(4)  #CA3 light cells. (ie tracking the hidden state of the world)
    light[lightState]=1
    
    #N_place = place.shape[0] 
    N_light = 4       
    N_hd=4

    light_hd = np.zeros((N_hd, N_light))
    for i_hd in range(0,4):
        for i_light in range(0,N_light):
            light_hd[i_hd,i_light] = light[i_light] * hd[i_hd]
    
    return CA3State(place,place_hd,light,light_hd)
    #return CA3State(place,place_hd,light,light_hd, dg.encodedValues) #ALAN apperantly CA3 doesn't need to know about Surfs? says: path.py line 56 as we are just getting the ground truths? This is backed up by the fact that touch sensors arnt used here

def CA3StateFromVector(v_ca3, N_places):

#TODO Another hard coded set here!!!!!!.... What does the light do!!!!!
    N_light=4
    N_hd=4

    place = v_ca3[0:N_places]

    place_hd = v_ca3[N_places:N_places + N_places*4]
    place_hd = place_hd.reshape((N_places,N_hd))
    # Luke again the bloody arms!
    light = v_ca3[N_places + N_places*4 : N_places + N_places*4 + N_light]   #which of 4 arms is lit

    light_hd = v_ca3[ N_places + N_places*4 + N_light : N_places + N_places*4 + N_light + N_light*N_hd ]
    light_hd = light_hd.reshape((N_hd,N_light)) #TODO check reshape is right way round

    return CA3State(place,place_hd,light,light_hd)


def ca3_states_to_matrix(ca3s):
    T=len(ca3s)
    N=ca3s[0].place.shape[0]
    out = np.zeros((T,N))
    for t in range(0,T):
        out[t,:] = ca3s[t].place
    #TODO convert to x,y coords here?

    return out



#Subbed in mine from DGStateAlan
"""
class DGState:
    def __init__(self, ec, dictGrids):

        N_place = 13
        N_hd = 4       

        l=Location()       #NEW, pure place cells in DG
        l.setGrids(ec.grids, dictGrids)
        self.place=np.zeros(N_place)
        self.place[l.placeId] = 1

        self.hd_lightAhead = np.zeros(4)
        if ec.lightAhead == 1:
            self.hd_lightAhead = ec.hd.copy()

        self.whisker_combis = np.zeros(3)  #extract multi-whisker features. 
        self.whisker_combis[0] = ec.whiskers[0] * ec.whiskers[1] * ec.whiskers[2]   #all on
        self.whisker_combis[1] = (1-ec.whiskers[0]) * (1-ec.whiskers[1]) * (1-ec.whiskers[2])   #none on
        self.whisker_combis[2] = ec.whiskers[0] * (1-ec.whiskers[1]) * ec.whiskers[2]   # both LR walls but no front

        #HOOK, needs to use EC data to define "combis" of features aswell

    def toVector(self):
        return np.hstack((self.place.flatten(), self.hd_lightAhead, self.whisker_combis))

    def toVectorSensesOnly(self):
        return np.hstack((self.whisker_combis))

    def toVectorOdomOnly(self):
        return np.hstack((self.place.flatten(), self.hd_lightAhead))

    def smartCollapse(self):                         #NEW
        self.place = smartCollapse(self.place)


def smartCollapse(xs):
    idx=argmax(xs)
    r = np.zeros(xs.flatten().shape)
    r[idx]=1
    return r.reshape(xs.shape)
"""





#converts a single place cell vector into an x,y coordinate
# Luke - This should be updated to recevie place cell inputs
def placeCells2placeID(_pcs, n_mazeSize):
    n_places = ((2*n_mazeSize)+1) **2 
    pcs = _pcs.copy()
    pcs=pcs[0:n_places]  #strip down to place cells only
    T = pcs.shape[0]
    grid = pcs.reshape(( ((2*n_mazeSize)+1),   ((2*n_mazeSize)+1) ))
    (xy) = np.where(grid==1)  
    return (xy[0][0], xy[1][0])   #return first (if several) matches






def ca3s2v(ca3s):    #CA3 states to vector
    N = ca3s[0].toVector().shape[0]
    T = len(ca3s)
    r = np.zeros((T,N))
    for t in range(0,T):
        r[t,0:N]=ca3s[t].toVector()
    return r


##with dentate 
def ecs2vd(ec_states):
    N_ec = ec_states[0].toVector().shape[0]
    N_dg = DGState(ec_states[0]).toVector().shape[0]
    N=N_ec+N_dg
    T = len(ec_states)
    r = np.zeros((T,N))
    for t in range(0,T):
        r[t,0:N]=ec_states[t].toVectorD()
    return r



##with dentate , senses obly
def ecs2vd_so(ec_states, dictGrids, dghelper=None):
    N = ec_states[0].toVectorSensesOnlyD(dictGrids,dghelper).shape[0]
    T = len(ec_states)
    r = np.zeros((T,N))
    for t in range(0,T):
        r[t,:]=ec_states[t].toVectorSensesOnlyD(dictGrids,dghelper)
    return r

##with dentate , odom only
def ecs2vd_oo(ec_states, dictGrids):
    N = ec_states[0].toVectorOdomOnlyD(dictGrids).shape[0]
    T = len(ec_states)
    r = np.zeros((T,N))
    for t in range(0,T):
        r[t,:]=ec_states[t].toVectorOdomOnlyD(dictGrids)
    return r
