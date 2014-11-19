import numpy as np
from SURFExtractor import makeSURFRepresentation
from location import Location

printMessages = False

#an ideal set of EC responses, to a particular agent state (excluding lightAhead)
#TODO: maybe classes should rep states in simplest possible way, and only convert to cells when requested??
class Senses:
    def __init__(self,N_mazeSize, x,y, ith, SURFdict):
        

        #self.placeCells = np.zeros((1+4*N_mazeSize))
        loc=Location()
        loc.setXY(x,y)
        #placeId = loc.placeId
        #self.placeCells[placeId] = 1   

        self.grids = loc.getGrids().copy()

        self.hd = np.array([0,0,0,0])
        self.hd[ith]=1

        self.rgb=np.array([0,0,0])  #assume a red poster in the east and a green poster in the north
        if ith==0:          #NB these differ from head direction cells, as HD odometry can go wrong!
            self.rgb[0]=1
        if ith==1:
            self.rgb[1]=1
        
        if SURFdict is not None:
            #HOOK ALAN - include SURF features in the senses of the dictionary
            #Problem with merging here is you can only have one image per direction?
            self.surfs=findSurfs(x,y,ith,SURFdict)
        else:
            self.surfs=np.array([])
            
        #print("Surf feature for %d,%d,%d:\n%s" % (x,y,ith,self.surfs))
        #x,y relate to surf features in SURFdict
        self.whiskers=np.array([0,0,0]) #to be filled in outside

STAY=0   #enum actions
FWD=1
LEFT=2
RIGHT=3
UTURN=4

def findSurfs(x,y,ith,SURFdict):
    #Need a dictionary to convert directions as extractor makes a dictionary with NSEW and Senses uses a direction 0 1 2 or 3
    directionDict = {0: 'E', 1: 'N', 2: 'W', 3: 'S'} 
    #Convert direction
    direction = directionDict[ith]
    
    key = ((x,y),direction)
    print key
    #Careful, there could be multiple images per location/direction mapping!
    if key in SURFdict.keys():
        surfFeatures = SURFdict[((x,y),direction)]
        if printMessages:
            print("Features for key: %s\n%s" % (key, surfFeatures))
            #FIX: Make this work for multiple images of the same location?
            #Currently just returns the first image in the dictionary
            print("Using the second image:\n%s" % surfFeatures[1])
        #If there is more than one surf feature for this direction, choose a different one from the one which it was trained on
        if len(surfFeatures) > 1:
            return surfFeatures[1]
        else:
            return surfFeatures[0]
    else:
        # LB switched this on for missing images        
        print("No feature for (x,y,direction) key: %s" % (key,))
        firstDesc = SURFdict.values()[0][0]
        #If no feature exists, just send back an empty feature set?
        
        return np.array([0]*len(firstDesc))
        #Or we could raise an exception since it should never really happen
        #raise NameError("There isn't a surf feature description for: %s" % (key,))

#n = number of locations per arm (so that (n,n) is the center point)
def makeMaze(n, b_useNewDG=False, prefixFolder = None):
    surfDict=None
    if b_useNewDG:
        print("Generating SURF representations...")
        surfDict = makeSURFRepresentation(prefixFolder)
        if printMessages:
            print("SURFDICTKEYS:%s" % surfDict.keys())
    dictSenses=dict()           # sensory info for a given location 
    dictAvailableActions=dict() # available actions for a given location -> forwards, left, right, stay (No backwards)
    dictNext=dict()             # where you would end up next having taken a given action (M.E. I think!)
    step_xs =  [1, 0, -1,  0]   #converts ith angles into x,y vector headings
    step_ys =  [0, 1,  0, -1]

    for ith in range(0,4):           #walk down each arm
        
        ith_u   = ((ith+2) % 4)  #heading in opposite direction/u turn
        ith_l   = ((ith+1) % 4)  #after a left turn
        ith_r   = ((ith-1) % 4)  #after a right turn

        x=y=n             #start at center -- add its data first, then walk outwards.    
        stateCenter = (x,y,ith)
        senses = Senses(n,x,y,ith,surfDict)
        dictSenses[stateCenter] = senses
        dictAvailableActions[stateCenter] = [STAY,FWD,LEFT,RIGHT]
        dictNext[stateCenter] = [(x,y,ith), (x+step_xs[ith],y+step_ys[ith],ith), (x,y,ith_l), (x,y,ith_r)] 

        for i in range(1,n+1):  #walk out to end of arm (excluding the center and arm end, handled separately)
            x+=step_xs[ith]
            y+=step_ys[ith]            

            state1 = (x,y,ith)                          #state1, facing towards the arm end...
            senses1 = Senses(n,x,y,ith,surfDict)
            if i==n:                                        #spec case at end of arm
                dictAvailableActions[state1] = [STAY,UTURN]
                dictNext[state1] = [(x,y,ith), (x,y,ith_u)]          
                senses1.whiskers=np.array([1,1,1])           #facing the wall
            else:
                dictAvailableActions[state1] = [STAY,UTURN,FWD]
                dictNext[state1] = [(x,y,ith), (x,y,ith_u), (x+step_xs[ith],y+step_ys[ith],ith)]
                senses1.whiskers=np.array([1,0,1])
            dictSenses[state1] = senses1
            
            state2 = (x,y,ith_u)                            #state2 is facing towards the center:
            senses2 = Senses(n,x,y,ith_u,surfDict)
            senses2.whiskers=np.array([1,0,1])
            dictSenses[state2] = senses2
            dictAvailableActions[state2] = [STAY,FWD,UTURN]
            dictNext[state2] = [(x,y,ith_u), (x-step_xs[ith],y-step_ys[ith],ith_u), (x,y,ith)]        

    #print("dictSenses\n%s\ndictAvailableActions\n%s\ndictNext\n%s\nThere are %d images\n" % (dictSenses.keys(), dictAvailableActions,dictNext, len(dictSenses.keys())))
    #print("\n\ndictSenses:\n%s" % dictSenses)
    return [dictSenses, dictAvailableActions, dictNext]

#No longer needed, sorted it out by passing it from go to hcq..
#[dictSenses, dictAvailableActions, dictNext] = makeMaze(3)

