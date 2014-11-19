import numpy as np
from SURFExtractor_LUKE import makeSURFRepresentation
#from SURFExtractor import TestComparisons
from location import Location
from generate_maze_from_data import maze_from_data
printMessages = True

# Luke Boorman November 2014 -> ADAPTED TO MOVE AWAY FROM FIXED OUTPUT PLUS MAZE!!!!!!

directionDict = {0: 'N', 1: 'E', 2: 'S', 3: 'W'} 
#an ideal set of EC responses, to a particular agent state (excluding lightAhead)
#TODO: maybe classes should rep states in simplest possible way, and only convert to cells when requested??
class Senses:
    def __init__(self):#,x,y, direction, SURFdict):
        
        self.hd=np.zeros(4)
        self.rgb=np.zeros(3)
        self.surfs=[]
        self.whiskers=np.zeros(3)
        self.grids=[]
#        #self.placeCells = np.zeros((1+4*N_mazeSize))
#        loc=Location()
#        loc.setXY(x,y)
#        #placeId = loc.placeId
#        #self.placeCells[placeId] = 1   
#
#        self.grids = loc.getGrids().copy()
#        
#        # Add attribute head direction - NSEW
#        self.headdirection = np.array([0,0,0,0])
#        self.headdirection[direction]=1
#
#        self.rgb=np.array([0,0,0])  #assume a red poster in the east and a green poster in the north
#        if direction==0:          #NB these differ from head direction cells, as headdirection odometry can go wrong!
#            self.rgb[0]=1
#        if direction==1:
#            self.rgb[1]=1
#        
#        if SURFdict is not None:
#            #HOOK ALAN - include SURF features in the senses of the dictionary
#            #Problem wdirection merging here is you can only have one image per direction?
#            self.surfs=findSurfs(x,y,direction,SURFdict)
#        else:
#            self.surfs=np.array([])
#            
#        #print("Surf feature for %d,%d,%d:\n%s" % (x,y,direction,self.surfs))
#        #x,y relate to surf features in SURFdict
#        self.whiskers=np.array([0,0,0]) #to be filled in outside
        

STAY=0   #enum actions
FWD=1
LEFT=2
RIGHT=3
UTURN=4

def findSurfs(x,y,direction,SURFdict):
    # LB: THIS SHOULD TSYA THE SAME!!!!!
    # OLD Need a dictionary to convert directions as extractor makes a dictionary wdirection NSEW and Senses uses a direction 0 1 2 or 3
    # OLD directionDict = {0: 'E', 1: 'N', 2: 'W', 3: 'S'}
    #  Changed
    # Dictionary to convert directions wdirection NESW and Senses uses a direction 0 1 2 or 3
    #directionDict = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}    
    #Convert direction
    direction = directionDict[direction]
    
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
        # LB switched this on for missing image test....        
        print("WARNING!!!!!!!!!! No feature for (x,y,direction) key: %s" % (key,))
        firstDesc = SURFdict.values()[0][0]
        #If no feature exists, just send back an empty feature set?
        
        return np.array([0]*len(firstDesc))
        #Or we could raise an exception since it should never really happen
        #raise NameError("There isn't a surf feature description for: %s" % (key,))

#n = number of locations per arm (so that (n,n) is the center point)
def makeMaze(b_useNewDG=False, prefixFolder = None):

    # Set up maze data loader
    maze_data=maze_from_data(prefixFolder)
    # Load data from folder     
    maze_data.index_image_files()
    # Display maze - TESTING
    #maze_data.display_maps_images()
    #maze_data.maze_interactive()
    # maze_data.maze_walk()
    
    # Run SURF features....
    surfDict=None
    
    surfDict = makeSURFRepresentation(prefixFolder,False)     
    print('SURF Completed')
    
    ## Store all sensory inputs in here, head direction, image surfs, whiskers
    dictSenses=dict()
    dictAvailableActions=dict()
    dictNext=dict()
    
    # Calculate NMax as grid cell encoding...
    # Setup as a 2xn grid, therfore need to encode for complete grid sapce, ie if x=4,y=5 grid x*y=20
    # Max size of peak2peak of x and y, log2 and maximum size.....     
    Nmax=int(np.ceil(np.log2(np.amax([maze_data.place_cell_id[1].ptp(),maze_data.place_cell_id[2].ptp()]))))   
    # Setup location class with nMax    
    loc=Location(Nmax)
    
    # Step for each move around Maze
    maze_step_size=1    
    
    # Calc step depending on direction...... PROBELMS HERE WITH LUKE AND CF MAZES!!!!               
    # Luke x = East is Positive, y = North = positive
    # 'N' x=0, y=+ /'E' x=+, y=0 / 'S' x=0, y=- / 'W' x=-, y=0
    # step_index=np.array([[0,maze_step_size],[maze_step_size,0],[0,-maze_step_size],[-maze_step_size,0]])
#    print 'WARNING RUNNING IN Luke MODE..... North positive!'  
    # CF x = East is Positive y = South = Positive          
    # 'N' x=0, y=- / 'E' x=-, y=0 / 'S' x=0, y=+ / 'W' x=+, y=0
    print 'WARNING RUNNING IN CHARLES FOX MODE..... North negative!'    
    step_index=np.array([[0,-maze_step_size],[maze_step_size,0],[0,maze_step_size],[-maze_step_size,0]])
    
    
    # Loop through all locations.... N,E,S,W    
    for current_id in range(0,maze_data.place_cell_id[0,:].size):
       # maze_data.find_next_set_images(0,26,1)
        current_x=maze_data.place_cell_id[1,current_id]
        current_y=maze_data.place_cell_id[2,current_id]
        
        print('Place cell ID:',str(current_id),' x:',str(current_x),' y:',str(current_y))
        ## Fn find_next_set_images
        ## returns: (images_to_combine,image_found,heading,direction_vector,picture_name,available_direction_vector)
        ## available_direction_vector = [forwards, backwards, left, right]
        
        # Get only those headings available....
        included_headings=np.where(maze_data.locations_unique[(current_x,current_y)]['Image_count']>0)
        
        for current_direction in included_headings[0]: # N=0,E=1,S=2,W=3
            #direction = directionDict[current_direction]
            # SURF HAS A DIFFERENT BLOODY DICTIONARY Key!!!!
            #current_surf_location=((current_x,current_y),direction)            
            current_location=(current_x,current_y,current_direction)             
            _,image_found,_,_,_,available_direction_vector=maze_data.find_next_set_images(current_x,current_y,current_direction)
            if image_found:
                # Sense dictionary
                dictSenses[current_location]=Senses()
                dictSenses[current_location].hd[current_direction]=1 # REVERSE and REMOVE BACKWARDS            
                dictSenses[current_location].whiskers=np.square(available_direction_vector[[2,0,3]]-1) # LEFT / FORWARDS / RIGHT
                dictSenses[current_location].rgb=np.array([0,0,0]) # Not set at the moment
                try:
                    dictSenses[current_location].surfs=findSurfs(current_x,current_y,current_direction,surfDict)
                except:
                    pass
                ##dictSenses[current_location].grids=1#SORT GRIDS   
                loc.setXY(current_x,current_y)
                dictSenses[current_location].grids = loc.getGrids().copy()
                        
                # Which ways can you go....
                # Take from available action vector [FWD=1, REVERSE(UTURN)=4, LEFT=2, RIGHT=3]
                        
                        
                #TODO: MAY need to spin this round - Charles fox different systems        
                dictAvailableActions[current_location] = np.concatenate(([0],np.array([1,4,2,3])[np.where(available_direction_vector>0)]))#[STAY,FWD,LEFT,RIGHT]

                # What is available next                
                #WTF!!!! dictNext[current_location] = [(x,y,direction), (x+step_xs[direction],y+step_ys[direction],direction), (x,y,direction_l), (x,y,direction_r)] 
                # Stay in the same place!!!!                
                # = 0                
                #dictNext[current_location] = [current_location]
                next_locations=(current_x,current_y,current_direction)
                # Calc step depending on direction...... x, y             
                current_stepping=step_index[current_direction]            
                # Forward                
                if 1 in dictAvailableActions[current_location]:
                    #dictNext[current_location].append([(current_x+current_stepping[0],current_y+current_stepping[1]),current_direction])    
                    next_locations=next_locations+(current_x+current_stepping[0],current_y+current_stepping[1],current_direction)
                # Left               
                if 2 in dictAvailableActions[current_location]:
                    new_heading,_=maze_data.phase_wrap_heading(current_direction-1)
                    #dictNext[current_location].append([(current_x,current_y),directionDict[new_heading]])
                    next_locations=next_locations+(current_x,current_y,new_heading)
                # Right              
                if 3 in dictAvailableActions[current_location]:
                    new_heading,_=maze_data.phase_wrap_heading(current_direction+1)
                    #dictNext[current_location].append([(current_x,current_y),directionDict[new_heading]])
                    next_locations=next_locations+(current_x,current_y,new_heading)
                # U-TURN, reverse             
                if 4 in dictAvailableActions[current_location]:
                    new_heading,_=maze_data.phase_wrap_heading(current_direction+2)
                    #dictNext[current_location].append([(current_x,current_y),directionDict[new_heading]])
                    next_locations=next_locations+(current_x,current_y,new_heading)
                dictNext[current_location] = zip(next_locations[::3],next_locations[1::3],next_locations[2::3])
                
                
                #, (x+step_xs[direction],y+step_ys[direction],direction), (x,y,direction_l), (x,y,direction_r)] 
                
                
    print ('Completed Dict for current data...')    
    # Walk around maze.... just use place cells    
    
    
#==============================================================================
#     
#     if b_useNewDG:
#         print("Generating SURF representations...")
#         surfDict = makeSURFRepresentation(prefixFolder,False) # Last argument = draw output....
#         if printMessages:
#             print("SURFDICTKEYS:%s" % surfDict.keys())
#     # 1. Dict input information a) SURF features b) head direction c) RGB d) Grids for this location e) whiskers       
#     dictSenses=dict()           # sensory info for a given location 
#     # 2. Dict of which you can go for each tile.....
#     dictAvailableActions=dict() # available actions for a given location -> forwards, left, right, stay (No backwards)
#     # 3. Next step.... how is this coded  (LB)  
#     dictNext=dict()             # where you would end up next having taken a given action (M.E. I think!)
#     
#     # LB Need to walk around the whole maze 
# 
#     print 'Cheese'
#     
#     keys_surfDict=surfDict.keys()
#     # Run for every single tile in database....    
#     for current_tile in range(0,len(keys_surfDict)):
#         current_x=keys_surfDict[current_tile][0][0]
#         current_y=keys_surfDict[current_tile][0][1]
#         current_heading=keys_surfDict[current_tile][1]
#==============================================================================
        # 1. dictSenses
        # c) RGB
        
    
    
    
#==============================================================================
#     LB: All OLD hardwired for PLUS MAZE!!!!!!
#     
#     step_xs =  [1, 0, -1,  0]   #converts direction angles into x,y vector headings
#     step_ys =  [0, 1,  0, -1]
# 
#     for direction in range(0,4):           #walk down each arm
#         
#         direction_u   = ((direction+2) % 4)  #heading in opposite direction/u turn
#         direction_l   = ((direction+1) % 4)  #after a left turn
#         direction_r   = ((direction-1) % 4)  #after a right turn
# 
#         x=y=n             #start at center -- add its data first, then walk outwards.    
#         stateCenter = (x,y,direction)
#         senses = Senses(n,x,y,direction,surfDict)
#         dictSenses[stateCenter] = senses
#         dictAvailableActions[stateCenter] = [STAY,FWD,LEFT,RIGHT]
#         dictNext[stateCenter] = [(x,y,direction), (x+step_xs[direction],y+step_ys[direction],direction), (x,y,direction_l), (x,y,direction_r)] 
# 
#         for i in range(1,n+1):  #walk out to end of arm (excluding the center and arm end, handled separately)
#             x+=step_xs[direction]
#             y+=step_ys[direction]            
# 
#             state1 = (x,y,direction)                          #state1, facing towards the arm end...
#             senses1 = Senses(n,x,y,direction,surfDict)
#             if i==n:                                        #spec case at end of arm
#                 dictAvailableActions[state1] = [STAY,UTURN]
#                 dictNext[state1] = [(x,y,direction), (x,y,direction_u)]          
#                 senses1.whiskers=np.array([1,1,1])           #facing the wall
#             else:
#                 dictAvailableActions[state1] = [STAY,UTURN,FWD]
#                 dictNext[state1] = [(x,y,direction), (x,y,direction_u), (x+step_xs[direction],y+step_ys[direction],direction)]
#                 senses1.whiskers=np.array([1,0,1])
#             dictSenses[state1] = senses1
#             
#             state2 = (x,y,direction_u)                            #state2 is facing towards the center:
#             senses2 = Senses(n,x,y,direction_u,surfDict)
#             senses2.whiskers=np.array([1,0,1])
#             dictSenses[state2] = senses2
#             dictAvailableActions[state2] = [STAY,FWD,UTURN]
#             dictNext[state2] = [(x,y,direction_u), (x-step_xs[direction],y-step_ys[direction],direction_u), (x,y,direction)]        
#==============================================================================

    #print("dictSenses\n%s\ndictAvailableActions\n%s\ndictNext\n%s\nThere are %d images\n" % (dictSenses.keys(), dictAvailableActions,dictNext, len(dictSenses.keys())))
    #print("\n\ndictSenses:\n%s" % dictSenses)
    return [dictSenses, dictAvailableActions, dictNext, Nmax]

#No longer needed, sorted it out by passing it from go to hcq..
#[dictSenses, dictAvailableActions, dictNext] = makeMaze(3)

# Loads and plots the maze as a graphic + map of place cell id's and + map of current position
def displayMaze(prefixFolder = None):
    # Set up maze data loader
    maze_data=maze_from_data(prefixFolder)
    # Load data from folder     
    maze_data.index_image_files()
    # Display maze - TESTING
    maze_data.display_maps_images()
    ### User interative mode -> Use keys Exit:'ESC' FD:W BK:A LEFT:S RIGHT:D
    maze_data.maze_interactive()
    #maze_data.maze_walk()

    
def displayPaths(prefixFolder = None, paths=None):
    # Set up maze data loader
    maze_data=maze_from_data(prefixFolder)
    # Load data from folder     
    maze_data.index_image_files()
    # Display maze - TESTING
    maze_data.display_maps_images()
    ### User interative mode -> Use keys Exit:'ESC' FD:W BK:A LEFT:S RIGHT:D
    #maze_data.maze_interactive()
    #maze_data.maze_walk()
    # Run interactive mode but iterate around the locations    
    maze_data.maze_walk(False,paths)
    
    
    
    
