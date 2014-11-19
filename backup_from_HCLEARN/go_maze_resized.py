import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hcq import *
from gui import *
from location import *
from paths import * #CHANGED AS IPYTHON DOESN'T LIKE FILES/DIRS CALLED PATH
from makeMazeResizeable import * # Luke Altered to add new functions for displaying mze
#from makeMazeResizeable import displayMaze
from os import sys
import learnWeights
import sys
import cPickle as pickle
import os.path

#if len(sys.argv) == 3:
#    b_useNewDG = (sys.argv[1] == "True")
#    learningRate = float(sys.argv[2])
#else:
#    b_useNewDG = True
#    learningRate = 0.01

b_useNewDG = True
learningRate = 0.01
    
print("Sys:%d" % len(sys.argv))
print("DG:%s" % b_useNewDG)
print("LR:%f" % learningRate)

np.set_printoptions(threshold=sys.maxint)         #=format short in matlab

# Set the root folder. You have two options:
# a) Put a file with name rootFolder.txt inside your visible path. 
#    It has to contain the full path of your root folder.
#    Do not sync with git this file (as it's different for every user)
# b) Do nothing, the code will find the current directory and use it as the default root folder.
rootFolderName = 'rootFolder.txt'
if os.path.isfile(rootFolderName):
    with open(rootFolderName,'r') as f:
        rootFolder = f.read().rstrip('\n')
    assert(f.closed)
else:
    rootFolder = os.path.dirname(os.path.abspath(__file__)) + '/'

# Run generate_map_from_streetview.py if you want to get the streetview pics

#----- Configuration -------------
## N_mazeSize=3 -> THIS NOW AUTOMATICALLY GENERATED!!!!!!!!!!!!!!!!!!!!!!!!!!

T=30000   #trained on 30000   #better to have one long path than mult epochs on overfit little path
b_learnWeights=True
b_plot=True
b_inference=True
tr_epochs=10
# If true then it'll attempt to load the maze (corresponding to the same set of configurations) 
# from a file. If the file doesn't exist, the algorithm will save the maze for future use.
# Notice that makeMaze must have some radomness, because multiple runs of go.py return different results.

## luke -> Plot mazes after load and enter interactive mode...
plot_maze=True
plot_paths=True

pickle_maze = True # True # True
imFolder ="division_street_1" # "DCSCourtyard" #"division_street_1" #"DivisionCarver" #DCSCourtyard"
fullImageFolder = rootFolder + imFolder + "/"
#-------------------------------------------

# Luke added to graphically display maze....
if plot_maze:
    displayMaze(prefixFolder=fullImageFolder)
    

pickled_maze_name = "maze_SEED" + str(SEED) + "_DG" + str(int(b_useNewDG)) +  "_imdir" + imFolder + ".pickle"
if pickle_maze and os.path.isfile(pickled_maze_name):
    saved_maze = pickle.load( open( pickled_maze_name, "rb" ) )
    dictSenses = saved_maze[0]
    dictAvailableActions = saved_maze[1]
    dictNext = saved_maze[2]
    N_mazeSize = saved_maze[3] 
    
    print "# Found and loaded pickled maze!"
else:
    [dictSenses, dictAvailableActions, dictNext, N_mazeSize] = makeMaze(b_useNewDG, prefixFolder=fullImageFolder)  #make maze, including ideal percepts at each place
    if pickle_maze:
        saved_maze = [dictSenses, dictAvailableActions, dictNext, N_mazeSize]
        pickle.dump( saved_maze, open( pickled_maze_name, "wb" ) )

# DictGrids is from location.py. Sets up a dictionary of grid cell locations from XY locations (I think!)

##### Next problem....
dictGrids = DictGrids()

# Luke Modified -> use start from first location in DictSenses
#start_location=[3,3,0] # Original setting in paths.py
start_location=np.asarray(dictSenses.keys()[0])

###### N_mazesize generated!!!
path_config = Paths(dictNext,N_mazeSize, T, start_location)          #a random walk through the maze -- a list of world states (not percepts)

## Luke added to plot paths on maze..... Part of testing larger mazes.....
if plot_maze and plot_paths:
    displayPaths(fullImageFolder, path_config.posLog)
    
(ecs_gnd, dgs_gnd, ca3s_gnd) = path_config.getGroundTruthFirings(dictSenses, dictGrids, N_mazeSize)  #ideal percepts for path_config, for plotting only

if b_learnWeights:
    print "TRAINING..."
    #ALAN: Careful this won't exist if b_learnDGWeights is not true (I.e. we're not using SURF features
    dghelper = learnWeights.learn(path_config, dictSenses, dictGrids, N_mazeSize, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=True, b_learnTrained=True, b_learnDGWeights=b_useNewDG, learningRate=learningRate, tr_epochs=tr_epochs)
else:
    dghelper=None

WR_t = np.load('tWR.npy')       ##NB loading trained versions from genuine wake sleep
WO_t = np.load('tWO.npy')
WS_t = np.load('tWS.npy')
WB_t = np.load('tWB.npy')
WB_t = WB_t.reshape(WB_t.shape[0])   #in case was learned as 1*N array instead of just N.

WR_ideal = np.load('WR.npy')       ##NB loading trained versions from perfect look-ahead training
WO_ideal = np.load('WO.npy')
WS_ideal = np.load('WS.npy')
WB_ideal = np.load('WB.npy')

# This is not random, it's something like the mean?
WR_rand0 = 0+ 0*np.random.random(WR_ideal.shape)
WB_rand0 = 0+ 0*np.random.random(WB_ideal.shape)
WO_rand0 = 0+ 0*np.random.random(WO_ideal.shape)
WS_rand0 = 0+ 0*np.random.random(WS_ideal.shape)

WR_rand = np.random.random(WR_ideal.shape)
WB_rand = np.random.random(WB_ideal.shape)
WO_rand = np.random.random(WO_ideal.shape)
WS_rand = np.random.random(WS_ideal.shape)


b_inference = True

b_obsOnly = False
b_usePrevGroundTruthCA3 = False
b_useGroundTruthGrids = False
b_useSub = True
b_learn = False

if b_inference:
    print "INFERENCE..."

    random.seed(SEED) ;  np.random.seed(SEED)
    hist1    = makeMAPPredictions(path_config,dictGrids, dictSenses, WB_t, WR_t, WS_t, WO_t, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Learned", b_learn=b_learn)
    #HOOK test with ground truths on and off

    random.seed(SEED) ;  np.random.seed(SEED)
    hist2   = makeMAPPredictions(path_config,dictGrids, dictSenses, WB_rand,  WR_rand,  WS_rand, WO_rand, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Random", b_learn=b_learn)

    random.seed(SEED) ;  np.random.seed(SEED)
    hist3 = makeMAPPredictions(path_config,dictGrids, dictSenses, WB_ideal, WR_ideal, WS_ideal, WO_ideal, dghelper, b_obsOnly=b_obsOnly,  b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids, b_useSub=b_useSub, str_title="Handset", b_learn=b_learn)

    random.seed(SEED) ;  np.random.seed(SEED)
    hist4   = makeMAPPredictions(path_config,dictGrids, dictSenses, WB_rand0,  WR_rand0,  WS_rand0, WO_rand0, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Random0", b_learn=b_learn)


print "DONE"

if b_plot:

    ##weights are modified in place
 #   np.save('tWR',WR_rand)
 #   np.save('tWS',WS)
 #   np.save('tWB',WB_rand)
 #   np.save('tWO',WO)
    anote = "obsonly%s_gndCA3%s_gndgrid%s_sub%s_bl%s_MERGED0102_SUBT026" % (b_obsOnly, b_usePrevGroundTruthCA3, b_useGroundTruthGrids, b_useSub, b_learn)
    
    (lost1,xys1) = plotResults(path_config, hist1, dictGrids, b_useNewDG, learningRate, note=anote)
    (lost2,xys2) = plotResults(path_config, hist2, dictGrids, b_useNewDG, learningRate, note=anote)
    (lost3,xys3) = plotResults(path_config, hist3, dictGrids, b_useNewDG, learningRate, note=anote)
    (lost4,xys4) = plotResults(path_config, hist4, dictGrids, b_useNewDG, learningRate, note=anote)

#    savefig('out/run.eps')
 
    #plotErrors(hist1, hist2, hist3, lost1, lost2, lost3, learningRate, surfTest=b_useNewDG, note=anote)
    plotErrors4(hist1, hist2, hist3, hist4, lost1, lost2, lost3, lost4, learningRate, surfTest=b_useNewDG, note=anote)

    ##figure()
    ##drawMaze()
    ##hold(True)
    ##drawPath(path_config.posLog[:,0:2],'k')
    ##savefig('out/maze.eps')
    ##drawPath(xys_pi, 'b')

    #clf()
    show()

    
    b_other = False
    if b_other:
        print "close the first plus maze window to begin the slideshow of place fields!"
        for i in range(0,100):
            (r, visits,firings) = plotPlaceCells(hist1, i, dictGrids)
            clf()
            gray()
            imagesc(r)
            show()

            fn = 'outPC/cell'+str(i)
            savefig(fn)



# --------- "Test" phase (new path_config, but reuse training weights etc)
T_test = 100
testPath = Paths(dictNext,N_mazeSize, T_test)          #a random walk through the maze -- a list of world states (not percepts)
(ecs_gnd, dgs_gnd, ca3s_gnd) = testPath.getGroundTruthFirings(dictSenses, dictGrids, N_mazeSize)  #ideal percepts for path_config, for plotting only

if b_inference:
    print "INFERENCE..."

    random.seed(SEED) ;  np.random.seed(SEED)
    hist1    = makeMAPPredictions(testPath,dictGrids, dictSenses, WB_t, WR_t, WS_t, WO_t, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Learned", b_learn=b_learn)
    #HOOK test with ground truths on and off

    random.seed(SEED) ;  np.random.seed(SEED)
    hist2   = makeMAPPredictions(testPath,dictGrids, dictSenses, WB_rand,  WR_rand,  WS_rand, WO_rand, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Random", b_learn=b_learn)

    random.seed(SEED) ;  np.random.seed(SEED)
    hist3 = makeMAPPredictions(testPath,dictGrids, dictSenses, WB_ideal, WR_ideal, WS_ideal, WO_ideal, dghelper, b_obsOnly=b_obsOnly,  b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids, b_useSub=b_useSub, str_title="Handset", b_learn=b_learn)
print "DONE"

if b_plot:
    anote = "obsonly%s_gndCA3%s_gndgrid%s_sub%s_bl%s_MERGED0102_SUBT026" % (b_obsOnly, b_usePrevGroundTruthCA3, b_useGroundTruthGrids, b_useSub, b_learn)
    (lost1,xys1) = plotResults(testPath, hist1, dictGrids, b_useNewDG, learningRate, note=anote)
    (lost2,xys2) = plotResults(testPath, hist2, dictGrids, b_useNewDG, learningRate, note=anote)
    (lost3,xys3) = plotResults(testPath, hist3, dictGrids, b_useNewDG, learningRate, note=anote)
    plotErrors(hist1, hist2, hist3, lost1, lost2, lost3, learningRate, surfTest=b_useNewDG, note=anote)
    show()