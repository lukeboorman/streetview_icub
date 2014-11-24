#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
# Test load each item to make sure it exists.....
from hcq import *
from gui import *
from locationLuke import * # using luke version....
from paths import * #CHANGED AS IPYTHON DOESN'T LIKE FILES/DIRS CALLED PATH
from makeMazeResizeable import * # Luke Altered to add new functions for displaying mze
import plotPlaceCells
#from makeMazeResizeable import displayMaze
#from os import sys
import learnWeights
import sys # repeat of above sys import
import cPickle as pickle
import os.path

#if len(sys.argv) == 3:
#    b_useNewDG = (sys.argv[1] == "True")
#    learningRate = float(sys.argv[2])
#else:
#    b_useNewDG = True
#    learningRate = 0.01

##################################################
################## Options #######################
##################################################
# LB: I have reorganised this to bring all to top of code to make it easier to understand and adjust....
#------------- Data input -> Maze -------------
imFolder ="division_street_1"   # Original: "DCSCourtyard" #"division_street_1" #"DivisionCarver"

#------------- Loading & Saving -------------
pickle_maze = True  # Original: True
# Andreas: If true then it'll attempt to load the maze (corresponding to the same set of configurations) 
# from a file. If the file doesn't exist, the algorithm will save the maze for future use.
# Andreas: Notice that makeMaze must have some radomness, because multiple runs of go.py return different results. 
## Load in previous ground truth data ??????????
b_usePrevGroundTruthCA3 = False  # Original: False
## ?????????????
b_useGroundTruthGrids = False  # Original: False


#------------- Plotting Options -------------
## Plot final outpts
b_plot = True # Original: True
## luke -> Plot mazes after load and enter interactive mode...
plot_maze = False  # Original: False
## luke -> Plot mazes and wonder round using learning paths...
plot_paths = True  # Original: False

#------------- Model Configuration -------------
#@@@@@@@@@@@@@ Learning @@@@@@@@@@@
## Number of learning steps around the maze...
T=30000   #trained on 30000   #better to have one long path than mult epochs on overfit little path
## Run new dentate gyrus code
b_useNewDG = True  # Original: True
## Set overall learning rate
learningRate = 0.01  # Original: 0.01
## Learn main model weights ( Go around the steps (T) built in paths)
b_learnWeights = True  # Original: True
## Use the learned model to infer locations using original path....    
b_inference_learn = True  # Original: True

#@@@@@@@@@@@@@ Controls @@@@@@@@@@@
# 1. Randomise (original) ~ Equivalent to ~Zero Mean
ctrl_randomised_zero = True  # Original: True 
# 2. Hand-set / ideal weights -> should give best value as uses ~GPS
ctrl_handset_ideal = True  # Original: True 
# 3. Randomise (using real random value's) ~ Equivalent to ~Mean (see Andreas)
ctrl_randomised_random = True  # Original: True 

#@@@@@@@@@@@@@ Testing @@@@@@@@@@@
## Number of training epochs -> NOT SURE WHAT THIS IS?
tr_epochs = 10  # Original: 10
## Number of steps for testing the model ability to predict locations
T_test = 100  # Original: 100
## Do more learning while performing inference (e.g. roaming around paths)...
b_learn = False  # Original: False  
## Only use observations ????????????
b_obsOnly = False  # Original: False
## ??????????????????????
b_useSub = True  # Original: True
## Use the model to infer locations using a new path (test phase)....    
b_inference_test = True  # Original: True
## N_mazeSize=3 -> THIS NOW AUTOMATICALLY GENERATED from loaded image files!

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
fullImageFolder = rootFolder + imFolder + "/"
#-------------------------------------------

# Luke added exception catch
#try:
# Luke added to graphically display maze....
if plot_maze:
    displayMaze(prefixFolder=fullImageFolder)
# LB: N_mazeSize as Nmax in dictGrids    

pickled_maze_name = "maze_SEED" + str(SEED) + "_DG" + str(int(b_useNewDG)) +  "_imdir" + imFolder + ".pickle"
if pickle_maze and os.path.isfile(pickled_maze_name):
    saved_maze = pickle.load( open( pickled_maze_name, "rb" ) )
    dictSenses = saved_maze[0]
    dictAvailableActions = saved_maze[1]
    dictNext = saved_maze[2]
    #N_mazeSize = saved_maze[3] 
    dictGrids = saved_maze[3]
    
    print "# Found and loaded pickled maze!"
else:
#    [dictSenses, dictAvailableActions, dictNext, N_mazeSize] = makeMaze(b_useNewDG, prefixFolder=fullImageFolder)  #make maze, including ideal percepts at each place
    [dictSenses, dictAvailableActions, dictNext, dictGrids] = makeMaze(b_useNewDG, prefixFolder=fullImageFolder)  #make maze, including ideal percepts at each place

    if pickle_maze:
        saved_maze = [dictSenses, dictAvailableActions, dictNext, dictGrids]
        pickle.dump( saved_maze, open( pickled_maze_name, "wb" ) )

# DictGrids is from location.py. Sets up a dictionary of grid cell locations from XY locations (I think!)

##### Next problem....
#dictGrids = DictGrids(dictPlaceCells,N_mazeSize)

# Luke Modified -> use start from first location in DictSenses
#start_location=[3,3,0] # Original setting in paths.py
start_location=np.asarray(dictSenses.keys()[0])

###### N_mazesize generated!!!
path_config = Paths(dictNext,dictGrids.Nmax, T, start_location)          #a random walk through the maze -- a list of world states (not percepts)

## Luke added to plot paths on maze..... Part of testing larger mazes.....
if plot_paths:
    displayPaths(fullImageFolder, path_config.posLog)
    
#(ecs_gnd, dgs_gnd, ca3s_gnd) = path_config.getGroundTruthFirings(dictSenses, dictGrids, N_mazeSize)  #ideal percepts for path_config, for plotting only
(ecs_gnd, dgs_gnd, ca3s_gnd) = path_config.getGroundTruthFirings(dictSenses, dictGrids)  #ideal percepts for path_config, for plotting only

if b_learnWeights:
    print "TRAINING..."
    #ALAN: Careful this won't exist if b_learnDGWeights is not true (I.e. we're not using SURF features
#    dghelper = learnWeights.learn(path_config, dictSenses, dictGrids, N_mazeSize, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=True, b_learnTrained=True, b_learnDGWeights=b_useNewDG, learningRate=learningRate, tr_epochs=tr_epochs)
    dghelper = learnWeights.learn(path_config, dictSenses, dictGrids, ecs_gnd, dgs_gnd, ca3s_gnd, b_learnIdeal=True, b_learnTrained=True, b_learnDGWeights=b_useNewDG, learningRate=learningRate, tr_epochs=tr_epochs)
else:
    dghelper=None

WR_t = np.load('tWR.npy')       ##NB loading trained versions from genuine wake sleep
WO_t = np.load('tWO.npy')
WS_t = np.load('tWS.npy')
WB_t = np.load('tWB.npy')
WB_t = WB_t.reshape(WB_t.shape[0])   #in case was learned as 1*N array instead of just N.

# Always loads ideal as comparison # if ctrl_handset_ideal:
WR_ideal = np.load('WR.npy')       ##NB loading trained versions from perfect look-ahead training
WO_ideal = np.load('WO.npy')
WS_ideal = np.load('WS.npy')
WB_ideal = np.load('WB.npy')
WB_ideal = WB_ideal.reshape(WB_ideal.shape[0])   #in case was learned as 1*N array instead of just N.

if ctrl_randomised_zero:
    # This is not random, it's something like the mean?
    WR_rand0 = 0+ 0*np.random.random(WR_ideal.shape)
    WB_rand0 = 0+ 0*np.random.random(WB_ideal.shape)
    WO_rand0 = 0+ 0*np.random.random(WO_ideal.shape)
    WS_rand0 = 0+ 0*np.random.random(WS_ideal.shape)

if ctrl_randomised_random:    
    # Real randomise weigths -> same results as above!
    WR_rand = np.random.random(WR_ideal.shape)
    WB_rand = np.random.random(WB_ideal.shape)
    WO_rand = np.random.random(WO_ideal.shape)
    WS_rand = np.random.random(WS_ideal.shape)

if b_inference_learn:
    print "INFERENCE... LEARNING PATH"
    hists=dict()
    ## Main learned values from using T steps around map!!!!
    random.seed(SEED) ;  np.random.seed(SEED)
    hists[('Learned')]    = makeMAPPredictions(path_config,dictGrids, dictSenses, WB_t, WR_t, WS_t, WO_t, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Learned", b_learn=b_learn)
    #HOOK test with ground truths on and off
    ## Reandomised weights -> fully random
    if ctrl_randomised_random:
        random.seed(SEED) ;  np.random.seed(SEED)
        hists[('Random')]  = makeMAPPredictions(path_config,dictGrids, dictSenses, WB_rand,  WR_rand,  WS_rand, WO_rand, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Random", b_learn=b_learn)
    if ctrl_handset_ideal:
        random.seed(SEED) ;  np.random.seed(SEED)
        hists[('Ideal')] = makeMAPPredictions(path_config,dictGrids, dictSenses, WB_ideal, WR_ideal, WS_ideal, WO_ideal, dghelper, b_obsOnly=b_obsOnly,  b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids, b_useSub=b_useSub, str_title="Handset", b_learn=b_learn)
    if ctrl_randomised_zero:
        random.seed(SEED) ;  np.random.seed(SEED)
        hists[('RandomZero')]   = makeMAPPredictions(path_config,dictGrids, dictSenses, WB_rand0,  WR_rand0,  WS_rand0, WO_rand0, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Random0", b_learn=b_learn)
print "DONE"

if b_plot:

    ##weights are modified in place
 #   np.save('tWR',WR_rand)
 #   np.save('tWS',WS)
 #   np.save('tWB',WB_rand)
 #   np.save('tWO',WO)
    losts=dict()
    anote = "obsonly%s_gndCA3%s_gndgrid%s_sub%s_bl%s_MERGED0102_SUBT026" % (b_obsOnly, b_usePrevGroundTruthCA3, b_useGroundTruthGrids, b_useSub, b_learn)
    ## Learned values
    losts[('Learned')],xys1 = plotResults(path_config, hists['Learned'], dictGrids, b_useNewDG, learningRate, note=anote)
    ## randomised weights
    if ctrl_randomised_random:    
        losts[('Random')],xys2 = plotResults(path_config, hists['Random'], dictGrids, b_useNewDG, learningRate, note=anote)
    if ctrl_handset_ideal:
        losts[('Ideal')],xys3 = plotResults(path_config, hists['Ideal'], dictGrids, b_useNewDG, learningRate, note=anote)
    if ctrl_randomised_zero:
        losts[('RandomZero')],xys4 = plotResults(path_config, hists['RandomZero'], dictGrids, b_useNewDG, learningRate, note=anote)
    #savefig('out/run.eps')
    #plotErrors(hist1, hist2, hist3, lost1, lost2, lost3, learningRate, surfTest=b_useNewDG, note=anote)
    #plotErrors4(hist1, hist2, hist3, hist4, lost1, lost2, lost3, lost4, learningRate, surfTest=b_useNewDG, note=anote)
    plotErrorsN(hists, losts, learningRate, surfTest=b_useNewDG, note=anote)
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
        for i in range(0,T_test):
            (r, visits,firings) = plotPlaceCells(hists[('Learned')], i, dictGrids)
            clf()
            gray()
            imagesc(r)
            show()
            fn = 'outPC/cell'+str(i)
            savefig(fn)

# --------- "Test" phase (new path_config, but reuse training weights etc)
#T_test = 100

# Luke -> Choose a random start location.....
print "Using randomised start location for test path...."
start_location=np.asarray(dictSenses.keys()[np.random.randint(0,len(dictSenses))])
testPath = Paths(dictNext,dictGrids.Nmax, T_test,start_location)          #a random walk through the maze -- a list of world states (not percepts)
(ecs_gnd, dgs_gnd, ca3s_gnd) = testPath.getGroundTruthFirings(dictSenses, dictGrids)  #ideal percepts for path_config, for plotting only
#testPath = Paths(dictNext,N_mazeSize, T_test)          #a random walk through the maze -- a list of world states (not percepts)
#(ecs_gnd, dgs_gnd, ca3s_gnd) = testPath.getGroundTruthFirings(dictSenses, dictGrids, N_mazeSize)  #ideal percepts for path_config, for plotting only

if b_inference_test:
    print "INFERENCE... SHORT TEST PATH"
    hists_path=dict()    
    random.seed(SEED) ;  np.random.seed(SEED)
    hists_path['Learned'] = makeMAPPredictions(testPath,dictGrids, dictSenses, WB_t, WR_t, WS_t, WO_t, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Learned", b_learn=b_learn)
    #HOOK test with ground truths on and off
    if ctrl_randomised_random: 
        random.seed(SEED) ;  np.random.seed(SEED)
        hists_path['Random'] = makeMAPPredictions(testPath,dictGrids, dictSenses, WB_rand,  WR_rand,  WS_rand, WO_rand, dghelper, b_obsOnly=b_obsOnly, b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids,  b_useSub=b_useSub, str_title="Random", b_learn=b_learn)
    if ctrl_handset_ideal:
        random.seed(SEED) ;  np.random.seed(SEED)
        hists_path['Ideal'] = makeMAPPredictions(testPath,dictGrids, dictSenses, WB_ideal, WR_ideal, WS_ideal, WO_ideal, dghelper, b_obsOnly=b_obsOnly,  b_usePrevGroundTruthCA3=b_usePrevGroundTruthCA3,  b_useGroundTruthGrids=b_useGroundTruthGrids, b_useSub=b_useSub, str_title="Handset", b_learn=b_learn)
print "DONE"

if b_plot:
    losts_path=dict()
    anote = "obsonly%s_gndCA3%s_gndgrid%s_sub%s_bl%s_MERGED0102_SUBT026" % (b_obsOnly, b_usePrevGroundTruthCA3, b_useGroundTruthGrids, b_useSub, b_learn)
    losts_path['Learned'],xys1 = plotResults(testPath, hists_path['Learned'], dictGrids, b_useNewDG, learningRate, note=anote)
    if ctrl_randomised_random: 
        losts_path['Random'],xys2 = plotResults(testPath, hists_path['Random'], dictGrids, b_useNewDG, learningRate, note=anote)
    if ctrl_handset_ideal:
        losts_path['Ideal'],xys3 = plotResults(testPath,hists_path['Ideal'], dictGrids, b_useNewDG, learningRate, note=anote)    
    plotErrorsN(hists_path, losts_path, learningRate, surfTest=b_useNewDG, note=anote)
    show()
# Luke added catch exception
#except Exception, e:
#    handleException(e)
#    raise       