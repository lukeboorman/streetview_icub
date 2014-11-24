import numpy as np
#from location import *
from locationLuke import Location


def plotPlaceCells(hist, iCell, gridDict):
    
    T = len(hist.ecs)
    # Luke: again this is hardcoded!!!!!!
    # LB: code modified to use Nmax
    square_grid=2**gridDict.Nmax    
    visits  = 0.00001+np.zeros((square_grid,square_grid))   #how many times agent has been here
    firings = np.zeros((square_grid,square_grid))   #how many times cell has fired   (avoid div by zero)
    #Old code    
    #visits  = 0.00001+np.zeros((7,7))   #how many times agent has been here
    #firings = np.zeros((7,7))   #how many times cell has fired   (avoid div by zero)
    
    #NB although we use a std CA3 class container for the CA3s, it has no real  semantics
    #the container was just build from a flat vector, whose weights were wake-sleep learned
    loc = Location(gridDict) # Luke modified
    for t in range(0,T):
        loc.setGrids( hist.ecs[t].grids) # Luke modified
        (x,y)=loc.getXY()
        visits[x][y] += 1
        
        if hist.ca3s[t].toVector()[iCell]:
            firings[x][y] += 1

    return (firings/visits, firings, visits)
            
    
