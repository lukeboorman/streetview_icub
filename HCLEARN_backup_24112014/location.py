# This is the same as location.py, but has Nmax to be given, so it can create arbitrarily large mazes.
# Has to be changed more (e.g. see which functions use Nmax). For the moment, DictGrids is working with larger mazes and all functions 
# are using customizable Nmax. But have to change the grid creation to actually take into account this Nmax

#place cell code:          th headings:
#                           
#       6                            y
#       5                            ^ th<
#       4                     1      |    \
# 9 8 7 0 1 2 3             2 * 0    * -->x
#       10                    3
#       11
#       12

#conversion between x,y locations and place cell Ids.

import numpy as np

# If the maze is has x in [0, ..., xMax] and y in [0, ..., yMax], then xyMax = max(xMax, yMax)
# Then, we find the nearest to xyMax (rounding upwards) number K, so that 2^Nmax = K.
default_Nmax=3

# Returns mod 2^{n+1} < 2^n
# (see Fox paper before eq. 19)
def gComponent(x, i):
    return (  np.mod(x,2**(i+1)) < 2**i )



class DictGrids:
    def __init__(self, Nmax=default_Nmax):
        self.Nmax=int(Nmax)
        #self.basis = np.array([[1,2,4],[8,16,32]])
        # We want to put a unique int id per cell. E.g. 
        # (0,0) -> 0
        # (0,1) -> 8 etc. 
        # It doesn't matter if cell (x,y) is a valid maze location, we need to do this assignment for all x times y cells,
        # x in [0, ...., xMax] and y in [0, .. yMax]
        # The following function is taking two inputs and returns a unique reversible id.
        self.basis = np.vstack((2**np.arange(self.Nmax,dtype='i2'),2**(np.arange(self.Nmax,dtype='i2')+self.Nmax)))
        self.d=dict()
        for x in range(int(np.sqrt(np.amax(self.basis)*2))):#(0,2*self.Nmax): # LB modified as not using full range!
            for y in range(int(np.sqrt(np.amax(self.basis)*2))):#0,2*self.Nmax):
                grids=getGrids(x,y,self.Nmax)
                id = sum(sum(grids*self.basis))
                #print id
                self.d[id]=(x,y)
    def lookup(self, grids):
        id = sum(sum(grids*self.basis))
        #print(str(id))
        return self.d[id]


def getGrids(x,y, Nmax=default_Nmax):
    assert(x <= (Nmax*2)+1)
    assert(y <= (Nmax*2)+1)
    #Nmax = 3
    grids = np.zeros((2,Nmax))
    for i in range(0,Nmax):   #TODO should test this more, maybe a source of bugs?

        bx0=gComponent(x,i)
        by0=gComponent(y,i+1)
        grids[0,i] = bx0^by0     #xor

        bx1=gComponent(x,i+1)
        by1=gComponent(y,i)
        grids[1,i] = bx1^by1     #xor
    return grids


# Continue edits from here and below...

class Location:
    def __init__(self, Nmax=default_Nmax):
        foo=0
        self.Nmax=Nmax

    # TODO for resizable
    def setPlaceId(self, placeId):
        self.placeId = placeId
        if placeId<0 or placeId>12:
            print "ERROR, tried to set placeId outside maze!"
    
    # TODO for resizable
    def setXY(self, x, y):
        if x==3 and y==3:
            self.placeId=0
        elif y==3 and x>3:
            self.placeId = 0 + (x-3)
        elif x==3 and y>3:
            self.placeId = 3 + (y-3)
        elif y==3 and x<3:
            self.placeId = 6 + (3-x)
        elif x==3 and y<3:
            self.placeId = 9 + (3-y)
        else:
            print "ERROR, tried to set XY outside maze!"
        
    def setGrids(self, grids, dictGrids):
        (x,y) = dictGrids.lookup(grids)
        self.setXY(x,y)
# LUKE COMMENTED OUT HERE AS REPEATED
#    def getGrids(self):
#        (x,y)=self.getXY()
#        return getGrids(x,y)

    # TODO for resizable # LUKE -> will send in direct place cell rather than assigning by location 
    def getXY(self):
        if self.placeId > 9:
            d = (self.placeId-9)* np.array([0, -1])
        elif self.placeId > 6:
            d = (self.placeId-6)* np.array([-1,  0])
        elif self.placeId > 3:
            d = (self.placeId-3)* np.array([0, 1])
        elif self.placeId > 0:
            d = (self.placeId-0)* np.array([1, 0])
        else:
            d = np.array([0,0])
        center = np.array([3,3])
        return center+d

    # TODO for resizable
    def getGrids(self):

        Nmax = self.Nmax
        grids = np.zeros((2,Nmax))

        (x,y) = self.getXY()

        for i in range(0,Nmax):   #TODO should test this more, maybe a source of bugs?

            bx0=gComponent(x,i)
            by0=gComponent(y,i+1)
            grids[0,i] = bx0^by0     #xor

            bx1=gComponent(x,i+1)
            by1=gComponent(y,i)
            grids[1,i] = bx1^by1     #xor

        return grids


    


                 
  

dictGrids = DictGrids()
