
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


def gComponent(x, i):
    return (  np.mod(x,2**(i+1)) < 2**i )



class DictGrids:
    def __init__(self):
        self.basis = np.array([[1,2,4],[8,16,32]])
        self.d=dict()
        for x in range(0,8):
            for y in range(0,8):
                grids=getGrids(x,y)
                id = sum(sum(grids*self.basis))
                self.d[id]=(x,y)
    def lookup(self, grids):
        id = sum(sum(grids*self.basis))
        return self.d[id]


def getGrids(x,y):
    Nmax = 3
    grids = np.zeros((2,Nmax))
    for i in range(0,Nmax):   #TODO should test this more, maybe a source of bugs?

        bx0=gComponent(x,i)
        by0=gComponent(y,i+1)
        grids[0,i] = bx0^by0     #xor

        bx1=gComponent(x,i+1)
        by1=gComponent(y,i)
        grids[1,i] = bx1^by1     #xor
    return grids



class Location:
    def __init__(self):
        foo=0

    def setPlaceId(self, placeId):
        self.placeId = placeId
        if placeId<0 or placeId>12:
            print "ERROR, tried to set placeId outside maze!"

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

    def getGrids(self):
        (x,y)=self.getXY()
        return getGrids(x,y)

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

    def getGrids(self):

        Nmax = 3
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
