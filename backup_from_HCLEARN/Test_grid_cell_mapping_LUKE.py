# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 09:50:42 2014

Test script understand grid cell encoding!!!.... why is it so complex!!!!

Run DEBUG and stop in makeMaze at line 98 : senses = Senses(n,x,y,ith,surfDict)

n=steps along each arm....
ith = which arm to go down


@author: luke
"""
import numpy as np
from makeMaze import Senses
import pickle
#
#saved_maze = pickle.load( open( 'D:/robotology/hclearn/maze_SEED2942875_N3_DG1_imdirdivision_street_1.pickle', "rb" ) )
#dictSenses = saved_maze[0]
#dictAvailableActions = saved_maze[1]
#dictNext = saved_maze[2]
#print "# Found and loaded pickled maze!"
grid_out=dict()


surfDict = pickle.load( open('Luke_save_surfDict_DCSCourtyard.pickle', "rb" ) )

x_range=np.array([3,2,1,0])
y_range=np.array([3,2,1,0])
n=3
ith_range=np.array(range(0,4))

current_ith=0
for current_x in x_range:
    for current_y in y_range:
        try:
            #for current_ith in 0:#ith_range:
            print 'Sending values x:',str(current_x),' y:',str(current_y),' ith (leg):',str(current_ith),' n (steps per arm):',str(n)
            senses = Senses(n,current_x,current_y,current_ith,surfDict)
            current_loc=(current_x,current_y,current_ith)
            grid_out[current_loc]=senses.grids
            print(str(senses.grids))
        except:
            pass