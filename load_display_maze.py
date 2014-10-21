# -*- coding: utf-8 -*-
"""
Created on Fri Oct 03 12:12:43 2014
 Load all the images and build overall image....
@author: luke
"""

#import urllib
#import math
import numpy as np
import sys
#import filecmp
# import cv2.cv as cv
# import Tkinter
import cv2
import os
import glob
from win32api import GetSystemMetrics

# Path to images
pathname = 'D:/robotology/streetview_icub/division_street_1'
heading_index='NESW' #N=0, E=1, S=2, W=3

#==============================================================================
# root = Tkinter.Tk()
# 
# def callback(event):
#     print "Clicked at", event.x, event.y
# 
# frame = Tkinter.Frame(root, width=100, height=100)
# frame.bind("<Button-1>", callback)
# frame.pack()
# 
# root.mainloop()
#==============================================================================

# File list
piclist, combined_img = ([] for i in range(2))
image_count=0
# Windows screen size
screen_width = GetSystemMetrics (0)
screen_height = GetSystemMetrics (1)

no_files=len(glob.glob(os.path.join(pathname, '*.jpg')))

#file_database=np.empty([5,no_files],dtype=int)
file_database=np.empty(no_files,\
    dtype=[('file_id','u2'),('x_loc','u2'),('y_loc','u2'),('heading','u2'),('img_id','u2')])
#,no_files],dtype=int)

img_file=np.empty([no_files,640,640,3],dtype='u1') #unsigned 8 bit (1 byte)
file_database['file_id'][:]=range(0,no_files)


for infile in glob.glob(os.path.join(pathname, '*.jpg')):
    # ADD filename to list    
    piclist.append(infile)
    # Extract relevant file information.....
    # find start of filename section
    file_info=piclist[image_count][piclist[image_count].find("\\")+1:piclist[image_count].find("\\")+14]
    # img count , x, y, heading, img_num    
    # x grid
    file_database['x_loc'][image_count]=int(file_info[0:3])
    # y grid    
    file_database['y_loc'][image_count]=int(file_info[4:7])
    # Convert letter heading to index 1= N, 2=E, 3=S, 4=W
    file_database['heading'][image_count]=heading_index.find(file_info[8:9])
    # File identifier (optional extra e.g. two files at same location x,y and direction)
    file_database['img_id'][image_count]=int(file_info[10:13])
    # Massive data image block!!!
    img_file[image_count,:,:,:]=cv2.imread(infile)
    image_count += 1
    
#==============================================================================
# ### Mini task... get all northern images, in order of x location
# # Northern data,
# file_database_north=np.squeeze(file_database[:,np.transpose(np.where(file_database[3,:]==0)[0])])
# #Sub  sorted by x location.....
# file_database_north_sortx=file_database_north[:,np.argsort(file_database_north[1,:])]
# #### Combine images into panorama
# # First Image
# #cv2.imshow('FRED', img_file[1])
# combined_img = np.concatenate((img_file[file_database_north_sortx[0,0:5]]) , axis=1) #file_database_north_sortx[0,:]
# resized_img = cv2.resize(combined_img, (screen_width, screen_height)) 
# cv2.imshow('FRED', resized_img)
# ## ALTERNATIVE:: get NESW for location
#==============================================================================



### Mini task... get data for each location NSEW
## First sort by x location!!
#file_database_by_loc=np.squeeze(file_database[:,np.transpose(np.where(file_database[1,:]==0)[0])])
#Sub  sorted by y location.....
#file_database_by_loc_sorty=file_database_by_loc[:,np.argsort(file_database_by_loc[3,:])]
#### Combine images into panorama
file_database_sorted=np.sort(file_database,order=['x_loc','y_loc','heading'])
# First Image
#cv2.imshow('FRED', img_file[1])

# Choose location
location_x=5
location_y=0
direction='E'


# Convert heading
phase_wrap=np.array([3, 0, 1, 2, 3, 0],dtype='u1')
heading_ind=heading_index.find(direction)
heading_array=np.array([phase_wrap[heading_ind], phase_wrap[heading_ind+1],phase_wrap[heading_ind+2]])
# find x values
matched_x_loc=np.extract(file_database_sorted['x_loc']==location_x,file_database_sorted)
# Check values found!!!!!
if matched_x_loc.size<4:
    print "Not enough images at this x location!!, number img=\t", matched_x_loc.size 
    sys.exit()
# find y values
matched_y_loc=np.extract(matched_x_loc['y_loc']==location_y,matched_x_loc)
# Check values found!!!!!
if matched_y_loc.size<4:
    print "Not enough images at this y location!!, number img=\t", matched_y_loc.size 
    sys.exit()
    
images_2_display=matched_y_loc['file_id'][heading_array]

#images_2_display=[np.array([phase_wrap[heading_ind], phase_wrap[heading_ind+1],phase_wrap[heading_ind+2]])]
combined_img = np.concatenate(img_file[images_2_display] , axis=1) #file_database_north_sortx[0,:]
resized_img = cv2.resize(combined_img, (screen_width, screen_height)) 
cv2.imshow('FRED', resized_img)
## ALTERNATIVE:: get NESW for location




