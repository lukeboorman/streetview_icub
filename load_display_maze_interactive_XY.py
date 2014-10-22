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
import cv2
import os
import glob
from win32api import GetSystemMetrics

# Path to images
pathname = 'D:/robotology/streetview_icub/division_street_2'
heading_index='NESW' #N=0, E=1, S=2, W=3
window_name='Streetview'

##### Find matching 3 files to display
def find_next_set_images(location_x,location_y,heading,file_database_sorted,picture_name_list):
    
    image_found=0
    
    heading,direction_vector=phase_wrap_heading(heading)
    # Convert heading
    phase_wrap=np.array([3, 0, 1, 2, 3, 0],dtype='u1')
    heading_array=np.array([phase_wrap[heading], phase_wrap[heading+1],phase_wrap[heading+2]])
    # find x values
    matched_x_loc=np.extract(file_database_sorted['x_loc']==location_x,file_database_sorted)
    # Check values found!!!!!
    if matched_x_loc.size<4:
        print "Not enough images at this x location!!, number img=\t", matched_x_loc.size 
        return (0,0,heading,direction_vector,0)

    # find y values
    matched_y_loc=np.extract(matched_x_loc['y_loc']==location_y,matched_x_loc)
    # Check values found!!!!!
    if matched_y_loc.size<4:
        print "Not enough images at this y location!!, number img=\t", matched_y_loc.size 
        return (0,0,heading,direction_vector,0)
 
    images_2_display=matched_y_loc['file_id'][heading_array]
    combined_img = np.concatenate(img_file[images_2_display] , axis=1) #file_database_north_sortx[0,:]
    resized_img = cv2.resize(combined_img, (image_display_width, image_display_height)) 
    image_found=1
    picture_name=picture_name_list[images_2_display[1]]
    return (resized_img,image_found,heading,direction_vector,picture_name)
    
### Check heading is range.....
def phase_wrap_heading(heading):
    while True:
        if heading>3:
            heading=heading-4
        elif heading<0:
            heading=heading+4
        if heading<=3 and heading>=0:
            break
    # Work out direction vectors....
    if heading==0: # North = x=0, y=1
        direction_vector=[0,1]
    elif heading==1: # East = x=1, y=0
        direction_vector=[1,0]
    elif heading==2: # south = x=0, y=-1
        direction_vector=[0,-1]
    else: # west = x=-1, y=0
        direction_vector=[-1,0]
    return (heading, direction_vector)
########################################
# Make database of image files
#####################################


# File list
piclist = []
image_count=0
# Windows screen size
screen_width = GetSystemMetrics (0)
screen_height = GetSystemMetrics (1)

image_display_width=screen_width
image_display_height=int(round(screen_width/3,0))

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
location_x=0
location_y=0
direction='E'


heading_ind=heading_index.find(direction)

new_location_x=location_x
new_location_y=location_y
#new_heading_ind=heading_ind

resized_img,image_found,new_heading_ind,direction_vector,image_title=find_next_set_images(location_x,location_y,heading_ind,file_database_sorted,piclist)

if image_found==0:
    print "No image exiting"
    sys.exit()
cv2.putText(resized_img, image_title, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0);    
cv2.imshow(window_name, resized_img)
## ALTERNATIVE:: get NESW for location

### Wait for key to update
while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27: # ESC
        cv2.destroyAllWindows()
        break
#    elif k == ord('s'):
#        cv2.imwrite('/Users/chris/foo.png', gray_img)
#        cv2.destroyAllWindows()
#        break
    elif k == ord('w'): # w=forwards
        #image = image[::-1]
        new_location_x +=direction_vector[0]
        new_location_y +=direction_vector[1]
        resized_img, image_found,new_heading_ind,direction_vector,image_title=find_next_set_images(new_location_x,new_location_y,new_heading_ind,file_database_sorted,piclist)
        if image_found==0:
            print "No image"
            new_location_x -=direction_vector[0]
            new_location_y -=direction_vector[1]
        else:
            cv2.putText(resized_img, image_title, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0);
            cv2.imshow(window_name, resized_img)
    elif k == ord('s'): # s= backwards
        #image = image[::-1]
        new_location_x -=direction_vector[0]
        new_location_y -=direction_vector[1]
        resized_img, image_found,new_heading_ind,direction_vector,image_title=find_next_set_images(new_location_x,new_location_y,new_heading_ind,file_database_sorted,piclist)
        if image_found==0:
            print "No image"
            new_location_x +=direction_vector[0]
            new_location_y +=direction_vector[1]
        else:
            cv2.putText(resized_img, image_title, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0);
            cv2.imshow(window_name, resized_img)
    elif k == ord(','): # ,<= left
        #image = image[::-1]
        #new_location_x -=1
        new_heading_ind -=1
        resized_img, image_found,new_heading_ind,direction_vector,image_title=find_next_set_images(new_location_x,new_location_y,new_heading_ind,file_database_sorted,piclist)
        if image_found==0:
            print "No image"
            #new_location_x +=1
        else:
            cv2.putText(resized_img, image_title, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0);
            cv2.imshow(window_name, resized_img)
    elif k == ord('.'): # .>= right
        #image = image[::-1]
        #new_location_x -=1
        new_heading_ind +=1
        resized_img, image_found,new_heading_ind,direction_vector,image_title=find_next_set_images(new_location_x,new_location_y,new_heading_ind,file_database_sorted,piclist)
        if image_found==0:
            print "No image"
            #new_location_x +=1
        else:
            cv2.putText(resized_img, image_title, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0);
            cv2.imshow(window_name, resized_img)
            