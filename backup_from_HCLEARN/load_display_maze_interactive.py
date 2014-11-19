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

# Path to images FORWARD SLASHES
pathname = 'D:/robotology/streetview_icub/division_street_randomised_dir_pitch' # 
# Choose start location (division street)
location_x=0
location_y=0
direction='E'


#pathname = 'D:/robotology/hclearn/division_street_1'
## Choose start location (division street)
#location_x=5
#location_y=15
#direction='E'

heading_index='NESW' #N=0, E=1, S=2, W=3
window_name='Streetview'
maze_map='Maze_Map'
place_cell_map='Place_cell_Map'

##### Find matching 3 files to display
def find_next_set_images(location_x,location_y,heading,file_database_sorted,picture_name_list):
    
    image_found=0
    
    heading,direction_vector=phase_wrap_heading(heading)
    # Convert heading
    phase_wrap=np.array([3, 0, 1, 2, 3, 0],dtype='u1')
    heading_array=np.array([phase_wrap[heading], phase_wrap[heading+1],phase_wrap[heading+2]])
    
    # Find mtching images.. if they exist
    matched_image_index=find_quad_image_block(file_database_sorted,location_x,location_y)
    # find x values
    if matched_image_index==-1:
        print "Not enough images at this x location!!"
        return (0,0,heading,direction_vector,0,0)
    # Check values found!!!!!
    if matched_image_index==-2:
        print "Not enough images at this y location!!" 
        return (0,0,heading,direction_vector,0,0)
        
    images_2_display=matched_image_index['file_id'][heading_array]
    
    combined_img = np.concatenate(img_file[images_2_display] , axis=1) #file_database_north_sortx[0,:]
    resized_img = cv2.resize(combined_img, (image_display_width, image_display_height)) 
    image_found=1
    picture_name=picture_name_list[images_2_display[1]]
    
    ######## Check for alternative image options -> Can we go forwards / backwards / left / right
    available_direction_vector=np.zeros([4],dtype='i1')+1
    #  1. Forwards
    matched_image_index_test=find_quad_image_block(file_database_sorted,location_x+direction_vector[0],location_y+direction_vector[1])
    if matched_image_index_test==-1 or matched_image_index_test==-2:
        available_direction_vector[0]=0 
    #  2. Backwards
    matched_image_index_test=find_quad_image_block(file_database_sorted,location_x-direction_vector[0],location_y-direction_vector[1])
    if matched_image_index_test==-1 or matched_image_index_test==-2:
        available_direction_vector[1]=0     
    #  3. Left
    _, direction_vector_test=phase_wrap_heading(heading-1)
    matched_image_index_test=find_quad_image_block(file_database_sorted,location_x+direction_vector_test[0],location_y+direction_vector_test[1])
    if matched_image_index_test==-1 or matched_image_index_test==-2:
        available_direction_vector[2]=0 
    #  4. Right
    _, direction_vector_test=phase_wrap_heading(heading+1)
    matched_image_index_test=find_quad_image_block(file_database_sorted,location_x+direction_vector_test[0],location_y+direction_vector_test[1])
    if matched_image_index_test==-1 or matched_image_index_test==-2:
        available_direction_vector[3]=0         
        
    return (resized_img,image_found,heading,direction_vector,picture_name,available_direction_vector)

def find_quad_image_block(file_database_sorted,location_x,location_y):
    # find x values
    matched_x_loc=np.extract(file_database_sorted['x_loc']==location_x,file_database_sorted)
    # Check values found!!!!!
    if matched_x_loc.size<4:
#        print "NOWT in Y"
        return(np.zeros([1],dtype='i1')-1)
    # find y values
    matched_image_index=np.extract(matched_x_loc['y_loc']==location_y,matched_x_loc)
    # Check values found!!!!!
    if matched_image_index.size<4:
#       print "NOWT in Y"
        return(np.zeros([1],dtype='i1')-2)  
    if matched_image_index.size>4:
        print 'WARNING TOO MANY IMAGES AT LOCATION x:' + str(location_x) + ' y:' + str(location_y)
    return(matched_image_index)
    
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


### Display images
def display_image(image_in, text_in, available_directions_index, heading_ind):
    # File title - remove in final!!!
    cv2.putText(image_in, text_in, (screen_width/2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0);
    # Add arrows to represent avaiable directions
    #colour_vector=(0,255,0)
    # 1. Forward
    if available_directions_index[0]==1: # red
        cv2.fillConvexPoly(image_in,arrow_up_pts,(0,255,0)) 
#    else: #green
#        colour_vector=(0,255,0)
    # 2. Backward
    if available_directions_index[1]==1: # red
        cv2.fillConvexPoly(image_in,arrow_down_pts,(0,255,0))  
#        colour_vector=(0,0,255)
#    else: #green
#        colour_vector=(0,255,0)
    # 3. Left
    if available_directions_index[2]==1: # red
        cv2.fillConvexPoly(image_in,arrow_left_pts,(255,0,0))
#        colour_vector=(0,0,255)
#    else: #green
#        colour_vector=(0,255,0)
    # 4. Right
    if available_directions_index[3]==1: # red
        cv2.fillConvexPoly(image_in,arrow_right_pts,(255,0,0))
    #        colour_vector=(0,0,255)
    #    else: #green
    #        colour_vector=(0,255,0)     
    ### Direction label    
    textsize=cv2.getTextSize(heading_index[heading_ind],cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
#    cv2.putText(image_in, heading_index[heading_ind], (x_arrow_base_location-(textsize[0][1]/2),\
#        image_display_height-arrow_point_size-int(textsize[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2);
    cv2.fillConvexPoly(image_in,arrow_dir_pts,(0,0,255))
    
    cv2.putText(image_in, heading_index[heading_ind], (0+(textsize[0][1]/2),\
        30+int(textsize[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2);        
    cv2.imshow(window_name, image_in)
    
def make_grid_index(x=8,y=8, pixel_width=200):
    "Draw an x(i) by y(j) chessboard using PIL."
    #import Image, ImageDraw
    #from itertools import cycle
    # Increase by one to include 0 effect    
    x+=1
    y+=1
    def sq_start(i,n):
        "Return the x/y start coord of the square at column/row i."
        return i * pixel_width / n
    
    def square(i, j):
        "Return the square corners, suitable for use in PIL drawings"
        return sq_start(i,x), sq_start(j,y), sq_start(i+1,x), sq_start(j+1,y)
    
    #image = Image.new("L", (pixel_width, pixel_width)  
    
    squares_out=np.empty([x,y,4],dtype='i2')
    ##draw_square = ImageDraw.Draw(image).rectangle
    for ix in range(0,x):    
        for iy in range(0,y):        
            squares_out[ix,iy,:]=square(ix, iy)    
    return squares_out
    
def plot_exisiting_locations_on_grid(map_data,squares_grid,useable_grid_locations):    
    # Plot white boxes onto grid where locations exist
    # Work out middle location!
    min_x=useable_grid_locations[0].min()
    min_y=useable_grid_locations[1].min() 
    for current_loc in range(0,useable_grid_locations[0].size): 
        sq=squares_grid[useable_grid_locations[0][current_loc]-min_x,useable_grid_locations[1][current_loc]-min_y,:]
        cv2.rectangle(map_data,tuple(sq[0:2]),tuple(sq[2:4]),(255,255,255),-1)
    return map_data
    
    
def plot_current_position_on_map(map_template,useable_grid_locations,current_x,current_y):    
    # Plot red box where vehicle is....    
    min_x=useable_grid_locations[0].min()
    min_y=useable_grid_locations[1].min()     
    sq=squares_grid[current_x-min_x,current_y-min_y,:]
    map_image_display=np.copy(map_template); # FORCE COPY SO IT DOESNT KEEP OLD MOVES!!!!!
    cv2.rectangle(map_image_display,tuple(sq[0:2]),tuple(sq[2:4]),(0,0,255),-1)
    
    
    map_image_display=flip_rotate_color_image(map_image_display,heading_index.find(direction))
    
    #map_image_display=np.copy(np.rot90(np.flipud(map_image_display),heading_index.find(direction)))
    # Show direction
    textsize=cv2.getTextSize(direction,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
    cv2.fillConvexPoly(map_image_display,arrow_dir_pts,(0,0,255))
    cv2.putText(map_image_display,direction,(int((textsize[0][1]/2)-2),int(30+(textsize[1]/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2); 
#             
    cv2.imshow(maze_map,map_image_display)
    return map_image_display

def plot_old_position_on_map(map_template,useable_grid_locations,current_x,current_y):    
    # Plot red box where vehicle is....    
    min_x=useable_grid_locations[0].min()
    min_y=useable_grid_locations[1].min()     
    sq=squares_grid[current_x-min_x,current_y-min_y,:]
    cv2.rectangle(map_template,tuple(sq[0:2]),tuple(sq[2:4]),(0,255,0),-1)                  
    return map_template

# Plot map with place cell id's
def plot_place_cell_id_on_map(map_data,place_cell_id):    
    # Plot red box where vehicle is....    
    min_x=place_cell_id[1].min()
    min_y=place_cell_id[2].min()
    map_out=np.copy(map_data); # FORCE COPY SO IT DOESNT KEEP OLD MOVES!!!!!
    map_out=flip_rotate_color_image(map_out,0)
    # Loop through each place id
    for  current_place in range(0,place_cell_id[0].size): 
        sq=squares_grid[place_cell_id[1][current_place]-min_x,place_cell_id[2][current_place]-min_y,:]        
        # Place number at bottom of square in middle.... 
        x_pos=sq[0]#+np.round(np.diff([sq[2],sq[0]])/2)
        y_pos=pixel_width-sq[1]+np.round(np.diff([sq[3],sq[1]])/2)
        cv2.putText(map_out, str(int(place_cell_id[0][current_place])), (int(x_pos),int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1);
  
    
   
    textsize=cv2.getTextSize('N',cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
    cv2.fillConvexPoly(map_out,np.abs(np.array([[pixel_width,0],[pixel_width,0],[pixel_width,0]])-arrow_dir_pts),(0,0,255))
    cv2.putText(map_out, 'N', (pixel_width-int((textsize[0][1]/2)+10),int(30+(textsize[1]/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2); 
             
    cv2.imshow(place_cell_map,map_out)
    return map_out
  
  # Flip image (mirror) then rotate anti clockwise by @ 90 degrees
def flip_rotate_color_image(image,angles_90):
    for current_color in range(0,image[0,0,:].size):
        image[:,:,current_color]=np.rot90(np.flipud(image[:,:,current_color]),angles_90)
    return image


###### START OF MAIN
    
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

############################
######### Make arrow points
x_arrow_base_location=int(image_display_width/2)
y_arrow_base_location=int(image_display_height*0.90)
# shorter size
arrow_point_size=int(image_display_height*0.05)
arrow_half_width=int((image_display_height*0.10)/2)
# 1. ARROW UP!!!   
arrow_up_pts = np.array([[x_arrow_base_location,y_arrow_base_location-arrow_point_size],\
    [x_arrow_base_location-arrow_half_width,y_arrow_base_location],[x_arrow_base_location+arrow_half_width,y_arrow_base_location]], np.int32)
arrow_up_pts = arrow_up_pts.reshape((-1,1,2))
# 2. ARROW DOWN!!!
arrow_down_pts = np.array([[x_arrow_base_location,image_display_height-1],\
    [x_arrow_base_location-arrow_half_width,image_display_height-arrow_point_size],[x_arrow_base_location+arrow_half_width,image_display_height-arrow_point_size]], np.int32)
arrow_down_pts = arrow_down_pts.reshape((-1,1,2))

# 3. ARROW Left!!!
arrow_left_pts = np.array([[x_arrow_base_location-arrow_half_width-arrow_half_width,y_arrow_base_location+int(arrow_point_size/2)],\
    [x_arrow_base_location-arrow_half_width,image_display_height-arrow_point_size], [x_arrow_base_location-arrow_half_width,y_arrow_base_location]], np.int32)
arrow_left_pts = arrow_left_pts.reshape((-1,1,2))

# 4. ARROW Right!!!
arrow_right_pts = np.array([[x_arrow_base_location+arrow_half_width+arrow_half_width,y_arrow_base_location+int(arrow_point_size/2)],\
    [x_arrow_base_location+arrow_half_width,y_arrow_base_location], [x_arrow_base_location+arrow_half_width,image_display_height-arrow_point_size]], np.int32)
arrow_right_pts = arrow_right_pts.reshape((-1,1,2))

# 5. ARROW Direction!!!   
arrow_dir_pts = np.array([[15,2],\
    [10,12],[20,12]], np.int32)
arrow_die_pts = arrow_dir_pts.reshape((-1,1,2))
#################################


no_files=len(glob.glob(os.path.join(pathname, '*.jpg')))

#file_database=np.empty([5,no_files],dtype=int)
file_database=np.empty(no_files,\
    dtype=[('file_id','i2'),('x_loc','i2'),('y_loc','i2'),('heading','i2'),('img_id','i2')])
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


### Build map from google data
## Making a Map build grip index 
pixel_width=600#200

## Make grid index x,y, [coords]
squares_grid=make_grid_index(file_database_sorted['x_loc'].ptp(),file_database_sorted['y_loc'].ptp(), pixel_width)
## Make squares with map data white
file_database_north=file_database[np.where(file_database['heading']==0)]
# Only get one heading from each location .... ie NORTH only
useable_grid_locations=np.array([file_database_north['x_loc'],file_database_north['y_loc']])
## Set up image arrays for maps.....
map_template=np.zeros((pixel_width,pixel_width,3),dtype='u1') # default black
map_image_display=np.zeros((pixel_width,pixel_width,3),dtype='u1') # default black

## Windows to display graphics
# Updated map of maze and current location
cv2.namedWindow(maze_map)
# Main image display
cv2.namedWindow(window_name)
# Layout of place cells
cv2.namedWindow(place_cell_map)

map_template=plot_exisiting_locations_on_grid(map_template,squares_grid,useable_grid_locations)

### Initialise main image windows
heading_ind=heading_index.find(direction)
available_directions_index=0
new_location_x=location_x
new_location_y=location_y
resized_img,image_found,new_heading_ind,direction_vector,image_title,available_directions_index=find_next_set_images(location_x,location_y,heading_ind,file_database_sorted,piclist)
if image_found==0:
    print "No base location image... exiting"
    sys.exit()
display_image(resized_img, image_title, available_directions_index, new_heading_ind)
## ALTERNATIVE:: get NESW for location

## Add in place locations (just use image list).
## Build empty array with x and y values...
place_cell_id=np.array([range(0,useable_grid_locations[0].size),useable_grid_locations[0],useable_grid_locations[1]])

# plot these on the map
map_place_out=plot_place_cell_id_on_map(map_template,place_cell_id)

### Put current location on map
map_image_display=plot_current_position_on_map(map_template,useable_grid_locations,location_x,location_y)

try:
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
            old_location_x=new_location_x
            old_location_y=new_location_y
            new_location_x +=direction_vector[0]
            new_location_y +=direction_vector[1]
            resized_img, image_found,new_heading_ind,direction_vector,image_title,available_directions_index=find_next_set_images(new_location_x,new_location_y,new_heading_ind,file_database_sorted,piclist)
            if image_found==0:
                print "No image"
                new_location_x -=direction_vector[0]
                new_location_y -=direction_vector[1]
            else:
                display_image(resized_img, image_title, available_directions_index, new_heading_ind)
                map_template=plot_old_position_on_map(map_template,useable_grid_locations,old_location_x,old_location_y)
                map_image_display=plot_current_position_on_map(map_template,useable_grid_locations,new_location_x,new_location_y)
        elif k == ord('s'): # s= backwards
            #image = image[::-1]
            old_location_x=new_location_x
            old_location_y=new_location_y
            new_location_x -=direction_vector[0]
            new_location_y -=direction_vector[1]
            resized_img, image_found,new_heading_ind,direction_vector,image_title,available_directions_index=find_next_set_images(new_location_x,new_location_y,new_heading_ind,file_database_sorted,piclist)
            if image_found==0:
                print "No image"
                new_location_x +=direction_vector[0]
                new_location_y +=direction_vector[1]
            else:
                display_image(resized_img, image_title, available_directions_index, new_heading_ind)
                map_template=plot_old_position_on_map(map_template,useable_grid_locations,old_location_x,old_location_y)
                map_image_display=plot_current_position_on_map(map_template,useable_grid_locations,new_location_x,new_location_y)
        elif k == ord('a'): # ,<= left
            #image = image[::-1]
            #new_location_x -=1
            new_heading_ind -=1
            resized_img, image_found,new_heading_ind,direction_vector,image_title,available_directions_index=find_next_set_images(new_location_x,new_location_y,new_heading_ind,file_database_sorted,piclist)
            if image_found==0:
                print "No image"
                #new_location_x +=1
            else:
                display_image(resized_img, image_title, available_directions_index, new_heading_ind)
                #map_image_display=plot_current_position_on_map(map_template,useable_grid_locations,new_location_x,new_location_y)
        elif k == ord('d'): # .>= right
            #image = image[::-1]
            #new_location_x -=1
            new_heading_ind +=1
            resized_img, image_found,new_heading_ind,direction_vector,image_title,available_directions_index=find_next_set_images(new_location_x,new_location_y,new_heading_ind,file_database_sorted,piclist)
            if image_found==0:
                print "No image"
                #new_location_x +=1
            else:
                display_image(resized_img, image_title, available_directions_index, new_heading_ind)
                #map_image_display=plot_current_position_on_map(map_template,useable_grid_locations,new_location_x,new_location_y)
except KeyboardInterrupt:
    pass
            