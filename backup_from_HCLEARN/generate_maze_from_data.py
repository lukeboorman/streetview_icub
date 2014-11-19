# -*- coding: utf-8 -*-
"""
Created on Fri Oct 03 12:12:43 2014
 Load all the images and build overall image....
@author: luke
"""
### Build map from google data

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
from collections import Counter

class maze_from_data:
    #Requires the folder name
    def __init__(self, folderName = 'D:/robotology/hclearn/division_street_1' ):
        self.folder = folderName
        # Path to images FORWARD SLASHES 
        self.heading_index='NESW' #N=0, E=1, S=2, W=3
        # Names of image windows
        self.window_name='Streetview'
        self.maze_map='Maze_Map'
        self.place_cell_map='Place_cell_Map'
        # Working out number of pixels to display maze etc
        self.image_display_width=0;
        self.image_display_height=0;
        self.img_file=np.empty([],dtype='u1') #unsigned 8 bit (1 byte)
        # Init first direction
        self.direction='E'
        ## Making a Map build grip index 
        self.pixel_width=600#200
        self.locations_unique=dict()
        self.step_time_delay=100
        
        ### Check heading is range.....
    def phase_wrap_heading(self,heading):
        while True:
            if heading>3:
                heading=heading-4
            elif heading<0:
                heading=heading+4
            if heading<=3 and heading>=0:
                break
#        # Work out direction vectors....
         # Luke Original vectors       
#        if heading==0: # North = x=0, y=1
#            direction_vector=[0,1]
#        elif heading==1: # East = x=1, y=0
#            direction_vector=[1,0]
#        elif heading==2: # south = x=0, y=-1
#            direction_vector=[0,-1]
#        else: # west = x=-1, y=0
#            direction_vector=[-1,0]
            
        # Work out direction vectors....
        #print 'USING CHARLES FOX DIRECTION VECTORS!'
        if heading==0: # North = x=0, y=1
            direction_vector=[0,-1]
        elif heading==1: # East = x=1, y=0
            direction_vector=[1,0]
        elif heading==2: # south = x=0, y=-1
            direction_vector=[0,1]
        else: # west = x=-1, y=0
            direction_vector=[-1,0]                
        return (heading, direction_vector)
        
    ##### Find matching 3 files to display
    def find_next_set_images(self,location_x,location_y,heading):
        
        image_found=0
        
        heading,direction_vector=self.phase_wrap_heading(heading)
        # Convert heading
        phase_wrap=np.array([3, 0, 1, 2, 3, 0],dtype='u1')
        heading_array=np.array([phase_wrap[heading], phase_wrap[heading+1],phase_wrap[heading+2]])
        
        # Find mtching images.. if they exist
        matched_image_index=self.find_quad_image_block(location_x,location_y)
        # find x values
        if matched_image_index==-1:
            print "Not enough images at this x location!!"
            return (0,0,heading,direction_vector,0,0)
        # Check values found!!!!!
        if matched_image_index==-2:
            print "Not enough images at this y location!!" 
            return (0,0,heading,direction_vector,0,0)
        
        ###### New code here to deal with only partial image blocks (Not all present!)!!!!!
        images_to_combine=np.zeros(3,dtype='i2')-1 # -1 = no image
        for current_dir in range(0,3):
            if heading_array[current_dir] in matched_image_index['heading']:
                images_to_combine[current_dir]=matched_image_index['file_id'][np.where(matched_image_index['heading']==heading_array[current_dir])]

#        print('Images to display:/n')
#        print images_to_combine
#        print matched_image_index
        image_found=1
        
        if images_to_combine[1]==-1:
            picture_name='No image here'
        else:
            picture_name=self.file_database_sorted['img_fname'][np.where(self.file_database_sorted['file_id']==images_to_combine[1])][0]  #self.file_database_sorted['img_fname'][np.where(self.file_database_sorted['file_id']==images_to_combine[1])]##picture_name_list[images_to_combine[1]]
        
        ######## Check for alternative image options -> Can we go forwards / backwards / left / right
        available_direction_vector=np.zeros([4],dtype='i1')+1
        #  1. Forwards
        matched_image_index_test=self.find_quad_image_block(location_x+direction_vector[0],location_y+direction_vector[1])
        if matched_image_index_test==-1 or matched_image_index_test==-2:
            available_direction_vector[0]=0 
        #  2. Backwards
        matched_image_index_test=self.find_quad_image_block(location_x-direction_vector[0],location_y-direction_vector[1])
        if matched_image_index_test==-1 or matched_image_index_test==-2:
            available_direction_vector[1]=0     
        #  3. Left
        _, direction_vector_test=self.phase_wrap_heading(heading-1)
        matched_image_index_test=self.find_quad_image_block(location_x+direction_vector_test[0],location_y+direction_vector_test[1])
        if matched_image_index_test==-1 or matched_image_index_test==-2:
            available_direction_vector[2]=0 
        #  4. Right
        _, direction_vector_test=self.phase_wrap_heading(heading+1)
        matched_image_index_test=self.find_quad_image_block(location_x+direction_vector_test[0],location_y+direction_vector_test[1])
        if matched_image_index_test==-1 or matched_image_index_test==-2:
            available_direction_vector[3]=0         
            
        return (images_to_combine,image_found,heading,direction_vector,picture_name,available_direction_vector)
        
    def concatenate_resize_images(self,images_to_combine):
        _,height,width,depth= self.img_file.shape
        combined_img=np.zeros([3,int(height),int(width),int(depth)],dtype='u1')
        for current_image in range(0,3):        
            if images_to_combine[current_image] !=-1:
                combined_img[current_image]=self.img_file[images_to_combine[current_image]]
            else:
                print('Missing image file replaced with zeros....')
        resized_img =cv2.resize(np.concatenate(combined_img , axis=1), (self.image_display_width, self.image_display_height))
        return (resized_img)        
    
    def find_quad_image_block(self,location_x,location_y):
        # find x values
        matched_x_loc=np.extract(self.file_database_sorted['x_loc']==location_x,self.file_database_sorted)
        # Check values found!!!!!
        if matched_x_loc.size<1:
    #        print "NOWT in Y"
            return(np.zeros([1],dtype='i1')-1)
        # find y values
        matched_image_index=np.extract(matched_x_loc['y_loc']==location_y,matched_x_loc)
        # Check values found!!!!!
        if matched_image_index.size<1:
    #       print "NOWT in Y"
            return(np.zeros([1],dtype='i1')-2)  
        if matched_image_index.size>4:
            print 'WARNING TOO MANY IMAGES AT LOCATION x:' + str(location_x) + ' y:' + str(location_y)
        return(matched_image_index)
            
    ### Display images
    def display_image(self,image_in, text_in, available_directions_index, heading_ind):
        # File title - remove in final!!!
        cv2.putText(image_in, text_in, (self.screen_width/2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0);
        # Add arrows to represent avaiable directions
        #colour_vector=(0,255,0)
        # 1. Forward
        if available_directions_index[0]==1: # red
            cv2.fillConvexPoly(image_in,self.arrow[('up')],(0,255,0)) 
    #    else: #green
    #        colour_vector=(0,255,0)
        # 2. Backward
        if available_directions_index[1]==1: # red
            cv2.fillConvexPoly(image_in,self.arrow[('down')],(0,255,0))  
    #        colour_vector=(0,0,255)
    #    else: #green
    #        colour_vector=(0,255,0)
        # 3. Left
        if available_directions_index[2]==1: # red
            cv2.fillConvexPoly(image_in,self.arrow[('left')],(255,0,0))
    #        colour_vector=(0,0,255)
    #    else: #green
    #        colour_vector=(0,255,0)
        # 4. Right
        if available_directions_index[3]==1: # red
            cv2.fillConvexPoly(image_in,self.arrow[('right')],(255,0,0))
        #        colour_vector=(0,0,255)
        #    else: #green
        #        colour_vector=(0,255,0)     
        ### Direction label    
        textsize=cv2.getTextSize(self.heading_index[heading_ind],cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
    #    cv2.putText(image_in, self.heading_index[heading_ind], (x_arrow_base_location-(textsize[0][1]/2),\
    #        self.image_display_height-arrow_point_size-int(textsize[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2);
        cv2.fillConvexPoly(image_in,self.arrow[('heading')],(0,0,255))
        
        cv2.putText(image_in, self.heading_index[heading_ind], (0+(textsize[0][1]/2),\
            30+int(textsize[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2);        
        cv2.imshow(self.window_name, image_in)
        
    def make_grid_index(self,x=8,y=8, pixel_width=200):
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
        
    def plot_exisiting_locations_on_grid(self,map_data):    
        # Plot white boxes onto grid where locations exist
        # Work out middle location!
        min_x=self.place_cell_id[1].min()
        min_y=self.place_cell_id[2].min() 
        for current_loc in range(0,self.place_cell_id[1].size): 
            sq=self.squares_grid[self.place_cell_id[1][current_loc]-min_x,self.place_cell_id[2][current_loc]-min_y,:]
            cv2.rectangle(map_data,tuple(sq[0:2]),tuple(sq[2:4]),(255,255,255),-1)
        return map_data
        
        
    def plot_current_position_on_map(self,current_x,current_y):    
        # Plot red box where vehicle is....    
        min_x=self.place_cell_id[1].min()
        min_y=self.place_cell_id[2].min()     
        sq=self.squares_grid[current_x-min_x,current_y-min_y,:]
        map_image_display=np.copy(self.map_template); # FORCE COPY SO IT DOESNT KEEP OLD MOVES!!!!!
        cv2.rectangle(map_image_display,tuple(sq[0:2]),tuple(sq[2:4]),(0,0,255),-1)
        
        
        map_image_display=self.flip_rotate_color_image(map_image_display,self.heading_index.find(self.direction),False)
        
        #map_image_display=np.copy(np.rot90(np.flipud(map_image_display),self.heading_index.find(self.direction)))
        # Show direction
        textsize=cv2.getTextSize(self.direction,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        cv2.fillConvexPoly(map_image_display,self.arrow[('heading')],(0,0,255))
        cv2.putText(map_image_display,self.direction,(int((textsize[0][1]/2)-2),int(30+(textsize[1]/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2); 
    #             
        cv2.imshow(self.maze_map,map_image_display)
        return map_image_display
    
    def plot_old_position_on_map(self,current_x,current_y):    
        # Plot red box where vehicle is....    
        min_x=self.place_cell_id[1].min()
        min_y=self.place_cell_id[2].min()     
        sq=self.squares_grid[current_x-min_x,current_y-min_y,:]
        cv2.rectangle(self.map_template,tuple(sq[0:2]),tuple(sq[2:4]),(0,255,0),-1)                  
        return self.map_template
    
    # Plot map with place cell id's
    def plot_place_cell_id_on_map(self,map_data,place_cell_id):    
        # Plot red box where vehicle is....    
        min_x=place_cell_id[1].min()
        min_y=place_cell_id[2].min()
        ptp_y=place_cell_id[2].ptp()
        
        map_out=np.copy(map_data); # FORCE COPY SO IT DOESNT KEEP OLD MOVES!!!!!
        map_out=self.flip_rotate_color_image(map_out,0,False)
        # Loop through each place id
        for  current_place in range(0,place_cell_id[0].size): 
#            sq=self.squares_grid[place_cell_id[1][current_place]-min_x,place_cell_id[2][current_place]-min_y,:]        
            # Flipping this in y-plane            
            sq=self.squares_grid[place_cell_id[1][current_place]-min_x,np.absolute(place_cell_id[2][current_place]-min_y-ptp_y),:]        
            # Place number at bottom of square in middle.... 
            x_pos=sq[0]#+np.round(np.diff([sq[2],sq[0]])/2)
            y_pos=self.pixel_width-sq[1]+np.round(np.diff([sq[3],sq[1]])/2)
            cv2.putText(map_out, str(int(place_cell_id[0][current_place])), (int(x_pos),int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1);
      
        
       
        textsize=cv2.getTextSize('N',cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        #cv2.fillConvexPoly(map_out,np.abs(np.array([[self.pixel_width,0],[self.pixel_width,0],[self.pixel_width,0]])-self.arrow[('heading')]),(0,0,255))
        cv2.putText(map_out, 'N', (self.pixel_width-int((textsize[0][1]/2)+10),int(30+(textsize[1]/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2); 
                 
        cv2.imshow(self.place_cell_map,map_out)
        return map_out
      
      # Flip image (mirror) then rotate anti clockwise by @ 90 degrees
    def flip_rotate_color_image(self,image,angles_90, flip_on):
        for current_color in range(0,image[0,0,:].size):
            if flip_on:
                image[:,:,current_color]=np.rot90(np.flipud(image[:,:,current_color]),angles_90)            
            else:
                image[:,:,current_color]=np.rot90(image[:,:,current_color],angles_90)
        return image
    
    
    ###### START OF MAIN
        
    ########################################
    # Make database of image files
    #####################################
    
    def index_image_files(self):
        # File list
        #piclist = []
        #### Load all files from given folder and sort by x then y then direction....

        no_files=len(glob.glob(os.path.join(self.folder, '*.jpg')))
        
        #file_database=np.empty([5,no_files],dtype=int)
        file_database=np.empty(no_files,\
            dtype=[('orig_file_id','i2'),('file_id','i2'),('x_loc','i2'),('y_loc','i2'),('heading','i2'),('img_id','i2'),('img_text','a50'),('img_fname','a100')])

        #,no_files],dtype=int)
        
        file_database['orig_file_id'][:]=range(0,no_files)
        
        self.locations_unique=dict()        
        
        image_count=0
        
#        if fixed_extract==True: # Lukes original mode of cutting by fixed locations in string.....
#            for infile in glob.glob(os.path.join(self.folder, '*.jpg')):
#                # ADD filename to list    
#                #piclist.append(infile)
#                # Extract relevant file information.....
#                # find start of filename section
#                file_info=infile[infile.rfind("\\")+1:infile.rfind("\\")+14]
#                # img count , x, y, heading, img_num
#                file_database['img_fname'][image_count]=infile
#                # x grid
#                file_database['x_loc'][image_count]=int(file_info[0:3])
#                # y grid    
#                file_database['y_loc'][image_count]=int(file_info[4:7])
#                # Convert letter heading to index 1= N, 2=E, 3=S, 4=W
#                file_database['heading'][image_count]=self.heading_index.find(file_info[8:9])
#                # File identifier (optional extra e.g. two files at same location x,y and direction)
#                file_database['img_id'][image_count]=int(file_info[10:13])
#                # Massive data image block!!!
#                
#        else: # Use original mode from HCLEARN - Charles FOX
        import re
        if os.path.exists(self.folder):
            for file in os.listdir(self.folder):
                #print file
                parts = re.split("[-,\.]", file)
                #Test that it is (NUM-NUM-DIRECTION-whatever)
                # print str(parts)
                if len(parts)>=2 and parts[0].isdigit() and parts[1].isdigit() and parts[2][0].isalpha: # and len(parts[2]) == 1):
                    if parts[2][0] in self.heading_index:
#                            key = ((int(parts[0]), int(parts[1])),parts[2])
                        #If it doesnt already exist, make this key
#                            if key not in self.files.keys():
#                                self.files[key] = []
                        #fullFilePath = os.path.join(self.folder,file)
                        #Add the new file onto the end of the keys list (since there can be multiple images for one direction)
                        file_database['img_fname'][image_count]=file
                        # x grid
                        file_database['x_loc'][image_count]=int(parts[0])
                        # y grid    
                        file_database['y_loc'][image_count]=int(parts[1])
                        # Convert letter heading to index 1= N, 2=E, 3=S, 4=W
                        file_database['heading'][image_count]=self.heading_index.find(parts[2])
                        # File identifier (optional extra e.g. two files at same location x,y and direction)
                        
                        if parts[3].isdigit():
                            file_database['img_id'][image_count]=int(parts[3])
                            file_database['img_text'][image_count]='use_ID'
                            if image_count==0:
                                use_file_id=1 # uses the numbering of file instead of text!
                        elif parts[3].isalpha():
                            file_database['img_id'][image_count]=-1
                            file_database['img_text'][image_count]=parts[3]
                            if image_count==0:
                                use_file_id=0 # uses the string text of file instead of text!
                        else:
                            file_database['img_id'][image_count]=1
                            file_database['img_text'][image_count]=''
                            if image_count==0:
                                use_file_id=-1 # uses none!
                                                
                        #TODO: Add in Rain / Midday or image index sorting here
                        #### Build complete locations dictionary.......                        
                        current_location_key=(file_database['x_loc'][image_count],file_database['y_loc'][image_count])
                        # Setup new location entry if missing                        
                        if current_location_key not in self.locations_unique.keys():
                            self.locations_unique[current_location_key]={('Image_count'): np.zeros(4,dtype='i2')}
                        # Add Image count to location
                        self.locations_unique[current_location_key][('Image_count')][file_database['heading'][image_count]]+=1
                        ### Fill in location info
                        img_count=self.locations_unique[current_location_key][('Image_count')][file_database['heading'][image_count]]-1                        
                        # Add heading marker
                        self.locations_unique[current_location_key][(img_count,parts[2])]=file_database['img_fname'][image_count]                                       
                        
                        image_count += 1
                        #self.files[key].append(fullFilePath)
                    else:
                        raise NameError("Heading is: %s\nit should be N E S or W" % parts[2])
                else:
                    print self.folder
                    print file
                    #raise NameError("File: %s\ndoes not fit naming convention INT-INT-HEADING" % file)
        else:
            raise NameError("Folder does not exists")    
        #==============================================================================
        # ### Mini task... get all northern images, in order of x location
        # # Northern data,
        # file_database_north=np.squeeze(file_database[:,np.transpose(np.where(file_database[3,:]==0)[0])])
        # #Sub  sorted by x location.....
        # file_database_north_sortx=file_database_north[:,np.argsort(file_database_north[1,:])]
        # #### Combine images into panorama
        # # First Image
        # #cv2.imshow('FRED', self.img_file[1])
        # combined_img = np.concatenate((self.img_file[file_database_north_sortx[0,0:5]]) , axis=1) #file_database_north_sortx[0,:]
        # resized_img = cv2.resize(combined_img, (self.screen_width, self.screen_height)) 
        # cv2.imshow('FRED', resized_img)
        # ## ALTERNATIVE:: get NESW for location
        #==============================================================================

        ### Mini task... get data for each location NSEW
        ## First sort by x location!!
        #file_database_by_loc=np.squeeze(file_database[:,np.transpose(np.where(file_database[1,:]==0)[0])])
        #Sub  sorted by y location.....
        #file_database_by_loc_sorty=file_database_by_loc[:,np.argsort(file_database_by_loc[3,:])]

        #### just get images that belong to each image ID.....
        if use_file_id==1: # Use the fourth value (file id)
            print ('Just using file IDs: ',str(file_database['img_id'][0]) )
            file_database_primary=file_database[np.where(file_database['img_id']==file_database['img_id'][0])]
        elif use_file_id==0: # Use first string value
            print ('Just using file with id text: ',file_database['img_text'][0] )            
            file_database_primary=file_database[np.where(file_database['img_text']==file_database['img_text'][0])]
        else: # do nothing
            print ('Using all files') 
            file_database_primary=file_database
        #### Combine images into panorama
        self.file_database_sorted=np.sort(file_database_primary,order=['x_loc','y_loc','heading'])
        self.file_database_sorted['file_id']=range(0,len(self.file_database_sorted))
        # Not all directions included..... therefore cannot use NORTH only!!!!!!
        #np.array(list(set(tuple(p) for p in points)))
        
        useable_grid_locations=np.empty(len(self.locations_unique.keys()),dtype=[('x_loc','i2'),('y_loc','i2')])
        useable_grid_locations['x_loc']=np.transpose(np.asarray(self.locations_unique.keys(),dtype='i3'))[0]
        useable_grid_locations['y_loc']=np.transpose(np.asarray(self.locations_unique.keys(),dtype='i3'))[1]

        
        useable_grid_locations=np.sort(useable_grid_locations,order=['x_loc','y_loc'])
        
        ## Add in place locations.
        ## Build empty array with x and y values...
        self.place_cell_id=np.array([np.zeros(useable_grid_locations['x_loc'].size,dtype='i2'),useable_grid_locations['x_loc'],useable_grid_locations['y_loc']])
        # 1. Order using longest x road (e.g. division street) => has most identical y values
        most_y=Counter(self.place_cell_id[2]).most_common()
        place_cell_id_x=np.zeros(useable_grid_locations['x_loc'].size,dtype='i2')
        # for each counter output.... run through 
        place_cell_counter=0
        for current_count_block in range(0,len(most_y)):
           line_locations_x=np.where(self.place_cell_id[2]==most_y[current_count_block][0])
           for current_map_tile in line_locations_x[0]:
               #print str(current_count_block), str(current_map_tile)
               place_cell_id_x[current_map_tile]=place_cell_counter
               place_cell_counter+=1
               
        #x_ok=np.where(np.diff(self.place_cell_id[1][self.place_cell_id_x])!=0)       
        self.place_cell_id[0]=place_cell_id_x
        # Sort by place cell ID!        
        self.place_cell_id=self.place_cell_id[:,self.place_cell_id[0,:].argsort()]
        #self.place_cell_id=np.array([range(0,useable_grid_locations[0].size),useable_grid_locations[0],useable_grid_locations[1]])        
        #print (str(self.place_cell_id))
                
        
    def load_image_files(self):
        
        num_images=len(self.file_database_sorted)

        if os.path.exists(self.folder):
            ### Load first image.... to get sizes
            dummy_img=cv2.imread(os.path.join(self.folder,self.file_database_sorted['img_fname'][0]))
            
            if dummy_img is False:
                print('Error nothing in image')
#            self.img_file=np.empty([num_images,640,640,3],dtype='u1') #unsigned 8 bit (1 byte)
            height, width, depth=dummy_img.shape
            ## Image for replacing when images missing!
#            self.zero_image=np.zeros([height, width, depth],dtype='u1')
            self.img_file=np.empty([num_images,height, width, depth],dtype='u1') #unsigned 8 bit (1 byte)
            # load all image files into array 
            for image_count in range(0,num_images):
                self.img_file[image_count,:,:,:]=cv2.imread(os.path.join(self.folder,self.file_database_sorted['img_fname'][image_count]))                
        else:
            print('CANNOT FIND IMAGES RETURNING!!')
            return(0)

        
    def display_maps_images(self):    
        
        ## Load image data from folder
        self.load_image_files()
        ## Set up image arrays for map plots.....
        self.map_template=np.zeros((self.pixel_width,self.pixel_width,3),dtype='u1') # default black
        #map_image_display=np.zeros((self.pixel_width,self.pixel_width,3),dtype='u1') # default black
        # Choose start location (take first place cell id)
        self.location_x=self.place_cell_id[1,0]
        self.location_y=self.place_cell_id[2,0]
        
        # Windows screen size
        self.screen_width = GetSystemMetrics (0)
        self.screen_height = GetSystemMetrics (1)
        
        # fit 3x images in window
        self.image_display_width=self.screen_width
        self.image_display_height=int(round(self.screen_width/3,0))
        
        ############################
        ######### Make arrow points (to show where to go....)
        x_arrow_base_location=int(self.image_display_width/2)
        y_arrow_base_location=int(self.image_display_height*0.90)
        # shorter size
        arrow_point_size=int(self.image_display_height*0.05)
        arrow_half_width=int((self.image_display_height*0.10)/2)
        
        self.arrow=dict()
        # 1. ARROW UP!!!   
        #arrow_up_pts 
        self.arrow[('up')]= np.array([[x_arrow_base_location,y_arrow_base_location-arrow_point_size],\
            [x_arrow_base_location-arrow_half_width,y_arrow_base_location],[x_arrow_base_location+arrow_half_width,y_arrow_base_location]], np.int32)
        self.arrow[('up')] = self.arrow[('up')].reshape((-1,1,2))
        # 2. ARROW DOWN!!!
        self.arrow[('down')] = np.array([[x_arrow_base_location,self.image_display_height-1],\
            [x_arrow_base_location-arrow_half_width,self.image_display_height-arrow_point_size],[x_arrow_base_location+arrow_half_width,self.image_display_height-arrow_point_size]], np.int32)
        self.arrow[('down')] = self.arrow[('down')].reshape((-1,1,2))
        
        # 3. ARROW Left!!!
        self.arrow[('left')] = np.array([[x_arrow_base_location-arrow_half_width-arrow_half_width,y_arrow_base_location+int(arrow_point_size/2)],\
            [x_arrow_base_location-arrow_half_width,self.image_display_height-arrow_point_size], [x_arrow_base_location-arrow_half_width,y_arrow_base_location]], np.int32)
        self.arrow[('left')] = self.arrow[('left')].reshape((-1,1,2))
        
        # 4. ARROW Right!!!
        self.arrow[('right')] = np.array([[x_arrow_base_location+arrow_half_width+arrow_half_width,y_arrow_base_location+int(arrow_point_size/2)],\
            [x_arrow_base_location+arrow_half_width,y_arrow_base_location], [x_arrow_base_location+arrow_half_width,self.image_display_height-arrow_point_size]], np.int32)
        self.arrow[('right')] = self.arrow[('right')].reshape((-1,1,2))
        
        # 5. ARROW Direction!!!   
        self.arrow[('heading')] = np.array([[15,2],\
            [10,12],[20,12]], np.int32)
        self.arrow[('heading')] = self.arrow[('heading')].reshape((-1,1,2))
        #################################
        

                ## Make grid index x,y, [coords]
        self.squares_grid=self.make_grid_index(self.file_database_sorted['x_loc'].ptp(),self.file_database_sorted['y_loc'].ptp(), self.pixel_width)
        
        
        
        ### Initialise main image windows
        heading_ind=self.heading_index.find(self.direction)
        available_directions_index=0
        #new_location_x=self.location_x
        #new_location_y=self.location_y
        
        ### Initialise interative environment
        images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(self.location_x,self.location_y,heading_ind)
        if image_found==0:
            print "No base location image... exiting"
            sys.exit()
        
        # Build images to display
        resized_img=self.concatenate_resize_images(images_to_combine)        
        
        ## Windows to display graphics
        # Updated map of maze and current location
        cv2.namedWindow(self.maze_map)
        self.map_template=self.plot_exisiting_locations_on_grid(self.map_template)
        cv2.waitKey(100)
        # Main image display
        #cv2.namedWindow(self.window_name)
        # Layout of place cells
        cv2.namedWindow(self.place_cell_map)                
        self.plot_place_cell_id_on_map(self.map_template,self.place_cell_id)
        cv2.waitKey(100)
        ## ALTERNATIVE:: get NESW for location
        self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
        # plot Place cells on the map
        cv2.waitKey(100)
        ### Put current location on map
        self.plot_current_position_on_map(self.location_x,self.location_y)
        cv2.waitKey(100)
        
    def maze_interactive(self):
        
        # get base x, y locations
        new_location_x=self.location_x
        new_location_y=self.location_y
        
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
                    new_location_x +=self.direction_vector[0]
                    new_location_y +=self.direction_vector[1]
                    images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                    if image_found==0:
                        print "No image"
                        new_location_x -=self.direction_vector[0]
                        new_location_y -=self.direction_vector[1]
                    else:
                        resized_img=self.concatenate_resize_images(images_to_combine)
                        self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                        self.map_template=self.plot_old_position_on_map(old_location_x,old_location_y)
                        self.plot_current_position_on_map(new_location_x,new_location_y)
                elif k == ord('s'): # s= backwards
                    #image = image[::-1]
                    old_location_x=new_location_x
                    old_location_y=new_location_y
                    new_location_x -=self.direction_vector[0]
                    new_location_y -=self.direction_vector[1]
                    images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                    if image_found==0:
                        print "No image"
                        new_location_x +=self.direction_vector[0]
                        new_location_y +=self.direction_vector[1]
                    else:
                        resized_img=self.concatenate_resize_images(images_to_combine)
                        self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                        self.map_template=self.plot_old_position_on_map(old_location_x,old_location_y)
                        self.plot_current_position_on_map(new_location_x,new_location_y)
                elif k == ord('a'): # ,<= left
                    #image = image[::-1]
                    #new_location_x -=1
                    self.new_heading_ind -=1
                    images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                    if image_found==0:
                        print "No image"
                        #new_location_x +=1
                    else:
                        resized_img=self.concatenate_resize_images(images_to_combine)
                        self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                        #map_image_display=plot_current_position_on_map(self.map_template,useable_grid_locations,new_location_x,new_location_y)
                elif k == ord('d'): # .>= right
                    #image = image[::-1]
                    #new_location_x -=1
                    self.new_heading_ind +=1
                    images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                    if image_found==0:
                        print "No image"
                        #new_location_x +=1
                    else:
                        resized_img=self.concatenate_resize_images(images_to_combine)
                        self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                        #map_image_display=plot_current_position_on_map(self.map_template,useable_grid_locations,new_location_x,new_location_y)
        except KeyboardInterrupt:
            pass
    
    # Iterate around the maze either using random stepping or generated from paths.poslog    
    def maze_walk(self, random=True, paths=0):
        
        # Depending on mode
        if random:
            new_location_x=self.location_x
            new_location_y=self.location_y
            
            try:
                ### Wait for key to update
                while True:
                # k = cv2.waitKey(0) & 0xFF
                # Delay here for each cycle through the maze.....
                    k=cv2.waitKey(self.step_time_delay) & 0xFF
                    # Depending on mode
                    #if random: # Generate random direction NESW
                    next_step=np.random.choice(np.array([0,1,2,3]))
                    # Test for button press or location value
                    if k == 27: # ESC
                        cv2.destroyAllWindows()
                        break
                #    elif k == ord('s'):
                #        cv2.imwrite('/Users/chris/foo.png', gray_img)
                #        cv2.destroyAllWindows()
                #        break
                    elif next_step == 0: # w=forwards
                        #image = image[::-1]
                        old_location_x=new_location_x
                        old_location_y=new_location_y
                        new_location_x +=self.direction_vector[0]
                        new_location_y +=self.direction_vector[1]
                        images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                        if image_found==0:
                            print "No image"
                            new_location_x -=self.direction_vector[0]
                            new_location_y -=self.direction_vector[1]
                        else:
                            resized_img=self.concatenate_resize_images(images_to_combine)
                            self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                            self.map_template=self.plot_old_position_on_map(old_location_x,old_location_y)
                            self.plot_current_position_on_map(new_location_x,new_location_y)
                    elif next_step == 1: # s= backwards
                        #image = image[::-1]
                        old_location_x=new_location_x
                        old_location_y=new_location_y
                        new_location_x -=self.direction_vector[0]
                        new_location_y -=self.direction_vector[1]
                        images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                        if image_found==0:
                            print "No image"
                            new_location_x +=self.direction_vector[0]
                            new_location_y +=self.direction_vector[1]
                        else:
                            resized_img=self.concatenate_resize_images(images_to_combine)
                            self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                            self.map_template=self.plot_old_position_on_map(old_location_x,old_location_y)
                            self.plot_current_position_on_map(new_location_x,new_location_y)
                    elif next_step == 2: # ,<= left
                        #image = image[::-1]
                        #new_location_x -=1
                        self.new_heading_ind -=1
                        images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                        if image_found==0:
                            print "No image"
                            #new_location_x +=1
                        else:
                            resized_img=self.concatenate_resize_images(images_to_combine)
                            self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                            #map_image_display=plot_current_position_on_map(self.map_template,useable_grid_locations,new_location_x,new_location_y)
                    elif next_step == 3: # .>= right
                        #image = image[::-1]
                        #new_location_x -=1
                        self.new_heading_ind +=1
                        images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                        if image_found==0:
                            print "No image"
                            #new_location_x +=1
                        else:
                            resized_img=self.concatenate_resize_images(images_to_combine)
                            self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                            #map_image_display=plot_current_position_on_map(self.map_template,useable_grid_locations,new_location_x,new_location_y)
            except KeyboardInterrupt:
                pass

        else: # use paths
            old_location_x=self.location_x.copy()
            old_location_y=self.location_y.copy()
            new_location_x=paths[0][0]
            new_location_y=paths[0][1]
            self.new_heading_ind=paths[0][2]
            location_count=0 # start from as first location set.....
            max_steps=paths.shape[0]
            #image = image[::-1]
            # new_location_x +=self.direction_vector[0]
            # new_location_y +=self.direction_vector[1]
            images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
            if image_found==0:
                print "No image"
                new_location_x -=self.direction_vector[0]
                new_location_y -=self.direction_vector[1]
            else:
                resized_img=self.concatenate_resize_images(images_to_combine)
                self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                self.map_template=self.plot_old_position_on_map(old_location_x,old_location_y)
                self.plot_current_position_on_map(new_location_x,new_location_y)            
            
            # This needs to be sorted to allowing sending on values for the next location to move to....
            try:
                ### Wait for key to update
                while True:
                # k = cv2.waitKey(0) & 0xFF
                # Delay here for each cycle through the maze.....
                    k=cv2.waitKey(self.step_time_delay) & 0xFF
                    if location_count>=max_steps:                    
                        k=27
                    # Test for button press or location value
                    if k == 27: # ESC
                        cv2.destroyAllWindows()
                        break
                    # Continue
                    #next_step=paths[location_count]
                    location_count+=1
                    old_location_x=new_location_x
                    old_location_y=new_location_y
                    new_location_x=paths[location_count][0]
                    new_location_y=paths[location_count][1]
                    self.new_heading_ind=paths[location_count][2]
                    images_to_combine,image_found,self.new_heading_ind,self.direction_vector,image_title,available_directions_index=self.find_next_set_images(new_location_x,new_location_y,self.new_heading_ind)
                    if image_found==0:
                        print 'ERROR -> NO IMAGE FOUND @' + str(paths[location_count])
                        cv2.destroyAllWindows()
                        break                                                
#                        print "No image"
#                        new_location_x -=self.direction_vector[0]
#                        new_location_y -=self.direction_vector[1]
#                    else:
                    resized_img=self.concatenate_resize_images(images_to_combine)
                    self.display_image(resized_img, image_title, available_directions_index, self.new_heading_ind)
                    self.map_template=self.plot_old_position_on_map(old_location_x,old_location_y)
                    self.plot_current_position_on_map(new_location_x,new_location_y)            
            except KeyboardInterrupt:
                pass            
        
if __name__ == '__main__':
    print('FRED')
    # Configure class   
    ttt=maze_from_data()
    # Read available files
    ttt.index_image_files()
    # Display interactively
    ttt.display_maps_images()
    # Run interactive mode....
    ttt.maze_interactive()
    
    

