# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:22:52 2014

Process maze images for HClearn code....

1. Cant have negative grid locations...
2. Cant have additional '-' 

Code
1. Shifts all image grid locations to positive
2. Removes long and lat descriptors


@author: luke
"""
import numpy as np
#import sys
import os
import glob

### Again to match HC Learn -> flip y values -> Need South positive / North Negative
flip_y=True
### Again to match HC Learn -> flip x values -> Need East positive / West Negative
flip_x=False

# Path to images # REMEMBER FORWARD SLASHES
in_folder_name = 'D:/robotology/hclearn/division_street_1'
# WOULD ADVISE YOU PUT DATA INTO A NEW FOLDER TO PREVENT OVERWRITING OF NAMES (e.g. during x, y flip)!!!!!!!!!!!
out_folder_name = 'D:/robotology/hclearn/division_street_2'
heading_index='NESW' #N=0, E=1, S=2, W=3

if not os.path.exists(out_folder_name):
    os.makedirs(out_folder_name) 

no_files=len(glob.glob(os.path.join(in_folder_name, '*.jpg')))

#file_database=np.empty([5,no_files],dtype=int)
file_database=np.empty(no_files,\
    dtype=[('file_id','i2'),('x_loc','i2'),('y_loc','i2'),('heading','i2'),('img_id','i2')])
#,no_files],dtype=int)
file_database['file_id'][:]=range(0,no_files)
piclist = []
image_count=0
IMG_SUFFIX = ".jpg"  

for infile in glob.glob(os.path.join(in_folder_name, '*.jpg')):
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
    #img_file[image_count,:,:,:]=cv2.imread(infile)
    image_count += 1

# Find minimum x value
x_min=file_database['x_loc'].min()
y_min=file_database['y_loc'].min()

x_ptp=0
y_ptp=0

if flip_y:
    y_ptp=file_database['y_loc'].ptp()
if flip_x:
    x_ptp=file_database['x_loc'].ptp()
#########


print('Files will be adjsuted with x +',str(-x_min),' and y +', str(-y_min))
print('Files will be flipped with x +',str(-x_ptp),' and y +', str(-y_ptp))

###### Rename each file with x,y offset and no lon/lat
for infile in range(0,len(piclist)):
    new_fn = os.path.join(out_folder_name, str(np.absolute(file_database['x_loc'][infile]-x_min-x_ptp)).zfill(3) + "-" + str(np.absolute(file_database['y_loc'][infile]-y_min-y_ptp)).zfill(3) + "-" + heading_index[file_database['heading'][infile]]+ "-" + str(file_database['img_id'][infile]).zfill(3) + IMG_SUFFIX)  
    #if piclist[infile]!=new_fn:    
    os.rename(piclist[infile],new_fn)
    #else:
    #    print 'File names are the same: ' + piclist[infile] + ' and ' + new_fn




          





