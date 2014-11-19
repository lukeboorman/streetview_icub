# -*- coding: utf-8 -*-
"""
Created on Wed Nov 05 15:20:20 2014
Replacemnt of bad grid locations... due to googles nearest find functionality
@author: luke
"""
import os, re
import urllib
#import math
import numpy as np
import glob
# Replace bad locations - because google finds closest!!!!!!!!
dir_name_overall="division_street_randomised_dir_pitch"
heading_index='NESW' #N=0, E=1, S=2, W=3

# Choose randomisation factor
randomise_pitch=0.1 # 0= off, otherwise frational percentage
randomise_direction=0.1 # 0= off, otherwise frational percentage
SEED=6786575 # Fixed seed for fixed randomisation
np.random.seed(SEED)

def get_image_from_google_streetview_api(current_x_location,current_y_location,current_image_count,current_heading,current_lat,current_lon,dir_name):
    # Google Street View Image API
    # 25,000 image requests per 24 hours
    # See https://developers.google.com/maps/documentation/streetview/
    # Register your uni account for the streetview api and generate a code and add it here      
    API_KEY = "AIzaSyBaDx0uEvdsayv1z94p0a0YIatmtNwQcBI"
    GOOGLE_URL = "http://maps.googleapis.com/maps/api/streetview?sensor=false&size=640x640&key=" + API_KEY
    IMG_SUFFIX = ".jpg"  
    # Heading - as degrees!!!!
    #heading = 0 # 0 to 360 (both values indicating North, with 90 indicating East, and 180 South).    
    # Add in randomisation......    
    if randomise_direction!=0:
        headings_labels=["N","E","S","W"],[0,90,180,270]+360*randomise_direction*(np.random.rand(4)-0.5)        
    else:
        headings_labels=["N","E","S","W"],[0,90,180,270]
    
    if randomise_pitch!=0:        
    # This is often, but not always, flat horizontal. 
    # Positive values angle the camera up (with 90 degrees indicating straight up); 
    # negative values angle the camera down (with -90 indicating straight down).
        pitch = 0+180*randomise_direction*(np.random.rand()-0.5) # pitch (default is 0) specifies the up or down angle of the camera relative to the Street View vehicle.     
    else:    
        pitch = 0 # pitch (default is 0) specifies the up or down angle of the camera relative to the Street View vehicle. 
  
    # The field of view is expressed in degrees, with a maximum allowed value of 120    
    fov = 90 # fov (default is 90) determines the horizontal field of view of the image.
    outfile = os.path.join(
        dir_name, str(int(current_x_location)).zfill(3) + "-" + str(int(current_y_location)).zfill(3) +
        "-" + headings_labels[0][current_heading]+ "-" + str(current_image_count).zfill(3) + "-" +
        str(current_lat) + "-" + str(current_lon) + IMG_SUFFIX)               
    url = GOOGLE_URL + "&location=" + str(current_lat) + "," + str(current_lon) + "&heading=" + str(headings_labels[1][current_heading]) + "&fov=" + str(fov) + "&pitch=" + str(pitch) 
    image_captured = 1                            
    try:
        urllib.urlretrieve(url, outfile)
    except:
        pass
        image_captured = 0
        print " WARNING Failed to get google image:" + ",ID" + \
            str(current_x_location) + "," + str(headings_labels[1][current_heading]) + "," + str(pitch) + "," + str(fov) + "," + str(current_lat) + "," + str(current_lon)
    return (outfile,image_captured)

def replace_bad_locations(dir, x, y, new_lat, new_lon):
    # Make pattern    
    pattern=str(int(x)).zfill(3)+'-'+str(int(y)).zfill(3)  
    print 'Replacing files at:', pattern
    count=0
    for f in os.listdir(dir_name_overall):
        if re.search(pattern, f):
            print 'Warning removing', os.path.join(dir, f)
            # Get current file count (fourth value!!!!)
            current_image_version=int(f[10:13])            
            os.remove(os.path.join(dir, f))
            count+=1
    print str(count), ' files removed'
    for i in range(0,4):
        # New image
        outfile,image_captured=get_image_from_google_streetview_api(x,y,current_image_version,i,new_lat,new_lon,dir)
        if not image_captured:
            print 'NO IMAGE FOUND!'

# Need to delete old....
file_list=glob.glob(os.path.join(dir_name_overall, '*.jpg'))
#file_database=numpy.empty([len(file_list),7])  
#file_database=numpy.empty(len(file_list),dtype=[('file_id','i2'),('x_loc','i2'),('y_loc','i2'),('heading','i2'),('fname','a100')])
      
#for infile in range(0,len(file_list)):
#    file_info=file_list[infile][file_list[infile].find("\\")+1:file_list[infile].find("\\")+12]
#    # img count , x, y, heading, img_num 
#    file_database['file_id'][infile]=int(infile)
#    # x grid
#    file_database['x_loc'][infile]=int(file_info[0:3])
#    # y grid    
#    file_database['y_loc'][infile]=int(file_info[4:7])
#    # Save filename
#    file_database['fname'][infile]=str(file_list[infile])
#        # Convert letter heading to index 1= N, 2=E, 3=S, 4=W
#    file_database['heading'][infile]=heading_index.find(file_info[8:9])
#    
#file_database_sorted=numpy.sort(file_database,order=['x_loc','y_loc','heading'])

#x_matched=numpy.array(numpy.nonzero(file_database_sorted['x_loc']==16))
#xy_matched=x_matched[numpy.nonzero(file_database_sorted['y_loc'][x_matched]==11)]
#os.remove(file_database_sorted['fname'][xy_matched])
## Need to replace 021-000-NSEW with different location: 
## @53.3809293,-1.475803,3a,75y,65.76h,72.54t/data=!3m4!1e1!3m2!1sTre41RkW_NUZ1zh-C_qF4Q!2e0
replace_bad_locations(dir_name_overall,21,0,53.3799304,-1.4745786)#53.3809293,-1.475803) 


## Need to replace 016-011-NSEW with different location: 
## @53.3809293,-1.475803,3a,75y,65.76h,72.54t/data=!3m4!1e1!3m2!1sTre41RkW_NUZ1zh-C_qF4Q!2e0
replace_bad_locations(dir_name_overall,16,11,53.3809358,-1.4758533)#53.3809293,-1.475803) 
## Need to replace 015-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,15,11,53.3809182,-1.4758523) 
## Need to replace 017-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,17,11,53.3809293,-1.475803) 
## Need to replace 018-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,18,11,53.3809658,-1.4756437) 
## Need to replace 019-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,19,11,53.3810046,-1.4754805) 
## Need to replace 020-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,20,11,53.3810455,-1.4752969) 
### Need to replace 021-011-NSEW with different location: 
#replace_bad_locations(dir_name_overall,21,11,53.381063,-1.4750983) 
## Need to replace 022-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,21,11,53.381063,-1.4750983) 
## Need to replace 023-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,22,11,53.3810864,-1.4751133) 
## Need to replace 024-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,23,11,53.3811011,-1.4750549) 
## Need to replace 025-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,24,11,53.3811397,-1.4749022) 
## Need to replace 026-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,25,11,53.3811749,-1.4747635) 
## Need to replace 027-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,26,11,53.38121,-1.4746247) 
## Need to replace 028-011-NSEW with different location: 
replace_bad_locations(dir_name_overall,27,11,53.3812389,-1.4745107) 


## Replace 032-000
# @53.3801694,-1.4731993,3a,75y,92.66h,66.32t/data=!3m4!1e1!3m2!1sHIP9RadmwGhqulK3s_DbsQ!2e0    
replace_bad_locations(dir_name_overall,32,0,53.3801694,-1.4731993) 
## Replace 037-000
# @53.3802642,-1.4724617,3a,75y,88.18h,61.12t/data=!3m4!1e1!3m2!1sd6uUu7ZngTmEH063JQUXFw!2e0   
replace_bad_locations(dir_name_overall,37,0,53.3802642,-1.4724617) 
## Replace 006-011
# @53.3806208,-1.4771689,3a,75y,60.38h,76.67t/data=!3m4!1e1!3m2!1szAlcpnzfq0F_lytAXLvarg!2e0!6m1!1e1
replace_bad_locations(dir_name_overall,6,11,53.3806208,-1.4771689) 
## Replace 007-011
# @53.3806606,-1.4769946,3a,75y,170.62h,77t/data=!3m4!1e1!3m2!1smY2x6nF3MRYhLfWtRXPxpg!2e0!6m1!1e1
replace_bad_locations(dir_name_overall,7,11,53.3806606,-1.4769946)    

## Replace -02-011
replace_bad_locations(dir_name_overall,-2,11,53.3803507,-1.4784343)

## Replace -03-011
replace_bad_locations(dir_name_overall,-3,11,53.380314,-1.4786175)  

## Replace -04-011
replace_bad_locations(dir_name_overall,-4,11,53.3802853,-1.4787604)

## Replace -04-011
replace_bad_locations(dir_name_overall,-5,11,53.380271,-1.4788319)

