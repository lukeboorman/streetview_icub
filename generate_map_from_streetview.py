#import argparse
import os
#import random
#import shapefile  # http://code.google.com/p/pyshp/
#import sys
import urllib
import math
import numpy
import filecmp
# Optional, http://stackoverflow.com/a/1557906/724176
#try:
#    import timing
#except:
#    pass


def get_line_of_images(base_location, start_location ,stop_location, dir_name="img_data"):

    current_y_location,current_image_count = 0, 1
    print "Getting images"
    location_attempts, total_tries, imagery_hits, imagery_misses, current_x_location = 0, 0, 0, 0, 0
    MAX_URLS = 250 #25000 #Set to 100 for testing Luke
    
    headings_labels=["N","E","S","W"],[0,90,180,270]
    
    #distance_degrees_per_pano=0.000125 # Starting distance between each image
    distance_degrees_per_pano=0.000100 # Starting distance between each image
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  
    
    diff_lat=numpy.diff([start_location[0],stop_location[0]])        
    diff_lon=numpy.diff([start_location[1],stop_location[1]]) 
    
    distance=math.sqrt(math.pow(diff_lat,2)+math.pow(diff_lon,2))
    
    approx_num_steps=math.ceil(distance/distance_degrees_per_pano)
    
    IMAGES_WANTED = int(approx_num_steps)+1
    print "distance:", distance, " num image steps:", IMAGES_WANTED
    step_lat=diff_lat/approx_num_steps
    step_lon=diff_lon/approx_num_steps
    
    step_array=range(IMAGES_WANTED)
    steps_lon_all=(step_array*step_lon)+start_location[1]
    steps_lat_all=(step_array*step_lat)+start_location[0]
    print "Longtitudes:", steps_lon_all
    print "Latitudes:", steps_lat_all

    print "Longtitude step:", step_lon
    print "Latitude step:", step_lat
    
    # Define these for saving sucessful lon and lats
    steps_lon_success=[]
    steps_lat_success=[]
    
    try:
        while(True):
            
            location_attempts += 1
            total_tries += 1 
            outfile,image_captured = get_image_from_google_streetview_api(current_x_location,current_y_location,current_image_count,headings_labels[1][0],steps_lat_all[imagery_hits],steps_lon_all[imagery_hits],dir_name) 
            if image_captured==0:
                imagery_misses += 1
            if os.path.isfile(outfile):
                # Check size and delete "Sorry, we have no imagery here".
                # Note: hardcoded based on current size of default.
                # Might change.
                # Will definitely change if you change requested image size.
                # This number here will change on different platforms.....
                if os.path.getsize(outfile) == 8381:  # bytes
                    print "    No imagery"
                    imagery_misses += 1
                    os.remove(outfile)
                else:
                    print "    ========== Got one! =========="
                    # Compare current and previous files (MAKING SURE ITS A NEW GOOGLE IMAGE LOCATION!)
                    # First tim just load current image....                
                    get_headings=1                
                    if imagery_hits != 0:  
                        if filecmp.cmp(tempfile,outfile):#os.path.getsize(outfile)==tempfilesize: #
                            get_headings=0
                            #imagery_hits -= 1
                            os.remove(outfile)
                            print "Files same size! -> removing, adding 10% extra step to lat / lon data...."
                            # Add on 10% extra of step lon and lat to                             
                            steps_lat_all[imagery_hits-steps_lat_all.size:]=steps_lat_all[imagery_hits-steps_lat_all.size:]+numpy.tile(step_lat*0.3,[1,steps_lat_all[imagery_hits-steps_lat_all.size:].size])
                            steps_lon_all[imagery_hits-steps_lon_all.size:]=steps_lon_all[imagery_hits-steps_lon_all.size:]+numpy.tile(step_lon*0.3,[1,steps_lon_all[imagery_hits-steps_lon_all.size:].size])
                            imagery_hits -= 1                            
                    if get_headings:
                        tempfile=outfile
                        # Got north already - now get other directions....                        
                        for current_heading in range(1,4):  # Headings E,S,W    #headings_labels[1][1:-1]:    
                            total_tries += 1 
                            outfile,image_captured = get_image_from_google_streetview_api(current_x_location,current_y_location,current_image_count,current_heading,steps_lat_all[imagery_hits],steps_lon_all[imagery_hits],dir_name) 
                            if image_captured==0:
                                imagery_misses += 1                 
                        steps_lat_success.append(round(steps_lat_all[imagery_hits],6))
                        steps_lon_success.append(round(steps_lon_all[imagery_hits],6)) 
                        current_x_location += 1
                    imagery_hits += 1                
                    if imagery_hits == IMAGES_WANTED:
                        print "Exiting as got maximum number of image locations wanted:\t", imagery_hits
                        return
            if total_tries >= MAX_URLS:
                print "Exiting as over total URL limit!!!!!!:\t", total_tries           
                return
    except KeyboardInterrupt:
        print "Keyboard interrupt"
    
    print "location_attempts:\t", location_attempts
    print "Imagery misses:\t", imagery_misses
    print "Imagery hits:\t", imagery_hits
    
    print "Imagery Success:\t", current_x_location
    print "Successful Longtitudes:", steps_lon_success
    print "Successful Latitudes:", steps_lat_success
    return
    
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
    headings_labels=["N","E","S","W"],[0,90,180,270]
    # The field of view is expressed in degrees, with a maximum allowed value of 120    
    fov = 90 # fov (default is 90) determines the horizontal field of view of the image.
    # This is often, but not always, flat horizontal. 
    # Positive values angle the camera up (with 90 degrees indicating straight up); 
    # negative values angle the camera down (with -90 indicating straight down).  
    pitch = 0 # pitch (default is 0) specifies the up or down angle of the camera relative to the Street View vehicle. 
  
    outfile = os.path.join(
        dir_name, str(current_x_location).zfill(3) + "-" + str(current_y_location).zfill(3) +
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

    
    
    
##    # Lukes house Yeomans road
## Add in a start point here (use google streetview and extract longs and lats from url
#start_location[0] = 53.391681 
#start_location[1] = -1.487513 
## Add in a stop point here - straight line from start (use google streetview and extract longs and lats from url
#stop_location[0] = 53.391256  
#stop_location[1] = -1.487248
#get_line_of_images(start_location[0],start_location[1],stop_location[0], stop_location[1], "yeomans road")    

# Division street one end to the other....    

# Start Point for grid indexing!!!!!!!
# Intersection of division street and Westfield terrace
base_location = [53.3795934, -1.4764689]

# Include each intersection in turn!!!!!
# Add in a start point here (use google streetview and extract longs and lats from url
start_location= [53.3795934, -1.4764689]
# Add in a stop point here - straight line from start (use google streetview and extract longs and lats from url
# LUKE SHORTENED HERE!!!!!
#stop_location = [53.3802642, -1.4754617]

stop_location = [53.3802642, -1.4724617]
#  
get_line_of_images(base_location, start_location ,stop_location, "division_street_1")
    
    
# End of file
    


