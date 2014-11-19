import os
import urllib
import math
import numpy as np
import filecmp
#import glob

# Define these for saving sucessful lon and lats
headings_labels=["N","E","S","W"],[0,90,180,270]
distance_degrees_per_pano_division=0.000125 # Starting distance between each image
distance_degrees_per_pano=0.000100 # Starting distance between each image

dir_name_overall="division_street_randomised_dir_pitch"

# set to two so we know on the 'Randomised' set of images
current_image_count=2

# Choose randomisation factor
randomise_pitch=0.1 # 0= off, otherwise frational percentage
randomise_direction=0.1 # 0= off, otherwise frational percentage
SEED=6786575 # Fixed seed for fixed randomisation
np.random.seed(SEED)


# This function loads first road of images and sorts x grid locations!!!
def get_first_line_of_images_along_x(start_location ,stop_location, dir_name="img_data"):

    current_y_location = 0
    print "Getting basline x images"
    total_tries, imagery_hits, imagery_misses, current_x_location = 0, 0, 0, 0
    MAX_URLS = 250 #25000 #Set to 100 for testing Luke    
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  
        
    diff_lat=np.diff([start_location[0],stop_location[0]])        
    diff_lon=np.diff([start_location[1],stop_location[1]]) 
    distance=math.sqrt(math.pow(diff_lat,2)+math.pow(diff_lon,2)) 
    approx_num_steps=math.ceil(distance/distance_degrees_per_pano_division)
    
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
    steps_lon_success=np.empty(1)
    steps_lat_success=np.empty(1)

    
    try:
        while(True):
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
                    # First time just load current image....                
                    get_headings=1                
                    if imagery_hits != 0:  
                        if filecmp.cmp(tempfile,outfile):#os.path.getsize(outfile)==tempfilesize: #
                            get_headings=0
                            #imagery_hits -= 1
                            os.remove(outfile)
                            print "Files same size! -> removing, adding 10% extra step to lat / lon data...."
                            # Add on 10% extra of step lon and lat to                             
                            steps_lat_all[imagery_hits-steps_lat_all.size:]=steps_lat_all[imagery_hits-steps_lat_all.size:]+np.tile(step_lat*0.3,[1,steps_lat_all[imagery_hits-steps_lat_all.size:].size])
                            steps_lon_all[imagery_hits-steps_lon_all.size:]=steps_lon_all[imagery_hits-steps_lon_all.size:]+np.tile(step_lon*0.3,[1,steps_lon_all[imagery_hits-steps_lon_all.size:].size])
                            
                            ## Need to cut additions from end as the whole thing is expanded
                            # Find if the end point is smaller or greater than the stop point
                            if np.diff(steps_lat_all[-2:])>0:
                                steps_lat_all=steps_lat_all[steps_lat_all<=stop_location[0]]
                                steps_lon_all=steps_lon_all[steps_lat_all<=stop_location[0]]
                            else:
                                steps_lat_all=steps_lat_all[steps_lat_all>=stop_location[0]]
                                steps_lon_all=steps_lon_all[steps_lat_all>=stop_location[0]]                                 
                            
                            if np.diff(steps_lon_all[-2:])>0:
                                steps_lat_all=steps_lat_all[steps_lon_all<=stop_location[1]]
                                steps_lon_all=steps_lon_all[steps_lon_all<=stop_location[1]]
                            else:
                                steps_lat_all=steps_lat_all[steps_lon_all>=stop_location[1]]
                                steps_lon_all=steps_lon_all[steps_lon_all>=stop_location[1]]                             
                            
                            imagery_hits -= 1                            
                    if get_headings:
                        tempfile=outfile
                        # Got north already - now get other directions....                        
                        for current_heading in range(1,4):  # Headings E,S,W    #headings_labels[1][1:-1]:    
                            total_tries += 1 
                            outfile,image_captured = get_image_from_google_streetview_api(current_x_location,current_y_location,current_image_count,current_heading,steps_lat_all[imagery_hits],steps_lon_all[imagery_hits],dir_name) 
                            if image_captured==0:
                                imagery_misses += 1                 
                        steps_lat_success=np.append(steps_lat_success,round(steps_lat_all[imagery_hits],7))
                        steps_lon_success=np.append(steps_lon_success,round(steps_lon_all[imagery_hits],7))
                        current_x_location += 1
                    imagery_hits += 1                
                    if imagery_hits == IMAGES_WANTED or imagery_hits == steps_lat_all.size:
                        print "Exiting as got maximum number of image locations wanted:\t", imagery_hits
                        break
            if total_tries >= MAX_URLS:
                print "Exiting as over total URL limit!!!!!!:\t", total_tries           
                return(steps_lat_success,steps_lon_success)
    except KeyboardInterrupt:
        print "Keyboard interrupt"
    
    print "Imagery misses:\t", imagery_misses
    print "Imagery hits:\t", imagery_hits    
    print "Imagery Success:\t", current_x_location
#    print "Successful Longtitudes:", steps_lon_success
#    print "Successful Latitudes:", steps_lat_success
    return (steps_lat_success,steps_lon_success)
    
    # This loads in vertical roads - only y changes
def get_line_of_images_fixed_x_grid(grid_start,start_location,stop_location, max_num_steps, dir_name):
    # x location fixed by incoming value!!!!!!    = LONG (EW) -> going east positive??
    current_x_location = grid_start[0,0]
    # y location will depend on position or negative first step!    = LAT (NS), going north positive!
    # current_y_location = 0
    # imagery_hits STARTS FROM SECOND VALUE (1) as we dont want the same img block twice! e.g. at division street
    imagery_hits = 1
    
    print "Getting basline x images"
    total_tries, imagery_misses= 0, 0
    MAX_URLS = 250 #25000 #Set to 100 for testing Luke    
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  
        
    diff_lat=np.diff([start_location[0],stop_location[0]])        
    diff_lon=np.diff([start_location[1],stop_location[1]]) 
    distance=math.sqrt(math.pow(diff_lat,2)+math.pow(diff_lon,2)) 
    approx_num_steps=math.ceil(distance/distance_degrees_per_pano)
    
    if max_num_steps>0: # No effect is set to zero!
        if max_num_steps>approx_num_steps:
            print 'Warning, max number of steps too high for road length.... STEPS likely to BE TOO FAR APART'
        approx_num_steps=max_num_steps; # Fix number of steps to link with other locations (secondary intersections!)    
    
    IMAGES_WANTED = int(approx_num_steps)+1
    print "distance:", distance, " num image steps:", IMAGES_WANTED
    step_lat=diff_lat/approx_num_steps
    step_lon=diff_lon/approx_num_steps
    
    # y_location start...    
    if diff_lat>0:
        y_increment=1
        current_y_location=1
    else:
        y_increment=-1
        current_y_location=-1
    
    step_array=range(IMAGES_WANTED)
    steps_lon_all=(step_array*step_lon)+start_location[1]
    steps_lat_all=(step_array*step_lat)+start_location[0]
    print "Longtitudes:", steps_lon_all
    print "Latitudes:", steps_lat_all
    print "Longtitude step:", step_lon
    print "Latitude step:", step_lat
    steps_lon_success=np.empty(1)
    steps_lat_success=np.empty(1)

    
    try:
        while(True):
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
                    # First time just load current image....                
                    get_headings=1                
                    if imagery_hits >1:  # changed here as imagery starts at second location to prevent duplicate images! 
                        if filecmp.cmp(tempfile,outfile):#os.path.getsize(outfile)==tempfilesize: #
                            get_headings=0
                            #imagery_hits -= 1
                            os.remove(outfile)
                            print "Files same size! -> removing, adding 10% extra step to lat / lon data...."
                            # Add on 10% extra of step lon and lat to                             
                            steps_lat_all[imagery_hits-steps_lat_all.size:]=steps_lat_all[imagery_hits-steps_lat_all.size:]+np.tile(step_lat*0.15,[1,steps_lat_all[imagery_hits-steps_lat_all.size:].size])
                            steps_lon_all[imagery_hits-steps_lon_all.size:]=steps_lon_all[imagery_hits-steps_lon_all.size:]+np.tile(step_lon*0.15,[1,steps_lon_all[imagery_hits-steps_lon_all.size:].size])
                            
                            ## Need to cut additions from end as the whole thing is expanded
                            # Find if the end point is smaller or greater than the stop point
                            if np.diff(steps_lat_all[-2:])>0:
                                steps_lat_all=steps_lat_all[steps_lat_all<=stop_location[0]+(step_lat*0.1)]
                                steps_lon_all=steps_lon_all[steps_lat_all<=stop_location[0]+(step_lon*0.1)]
                            else:
                                steps_lat_all=steps_lat_all[steps_lat_all>=stop_location[0]-(step_lat*0.1)]
                                steps_lon_all=steps_lon_all[steps_lat_all>=stop_location[0]-(step_lon*0.1)]                                 
                            
                            if np.diff(steps_lon_all[-2:])>0:
                                steps_lat_all=steps_lat_all[steps_lon_all<=stop_location[1]+(step_lat*0.1)]
                                steps_lon_all=steps_lon_all[steps_lon_all<=stop_location[1]+(step_lon*0.1)]
                            else:
                                steps_lat_all=steps_lat_all[steps_lon_all>=stop_location[1]-(step_lat*0.1)]
                                steps_lon_all=steps_lon_all[steps_lon_all>=stop_location[1]-(step_lon*0.1)]                             
                            
                            imagery_hits -= 1                            
                    if get_headings:
                        tempfile=outfile
                        # Got north already - now get other directions....                        
                        for current_heading in range(1,4):  # Headings E,S,W    #headings_labels[1][1:-1]:    
                            total_tries += 1 
                            outfile,image_captured = get_image_from_google_streetview_api(current_x_location,current_y_location,current_image_count,current_heading,steps_lat_all[imagery_hits],steps_lon_all[imagery_hits],dir_name) 
                            if image_captured==0:
                                imagery_misses += 1                 
                        steps_lat_success=np.append(steps_lat_success,round(steps_lat_all[imagery_hits],7))
                        steps_lon_success=np.append(steps_lon_success,round(steps_lon_all[imagery_hits],7))
                        # Increment y in positive or negative direction!
                        current_y_location += y_increment
                    imagery_hits += 1                
                    if imagery_hits == IMAGES_WANTED or imagery_hits == steps_lat_all.size:
                        print "Exiting as got maximum number of image locations wanted:\t", imagery_hits
                        break
            if total_tries >= MAX_URLS:
                print "Exiting as over total URL limit!!!!!!:\t", total_tries           
                return (steps_lat_success,steps_lon_success, [current_x_location,current_y_location])
    except KeyboardInterrupt:
        print "Keyboard interrupt"
    
    print "Imagery misses:\t", imagery_misses
    print "Imagery hits:\t", imagery_hits    
    print "Imagery Success:\t", current_x_location
#    print "Successful Longtitudes:", steps_lon_success
#    print "Successful Latitudes:", steps_lat_success
    return (steps_lat_success,steps_lon_success, [current_x_location,current_y_location])#(steps_lat_success,steps_lon_success)    
    
    # This loads in vertical roads - only y changes
def get_line_of_images_fixed_y_grid(grid_start,start_location,stop_location, max_num_steps ,dir_name,degs_per_pano):
    # x location fixed by incoming value!!!!!!    = LONG (EW) -> going east positive??
    current_y_location = grid_start[1]
    current_x_location = grid_start[0]
    # y location will depend on position or negative first step!    = LAT (NS), going north positive!
    # current_y_location = 0
    # imagery_hits STARTS FROM SECOND VALUE (1) as we dont want the same img block twice! e.g. at division street
    imagery_hits = 1
    
    print "Getting basline x images"
    total_tries, imagery_misses = 0, 0
    MAX_URLS = 250 #25000 #Set to 100 for testing Luke    
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  
        
    diff_lat=np.diff([start_location[0],stop_location[0]])        
    diff_lon=np.diff([start_location[1],stop_location[1]]) 
    distance=math.sqrt(math.pow(diff_lat,2)+math.pow(diff_lon,2)) 
    approx_num_steps=math.ceil(distance/degs_per_pano)
    
    if max_num_steps>0: # No effect is set to zero!
        if max_num_steps>approx_num_steps:
            print 'Warning, max number of steps too high for road length.... STEPS likely to BE TOO FAR APART'
        approx_num_steps=max_num_steps; # Fix number of steps to link with other locations (secondary intersections!)
    
    IMAGES_WANTED = int(approx_num_steps)+1
    print "distance:", distance, " num image steps:", IMAGES_WANTED
    step_lat=diff_lat/approx_num_steps
    step_lon=diff_lon/approx_num_steps
    
    # y_location start...    
    if diff_lon>0:
        x_increment=1
        current_x_location+=1
    else:
        x_increment=-1
        current_x_location-=1
    
    step_array=range(IMAGES_WANTED)
    steps_lon_all=(step_array*step_lon)+start_location[1]
    steps_lat_all=(step_array*step_lat)+start_location[0]
    print "Longtitudes:", steps_lon_all
    print "Latitudes:", steps_lat_all
    print "Longtitude step:", step_lon
    print "Latitude step:", step_lat
    steps_lon_success=np.empty(1)
    steps_lat_success=np.empty(1)

    
    try:
        while(True):
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
                    # First time just load current image....                
                    get_headings=1                
                    if imagery_hits >1:  # changed here as imagery starts at second location to prevent duplicate images! 
                        if filecmp.cmp(tempfile,outfile):#os.path.getsize(outfile)==tempfilesize: #
                            get_headings=0
                            #imagery_hits -= 1
                            os.remove(outfile)
                            print "Files same size! -> removing, adding 15% extra step to lat / lon data...."
                            # Add on 10% extra of step lon and lat to                             
                            steps_lat_all[imagery_hits-steps_lat_all.size:]=steps_lat_all[imagery_hits-steps_lat_all.size:]+np.tile(step_lat*0.15,[1,steps_lat_all[imagery_hits-steps_lat_all.size:].size])
                            steps_lon_all[imagery_hits-steps_lon_all.size:]=steps_lon_all[imagery_hits-steps_lon_all.size:]+np.tile(step_lon*0.15,[1,steps_lon_all[imagery_hits-steps_lon_all.size:].size])
                            
                            ## Need to cut additions from end as the whole thing is expanded
                            # Find if the end point is smaller or greater than the stop point
                            if np.diff(steps_lat_all[-2:])>0:
                                steps_lat_all=steps_lat_all[steps_lat_all<=stop_location[0]]
                                steps_lon_all=steps_lon_all[steps_lat_all<=stop_location[0]]
                            else:
                                steps_lat_all=steps_lat_all[steps_lat_all>=stop_location[0]]
                                steps_lon_all=steps_lon_all[steps_lat_all>=stop_location[0]]                                 
                            
                            if np.diff(steps_lon_all[-2:])>0:
                                steps_lat_all=steps_lat_all[steps_lon_all<=stop_location[1]]
                                steps_lon_all=steps_lon_all[steps_lon_all<=stop_location[1]]
                            else:
                                steps_lat_all=steps_lat_all[steps_lon_all>=stop_location[1]]
                                steps_lon_all=steps_lon_all[steps_lon_all>=stop_location[1]]                             
                            
                            imagery_hits -= 1                            
                    if get_headings:
                        tempfile=outfile
                        # Got north already - now get other directions....                        
                        for current_heading in range(1,4):  # Headings E,S,W    #headings_labels[1][1:-1]:    
                            total_tries += 1 
                            outfile,image_captured = get_image_from_google_streetview_api(current_x_location,current_y_location,current_image_count,current_heading,steps_lat_all[imagery_hits],steps_lon_all[imagery_hits],dir_name) 
                            if image_captured==0:
                                imagery_misses += 1                 
                        steps_lat_success=np.append(steps_lat_success,round(steps_lat_all[imagery_hits],7))
                        steps_lon_success=np.append(steps_lon_success,round(steps_lon_all[imagery_hits],7))
                        # Increment y in positive or negative direction!
                        current_x_location += x_increment
                    imagery_hits += 1                
                    if imagery_hits == IMAGES_WANTED or imagery_hits == steps_lat_all.size:
                        print "Exiting as got maximum number of image locations wanted:\t", imagery_hits
                        break
            if total_tries >= MAX_URLS:
                print "Exiting as over total URL limit!!!!!!:\t", total_tries           
                return (steps_lat_success,steps_lon_success, [current_x_location,current_y_location])
    except KeyboardInterrupt:
        print "Keyboard interrupt"
    
    print "Imagery misses:\t", imagery_misses
    print "Imagery hits:\t", imagery_hits    
    print "Imagery Success:\t", current_y_location
#    print "Successful Longtitudes:", steps_lon_success
#    print "Successful Latitudes:", steps_lat_success
    return (steps_lat_success,steps_lon_success, [current_x_location,current_y_location])#(steps_lat_success,steps_lon_success)    
        
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
# base_location = [53.3795934, -1.4764689]

# DIVISION STREET -> Full RUN
# This is used as the main base for the x grid!!!!!!
# Div / westfield terrace
start_location_division_street = [53.3794166, -1.4774948] #[53.3795934, -1.4764689] # = 0,0
# Div cambridge street (Barkers pool)
stop_location_division_street = [53.3802642, -1.4724617]

############################################################ssssww###
############ Shortened stop at carver street.....
#stop_location_division_street = [53.3800566, -1.4738162]


steps_lat_division,steps_lon_division=get_first_line_of_images_along_x(start_location_division_street ,stop_location_division_street, dir_name_overall)

# This is used as the main base for the y grid!!!
# Find locations in 
primary_street_intersection=np.empty([9,2])
primary_street_intersection_x=np.empty([9,1])
# Division street INTERSECTIONS
primary_street_intersection[0,:]=[53.3795934, -1.4764689] # Westfield terrace_trafalgar_street_NS
primary_street_intersection[1,:]=[53.3797974, -1.4753467] # rockingham_street_NS
primary_street_intersection[2,:]=[53.3799304, -1.4745786] # rockingham_lane_N
primary_street_intersection[3,:]=[53.3800566, -1.4738162] # carver_street_NS
# Not connected to west street
# South versions!!!!!
primary_street_intersection[4,:]=[53.3795934, -1.4764689] # Westfield terrace_trafalgar_street_NS
primary_street_intersection[5,:]=[53.3797974, -1.4753467] # rockingham_street_NS
primary_street_intersection[6,:]=[53.3800566, -1.4738162] # carver_street_NS
primary_street_intersection[7,:]=[53.3801694, -1.4731993] # carver_lane_backfields_NS
primary_street_intersection[8,:]=[53.3802642, -1.4724617] # cambridge_street_S

vertical_y_street_intersection=np.empty([9,2])
vertical_y_street_intersection_x=np.empty([9,1])

# North intersect with west street
vertical_y_street_intersection[0,:]=[53.3806606, -1.4769946] # [53.3806765,-1.4769241] # Westfield terrace_trafalgar_street_NS
vertical_y_street_intersection[1,:]=[53.380979,-1.475874]    #[53.3809358, -1.4758533] # rockingham_street_NS
vertical_y_street_intersection[2,:]=[53.3810864, -1.4751133] # rockingham_lane_N
vertical_y_street_intersection[3,:]=[53.3812269, -1.4744049] # carver_street_NS
# South
vertical_y_street_intersection[4,:]=[53.3784121, -1.4758985] #Trafalgar street
vertical_y_street_intersection[5,:]=[53.3786444, -1.4747796] #Rockingham street (South)
vertical_y_street_intersection[6,:]=[53.3789671, -1.4733142] #Carver street (South)
vertical_y_street_intersection[7,:]=[53.3788670, -1.4726575] #Backfields
vertical_y_street_intersection[8,:]=[53.3791734, -1.4719521] #Cambridge street

# west street INTERSECTIONS
secondary_street_start= [53.380271,-1.4788319] # [53.3803826, -1.4774948] # [53.3806606, -1.4769946]
secondary_street_end= [53.3812389,-1.4745107] #[53.3812269, -1.4744049]

# Index of streets which connect from division street to west street
secondary_street_intersections=[0,1,2,3]



# Extract x locations -> x is longtitude east - west
for find_close in range(0,primary_street_intersection[:,1].size):
    ################## WARNING USES -1 to adjust position!!!!
    primary_street_intersection_x[find_close]=np.abs(steps_lon_division-primary_street_intersection[find_close,1]).argmin()-1

grid_start=np.zeros([1,2])

# Has grid start and grid end of each intersection!!!
x_grid_index_intersections=np.zeros([vertical_y_street_intersection[:,1].size,4])


##### Link number of steps between each street for west so they link up!!!!!!
# 1. Run first street find number of steps and use for next streets!
steps_lat_intersections=[[]];
steps_lon_intersections=[[]];


for current_intersection in range(0,vertical_y_street_intersection[:,1].size):
    grid_start[0,0]=primary_street_intersection_x[current_intersection]
    x_grid_index_intersections[current_intersection,0:2]=grid_start
    if current_intersection==0:
        steps_lat_intersections[current_intersection],steps_lon_intersections[current_intersection],x_grid_index_intersections[current_intersection,2:4]=get_line_of_images_fixed_x_grid(grid_start,primary_street_intersection[current_intersection,:] ,vertical_y_street_intersection[current_intersection,:],0, dir_name_overall)
    else:
        print 'Using limited number of y steps as:\t' + str(x_grid_index_intersections[0,3])
        steps_lat_intersections.append([])
        steps_lon_intersections.append([])
        if current_intersection in secondary_street_intersections:
            # Limit steps between intersections
            steps_lat_intersections[current_intersection],steps_lon_intersections[current_intersection],x_grid_index_intersections[current_intersection,2:4]=get_line_of_images_fixed_x_grid(grid_start,primary_street_intersection[current_intersection,:] ,vertical_y_street_intersection[current_intersection,:],x_grid_index_intersections[0,3]-1, dir_name_overall)
        else:
            # Unlimited steps as no intersections....
            steps_lat_intersections[current_intersection],steps_lon_intersections[current_intersection],x_grid_index_intersections[current_intersection,2:4]=get_line_of_images_fixed_x_grid(grid_start,primary_street_intersection[current_intersection,:] ,vertical_y_street_intersection[current_intersection,:],0, dir_name_overall)
            
######## Get images for west street! -> FAILS AS DIFFERENT NUMBER OF IMAGES IN EACH STREET
# Start at end of first vertical line       
        
# This needs to start at west street start and continue between each available pre built intersection!        
# Need Lat, lon and grid ref of each start street!
        # Store grid info as: x_loc,y_loc,lat,lon

grid_intersection_ref=np.empty([4,len(secondary_street_intersections)],dtype='f4')        
for current_intersection in range(0,len(secondary_street_intersections)):   
    # max x       
    grid_intersection_ref[current_intersection][0]=x_grid_index_intersections[secondary_street_intersections[current_intersection],2:3]
    # max y    
    grid_intersection_ref[current_intersection][1]=x_grid_index_intersections[secondary_street_intersections[current_intersection],3:4]
    # max lat limit
    grid_intersection_ref[current_intersection][2]=steps_lat_intersections[secondary_street_intersections[current_intersection]][-1]
    # max lon
    grid_intersection_ref[current_intersection][3]=steps_lon_intersections[secondary_street_intersections[current_intersection]][-1]


# Sort values by x grid locations (smallest first!!!!)
#np.sort(grid_intersection_ref,axis=1)
I = np.argsort(grid_intersection_ref[:,0])
grid_intersection_ref=grid_intersection_ref[I,:]

# Has grid start and grid end of each intersection!!!
y_grid_index_intersections=np.zeros([grid_intersection_ref.shape[0],4]) # add another row for street end (none as west street finishes)
steps_secondary_lat_intersections=[[]];
steps_secondary_lon_intersections=[[]];

# 1. Go images between each point!!!!
for current_intersection in range(0,grid_intersection_ref.shape[0]-1):
    steps_secondary_lat_intersections[current_intersection],steps_secondary_lon_intersections[current_intersection],y_grid_index_intersections[current_intersection,2:4]=get_line_of_images_fixed_y_grid(grid_intersection_ref[current_intersection,0:2]-[0,1],grid_intersection_ref[current_intersection,2:4],grid_intersection_ref[current_intersection+1,2:4],grid_intersection_ref[current_intersection+1,0]-grid_intersection_ref[current_intersection,0], dir_name_overall,distance_degrees_per_pano)
    steps_secondary_lat_intersections.append([])
    steps_secondary_lon_intersections.append([])
# 2. Add in ends of road!!!!!!
    # Secondary start
steps_secondary_lat_intersections[grid_intersection_ref.shape[0]-1],steps_secondary_lon_intersections[grid_intersection_ref.shape[0]-1],y_grid_index_intersections[grid_intersection_ref.shape[0]-1,2:4]=get_line_of_images_fixed_y_grid(grid_intersection_ref[0,0:2]-[1,1],[53.3805726,-1.4773802],secondary_street_start,0, dir_name_overall,distance_degrees_per_pano_division)

#get_line_of_images_fixed_y_grid(grid_start,start_location,stop_location, max_num_steps ,dir_name)
#steps_secondary_lat_intersections.append([])53.3806208,-1.4771689
#steps_secondary_lon_intersections.append([])
#    # Secondary end
#steps_secondary_lat_intersections[grid_intersection_ref.shape[0]+1],steps_secondary_lon_intersections[grid_intersection_ref.shape[0]+1],y_grid_index_intersections[grid_intersection_ref.shape[0]+1,2:4]=get_line_of_images_fixed_y_grid(grid_intersection_ref[0,0:2]-[0,1],grid_intersection_ref[0,2:4],secondary_street_start,0, dir_name_overall)
#steps_secondary_lat_intersections.append([])
#steps_secondary_lon_intersections.append([])    
               
  #steps_secondary_lat_intersections[current_intersection],steps_secondary_lon_intersections[current_intersection],y_grid_index_intersections[current_intersection,2:4]=get_line_of_images_fixed_y_grid(x_grid_index_intersections[0,2:4],secondary_street_start ,secondary_street_end, 0, dir_name_overall)



    
# Save out data!!!!
if not os.path.exists(dir_name_overall):
    os.makedirs(dir_name_overall)
file_grid_x=open(os.path.join(dir_name_overall,"division_street_grid_vals.txt"),"w")
file_grid_x.write('Primary street intersections')
file_grid_x.write("\n")
file_grid_x.write(str(primary_street_intersection))
file_grid_x.write("\n")
file_grid_x.write('steps_lon_division')
file_grid_x.write("\n")
file_grid_x.write(str(steps_lon_division))
file_grid_x.write("\n")
file_grid_x.write('steps_lat_division')
file_grid_x.write("\n")
file_grid_x.write(str(steps_lat_division))
file_grid_x.write("\n")
file_grid_x.write('x_grid_index_intersections')
file_grid_x.write("\n")
file_grid_x.write(str(x_grid_index_intersections))
file_grid_x.write("\n")
file_grid_x.write('y_grid_index_intersections')
file_grid_x.write("\n")
file_grid_x.write(str(y_grid_index_intersections))
file_grid_x.write("\n")
file_grid_x.write('steps_secondary_lon_intersections')
file_grid_x.write("\n")
file_grid_x.write(str(steps_secondary_lon_intersections))
file_grid_x.write("\n")
file_grid_x.write('grid_intersection_ref')
file_grid_x.write("\n")
file_grid_x.write(str(grid_intersection_ref))
file_grid_x.write("\n")
file_grid_x.close()
# Add in a start point here (use google streetview and extract longs and lats from url

# Add in a stop point here - straight line from start (use google streetview and extract longs and lats from url
# LUKE SHORTENED HERE!!!!!
#stop_location = [53.3802642, -1.4754617]

#  
#get_line_of_images(base_location, start_location ,stop_location, dir_name_overall)
    
    
# End of file
    


