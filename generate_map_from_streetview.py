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

# Google Street View Image API
# 25,000 image requests per 24 hours
# See https://developers.google.com/maps/documentation/streetview/
# Register your uni account for the streetview api and generate a code and add it here
API_KEY = "AIzaSyBaDx0uEvdsayv1z94p0a0YIatmtNwQcBI"
GOOGLE_URL = "http://maps.googleapis.com/maps/api/streetview?sensor=false&size=640x640&key=" + API_KEY
IMG_PREFIX = "GS"
IMG_SUFFIX = ".jpg"

# parser = argparse.ArgumentParser(
#    description="Get random Street View images from a given country")
#parser.add_argument('country',  help='ISO 3166-1 Alpha-3 Country Code')
#args = parser.parse_args()
dir_name="img_data"


print "Getting images"
attempts, country_hits, imagery_hits, imagery_misses, imagery_sucess = 0, 0, 0, 0, 0
MAX_URLS = 40 #25000 #Set to 100 for testing Luke

#file_labels=["North","East","South","West"],[0,90,180,270]
headings_list=[90,180,270]
#print file_labels[2]
#IMAGES_WANTED = 1 #10
#sys.exit(0)

distance_degrees_per_pano=0.000125 # Starting distance between each image

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Add in a start point here (use google streetview and extract longs and lats from url
start_lat = 53.391681  #random.uniform(min_lat, max_lat)
start_lon = -1.487513  #random.uniform(min_lon, max_lon)
# Add in a stop point here - straight line from start (use google streetview and extract longs and lats from url
stop_lat = 53.391256  #random.uniform(min_lat, max_lat)
stop_lon = -1.487248  #random.uniform(min_lon, max_lon)        

diff_lat=numpy.diff([start_lat,stop_lat])        
diff_lon=numpy.diff([start_lon,stop_lon]) 

distance=math.sqrt(math.pow(diff_lat,2)+math.pow(diff_lon,2))

approx_num_steps=math.ceil(distance/distance_degrees_per_pano)

IMAGES_WANTED = int(approx_num_steps)+1
print "distance:", distance, " num image steps:", IMAGES_WANTED
step_lat=diff_lat/approx_num_steps
step_lon=diff_lon/approx_num_steps

step_array=range(IMAGES_WANTED)
steps_lon_all=(step_array*step_lon)+start_lon
steps_lat_all=(step_array*step_lat)+start_lat
print "Longtitudes:", steps_lon_all
print "Latitudes:", steps_lat_all

# Define these for saving sucessful lon and lats
steps_lon_success=[]
steps_lat_success=[]

try:
    while(True):
        attempts += 1
        # print attempts, start_lat, start_lon
        # Is (lat,lon) inside borders?
        # if point_inside_polygon(start_lon, start_lat, borders):
        #   print "  In country"
        country_hits += 1
        # Luke additions
        # Heading - as degrees!!!!
        heading = 0 # 0 to 360 (both values indicating North, with 90 indicating East, and 180 South).
        fov = 90 # fov (default is 90) determines the horizontal field of view of the image.
        # The field of view is expressed in degrees, with a maximum allowed value of 120
        pitch = 0 # pitch (default is 0) specifies the up or down angle of the camera relative to the Street View vehicle. 
        # This is often, but not always, flat horizontal. 
        # Positive values angle the camera up (with 90 degrees indicating straight up); 
        # negative values angle the camera down (with -90 indicating straight down).        
        
        lat_lon = str(steps_lat_all[imagery_hits]) + "," + str(steps_lon_all[imagery_hits])
        # outfile = os.path.join(
        #    dir_name, IMG_PREFIX + lat_lon + IMG_SUFFIX)
        # outfile = os.path.join(
        #    dir_name, IMG_PREFIX + lat_lon + "," + str(heading) + "," + str(fov) + "," + str(pitch) + IMG_SUFFIX)
        outfile = os.path.join(
            dir_name, IMG_PREFIX + ",ID" + str(imagery_sucess) + "," + str(heading) + "," + str(pitch) + "," + str(fov) + lat_lon + IMG_SUFFIX)       
        # url = GOOGLE_URL + "&location=" + lat_lon            
        url = GOOGLE_URL + "&location=" + lat_lon + "&heading=" + str(heading) + "&fov=" + str(fov) + "&pitch=" + str(pitch) 
        try:
            urllib.urlretrieve(url, outfile)
        except:
            pass
            print " WARNING Failed to get google image:" + ",ID" + str(imagery_sucess) + "," + str(heading) + "," + str(pitch) + "," + str(fov) + lat_lon

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
                        #imagery_hits -= 1
                        os.remove(outfile)
                        print "Files same size! -> removing"
                        get_headings=0
                if get_headings:
                    tempfile=outfile
                    # Got north first - now get other directions....                        
                    for current_heading in headings_list:                      
                        outfile = os.path.join(
                        dir_name, IMG_PREFIX + ",ID" + str(imagery_sucess) + "," + str(current_heading) + "," + str(pitch) + "," + str(fov) + lat_lon + IMG_SUFFIX)     
                        # url = GOOGLE_URL + "&location=" + lat_lon            
                        url = GOOGLE_URL + "&location=" + lat_lon + "&heading=" + str(current_heading) + "&fov=" + str(fov) + "&pitch=" + str(pitch) 
                        try:
                            urllib.urlretrieve(url, outfile)
                        except:
                            pass
                            print " WARNING Failed to get google image:" + ",ID" + str(imagery_sucess) + "," + str(current_heading) + "," + str(pitch) + "," + str(fov) + lat_lon
                    steps_lat_success.append(round(steps_lat_all[imagery_hits],6))
                    steps_lon_success.append(round(steps_lon_all[imagery_hits],6))
                    
                    imagery_sucess += 1
                imagery_hits += 1                
                if imagery_hits == IMAGES_WANTED:
                        break
        if country_hits == MAX_URLS:
            break
except KeyboardInterrupt:
    print "Keyboard interrupt"

print "Attempts:\t", attempts
#print "Country hits:\t", country_hits
print "Imagery misses:\t", imagery_misses
print "Imagery hits:\t", imagery_hits

print "Imagery Success:\t", imagery_sucess
print "Successful Longtitudes:", steps_lon_success
print "Successful Latitudes:", steps_lat_success



# End of file
