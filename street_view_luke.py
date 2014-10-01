#import argparse
import os
#import random
#import shapefile  # http://code.google.com/p/pyshp/
import sys
import urllib
import math
import numpy
import filecmp
# Optional, http://stackoverflow.com/a/1557906/724176
try:
    import timing
except:
    pass

# Google Street View Image API
# 25,000 image requests per 24 hours
# See https://developers.google.com/maps/documentation/streetview/
# Register your uni account for the streetview api and generate a code and add it here
API_KEY = "AIzaSyBaDx0uEvdsayv1z94p0a0YIatmtNwQcBI"
GOOGLE_URL = "http://maps.googleapis.com/maps/api/streetview?sensor=false&size=640x640&key=" + API_KEY
IMG_PREFIX = "img_"
IMG_SUFFIX = ".jpg"

# parser = argparse.ArgumentParser(
#    description="Get random Street View images from a given country")
#parser.add_argument('country',  help='ISO 3166-1 Alpha-3 Country Code')
#args = parser.parse_args()
dir_name="img_data"

# Determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.
# http://www.ariel.com.au/a/python-point-int-poly.html
#def point_inside_polygon(x, y, poly):
#    n = len(poly)
#    inside = False
#    p1x, p1y = poly[0]
#    for i in range(n+1):
#        p2x, p2y = poly[i % n]
#        if y > min(p1y, p2y):
#            if y <= max(p1y, p2y):
#                if x <= max(p1x, p2x):
#                    if p1y != p2y:
#                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
#                    if p1x == p2x or x <= xinters:
#                        inside = not inside
#        p1x, p1y = p2x, p2y
#    return inside

#print "Loading borders"
#shape_file = "D:/VVV14/TM_WORLD_BORDERS/TM_WORLD_BORDERS-0.3.shp"
#if not os.path.exists(shape_file):
#    print "Cannot find " + shape_file + ". Please download it from "
#    "http://thematicmapping.org/downloads/world_borders.php and try again."
#    sys.exit()

#==============================================================================
# sf = shapefile.Reader(shape_file)
# shapes = sf.shapes()
#==============================================================================

#==============================================================================
# print "Finding country"
# for i, record in enumerate(sf.records()):
#     if record[2] == dir_name.upper():
#         print record[2], record[4]
#         print shapes[i].bbox
#         min_lon = shapes[i].bbox[0]
#         min_lat = shapes[i].bbox[1]
#         max_lon = shapes[i].bbox[2]
#         max_lat = shapes[i].bbox[3]
#         borders = shapes[i].points
#         break
# 
#==============================================================================

print "Getting images"
attempts, country_hits, imagery_hits, imagery_misses = 0, 0, 0, 0
MAX_URLS = 40 #25000 #Set to 100 for testing Luke

file_labels=["North","East","South","West"],[0,90,180,270]

print file_labels
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
        outfile = os.path.join(
            dir_name, IMG_PREFIX + lat_lon + "," + str(heading) + "," + str(fov) + "," + str(pitch) + IMG_SUFFIX)
        # url = GOOGLE_URL + "&location=" + lat_lon            
        url = GOOGLE_URL + "&location=" + lat_lon + "&heading=" + str(heading) + "&fov=" + str(fov) + "&pitch=" + str(pitch) 
        try:
            urllib.urlretrieve(url, outfile)
        except:
            pass
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
                # Compare current and previous files
                if imagery_hits == 0:
                    #tempfilesize=os.path.getsize(outfile)
                    tempfile=outfile
                else:
                    if filecmp.cmp(tempfile,outfile):#os.path.getsize(outfile)==tempfilesize: #
                        #imagery_hits -= 1
                        os.remove(outfile)
                        print "Files same size! -> removing"
                    else:
                        #tempfilesize=os.path.getsize(outfile)
                        tempfile=outfile
                imagery_hits += 1                
                if imagery_hits == IMAGES_WANTED:
                    break
        if country_hits == MAX_URLS:
            break
except KeyboardInterrupt:
    print "Keyboard interrupt"

print "Attempts:\t", attempts
print "Country hits:\t", country_hits
print "Imagery misses:\t", imagery_misses
print "Imagery hits:\t", imagery_hits

# End of file
