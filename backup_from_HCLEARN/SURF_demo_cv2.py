# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 08:58:09 2014

@author: luke
"""

import cv2
import numpy as np

# Load the images
img =cv2.imread('D:/robotology/hclearn/window.jpg')#'messi4.jpg')

# Convert them to grayscale
imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# SURF extraction
surf = cv2.xfeatures2d.SURF_create() #cv2.SURF()
kp, descriptors = surf.detectAndCompute(imgg,None,useProvidedKeypoints = False)#surf.detect(imgg,None,useProvidedKeypoints = False)

# Setting up samples and responses for kNN
samples = np.array(descriptors)
responses = np.arange(len(kp),dtype = np.float32)

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)


######## FLANN

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors,descriptors,k=2) #flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img,kp1,img2,kp2,matches,None,**draw_params)


## kNN training
#knn = cv2.KNearest()
#knn.train(samples,responses)
#
## Now loading a template image and searching for similar keypoints
#template = cv2.imread('template.jpg')
#templateg= cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
#keys,desc = surf.detect(templateg,None,useProvidedKeypoints = False)
#
#for h,des in enumerate(desc):
#    des = np.array(des,np.float32).reshape((1,128))
#    retval, results, neigh_resp, dists = knn.find_nearest(des,1)
#    res,dist =  int(results[0][0]),dists[0][0]
#
#    if dist<0.1: # draw matched keypoints in red color
#        color = (0,0,255)
#    else:  # draw unmatched in blue color
#        print dist
#        color = (255,0,0)
#
#    #Draw matched key points on original image
#    x,y = kp[res].pt
#    center = (int(x),int(y))
#    cv2.circle(img,center,2,color,-1)
#
#    #Draw matched key points on template image
#    x,y = keys[h].pt
#    center = (int(x),int(y))
#    cv2.circle(template,center,2,color,-1)

cv2.imshow('img',img)
cv2.imshow('barry',img2)
#cv2.imshow('tm',template)
cv2.waitKey(0)
cv2.destroyAllWindows()