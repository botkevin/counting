import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils

"""
Takes in a method("orb","surf","sift")
returns a detector
"""
def detector(method):
    dec = None
    if method == "orb":
        dec = cv2.ORB_create()
    elif method == "surf":
        dec = cv2.xfeatures2d.SURF_create(400)
    elif method == "sift":
        dec = cv2.xfeatures2d.SIFT_create()
    else:
        raise Exception("use surf, sift, or orb")
    return dec

"""
Takes in a detector and image
returns keypoints and descriptors
"""
def detect(dec, img):
    # find the keypoints and descriptors
    kp, des = dec.detectAndCompute(img,None)
    return kp, des
   
"""
Takes in descriptors
returns 2 best matches using FLANN or brute force NN
"""
def match(des1, des2, modus, method):
    matches = None
    if modus == "bf":
        # NORM_HAMMING for ORB, NORM_L2(default) for SURF/else
        bf = None
        if method == "orb":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

    elif modus == "FLANN":
        index_params, search_params = None, None
        if method == "orb":
            FLANN_INDEX_LSH = 6
            index_params= dict (algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
        else:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
        # This is FLANN # of times recursive search 
        # Higher number is more accurate at the cost
        # of more computation. Below is default value.
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
    return matches

"""
Takes in matches and applies ratio test
David Lowe's Ratio test 
http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20
returns good points
"""
def ratio(matches):
    good = []
    for m,n in matches:
        if m.distance < .75 * n.distance: #.8 for orb mebe?
            good.append([m])
    return good
 