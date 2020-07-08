import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils

# Detect Keypoints
# Help: https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
#       https://achuwilson.wordpress.com/2011/08/05/object-detection-using-surf-in-opencv-part-1/
#       https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
def detect(img1, img2, method):
    # Using ORB here but we can also use surf/sift
    dec = None
    if method == "orb":
        dec = cv2.ORB_create()
    elif method == "surf":
        dec = cv2.xfeatures2d.SURF_create(400)
    elif method == "sift":
        dec = cv2.xfeatures2d.SIFT_create()
    else:
        raise Exception("use surf, sift, or orb")
    # find the keypoints and descriptors with ORB
    kp1, des1 = dec.detectAndCompute(img1,None)
    kp2, des2 = dec.detectAndCompute(img2,None)
    
#     # If using FLANN based matcher, below are ORB recommended
#     # parameters as per the FLANN doc.
#     FLANN_INDEX_LSH = 6
#     index_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
#     # This is FLANN # of times recursive search 
#     # Higher number is more accurate at the cost
#     # of more computation. Below is default value.
#     search_params = dict(checks=50) 
#     flann = cv2.FlannBasedMatcher(index_params,search_params)
#     matches = flann.knnMatch(des1,des2,k=2)

    # create BFMatcher object
    # NORM_HAMMING for ORB, NORM_L2(default) for SURF
    bf = None
    if method == "orb":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # David Lowe's Ratio test 
    # http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20
#     good = []
#     for points in matches:
#         prev_dist = 0
#         prev = None
#         for match in points:
#             if prev_dist < 0.75*match.distance:
#                 if prev != None:
#                     good.append([prev])
#             else:
#                 break
#         # if everything is close enough in distance,
#         # that means none of the matches are good.
#         good = []

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < .75 * n.distance: #.8 for orb
            good.append([m])

            
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, outImg = None, flags=2)
    show_img = imutils.resize(img3, width=1000) 
    cv2.imshow('image',show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect(cv2.imread('./images/screen.jpg',0),
       cv2.imread('./images/find_the_screen.jpg',0),
       'surf')
