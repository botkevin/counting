import sys
sys.path.append("../")
import score

import numpy as np
import skimage.io
import cv2
from matplotlib import pyplot as plt
from selective_search import selective_search
import detect_organized as det

"""
Finds and returns regions of interest(roi) for further processing
If looking for definition of rois
---------
rois : [[box, kp_child, good, dst(?), matchesMask(?), angles(?)], ...]
    kp_master: keypoints of master image
    box : ROI given by selective search that is not empty of matches or too small
    kp_child : keypoints of child image
    good : points found with detect_organized. be careful because crosscheck leaves
            empty arrays for nonmatched descriptors, which may have to be filtered out
            See trim.homography and the homography related methods in display
    dst : [[[int32, int32]], ...x4]
        array of 4 points that make the homography box
    matchesMask : mask of the homography box
    angles : [int, ...x4]
        four angles of quadrilateral of dst

    IMPORTANT: dst, matchesMask will only appear after homography_all is called
               angles will only appear after calling angle_cutoff
    Trying to run functions that depend on these will raise error because they are NoneType
"""

def s_search(image, mode='fast'):
    """ 
    Utilizes selective search to find the objects in the image
    Returns boxes:(x1,y1,x2,y2)
    """
    # print("s_search start")
    boxes = selective_search(image, mode=mode)
    # print("s_search end")
    return boxes

def check_roi_good(master_img, search_img, boxes, method, ratio=.75, modus="FLANN", crosscheck=False):
    """Finds features and descriptors(FAD) in master
    Feeds search image into s_search; returned boxes
    Find FAD in roi
    Should use with crosscheck=True almost always

    Parameters
    ----------
    master_img : image matrix
        master image of object to find
    search_img : 
        target image with object
    method : string
        "orb","surf","sift"
    ratio : float
        ratio for lowe's ratio test
        DEFAULT = .75
    modus : string
        method to calculate nearest neighbors. "bf" or "FLANN"
        See detect_organized.match()
        DEFAULT = "FLANN"
    crosscheck : boolean
        crosscheck if true or ratio if false
        Crosscheck works much better than ratio
        DEFAULT = False

    Returns
    -------
    kp_master, [[box, kp_child, good], ...]
        kp_master: keypoints of master image
        box : ROI given by selective search that is 
              not empty of matches or too small
        kp_child : keypoints of child image
        good : points found with detect_organized
    """
    # TODO: add jobs/multithreading
    # just use surf as method for now

    dec = det.detector(method)
    kp_master, des_master = det.detect(dec, master_img)
    rois = []
    for box in boxes:  
        mask = _make_mask(search_img.shape, box)
        kp_child, des_child = det.detect(dec, search_img, mask)
        good = None
        if crosscheck:
            good = det.match_and_crosscheck(des_master, des_child, modus, method)
        else:
            # use ratio test
            matches = det.match(des_master, des_child, modus, method) 
            good = det.ratio(matches, ratio)
        if good: # list is not empty, we dont want empty matchboxes
            rois.append([box, kp_child, good, None, None, None])
    return kp_master, rois
    
def _make_mask(img_shape, rectangle):
    """
    simple method to make a mask to only use inside of rectangle
    """
    x1 = rectangle[0]
    y1 = rectangle[1]
    x2 = rectangle[2]
    y2 = rectangle[3]
    assert x2>x1
    assert y2>y1

    mask = np.zeros(img_shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

    
