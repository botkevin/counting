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
Finds features and descriptors(FAD) in master
Feeds search image into s_search; returned roi
Find FAD in roi
Given FAD, check score: score.py
score cutoff?
Put all of roi back into search_img parent
Non Maximum Suppression (NMS)
"""

def s_search(image):
    """ 
    Utilizes selective search to find the objects in the image
    Returns boxes:(x1,y1,x2,y2)
    """
    print("s_search start")
    boxes = selective_search(image, mode='single')
    print("s_search end")
    return boxes

def check_roi_good(master_img, search_img, boxes, method, ratio=.75, modus="FLANN", crosscheck=False):
    """Finds features and descriptors(FAD) in master
    Feeds search image into s_search; returned boxes
    Find FAD in roi

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
        crosscheck or ratio
        DEFAULT = False

    Returns
    -------
    kp_master, [(box, kp_child, good), ...]
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
    print_i = 0
    for box in boxes:
        # print(print_i)
        # print_i+=1    
        mask = _make_mask(search_img.shape, box)
        kp_child, des_child = det.detect(dec, search_img, mask)
        # TEST: changing from "FLANN" to "bf"
        good = None
        if crosscheck:
            good = det.match_and_crosscheck(des_master, des_child, modus, method)
        else:
            matches = det.match(des_master, des_child, modus, method) 
            good = det.ratio(matches, ratio)
        if good: # list is not empty, we dont want empty matchboxes
            rois.append((box, kp_child, good))
        # else:
            # print(print_i)
        print_i+=1
    return kp_master, rois

def check_roi_all(search_img, method):
    # TODO: make this the roi for BOVW
    return

def roi_prune(rois, scoring_method, scoring_cutoff):
    # TODO: finish this
    return

# TEMP
def roi_prune_basic(rois, scoring_cutoff):
    # TODO: add nms
    return
    
def _make_mask(img_shape, rectangle):
    x1 = rectangle[0]
    y1 = rectangle[1]
    x2 = rectangle[2]
    y2 = rectangle[3]
    assert x2>x1
    assert y2>y1

    mask = np.zeros(img_shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

    
