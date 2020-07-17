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
    boxes = selective_search(image, mode='fast')
    return boxes

def check_roi(master_img, search_img, method):
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

    Returns
    -------
    [(box,good), ...]
        box is the ROI given by selective search
        good are the points found with detect_organized
    """
    # TODO: add jobs/multithreading
    # just use surf as method for now

    dec = det.detector(method)
    kp_master, des_master = det.detect(dec, master_img)
    print("s_search start")
    boxes = s_search(search_img)
    print("s_search end")
    rois = []
    for box in boxes:
        mask = make_mask(search_img.shape, box)
        kp_child, des_child = det.detect(dec, search_img, mask)
        matches = det.match(des_master, des_child, "FLANN", method)
        good = det.ratio(matches)
        rois.append((box,good))
    return rois

def roi_prune(rois, scoring_method):
    # TODO: finish this
    return
    
def make_mask(img_shape, rectangle):
    x1 = rectangle[0]
    y1 = rectangle[1]
    x2 = rectangle[2]
    y2 = rectangle[3]

    mask = np.zeros(img_shape, np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

    
