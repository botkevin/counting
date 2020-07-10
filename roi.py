import numpy as np
import skimage.io
import cv2
from matplotlib import pyplot as plt
from selective_search import selective_search
import detect_organized as det


def s_search(image):
    """ 
    Utilizes selective search to find the objects in the image
    Returns box and image
    """
    boxes = selective_search(image, mode='single')
    return boxes

"""
Finds features and descriptors(FAD) in master
Feeds search image into s_search; returned roi
Find FAD in roi
Given FAD, check score: score.py
score cutoff?
Put all of roi back into search_img parent
Non Maximum Suppression (NMS)
"""
def check_roi(master_img, search_img, method):
    # just use surf as method for now
    # initiate detector
    dec = det.detector(method)
    kp_master, des_master = det.detect(dec, master_img)
    
def make_mask(img_shape, rectangle): #TODO: add parameter rectangle
    # TODO: finish this broken thing
    x1 = rectangle[0]
    y1 = rectangle[1]
    x2 = rectangle[2]
    y2 = rectangle[3]

    mask = np.zeros(img_shape, np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

    
