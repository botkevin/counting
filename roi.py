import numpy as np
import skimage.io
import cv2
from matplotlib import pyplot as plt
from selective_search import selective_search
import detect


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
    dec = detect_organized.detector(method)
    kp_master, des_master = detect_organized.detect(dec, master_img)

    
