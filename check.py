import numpy as np
import skimage.io
import cv2
from matplotlib import pyplot as plt
from selective_search import selective_search

"""
Utilizes selective search to find the objects in the image
Returns box and image
"""
def s_search(image):
    boxes = selective_search(image, mode='single')
    return boxes

def check_imgs(img1, img2):
    

    # Initiate detector 
    # FUT: add different detectors
    dec = cv2.SIFT()

