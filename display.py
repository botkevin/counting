import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
displays feature matches in various ways thru pyplot
Should really only use the show() method in this file
flags:
 - 0: matches only
 - 1: matches w/ homography
 - 2: homography only
"""
def show(kp1, kp2, img1, img2, good, flag=0):
    """displays feature matches in various ways thru pyplot

    Parameters
    ----------
    kp1 : keypoints 
        keypoints of img1
    kp2 : 
        see above
    img1 : image matrix
        first image
    img2 : 
        see above
    good : [[m], ...]
        good matches -> m
    flag : int, optional
        display flag
         - 0: matches only
         - 1: matches w/ homography
         - 2: homography only, 
        by default 0
    """
    # params = dict(kp1=kp1, kp2=kp2, img1=img1, img2=img2, good=good)
    flag_functions={
        0 : _matches_d,
        1 : _homography_d,
        2 : _homography_nm_d
    }
    flag_functions.get(flag)(kp1, kp2, img1.copy(), img2.copy(), good)

def show_multiple(kp1, kp2, img1, img2, good, flag=0):
    # TODO: make this into a parent function
    flag_functions={
        0 : temp
    }
    raise NotImplementedError
    return

# a box full of matches
# img1 is master, img2 is child
def matchbox(kp_master, img1, img2, rois, idxs, n=-1):
    rois = [rois[i] for i in idxs]
    if n == -1:
        n = len(rois)

    img2 = img2.copy()
    for roii in rois[:n]:
        box = roii[0]
        start_point = box[:2] 
        end_point = box[2:]
        color = (255, 0, 0) 
        thickness = 1
        box_img = cv2.rectangle(img2.copy(), start_point, end_point, color, thickness)
        
        kp_child = roii[1]
        good = roii[2]
        img3 = cv2.drawMatchesKnn(img1, kp_master, box_img, kp_child, good,
                                  outImg=None, flags=2)
        plt.imshow(img3),plt.show()
    

def just_boxes_r(rois, img, idxs):
    just_boxes([rois[i][0] for i in idxs], img)

def just_boxes(boxes, img):
    image = img.copy()
    for box in boxes:
        start_point = box[:2] 
        end_point = box[2:]
        color = (255, 0, 0) 
        thickness = 1
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    plt.imshow(image),plt.show()

def _matches_d(kp1, kp2, img1, img2, good):
    """
    shows matches only
    """
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,outImg=None,flags=2)
    plt.imshow(img3, 'gray'),plt.show()


def _homography_d(kp1, kp2, img1, img2, good, show_matches=True):
    """
    shows homography bound and matches (dependent on show_matches)
    """
    # if statment is for crosscheck
    # crosscheck will return empty lists for nonmatched terms
    good = [m[0] for m in good if m] 
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    if not show_matches:
        matchesMask = [0]*len(matchesMask)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()

def _homography_nm_d(kp1, kp2, img1, img2, good):
    """
    shows only homography bound
    """
    _homography_d(kp1, kp2, img1, img2, good, show_matches=False)