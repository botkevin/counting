import trim

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""displays feature matches in various ways thru pyplot.
"""

def show(kp1, kp2, img1, img2, good, flag=0, show=True):
    """ returns and displays feature matches
        in various ways thru pyplot

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
    show : Bool, optional
        use plt to show the image or not
        True because of legacy reasons, 
        should really false for most applications

    Returns
    -------
    image matrix
        image matrix of image to show
    """
    # params = dict(kp1=kp1, kp2=kp2, img1=img1, img2=img2, good=good)
    flag_functions={
        0 : _matches_d,
        1 : _homography_d,
        2 : _homography_nm_d
    }
    return flag_functions.get(flag)(kp1, kp2, img1.copy(), img2.copy(), good, show)

# a box full of matches
# img1 is master, img2 is child
def matchbox(kp_master, img1, img2, rois, n=-1, homography=False, mMask=False, show=True):
    """ shows each box from selective search and homography(optional)
        with respective matches individually for examination. 
        Used for testing purposes

    Parameters
    ----------
    n : int, optional
        number of boxes to show, by default -1
    homography : bool, optional
        flag to show homography, by default False
    mMask : bool, optional
        flag to apply matcheMask - shows only matches within homography, by default False
    """
    if n == -1:
        n = len(rois)

    images = [] # initialize images to return

    img2 = img2.copy()
    for i in range(n):
        roii = rois[i]
        box = roii[0]
        start_point = box[:2] 
        end_point = box[2:]
        color = (255, 0, 0) 
        thickness = 1
        
        kp_child = roii[1]
        good = roii[2]

        # crosscheck will return empty lists for nonmatched terms
        # need to prune for matchesMask to work. len matchesMask == len good_n
        good_n = [a[0] for a in good if a]

        # bounding box
        box_img = cv2.rectangle(img2.copy(), start_point, end_point, color, thickness)

        draw_params = dict(singlePointColor = None,
                        matchesMask = None,
                        flags = 2)

        if homography:
            dst, matchesMask = roii[3], roii[4]
            box_img = cv2.polylines(box_img,[np.int32(dst)],True,(0,0,255),1, cv2.LINE_AA)
            if mMask:
                # for some reason drawMatches does not like if I add or change
                # this dictionary, but making another one works.
                draw_params = dict(singlePointColor = None,
                        matchesMask = matchesMask,
                        flags = 2)

        img3 = cv2.drawMatches(img1, kp_master, box_img, kp_child, good_n,None,**draw_params)
        if show:
            plt.imshow(img3),plt.show()
        images.append(img3)

    return images

def just_boxes_r(rois, img, idxs, show=True):
    """
    given an array of rois and indexes, displays specific boxes on the image
    """
    return just_boxes([rois[i][0] for i in idxs], img, show)

def just_boxes(boxes, img, show=True):
    """
    given an array of boxes, displays them on the image
    """
    image = img.copy()
    for box in boxes:
        start_point = box[:2] 
        end_point = box[2:]
        color = (255, 0, 0) 
        thickness = 1
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    if show:
        plt.imshow(image),plt.show()
    return image

def homography_boxes(roi, img2, show=True):
    """displays all homography boxes on the image.

    Parameters
    ----------
    roi : [[box, kp_child, good, dst, matchesMask], ...]
        appends to roii in place and returns roi
        dst : [[[int32, int32]], ...x4]
            array of 4 points that make the homography box
        matchesMask : mask of the homography box
    """
    img2 = img2.copy()
    for roii in roi:
        dst = roii[3]
        img2 = cv2.polylines(img2,[dst],True,(0,0,255),1, cv2.LINE_AA)
    
    if show:
        plt.imshow(img2),plt.show()
    return img2

def _matches_d(kp1, kp2, img1, img2, good, show=True):
    """
    shows matches only
    """
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,outImg=None,flags=2)
    if show:
        plt.imshow(img3, 'gray'),plt.show()
    return img3


def _homography_d(kp1, kp2, img1, img2, good, show=True, show_matches=True):
    """
    shows homography bound and matches (dependent on show_matches)
    """
    # crosscheck will return empty lists for nonmatched terms
    # need to prune for matchesMask to work. len matchesMask == len good_n
    good_n = [a[0] for a in good if a] 

    dst, _, matchesMask = trim.homography(kp1, kp2, img1.shape, good)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # show matches
    if not show_matches:
        matchesMask = [0]*len(matchesMask)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_n,None,**draw_params)

    if show:
        plt.imshow(img3, 'gray'),plt.show()
    return img3

def _homography_nm_d(kp1, kp2, img1, img2, good, show=True):
    """
    shows only homography bound
    """
    return _homography_d(kp1, kp2, img1, img2, good, show, show_matches=False)