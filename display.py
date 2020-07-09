import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
displays feature matches in various ways thru pyplot
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
        0 : matches_d,
        1 : homography_d,
        2 : homography_nm_d
    }
    flag_functions.get(flag)(kp1, kp2, img1, img2, good)


def matches_d(kp1, kp2, img1, img2, good):
    """
    shows matches only
    """
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,outImg=None,flags=2)
    plt.imshow(img3, 'gray'),plt.show()


def homography_d(kp1, kp2, img1, img2, good, show_matches=True):
    """
    shows homography bound and matches (dependent on show_matches)
    """
    good = [m[0] for m in good]
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

def homography_nm_d(kp1, kp2, img1, img2, good):
    """
    shows only homography bound
    """
    homography_d(kp1, kp2, img1, img2, good, show_matches=False)