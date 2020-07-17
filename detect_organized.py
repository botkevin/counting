import cv2
import warnings

def detector(method):
    """Sets up a detector based on a method

    Parameters
    ----------
    method : string
        "orb","surf","sift"

    Returns
    -------
    dec
        feture detector using [method]
    """    
    dec = None
    if method == "orb":
        dec = cv2.ORB_create()
    elif method == "surf":
        dec = cv2.xfeatures2d.SURF_create(400)
    elif method == "sift":
        dec = cv2.xfeatures2d.SIFT_create()
    else:
        raise Exception("use surf, sift, or orb")
    return dec

def detect(dec, img, mask=None):
    """Takes in a detector and image
        returns keypoints and descriptors

    Parameters
    ----------
    dec : cv::Feature2D class
    img : image matrix

    Returns
    -------
    kp : cv2 keypoints
    des : cv2 descriptors
    """    
    # find the keypoints and descriptors
    kp, des = dec.detectAndCompute(img, mask)
    return kp, des
   
def match(des1, des2, modus, method):
    """Takes in descriptors
        returns 2 best matches using FLANN
        or brute force NN

    Parameters
    ----------
    des1 : descriptor
        descriptor of image 1
    des2 : 
        see above
    modus : string
        method to find NN ("bf", "FLANN")
    method : string
        "orb","surf","sift"

    Returns
    -------
    [(m,n), ...]
        list of 2 best matches int tuple form
    """
    # if there are less than 2 descriptors
    # then NN will fail to find 2 neighbors
    # really shouldnt return anything anyways

    if len(des1)<=2:
        # If the source image only has 2 descriptors its
        # probably the wrong image
        warnings.warn("Source image only has 2 descriptors.")
        return []
    # getting none from roi. Let it fall thru but log it
    if des2 is None:
        return []
    if len(des2)<=2:
        return []


    
    matches = None
    if modus == "bf":
        # NORM_HAMMING for ORB, NORM_L2(default) for SURF/else
        bf = None
        if method == "orb":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

    elif modus == "FLANN":
        index_params, search_params = None, None
        if method == "orb":
            FLANN_INDEX_LSH = 6
            index_params = dict (algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
        else:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
        # This is FLANN # of times recursive search 
        # Higher number is more accurate at the cost
        # of more computation. Below is default value.
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
    return matches

def ratio(matches):
    """
    Takes in matches and applies ratio test
    David Lowe's Ratio test 
    http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20
    returns good matches
    """
    # matches empty
    if not matches:
        return []

    good = []
    for m,n in matches:
        if m.distance < .75 * n.distance: #.8 for orb mebe?
            good.append([m])
    return good
 