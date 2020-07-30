import cv2
import numpy as np

def idx_trim(rois, idxs):
    rois = [rois[i] for i in idxs]
    return rois

def homography(kp1, kp2, img1_shape, good):
    good_n = [a[0] for a in good if a] 
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_n ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_n ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1_shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    # M is just used for perspective transform. I dont think we ever need it
    return dst, M, matchesMask

def homography_all(kp_master, img1, img2, rois):
    """
    returns appends homography box and matchesMask onto rois for each roii
    Changes roi in place

    Returns
    -------
    roi : [[box, kp_child, good, dst, matchesMask], ...]
        appends to roii in place and returns roi
        dst : [[[int32, int32]], ...x4]
            array of 4 points that make the homography box
        matchesMask : mask of the homography box
    """
    img2 = img2.copy()
    homography_boxes = [] # return value
    for i in range(len(rois)):
        roii = rois[i]
        box = roii[0]
        
        kp_child = roii[1]
        good = roii[2]
        dst, _, matchesMask = homography(kp_master, kp_child, img1.shape[:2], good)
        roii.append(np.int32(dst))
        roii.append(matchesMask)
    return rois

# TODO: add score
def nms(boxes, thresh):
    """Non-Maximum Suppression
        Given overlapping bounding boxes and a tuneable threshold
        finds correct bounding boxes
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Parameters
    ----------
    boxes : [(x1,x2,y1,y2), ...]
        (x1,x2,y1,y2) - bounding coordinates
    thresh : float
        percentage of overlap threshold to bound on

    Returns
    -------
    boxes : [(x1,x2,y1,y2), ...]
        bounding coordinates without overlap
    """
    if len(boxes) == 0:
        return boxes
    
    # we need floats for division
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	to return
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # TODO: sort this based on score
    # NOTE: if score is naive score, this will work poorly
    #        because the largest bounding box will eat everything
    # idxs = np.argsort(y2)
    # try sorthing this based on smallest first
    idxs = np.argsort(area)[::-1]

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        # vectorized code; returns a list
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > thresh)[0])))

    return boxes[pick].astype("int")