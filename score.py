import numpy as np

    """scoring methods and cutoff methods are in here
    """

# cutoff function should be in here and not roi.py
# because roi shouldnt have to account for different
# scoring methods, and differente cutoffs require
# different inputs and processing for scoring, which
# is score.py's job

def _basic_s(roii):
    """
    determines score based on number of good points
    """
    # take out empty arrays
    # crosscheck returns empty arrays for nonmatched descriptors
    good = roii[2]
    good_pruned = [m for m in good if m] 
    return len(good_pruned)


def basic_cutoff(rois, scoring_cutoff):
    """basic scoring cutoff functions

    Parameters
    ----------
    rois : [(box, kp_child, good), ...]
        kp_master: keypoints of master image
        box : ROI given by selective search that is 
              not empty of matches or too small
        kp_child : keypoints of child image
        good : points found with detect_organized
    scoring_cutoff : int

    Returns
    -------
    array
        array of indexes in rois that pass the cutoff
    """
    idxs = []
    for i in range(len(rois)):
        roii = rois[i]
        score = _basic_s(roii)
        if score >= scoring_cutoff:
            idxs.append(i)
    return idxs

def _angle(p1,p2,p3):
    """
    finds angle between p1 and p3 with p2 as the vertex 
    (lines from p1 -> p2 -> p3)
    Utilizes definition of vector dot product
    """
    v1 = [p1[0]-p2[0], p1[1]-p2[1]] # p1-p2
    v2 = [p3[0]-p2[0], p3[1]-p2[1]] # p3-p2
    num = np.dot(v1,v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(num/denom))

def angle_cutoff(rois, ang_thresh):
    """
    IMPORTANT: remember to use trim.idx_trim before you use this method
    cuts of angles 90 +- ang_thresh
    ang_thresh is in degrees
    """
    idxs = []
    for i in range(len(rois)):
        b = rois[i][3]
        ang = [0,0,0,0]
        ang[0] = _angle(b[0][0], b[1][0], b[2][0])
        ang[1] = _angle(b[1][0], b[2][0], b[3][0])
        ang[2] = _angle(b[2][0], b[3][0], b[0][0])
        ang[3] = 360 - ang[0] - ang[1] - ang[2]
        in_bounds = True
        for a in ang:
            if a > 90+ang_thresh or a < 90-ang_thresh:
                in_bounds = False
        if in_bounds:
            idxs.append(i)
    return idxs
    
def _mask_basic_s(roii):
    """
    score based on basic score, but only counts good points
    inside the homography mask
    """
    matchesMask = roii[4]
    score = 0
    for m in matchesMask:
        if m == 1:
            # matchesMask is 1 if in, 0 if not
            score += 1
    return score

# we can build fancy clustering scoring methods here
# unsure if the make skeleton code would also be here
# would probably need some type of class?
# TODO: read those papers