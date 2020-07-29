
# TODO: add master methods for all the cutoffs

def basic(good):
    """
    determines score based on number of good points
    """
    # take out empty arrays
    # crosscheck returns empty arrays for nonmatched descriptors
    good_pruned = [m for m in good if m] 
    return len(good_pruned)

# cutoff function should be in here and not roi.py
# because roi shouldnt have to account for different
# scoring methods, and differente cutoffs require
# different inputs and processing for scoring, which
# is score.py's job
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
        good = rois[i][2]
        score = basic(good)
        if score >= scoring_cutoff:
            idxs.append(i)
    return idxs

# we can build fancy clustering scoring methods here
# unsure if the make skeleton code would also be here
# would probably need some type of class?
# TODO: read those papers