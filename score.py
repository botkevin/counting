
# TODO: add master methods for all the cutoffs

def basic(good):
    """
    determines score based on number of good points
    """
    return len(good)

# cutoff function should be in here and not roi.py
# because roi shouldnt have to account for different
# scoring methods, and differente cutoffs require
# different inputs and processing for scoring, which
# is score.py's job
def basic_cutoff(rois, scoring_cutoff):
    """basic scoring cutoff functions

    Parameters
    ----------
    rois : [(box,good), ...]
        box is the ROI given by selective search
        good are the points found with detect_organized
    scoring_cutoff : int

    Returns
    -------
    array
        array of indexes in rois that pass the cutoff
    """
    idxs = []
    for i in range(len(rois)):
        good = rois[i][1]
        score = basic(good)
        if score >= scoring_cutoff:
            idxs.append(i)
    return idxs

# we can build fancy clustering scoring methods here
# unsure if the make skeleton code would also be here
# would probably need some type of class?
# TODO: read those papers