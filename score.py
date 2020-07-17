

def basic(kp1, kp2, good):
    """
    determines score based on number of good points
    """
    return good

# cutoff function should be in here and not roi.py
# because roi shouldnt have to account for different
# scoring methods, and differente cutoffs require
# different inputs and processing for scoring, which
# is score.py's job
def basic_cutoff():
    # TODO: finish this, but first finish roi
    return

# we can build fancy clustering scoring methods here
# unsure if the make skeleton code would also be here
# would probably need some type of class?
# TODO: read those papers