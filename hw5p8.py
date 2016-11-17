import numpy as np
import matplotlib.pyplot as plt

def make_weights_for_fx():
    # this is a random f(x) line function. Creates the d=2 weight vector for f(x) from the random two points selected.

    #Returns: a tuple  2-items tuple representing  (intercept, slope)

    point1 = np.random.uniform(-1, 1, (1, 2))[0]
    print(point1)
    return point1

def create_random_points_and_labels(N, weights_fx)
    """Creates N random points in the space of [-1, 1] x [-1, 1] and then labels them based on the side if the line they
    land on based on make_weights_for_fx"""

    Args:
        N: Number of random points
        weights_fx: tuple representing (intercept, slope) of f(x)


