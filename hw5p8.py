import numpy as np
import matplotlib.pyplot as plt

def make_weights_for_fx():
    # this is a random f(x) line function. Creates the d=2 weight vector for f(x) from the random two points selected.

    #Returns: a tuple  2-items tuple representing  (intercept, slope)

    point1 = np.random.uniform(-1, 1, (1, 2))[0]
    point2 = np.random.uniform(-1, 1, (1, 2))[0]
    w1 = (point2 - point1)[1] / (point2 - point1)[0]
    w0 = point1[1] - w1 * point1[0]
    w_fx = np.array((w0, w1))

    print(point1)
    return w_fx

w_3dim_to_2dim = lambda w_3dim: -(w_3dim/w_3dim[2])[:2]
w_2dim_to_3_dim = lambda w_2dim: np.hstack((-w_2dim, 1))

def create_random_points_and_labels(N, weights_fx):
    """Creates N random points in the space of [-1, 1] x [-1, 1] and then labels them based on the side if the line they
    land on based on make_weights_for_fx

    Args:
        N: Number of random points
        weights_fx: tuple representing (intercept, slope) of f(x)

    Returns:
        A tuple of the following items:
        X: an NxD array representing the coordinates of the points
        Y: an Nx1 array representing +1 or -1 depending on which side of the points land in relation to f_x

        """

    X = np.hstack((np.ones(N,1), np.random.uniform(-1, 1, (N,2)))) #hstack stacks arrays in sequence horizontally, then
                                                                    #ones creates an Nx1 matrix. Then we have a random
                                                                    # uniform matrix between -1,1 size Nx2.

    w_fx = w_2dim_to_3dim(weights_fx)
    score = np.dot(X, w_fx)

    Y = np.where(np.exp(score) / (1 + np.exp(score)) > 0.5, 1, -1)#returns 1 where >0.5, -1 otherwise
    Y = np.expand_dims(Y,1) #changing array into a vector for calculations to work

    return(X, Y, w_fx)

def plot_points_and_lines(weights_fx, X, Y, color="blue", label="f(x)"):
    """Creates the plot of f(x) and g(x) lines along with labeled points

        Args:
            weights_fx: tuple representing (intercept, slope) of f(x)
            weights_gx: tuple representing (intercept, slope) of g(x)
            X: an NxD array representing the coordinates of the points
            Y: an Nx1 array representing +1 or -1

            """

    line = np.linspace(-1,1, 1001) # points from 0 to 1000
    plt.plot(line, line * weights_fx[1] + weights_fx[0], label=label, color=color) #makes the f(x) line

plt.ylim(-1,1)
plt.xlim(-1,1)

plt.scatter(X[:,1][Y.ravel()==1], X[:,2][Y.ravel()==1], marker="+", c=("r"), label="+")
plt.scatter(X[:,1][Y.ravel()==-1], X[:,2][Y.ravel()==1], marker="_", c=("b"), label="-")

w_fx = make_weights_for_fx()
X, Y, w, create_random_points_and_labels(100, w_fx)
plot_points_and_lines(w_fx, X, Y)



