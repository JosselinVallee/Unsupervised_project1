import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import defaultdict
from hashable import *

pd.options.mode.chained_assignment = None  # default='warn'

def kohonen(max_iter=5, init_eta=0.1, movment_limit=0.1, init_sigma=3.0,
                 Wcmmean=2000,
                 Wcmstd=250,
                 learning_rate_fun=lambda x: x,
                 sigma_fun=lambda x, y: x,
                 data=None, labels=None, targetdigits=None, size_k=6):


    dim = 28*28
    data_range = 255.0

    # load in data and labels
    if data is None:
        data = np.array(np.loadtxt('data.txt'))
    if labels is None:
        labels = np.loadtxt('labels.txt')

    # select 4 digits
    if targetdigits is None:
        name = 'Josselin Vallée'
        targetdigits = name2digits(name) # assign the four digits that should be used

    print(targetdigits) # output the digits that were selected
    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
    # select all the labels correponding to targetdigits
    labels = labels[np.logical_or.reduce([labels==x for x in targetdigits])]

    dy, dx = data.shape

    #set the size of the Kohonen map. In this case it will be 6 X 6
    #size_k = 6

    #set the width of the neighborhood via the width of the gaussian that
    #describes it
    sigma = init_sigma

    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range

    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))

    #set the learning rate
    eta = init_eta

    #set the maximal iteration count
    tmax = max_iter

    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)

    # initialize previous center position
    prev_centers = np.copy(centers)

    c_m_array = np.zeros((len(centers), tmax))
    c_m_mean_array = np.zeros((len(centers), tmax))
    c_m_std_array = np.zeros((len(centers), tmax))
    cmmeanlen = 0
    cmstdlen = 0


    #Initialisation of the eta and
    eta_array = [(0, eta)]
    sigma_array = [(0, sigma)]

    do_stop = False
    T = -1
    reset_count = True
    do_stop_count = 0

    for t, i in enumerate(i_random):

        T += 1
        #Copy the previous center position
        prev_centers = np.copy(centers)

        #Perform one step of the SOM
        som_step(centers, data[i,:],neighbor,eta,sigma)

        # update learning rate
        eta = learning_rate_fun(eta)
        if eta != eta_array[-1][1]:
            eta_array.append((t, eta))

        #update sigma
        sigma = sigma_fun(sigma, eta)
        if sigma != sigma_array[-1][1]:
            sigma_array.append((t, sigma))

        # center movement
        diff_centers = centers - prev_centers
        norm_movments = np.linalg.norm(diff_centers, axis=1)
        c_m_array[:, t] = norm_movments

        #Window size
        if t > Wcmmean:
            c_m_mean_array[:, cmmeanlen] = np.mean(c_m_array[:, (t-Wcmmean):t], axis=1)
            cmmeanlen += 1

        reset_count = True
        if cmmeanlen > Wcmstd:
            stds = np.std(c_m_mean_array[:, (cmmeanlen-Wcmstd):cmmeanlen], axis=1)
            reset_count = np.max(stds) > movment_limit
            c_m_std_array[:, cmstdlen] = stds
            cmstdlen += 1

        if reset_count:
            do_stop_count = 0
        else:
            do_stop_count += 1

        if do_stop_count > 2000:
            do_stop = True

        # show stuff
        if t % 1000 == 998:
            print("\n", t, end="")
            print(" > sigma:", sigma, end="")
            print(" > eta:", eta, end="")

        if do_stop:
            print("\n converged at iteration", t)
            break

    print("Total iterations: ", T)

    print(label_digits(data, labels, centers, targetdigits, size_k))

    # for visualization
    plt.figure("centers"+str(eta))
    tle = "eta:"+str(eta)
    plt.title(tle)
    for i in range(size_k**2):
        plt.subplot(size_k,size_k,i+1)
        plt.axis('off')
        plt.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')

    plt.show()
    plt.draw()

    return {}

def label_digits(data, labels, centers, targetdigits, size_k):
    """ Assigns labels to the Kohonen map by looking for the closest
        data point from each center and assigning its label to it.
    """
    # Initialize indexes and digits
    closest = 0
    digits = []
    # iterate over size of kohonen map
    for i in range(size_k**2):
            #find the index of the closest point to ith center
            closest = np.argmin(np.sum((data[:] - centers[i, :])**2, axis=1))
            digits.append(labels[closest])
    #Create a size_k*size_k matrix
    return np.reshape(digits, (size_k, size_k))

def center_position(points):
    c = np.mean(points, axis=0)
    assert len(c) == len(points[0])
    return c

def som_step(centers,data,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a
    self-organized map (SOM).

      centers = som_step(centers,data,neighbor,eta,sigma)

      Input and output arguments:
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """

    size_k = int(np.sqrt(len(centers)))

    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))

    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)

    # update all units
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights
        centers[j,:] += disc * eta * (data - centers[j,:])

def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.

     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """

    name = name.lower()

    if len(name)>25:
        name = name[0:25]

    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]

    n = len(name)

    s = 0.0

    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = np.mod(s,x.shape[0])

    return np.sort(x[t,:])

def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))

if __name__ == "__main__":
    # load in data and labels
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')

    kohonen(max_iter=100000, init_eta=1, init_sigma=4.0,
                 movment_limit=0.01,
                 Wcmmean=2000,
                 Wcmstd=1500,
                 sigma_fun=lambda s, e: s-0.0001 if s > 2 else s*0.9999,
                 learning_rate_fun=lambda e: e-0.0001 if e > 0.5 else e*0.9999,
                 data=data, labels=labels)
