"""Test out cvxpy method?

"""

__authors__ = "D. Knowles"
__date__ = "14 Sep 2023"

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def main():

    # number of satellites
    n_sats = 5

    S, ranges, noise = initialize(n_sats)

    rb = cp.Variable(1)
    D = cp.Variable((n_sats+1,n_sats+1),symmetric=True)


    # D = _edm_from_satellites_ranges(S, ranges, rb)


    # print(get_gram(D))

    # G, norm_factor = get_gram(D)
    # norm_factor = 1E14
    # G = get_gram(D)/norm_factor
    # print("original G",G[0,1:])
    #
    # # X = one_hot*x + D
    #
    # one_hot = np.array([0]+[1]*n_sats).reshape(-1,1) * np.array([1]+[0]*n_sats).reshape(1,-1) \
    #         + np.array([1]+[0]*n_sats).reshape(-1,1) * np.array([0]+[1]*n_sats).reshape(1,-1)

    prob = cp.Problem(
                      # cp.Minimize(cp.lambda_sum_largest(x,5)),
                      cp.Minimize(cp.sum(D**2)+rb),
                      [
                       cp.diag(D) == 0.,
                       D[1:,1:] == _edm(S),
                       # D[0,1] == cp.power(ranges[0]+rb),2),
                       D[0,2:] == cp.power((ranges[1:]),2),
                       D[1:,0] == cp.power((ranges),2),
                       D[0,1] == (ranges[0])**2,
                       cp.power(rb,2) == 2.,
                       # x[1:,1:] == G[1:,1:],
                       # x = cp.abs(x[0,1:]-G[0,1:]
                       # x[0,2:] == D[0,2:],
                       # x[2:,0] == D[2:,0],
                       # x[0,0] == 0,
                      ]
                      #+ [cp.abs(x[0,i]-G[0,i]) <= 1/norm_factor for i in range(1,n_sats+1)]
                      )
    prob.solve()
    print("\nMinimize:")
    print("The optimal value is", prob.value)
    print("A solution rb is",rb.value)
    print("A solution D is",D.value)

    print(np.linalg.norm(x[0,1:].value-G[0,1:]))
    print(np.linalg.norm(x[1:,0].value-G[1:,0]))

    print("diff:",recover_edm(norm_factor*G)[0,1:] - D[0,1:])
    print("real D",D[0,1:])
    print("recovered",recover_edm(norm_factor*G)[0,1:])
    print("cvxpy value",recover_edm(norm_factor*x.value)[0,1:])
    # print("real D",D[0,1:])

    # Plotting
    fig = plt.figure(figsize=(9.5,4))

    ax1 = plt.subplot(131)
    sing_scatter = ax1.scatter(range(D.shape[0]),sing_values(G))
    plt.yscale("log")

    ax2 = plt.subplot(132)
    sing_scatter = ax2.scatter(range(D.shape[0]),sing_values(x.value))
    plt.yscale("log")

    # singular values plot
    ax3 = plt.subplot(133)
    ax3.scatter(range(D.shape[0]-1),noise,label="true noise")
    ax3.scatter(range(D.shape[0]-1),recover_edm(norm_factor*x.value)[0,1:]-recover_edm(norm_factor*G)[0,1:],label="cvxpy noise")
    plt.legend()

    plt.show()

def get_gram(D):
    n = D.shape[0]

    I = np.eye(n)
    J = I - (1./n)*np.ones((n,n))
    # G = -0.5*J.dot(D).dot(J)
    G = -0.5*cp.quad_form(J,D)
    # norm_factor = np.max(G)
    # G /= norm_factor

    # return G, norm_factor
    return G

def sing_values(G):
    U, S, Vh = np.linalg.svd(G)

    return S

def recover_edm(G):
    n = G.shape[0]
    D = np.diag(G).reshape(-1,1).dot(np.ones((1,n))) \
        - 2.*G + np.ones((n,1)).dot(np.diag(G).reshape(1,-1))

    return D

def initialize(n_sats):
    """Initialize.

    Parameters
    ----------
    n_sats : int
        Number of satellites.

    """
    # np.random.seed(3)
    np.random.seed(10)

    # receiver positions
    R = np.array([[0.],
                  [0.],
                  [0.]])

    # dimensionality of euclidean space
    dims = 3

    # Define initial parameters
    init_fault = 10.

    fc_init = 1
    """fc_init : int initial fault count"""

    noise_std_init = 5.0
    """float : standard deviation of noise initially."""

    sat_indexes = np.arange(n_sats)
    np.random.shuffle(sat_indexes)
    fault_indexes = sat_indexes[:fc_init]

    # satellite positions
    S = np.zeros((dims,n_sats))
    S_direction = np.random.rand(dims,n_sats)-0.5
    # force z direction to be positive
    S_direction[2,:] = np.abs(S_direction[2,:])
    S_distance = np.random.normal(loc=23E6,scale=2E6,size=(n_sats,))
    # S_distance = np.random.normal(loc=50,scale=20,size=(n_sats,))

    # normalize to new distance
    for ns in range(n_sats):
        S[:,ns] = S_direction[:,ns] \
                * S_distance[ns]/np.linalg.norm(S_direction[:,ns])

    pranges = np.linalg.norm((S-R), axis = 0)
    """pseudoranges : np.ndarray"""
    noise = np.random.normal(loc=0, scale=noise_std_init, size=(n_sats,))
    # ranges = pranges
    ranges = pranges + noise
    # for ind in fault_indexes:
    #     ranges[ind] += init_fault

    return S, ranges, noise

def _edm(X):
    """Creates a Euclidean distance matrix (EDM) from point locations.

    See [1]_ for more explanation.

    Parameters
    ----------
    X : np.array
        Locations of points/nodes in the graph. Numpy array of shape
        state space dimensions x number of points in graph.

    Returns
    -------
    D : np.array
        Euclidean distance matrix as a numpy array of shape (n x n)
        where n is the number of points in the graph.
        creates edm from points

    References
    ----------
    ..  [1] I. Dokmanic, R. Parhizkar, J. Ranieri, and M. Vetterli.
        “Euclidean Distance Matrices: Essential Theory, Algorithms,
        and Applications.” 2015. arxiv.org/abs/1502.07541.

    """
    n = X.shape[1]
    G = (X.T).dot(X)
    D = np.diag(G).reshape(-1,1).dot(np.ones((1,n))) \
        - 2.*G + np.ones((n,1)).dot(np.diag(G).reshape(1,-1))
    return D

def _edm_from_satellites_ranges(S,ranges,rb):
    """Creates a Euclidean distance matrix (EDM) from points and ranges.

    Creates an EDM from a combination of known satellite positions as
    well as ranges from between the receiver and satellites.

    Parameters
    ----------
    S : np.array
        known locations of satellites packed as a numpy array in the
        shape state space dimensions x number of satellites.
    ranges : np.array
        ranges between the receiver and satellites packed as a numpy
        array in the shape 1 x number of satellites
    rb : cvxpy.Variable
        Receiver bias

    Returns
    -------
    D : np.array
        Euclidean distance matrix in the shape (1 + s) x (1 + s) where
        s is the number of satellites

    """
    num_s = S.shape[1]
    D = np.zeros((num_s+1,num_s+1))

    ranges = ranges - rb
    print(cp.power(ranges,2).shape)

    range_mat_col = cp.hstack([cp.reshape(cp.power(ranges,2),(num_s,1))]+[np.array([0]*num_s).reshape(-1,1) for i in range(num_s)])
    range_mat_col = cp.vstack([np.zeros((1,num_s+1)),range_mat_col])
    range_mat_row = cp.vstack([cp.power(ranges,2)]+[[0]*num_s for i in range(num_s)])
    range_mat_row = cp.hstack([np.zeros((num_s+1,1)),range_mat_row])


    print("row",range_mat_row.shape)
    print("col",range_mat_col.shape)

    # D[0,1:] = cp.power(ranges,2)
    # D[1:,0] = cp.power(ranges,2)
    D[1:,1:] = _edm(S)

    print(D.shape)

    D += range_mat_row + range_mat_col

    print("D here:",D.value)

    return D

if __name__ == "__main__":
    main()
