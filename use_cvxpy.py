"""Test out cvxpy method?

"""

__authors__ = "D. Knowles"
__date__ = "14 Sep 2023"

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from gnss_lib_py.algorithms.fde import _edm_from_satellites_ranges


def main():

    # number of satellites
    n_sats = 10

    S, ranges, noise = initialize(n_sats)

    D = _edm_from_satellites_ranges(S, ranges)


    x = cp.Variable((n_sats+1,n_sats+1),symmetric=True)
    # x = cp.Variable(1)

    # print(get_gram(D))

    G, norm_factor = get_gram(D)
    print("original G",G[0,1:])

    # X = one_hot*x + D
    orig_value = cp.lambda_sum_largest(G,3).value
    print("orig value:",orig_value)

    prob = cp.Problem(cp.Minimize(cp.lambda_sum_largest(x,5)),
                      [
                       x[1:,1:] == G[1:,1:],
                       # x[0,2:] == D[0,2:],
                       # x[2:,0] == D[2:,0],
                       x[0,0] == 0,
                      ] +
                      [cp.abs(x[0,i]-G[0,i]) <= 0.01 for i in range(1,n_sats+1)]
                      )
    prob.solve()
    print("\nMinimize:")
    print("The optimal value is", prob.value)
    print("new first three:",cp.lambda_sum_largest(G,3).value)
    print("new first three:",cp.lambda_sum_largest(x,3).value)
    print("A solution x is",x.value[0,1:])

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
    G = -0.5*J.dot(D).dot(J)
    norm_factor = np.max(G)
    G /= norm_factor

    return G, norm_factor

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
    # np.random.seed(10)

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



if __name__ == "__main__":
    main()
