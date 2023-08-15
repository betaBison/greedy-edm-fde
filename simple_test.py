import numpy as np
import cvxpy as cp

# number of measurements
n = 4
# assert n >= 4

# cvxpy variables
D = cp.Variable((n+1,n+1),symmetric=True)
rb = cp.Variable(1,)

# create random symmetric matrix with zeros along diagonal
s_x = np.random.random((3,n))
S = s_x.T @ s_x
np.fill_diagonal(S,0)

ranges = np.random.random(n)

prob = cp.Problem(
                  # cp.Minimize(),
                  cp.Minimize(cp.lambda_sum_largest(D,5)),
                  [
                   cp.diag(D) == 0.,
                   D[1:,1:] == S, # bottom right corner of matrix is known
                   D[1:,0] == cp.power(ranges+rb,2),
                   D[0,1:] == cp.power(ranges+rb,2),
                  ]
                  )
prob.solve()
print("The optimal value is", prob.value)
print("A solution D is:\n",D.value)
print("A solution rb is",rb.value)
