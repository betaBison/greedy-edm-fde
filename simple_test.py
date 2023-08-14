import cvxpy as cp

D = cp.Variable((2,2),symmetric=True)
rb = cp.Variable(1,)
x = cp.Variable(1,)

prob = cp.Problem(
                  # cp.Minimize(),
                  cp.Minimize(cp.lambda_sum_largest(D,2)),
                  [
                   cp.diag(D) == 0.,
                   D[0,1] == cp.power(rb,2),
                   D[1,0] == cp.power(rb,2),
                  ]
                  )
prob.solve()
print("The optimal value is", prob.value)
print("A solution D is:\n",D.value)
print("A solution rb is",rb.value)
